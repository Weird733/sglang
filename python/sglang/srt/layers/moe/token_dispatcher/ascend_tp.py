from __future__ import annotations

from typing import NamedTuple, Optional

import torch

from sglang.srt.hardware_backend.npu.moe.finalize_routing import (
    AllGatherFinalizeRoutingWrapper,
    NPUFinalizeRouting,
)
from sglang.srt.hardware_backend.npu.moe.init_routing import (
    NPUMoEInitRouting_v2,
    NPUMoEInitRouting_v22,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInputFormat,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import (
    DispatcherOutputDtype,
    get_ascend_dispatcher_output_dtype,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_bool_env_var

# MoE 前段融合包（moe_front_fusion/v1）总开关：BF16 路径用 v2.2 自写
# init_routing（0 容差逐位验收），并向后传 exclusive offsets 供
# persistent GMM2 直读。默认关，开启：SGLANG_MOE_FRONT_FUSION=1。
_moe_front_fusion = get_bool_env_var("SGLANG_MOE_FRONT_FUSION")


class AscendTPDispatchOutput(NamedTuple):
    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    expanded_row_idx: torch.Tensor
    expert_tokens: torch.Tensor
    group_list_type: int
    # v2.2 自写 init 原生 exclusive offsets（int32 [E]）；stock init 为 None
    expert_offsets: Optional[torch.Tensor] = None

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.ASCEND_TP


class AscendTPCombineInput(NamedTuple):
    hidden_states: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.ASCEND_TP


class AscendTPDispatcher(BaseDispatcher):
    def __init__(self, moe_runner_config: MoeRunnerConfig):
        super().__init__()
        self.num_experts = moe_runner_config.num_experts
        self.top_k = moe_runner_config.top_k
        self._dispatch_output: Optional[AscendTPDispatchOutput] = None

        self.quant_config: Optional[dict] = None

        # Initialise routing kernels with default (no quant config yet)
        self.set_ascend_dispatcher_output_dtype()

    def set_quant_config(self, quant_config: dict) -> None:
        self.quant_config = quant_config
        self.set_ascend_dispatcher_output_dtype()

        # If the quantisation is GGUF and TP is active, wrap the finalizer
        # with an all‑gather so that the dispatcher stays completely clean.
        if (
            isinstance(self.quant_config, dict)
            and self.quant_config.get("quant_type") == "gguf"
            and get_parallel().tp_size > 1
        ):
            self.finalize = AllGatherFinalizeRoutingWrapper(self.finalize, dim=-1)

    def set_ascend_dispatcher_output_dtype(self) -> None:
        """Choose init & finalize routing kernels based on quant config."""
        self.ascend_dispatcher_output_dtype = get_ascend_dispatcher_output_dtype(self)

        if self.ascend_dispatcher_output_dtype == DispatcherOutputDtype.BF16:
            if _moe_front_fusion:
                self.init = NPUMoEInitRouting_v22()
            else:
                self.init = NPUMoEInitRouting_v2(quant_mode=-1)
            self.finalize = NPUFinalizeRouting(drop_pad_mode=2)
            self.group_list_type = 1
        elif self.ascend_dispatcher_output_dtype == DispatcherOutputDtype.INT8:
            self.init = NPUMoEInitRouting_v2(quant_mode=1)
            self.finalize = NPUFinalizeRouting(drop_pad_mode=2)
            self.group_list_type = 1
        else:
            raise ValueError(
                f"Unsupported ascend_dispatcher_output_dtype: {self.ascend_dispatcher_output_dtype}"
            )

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> AscendTPDispatchOutput:
        topk_weights, topk_ids, _ = topk_output
        topk_weights = topk_weights.to(hidden_states.dtype)
        topk_ids = topk_ids.to(torch.int32)
        top_k = topk_weights.shape[-1]

        (
            permuted_hidden_states,
            expanded_row_idx,
            expert_tokens,
            hidden_states_scale,
        ) = self.init._init_routing(
            hidden_states,
            topk_ids,
            self.num_experts,
            top_k,
        )

        self._dispatch_output = AscendTPDispatchOutput(
            hidden_states=permuted_hidden_states,
            hidden_states_scale=hidden_states_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expanded_row_idx=expanded_row_idx,
            expert_tokens=expert_tokens,
            group_list_type=self.group_list_type,
            # v22 init 产出 exclusive offsets；stock init 无此属性 → None
            expert_offsets=getattr(self.init, "last_expert_offsets", None),
        )
        return self._dispatch_output

    def combine(self, combine_input: AscendTPCombineInput) -> torch.Tensor:
        if self._dispatch_output is None:
            raise RuntimeError("combine() called before dispatch()")

        dispatch_out = self._dispatch_output

        # The finalizer (possibly wrapped with TP all‑gather) does all the work.
        final_hidden_states = self.finalize._finalize_routing(
            combine_input.hidden_states,
            topk_weights=dispatch_out.topk_weights,
            expanded_row_idx=dispatch_out.expanded_row_idx,
            topk_ids=dispatch_out.topk_ids,
        )

        self._dispatch_output = None
        return final_hidden_states
