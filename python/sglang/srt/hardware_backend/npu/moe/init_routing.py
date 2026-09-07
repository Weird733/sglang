"""
NPU MoE init routing components.

Prepare token routing before expert computation. Two API versions are provided:
- v1: legacy routing using ``npu_moe_init_routing``.
- v2: improved routing using ``npu_moe_init_routing_v2``.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class BaseInitRouting(ABC):
    """Abstract base for NPU MoE init routing."""

    @abstractmethod
    def _init_routing(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: ...


class NPUMoEInitRouting_v1(BaseInitRouting):
    """
    NPU MoE init routing (v1 API).

    Uses ``npu_moe_init_routing`` with a manually constructed ``row_idx`` tensor.
    """

    def _init_routing(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        num_tokens = hidden_states.shape[0]
        row_idx_len = num_tokens * top_k
        row_idx = (
            torch.arange(0, row_idx_len, dtype=torch.int32, device=topk_ids.device)
            .view(topk_ids.shape[1], -1)
            .permute(1, 0)
            .contiguous()
        )

        hidden_states, expanded_row_idx, expanded_expert_idx = (
            torch.ops.npu.npu_moe_init_routing(
                hidden_states,
                row_idx=row_idx,
                expert_idx=topk_ids,
                active_num=num_tokens,
            )
        )
        expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(
            expanded_expert_idx, num_experts
        )
        expert_tokens = expert_tokens.to(torch.int64)
        return hidden_states, expanded_row_idx, expert_tokens, None


class NPUMoEInitRouting_v2(BaseInitRouting):
    """
    NPU MoE init routing (v2 API).

    Uses ``npu_moe_init_routing_v2``, which integrates expert token counting.
    """

    def __init__(self, quant_mode: int = -1):
        self.quant_mode = quant_mode

    def _init_routing(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        num_tokens = hidden_states.shape[0]
        hidden_states, expanded_row_idx, expert_tokens, pertoken_scale = (
            torch.ops.npu.npu_moe_init_routing_v2(
                hidden_states,
                topk_ids,
                active_num=num_tokens * top_k,
                expert_num=num_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[0, num_experts],
                quant_mode=self.quant_mode,
            )
        )
        if self.quant_mode == -1:
            pertoken_scale = None
        expert_tokens = expert_tokens.to(torch.int64)
        return hidden_states, expanded_row_idx, expert_tokens, pertoken_scale


class NPUMoEInitRouting_Quant(BaseInitRouting):
    """
    NPU MoE init routing (Quant API).

    Uses ``npu_moe_init_routing_quant``, which integrates expert token counting.
    """

    def _init_routing(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        num_tokens = hidden_states.shape[0]

        hidden_states, expanded_row_idx, expert_tokens, _, pertoken_scale = (
            torch.ops.npu.npu_moe_init_routing_quant(
                hidden_states,
                topk_ids,
                active_num=num_tokens * topk_ids.shape[1],
                expert_num=num_experts,
                expert_tokens_num_mode=1,
                expert_tokens_before_capacity_flag=False,
                quant_mode=1,
            )
        )
        expert_tokens = expert_tokens.to(torch.int64)
        return hidden_states, expanded_row_idx, expert_tokens, pertoken_scale


class NPUMoEInitRouting_v22(BaseInitRouting):
    """
    v2.2 自写 init routing（moe_front_fusion/v1，六轮单测 0 容差逐位验收）。

    走 ``sgl_kernel_npu.moe.moe_front_routing.moe_init_routing_v22``（Triton
    rank_hist + partials_cumsum + gather 三 launch），语义与
    ``npu_moe_init_routing_v2(type=1)`` 逐位一致，并原生产出 exclusive
    offsets（int32，``self.last_expert_offsets``）供 persistent GMM2
    ``offsets=`` 直用（省 _gmm2_offsets_kernel ~4.4µs/层）。

    仅支持 BF16 无量化路径（pertoken_scale=None）；形态门（decode 尺寸
    M=T*top_k ≤512 且 M%8==0、H 为 2 的幂、bf16）之外回退 stock v2 实现。
    """

    def __init__(self):
        self.last_expert_offsets: Optional[torch.Tensor] = None

    def _init_routing(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        self.last_expert_offsets = None
        num_tokens = hidden_states.shape[0]
        m = num_tokens * top_k
        h = hidden_states.shape[1]
        if (
            m <= 512
            and m % 8 == 0
            and (h & (h - 1)) == 0
            and hidden_states.dtype == torch.bfloat16
        ):
            from sgl_kernel_npu.moe.moe_front_routing import moe_init_routing_v22

            hidden_states, expanded_row_idx, expert_tokens, excl, _incl = (
                moe_init_routing_v22(
                    hidden_states, topk_ids, num_experts, top_k
                )
            )
            self.last_expert_offsets = excl
            return hidden_states, expanded_row_idx, expert_tokens, None

        # 形态门外（prefill 大尺寸 / 非验证 dtype）：回退 stock v2。
        hidden_states, expanded_row_idx, expert_tokens, pertoken_scale = (
            torch.ops.npu.npu_moe_init_routing_v2(
                hidden_states,
                topk_ids,
                active_num=m,
                expert_num=num_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[0, num_experts],
                quant_mode=-1,
            )
        )
        expert_tokens = expert_tokens.to(torch.int64)
        return hidden_states, expanded_row_idx, expert_tokens, None
