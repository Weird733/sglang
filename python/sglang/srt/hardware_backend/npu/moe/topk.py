from typing import TYPE_CHECKING, Optional

import torch
from sgl_kernel_npu.norm.l1_norm import l1_norm

from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location_dispatch import topk_ids_logical_to_physical
from sglang.srt.layers.moe.topk import (
    StandardTopKOutput,
    capture_routed_experts_if_allowed,
    select_experts,
)
from sglang.srt.utils import get_bool_env_var

# MoE 前段融合包（moe_front_fusion/v1，六轮单测定案）总开关：
# renorm=1 单算子路由 + v2.2 自写 init_routing。默认关，开启：
# SGLANG_MOE_FRONT_FUSION=1。仅作用于下方 fast path（无 group/无 bias）。
_moe_front_fusion = get_bool_env_var("SGLANG_MOE_FRONT_FUSION")

if TYPE_CHECKING:
    from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
    from sglang.srt.layers.moe.topk import TopKConfig, TopKOutput


def _apply_routed_scaling_after_renorm(
    topk_weights: torch.Tensor,
    topk_config: "TopKConfig",
) -> torch.Tensor:
    """Mirror GPU post-renorm scaling when apply_routed_scaling_factor_on_output is set."""
    if (
        topk_config.renormalize
        and topk_config.apply_routed_scaling_factor_on_output
        and topk_config.routed_scaling_factor is not None
    ):
        return topk_weights * topk_config.routed_scaling_factor
    return topk_weights


def fused_topk_npu(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    topk_config: "TopKConfig",
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional["ExpertLocationDispatchInfo"] = None,
    layer_id: Optional[int] = None,
) -> "TopKOutput":

    use_grouped_topk = topk_config.use_grouped_topk
    renormalize = topk_config.renormalize
    correction_bias = topk_config.correction_bias

    # Fast path: simple top-k without grouped routing and bias
    if not use_grouped_topk and correction_bias is None:
        if (
            _moe_front_fusion
            and renormalize
            and topk_config.num_fused_shared_experts == 0
        ):
            # renorm=1 单算子 = gating_top_k_softmax + l1_norm（六轮单测：
            # ids 与 stock 链逐位一致含构造并列对抗、weights 差 ≤3e-8，
            # 链路口径 -10.2µs/层）。
            # cast_elimination v1：bf16 直喂（删除原 router_logits.to(fp32) 的
            # host cast——aclnnMoeGatingTopK 契约支持 BF16 输入，kernel 内
            # CAST_NONE 精确扩张后 softmax 全程 fp32 计算，与 host 侧 fp32 化
            # 逐位等价 → ids 逐位不变；yOut 随输入变 bf16，kernel 内 CAST_RINT
            # 与 torch .to(bf16) 的 RNE 一致 → 下游 ascend_tp.py dispatch 的
            # .to(hidden_states.dtype) 变 no-op；净消 2 个 cast kernel/层）。
            # 出口不再 fp32 化（bf16 输出下该 .to(fp32) 会变成真 cast，负优化）；
            # deepep 线对 topk_weights 无 dtype 转换，其 fp32 契约由
            # deepep.py 两处 dispatch_a 的防御性 .to(fp32) 恢复（同包 hunk）。
            topk_weights, topk_ids, _ = torch.ops.npu.npu_moe_gating_top_k(
                router_logits,
                k=topk_config.top_k,
                bias=None,
                k_group=1,
                group_count=1,
                group_select_mode=0,
                renorm=1,
                norm_type=0,
                routed_scaling_factor=1.0,
                eps=float(1e-20),
            )
        else:
            topk_weights, topk_ids, _ = torch.ops.npu.npu_moe_gating_top_k_softmax(
                router_logits,
                k=topk_config.top_k,
            )

            if renormalize:
                topk_weights = l1_norm(
                    topk_weights
                    if topk_config.num_fused_shared_experts == 0
                    else topk_weights[:, :-1]
                )
            topk_weights = topk_weights.to(torch.float32)

    # sqrtsoftplus (DSV4 noaux_tc): the NPU op only scores sigmoid/softmax, so use
    # a torch path. top-k over (scores + bias); weights from un-biased scores.
    elif topk_config.scoring_func == "sqrtsoftplus":
        scores = torch.nn.functional.softplus(router_logits.float()).sqrt()
        scores_for_choice = (
            scores + correction_bias.unsqueeze(0).float()
            if correction_bias is not None
            else scores
        )
        _, topk_ids = torch.topk(
            scores_for_choice, k=topk_config.top_k, dim=-1, sorted=False
        )
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = scores.gather(1, topk_ids)
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        else:
            topk_weights = topk_weights * topk_config.routed_scaling_factor
        topk_weights = topk_weights.to(torch.float32)

    # Support grouped top-k or correction bias or sigmoid or routed_scaling_factor
    elif (
        correction_bias is not None
        or topk_config.scoring_func == "sigmoid"
        or num_token_non_padded is not None
    ):
        topk_weights, topk_ids, _ = torch.ops.npu.npu_moe_gating_top_k(
            router_logits.to(torch.float32),
            k=topk_config.top_k,
            bias=(
                correction_bias.to(torch.float32)
                if correction_bias is not None
                else None
            ),
            # num_expert_group and topk_group in some topk_config without group is None, (not supported by this ops)
            k_group=topk_config.topk_group if use_grouped_topk else 1,
            group_count=topk_config.num_expert_group if use_grouped_topk else 1,
            group_select_mode=(1 if use_grouped_topk else 0),
            renorm=0,
            # 1 for sigmoid, 0 for softmax
            norm_type=1,
            routed_scaling_factor=(
                topk_config.routed_scaling_factor
                if topk_config.apply_routed_scaling_factor_on_output
                else 1
            ),
            eps=float(1e-20),
        )
        topk_weights = topk_weights.to(torch.float32)

    # torch native is not yet supported num_token_non_padded
    # Fallback to torch native implementation
    else:
        topk_config.torch_native = True
        return select_experts(
            hidden_states=hidden_states,
            layer_id=layer_id,
            router_logits=router_logits,
            topk_config=topk_config,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )

    if expert_location_dispatch_info is not None:
        topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    get_global_expert_distribution_recorder().on_select_experts(topk_ids=topk_ids)
    capture_routed_experts_if_allowed(topk_config, layer_id, topk_ids)

    return StandardTopKOutput(topk_weights, topk_ids, router_logits)
