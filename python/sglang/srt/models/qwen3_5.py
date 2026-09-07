# Copyright 2025 Qwen Team
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Inference-only Qwen3.5 model and Qwen3.5 MoE model compatible with HuggingFace weights."""

import logging
import os
from functools import lru_cache
from typing import Iterable, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import triton

# Layers - Attention
from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm as RMSNormGated
from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkvzba_split_reshape_cat_contiguous,
)
from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

# Configs
from sglang.srt.configs.qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5MoeConfig,
    Qwen3_5TextConfig,
)

# Distributed
from sglang.srt.distributed import get_pp_group
from sglang.srt.environ import envs

# MoE 权重 L2 预取（v1.2，SGLANG_NPU_MOE_PREFETCH）：注册、本层 prepare_attn 后
# 自目标发射 w13(+w2) 预取、step 末 drain。
# 模块顶层不 import torch_npu，跨平台安全；未启用时为 no-op。
from sglang.srt.hardware_backend.npu.moe_weight_prefetch import (
    moe_prefetch_emit,
    moe_prefetch_register_model,
    moe_prefetch_step_drain,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.attention.mamba.mamba import mamba_v2_sharded_weight_loader
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import (
    is_dp_attention_enabled,
)

# Layers - Others
from sglang.srt.layers.layernorm import GemmaRMSNorm

# Layers - Linear
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.parameter import (
    BlockQuantScaleParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from sglang.srt.models.qwen2_moe import (
    Qwen2MoeMLP,
    Qwen2MoeSparseMoeBlock,
    can_fuse_shared_expert,
)

# Models
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.models.utils import (
    fused_qk_gemma_rmsnorm,
    fused_qk_gemma_rmsnorm_with_gate,
)
from sglang.srt.runtime_context import (
    get_forward,
    get_parallel,
    get_server_args,
    get_stream,
)

# Utils
from sglang.srt.utils import (
    LazyValue,
    add_prefix,
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_gfx95_supported,
    is_hip,
    is_npu,
    is_xpu,
    make_layers,
    set_weight_attrs,
    use_intel_amx_backend,
)
from sglang.srt.utils.hf_transformers_utils import get_processor, get_rope_config

logger = logging.getLogger(__name__)
_is_cuda = is_cuda()
_is_npu = is_npu()
_is_cpu = is_cpu()
_is_gfx95 = is_gfx95_supported()
_is_hip = is_hip()
_is_xpu = is_xpu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_hip_use_alt_stream = get_bool_env_var("SGLANG_ALT_STREAM") and _is_hip
_gdn_use_alt_stream = _is_cuda or (
    get_bool_env_var("SGLANG_GDN_QKVZ_BA_ALT_STREAM", "False") and _hip_use_alt_stream
)
_qknorm_use_alt_stream = _is_cuda or (
    get_bool_env_var("SGLANG_QK_NORM_ALT_STREAM", "False") and _hip_use_alt_stream
)
# GDN_QKVZBA_PACK（code/gdn_qkvzba_pack_v1_2 包）：NPU 上把 GDN 层 in_proj_qkvz /
# in_proj_ba 两个 MergedColumnParallelLinear 的权重沿 N（输出行）拼成单块
# [N_pack, K]（N_pack = N_qkvz+N_ba 向上补零行到 16 倍数，见
# _gdn_qkvzba_pack_materialize——补齐后打包输出的物理行距满足 op1
# fused_qkvzba_conv1d v2.1 strided 输入的 32B 对齐契约，qkvz/ba 两段以行距视图
# 直送 op1，零拷贝），一次 stock linear 取代两次 matmulV2。数学逐列等价——N 拼接
# 不改变任何输出列的 K 归约内容（pad 列输出无人消费）。默认开；=0 回退两次 GEMM
# 原链路。仅 NPU + 非量化 bf16/fp16 生效，其余形态由运行期守卫自动回退
# （_gdn_qkvzba_packable）。
_gdn_qkvzba_pack_env = get_bool_env_var("SGLANG_NPU_GDN_QKVZBA_PACK", "true")
# 打包仅在小 M 启用：首轮实测（exp 见包 src/all_ut_result.txt）大 M 下宽 N
# （如 3088，非 256 tile 整数倍）的 matmulV2 ragged tiling 惩罚超过省下的 ba
# GEMM——M=1024 回退 +12~28%；M<=256（decode 全域）图模式稳定领先 3~5µs/层。
_gdn_qkvzba_pack_max_m = int(os.getenv("SGLANG_NPU_GDN_QKVZBA_PACK_MAX_M", "256"))
_is_amx_available = cpu_has_amx_support()

cached_get_processor = lru_cache(get_processor)


def _disable_shared_experts_fusion() -> bool:
    # Resolved lazily: the global server args is not set at module import time
    # (e.g. when this module is imported by unit tests).
    return get_server_args().disable_shared_experts_fusion


if _is_cuda:
    from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
        fused_qk_gemma_rmsnorm_rope_gate,
    )

if _is_cpu:
    fused_sigmoid_mul = torch.ops.sgl_kernel.fused_sigmoid_mul_cpu
    fused_qk_gemma_rmsnorm = torch.ops.sgl_kernel.fused_qk_gemma_rmsnorm_cpu
    fused_qk_gemma_rmsnorm_with_gate = (
        torch.ops.sgl_kernel.fused_qk_gemma_rmsnorm_with_gate_cpu
    )
    fused_qkvzba_split_reshape_cat_contiguous = (
        torch.ops.sgl_kernel.fused_qkvzba_split_reshape_cat_contiguous_cpu
    )


@lru_cache(maxsize=1)
def _enable_qwen35_fused_ar_quant() -> bool:
    """Gate the fused AR+RMSNorm+per-group-FP8-quant path for Qwen3.5.

    The single-kernel backend is ROCm/aiter/gfx95-only. The model gate stays
    tied to ROCm/aiter so non-gfx95 HIP can keep the existing 2-kernel fallback
    behavior for tuple handoff when this branch is used. It replaces the
    existing ``--enable-aiter-allreduce-fusion`` 3-kernel path
    (AR → RMSNorm → per-group quant) with either a single fused kernel (when
    the fully-fused variant is eligible) or a 2-kernel path
    (fused AR+RMSNorm + separate per-group quant) that still saves one
    kernel launch vs. baseline. The LayerCommunicator gracefully falls back
    to ``forward_with_allreduce_fusion`` (plain AR+RMSNorm) when the fused
    quant helper returns ``None``, so turning this on never regresses the
    AR+RMSNorm fusion itself.

    Opt-out: set ``SGLANG_DISABLE_FUSED_AR_QUANT=1`` to fall back to the
    unmodified AR+RMSNorm fusion path.
    """
    if not _use_aiter:
        return False
    if get_bool_env_var("SGLANG_DISABLE_FUSED_AR_QUANT", default="false"):
        return False
    return bool(get_server_args().enable_aiter_allreduce_fusion)


def _linear_accepts_fp8_tuple(linear: nn.Module) -> bool:
    quant_method = getattr(linear, "quant_method", None)
    return quant_method.__class__.__name__ == "Fp8LinearMethod" and (
        getattr(quant_method, "block_quant", False)
        or getattr(quant_method, "use_mxfp8", False)
    )


def _select_fused_ar_input_for_linear(hidden_states, linear: nn.Module):
    if not isinstance(hidden_states, tuple):
        return hidden_states
    if len(hidden_states) == 3:
        hs_bf16, hs_fp8, hs_scale = hidden_states
        if _linear_accepts_fp8_tuple(linear):
            return (hs_fp8, hs_scale)
        return hs_bf16
    if len(hidden_states) == 2 and _linear_accepts_fp8_tuple(linear):
        return hidden_states
    raise TypeError(
        f"{linear.__class__.__name__} cannot consume fused AR quant tuple input"
    )


def _gdn_qkvzba_packable(proj_qkvz, proj_ba) -> bool:
    """GDN_QKVZBA_PACK 运行期守卫（任一不满足 → 回退两次 GEMM 原链路）。

    只允许最普通的非量化 2D 权重：FP8 等量化形态的 scale 布局与 packed GEMM
    的加性不兼容，直接排除（量化形态由 quant_method 类名判定）。
    """
    wq = proj_qkvz.weight
    wb = proj_ba.weight
    if wq.dim() != 2 or wb.dim() != 2 or wq.shape[1] != wb.shape[1]:
        return False
    if wq.dtype != wb.dtype or wq.dtype not in (torch.bfloat16, torch.float16):
        return False
    if proj_qkvz.bias is not None or proj_ba.bias is not None:
        return False
    for proj in (proj_qkvz, proj_ba):
        qm = getattr(proj, "quant_method", None)
        if qm is not None and qm.__class__.__name__ != "UnquantizedLinearMethod":
            return False
    return True


def _gdn_qkvzba_pack_materialize(proj_qkvz, proj_ba) -> torch.Tensor:
    """物化 [N_pack, K] 打包权重，并把两模块 weight.data 重绑为其行切片。

    v1.2：N_pack = (N_qkvz+N_ba) 向上补零行到 16 倍数（如 TP8 的 1544→1552；
    TP4 的 3088 本即 16 倍数、补 0 行）。打包 GEMM 输出的物理行距因此保持 16
    倍数，满足 op1 fused_qkvzba_conv1d v2.1 strided 输入对 z 拷贝的 32B 对齐
    契约（host TORCH_CHECK xRowStride%16==0），qkvz/ba 行距视图可直送 op1。
    pad 行权重为零、pad 列输出无任何消费者。

    重绑后两段参数的存储就是打包缓冲的行切片（packed[:n] 与 packed[n:n+m] 各自
    连续，注意 ba 段切片**不含 pad 尾行**）：
      - 权重 loader / RL 权重同步按 in-place narrow+copy_ 写入时（sglang 现状，
        layers/linear.py 的 weight_loader 均为 param.data.copy_ 系），写入直接
        落在打包缓冲上，天然同步、无需重物化；
      - 若某路径整体替换 weight.data（指针漂移），调用方以 data_ptr 校验在
        下一次 forward 自动重新物化。两种情形下打包缓冲都与模块权重一致。
    返回打包缓冲（调用方缓存，并以 weight.data_ptr() == packed.data_ptr() 校验）。
    """
    wq = proj_qkvz.weight
    wb = proj_ba.weight
    n_qkvz = wq.shape[0]
    n_ba = wb.shape[0]
    n_pack = (n_qkvz + n_ba + 15) // 16 * 16
    if n_pack == n_qkvz + n_ba:
        packed = torch.cat([wq.detach(), wb.detach()], dim=0)
    else:
        packed = torch.cat(
            [
                wq.detach(),
                wb.detach(),
                wq.detach().new_zeros(n_pack - n_qkvz - n_ba, wq.shape[1]),
            ],
            dim=0,
        )
    wq.data = packed[:n_qkvz]
    wb.data = packed[n_qkvz : n_qkvz + n_ba]
    return packed


if _is_npu:
    from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import (
        split_qkvgate_gemma_rmsnorm_rope,
    )
    # v1 变更（方案A）：NPU 启用融合 qkvzba split kernel，改用 sgl_kernel_npu 版
    # （graph 口径比 sglang 侧 triton_gdn_fused_proj 版快，且逐 bit 等价，
    #  见 resource/code/gdn_mamba/test_a 结果）
    from sgl_kernel_npu.fla.utils import (
        fused_qkvzba_split_reshape_cat_contiguous,
    )
    # v1 变更（full_attention A1b/A2）：全注意力段 decode 融合 kernel——
    # A1b split+norm+rope+KV scatter 合一、A2 sigmoid_mul 融合；调用点带形状
    # 守卫，未命中已验证形状自动回退 stock（见 resource/code/full_attention/）
    from sglang.kernels.ops.attention.full_attention_fusion_npu import (
        fa_sigmoid_mul,
        fa_sigmoid_mul_supported,
        fa_split_qkvgate_scatter,
        fa_split_qkvgate_scatter_supported,
        fa_v4_scatter_context,
    )
    # TP_FUSION: TP 线 AscendC 三算子接线（tp_ascendc_fusion 包；总开关
    # SGLANG_NPU_TP_ASCENDC_FUSION 默认关）。部署假设同 v1：先将
    # tp_ascendc_fusion_npu.py 落到 sglang/kernels/ops/ 再替换本文件。
    from sglang.kernels.ops.tp_ascendc_fusion_npu import (
        tp_fused_qkvzba_conv1d_shape_supported,
        tp_norm_qkv_scatter_context,
        tp_norm_qkv_scatter_shape_supported,
        tp_sigmoid_mul_mm_runtime_ok,
        tp_sigmoid_mul_mm_shape_supported,
    )


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(
        self,
        config: Qwen3_5TextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.attn_tp_rank = get_parallel().attn_tp_rank
        self.attn_tp_size = get_parallel().attn_tp_size
        self.hidden_size = config.hidden_size
        self.num_v_heads = (
            config.linear_num_value_heads
            if not _is_cpu
            else config.linear_num_value_heads_cpu
        )
        self.num_k_heads = (
            config.linear_num_key_heads
            if not _is_cpu
            else config.linear_num_key_heads_cpu
        )
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.alt_stream = alt_stream

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_id = layer_id
        self.activation = config.hidden_act
        self.output_gate_type = config.output_gate_type
        self.layer_norm_epsilon = config.rms_norm_eps

        # Conv1d layer
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            quant_config=None,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("conv1d", prefix),
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # projection of the input hidden states
        self.in_proj_qkvz = self.create_qkvz_proj(
            hidden_size=self.hidden_size,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            quant_config=quant_config,
            prefix=add_prefix("in_proj_qkvz", prefix),
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
        )

        self.in_proj_ba = self.create_ba_proj(
            hidden_size=self.hidden_size,
            num_v_heads=self.num_v_heads,
            quant_config=quant_config,
            prefix=add_prefix("in_proj_ba", prefix),
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
        )

        # Override weight loaders for packed checkpoint format.
        # Important: for FP8, this must cover not only `.weight` but also
        # `weight_scale_inv` / `weight_scale` / `input_scale` if present.
        self._bind_packed_weight_loaders(self.in_proj_qkvz)
        self._bind_packed_weight_loaders(self.in_proj_ba)
        self._fused_input_proj_cpu_enabled = LazyValue(
            lambda: (
                _is_cpu
                and self.in_proj_qkvz.weight.dtype == torch.bfloat16
                and self.in_proj_ba.weight.dtype == torch.bfloat16
                and self.in_proj_qkvz.bias is None
                and self.in_proj_ba.bias is None
                and use_intel_amx_backend(self.in_proj_qkvz)
                and use_intel_amx_backend(self.in_proj_ba)
            )
        )

        # GDN_QKVZBA_PACK：打包开关与惰性物化缓存。打包在权重就绪后的首次
        # forward 物化（_get_packed_qkvzba_weight），只改计算组织——两模块的
        # 参数、名称、loader、checkpoint 映射全部不变，9B dense 与 35B MoE
        # 共用本类、同享本开关。
        self._qkvzba_pack_enabled = _is_npu and _gdn_qkvzba_pack_env
        self._packed_qkvzba_weight: Optional[torch.Tensor] = None

        # Conv1d weight loader setup
        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        self._override_weight_loader(
            self.conv1d.weight,
            mamba_v2_sharded_weight_loader(
                [
                    query_key_settings,
                    query_key_settings,
                    value_settings,
                ],
                self.attn_tp_size,
                self.attn_tp_rank,
            ),
        )

        # State parameters
        # cast_elimination_v2（随本包携带，见 code/cast_elimination_v2）：dt_bias 与
        # A_log 一样按 fp32 物化（下行 A_log 既有先例）。Triton gating/recurrent 核内
        # 本就 .to(tl.float32)（无损加宽，值逐 bit 不变）；AscendC recurrent host 硬要
        # fp32——此前 wrapper 每层 .float() 真 cast（图内 1 cast/层），本改动后变 no-op。
        # 加载与 RL 权重同步均 copy_ 系（隐式转换），与 A_log 生产已验证路径相同。
        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads // self.attn_tp_size, dtype=torch.float32),
        )
        self.A_log = nn.Parameter(
            torch.empty(self.num_v_heads // self.attn_tp_size, dtype=torch.float32),
        )

        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        self.attn = RadixLinearAttention(
            layer_id=layer_id,
            num_q_heads=self.num_k_heads // self.attn_tp_size,
            num_k_heads=self.num_k_heads // self.attn_tp_size,
            num_v_heads=self.num_v_heads // self.attn_tp_size,
            head_q_dim=self.head_k_dim,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv_weights=conv_weights,
            bias=self.conv1d.bias,
            activation=self.activation,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
        )

        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            device=torch.get_device_module().current_device(),
            dtype=torch.get_default_dtype(),
            **(
                {"activation": self.output_gate_type}
                if self.output_gate_type is not None
                else {}
            ),
        )

        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_results=False,
            quant_config=quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("out_proj", prefix),
        )

        # TP_FUSION 候选1：fused_qkvzba_conv1d（split+causal_conv1d 单 kernel）的
        # init 期形状守卫烘定（头数传 TP 切分后 per-rank 值，与算子语义一致）；
        # 未命中/总开关关闭 → forward 走原 split+causal_conv1d 链路（逐字保留在下方）
        self._tp_fused_qkvzba_ok = _is_npu and tp_fused_qkvzba_conv1d_shape_supported(
            triton.cdiv(self.num_k_heads, self.attn_tp_size),
            triton.cdiv(self.num_v_heads, self.attn_tp_size),
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
        )

    @staticmethod
    def _override_weight_loader(param, loader):
        """Robustly override loader for:
        1) BasevLLMParameter subclasses: real storage is `_weight_loader`
        2) regular Parameters that already have mutable `weight_loader`
        3) regular Parameters without `weight_loader` yet
        """
        if hasattr(param, "_weight_loader"):
            # FP8 / quantized BasevLLMParameter path
            param._weight_loader = loader
            return

        if hasattr(param, "weight_loader"):
            # Regular parameter/tensor that already has a mutable attr.
            # Do NOT call set_weight_attrs here, because it asserts when
            # overwriting an existing attribute.
            param.weight_loader = loader
            return

        # Fresh attribute on a normal tensor/Parameter
        set_weight_attrs(param, {"weight_loader": loader})

    def _bind_packed_weight_loaders(self, module):
        """Bind packed-checkpoint-aware loaders to all relevant params of a merged module."""
        for attr_name in ("weight", "weight_scale_inv", "weight_scale", "input_scale"):
            param = getattr(module, attr_name, None)
            if param is None:
                continue
            original_loader = getattr(param, "weight_loader", None)
            if original_loader is None:
                continue
            wrapped_loader = self._make_packed_weight_loader(module, original_loader)
            self._override_weight_loader(param, wrapped_loader)

    @staticmethod
    def _get_split_sizes_for_param(module, param, loaded_shard_id):
        """Return checkpoint-side split sizes for this param type."""
        if isinstance(param, BlockQuantScaleParameter):
            # Split by output blocks, not raw output sizes.
            block_n, _ = module.quant_method.quant_config.weight_block_size
            block_n = 1 if getattr(param, "format_ue8m0", False) else block_n
            return [
                (module.output_sizes[idx] + block_n - 1) // block_n
                for idx in loaded_shard_id
            ]

        if isinstance(param, PerTensorScaleParameter):
            # One logical scale per logical shard.
            return [1 for _ in loaded_shard_id]

        # Normal weight / non-block quant tensor
        return [module.output_sizes[idx] for idx in loaded_shard_id]

    @classmethod
    def _make_packed_weight_loader(cls, module, original_weight_loader):
        """Wrap the param's original loader so split checkpoints:
          - in_proj_qkv + in_proj_z -> merged in_proj_qkvz
          - in_proj_b + in_proj_a   -> merged in_proj_ba
        can load correctly for both normal and FP8 params.
        """

        def weight_loader(param, loaded_weight, loaded_shard_id=None):
            # Only intercept split-checkpoint tuple shards.
            # int shard_id and None should preserve original behavior.
            if isinstance(loaded_shard_id, tuple):
                split_sizes = cls._get_split_sizes_for_param(
                    module, param, loaded_shard_id
                )

                if loaded_weight.numel() == 1:
                    # Single-element tensor (scalar or [1]):
                    # broadcast to each logical shard.
                    chunks = [loaded_weight.view(-1)] * len(loaded_shard_id)
                else:
                    split_dim = getattr(param, "output_dim", 0)
                    if _is_cpu:
                        cpu_split_sizes = []
                        split_size_sum = sum(split_sizes)
                        target_size_sim = loaded_weight.size(split_dim)
                        for i in range(len(split_sizes)):
                            cpu_split_sizes.append(
                                int(target_size_sim * split_sizes[i] / split_size_sum)
                            )
                        assert (
                            sum(cpu_split_sizes) == target_size_sim
                        ), f"Padding the loaded weight failed due to sizes are not divisible cleanly from {cpu_split_sizes} to {target_size_sim}"
                        chunks = loaded_weight.split(cpu_split_sizes, dim=split_dim)
                    else:
                        chunks = loaded_weight.split(split_sizes, dim=split_dim)

                assert len(chunks) == len(loaded_shard_id), (
                    f"Chunk/shard mismatch: {len(chunks)=}, "
                    f"{len(loaded_shard_id)=}, {split_sizes=}"
                )

                for idx, chunk in zip(loaded_shard_id, chunks):
                    # Delegate each chunk to the param's original int-shard loader.
                    original_weight_loader(param, chunk, idx)
                return

            return original_weight_loader(param, loaded_weight, loaded_shard_id)

        return weight_loader

    def create_qkvz_proj(
        self,
        hidden_size: int,
        key_dim: int,
        value_dim: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> MergedColumnParallelLinear:
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[key_dim, key_dim, value_dim, value_dim],
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

    def create_ba_proj(
        self,
        hidden_size: int,
        num_v_heads: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> MergedColumnParallelLinear:
        # Qwen3.5 has separate in_proj_b and in_proj_a weights in the
        # checkpoint, which are loaded into the fused in_proj_ba parameter
        # via stacked_params_mapping with shard_id 0 and 1 respectively.
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[num_v_heads, num_v_heads],
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
    ):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvzba`.
        """
        k_tp = self.key_dim // self.attn_tp_size
        v_tp = self.value_dim // self.attn_tp_size
        nv_tp = self.num_v_heads // self.attn_tp_size

        # Directly split, no head group reshape
        query, key, value, z = mixed_qkvz.split([k_tp, k_tp, v_tp, v_tp], dim=-1)
        b, a = mixed_ba.split([nv_tp, nv_tp], dim=-1)

        # value / z reshape to (seq, num_v_heads/tp, head_v_dim)
        value = value.reshape(value.size(0), -1, self.head_v_dim)
        z = z.reshape(z.size(0), -1, self.head_v_dim)

        return query, key, value, z, b, a

    def _get_packed_qkvzba_weight(self) -> torch.Tensor:
        """GDN_QKVZBA_PACK：返回打包权重 [N_qkvz+N_ba, K]。

        惰性物化（权重加载完成后首次 forward 触发）；此后每次 forward 以
        data_ptr 校验——in-place 写入（loader / RL 权重同步）下指针不变、
        缓冲天然同步，整体替换 .data 时指针漂移、自动重新物化。
        """
        packed = self._packed_qkvzba_weight
        if (
            packed is not None
            and self.in_proj_qkvz.weight.data_ptr() == packed.data_ptr()
        ):
            return packed
        packed = _gdn_qkvzba_pack_materialize(self.in_proj_qkvz, self.in_proj_ba)
        self._packed_qkvzba_weight = packed
        return packed

    def _forward_input_proj(self, hidden_states: torch.Tensor):
        # AMD/aiter fused AR+RMSNorm+per-group-quant path ships a
        # ``(bf16, fp8, scale)`` 3-tuple so the FP8 ``in_proj_qkvz`` can
        # consume ``(fp8, scale)`` (skipping its internal quant) while the
        # bf16 ``in_proj_ba`` consumes the unquantized bf16. Non-aiter runs skip
        # the tuple branch and keep the original control flow below unchanged.
        if _use_aiter and isinstance(hidden_states, tuple):
            return self._forward_input_proj_fused_quant_amd(hidden_states)

        if (
            _is_cpu
            or _is_npu
            or check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)
        ):
            DUAL_STREAM_TOKEN_THRESHOLD = 0
        else:
            DUAL_STREAM_TOKEN_THRESHOLD = 1024

        seq_len, _ = hidden_states.shape
        if (
            self.alt_stream is not None
            and get_is_capture_mode()
            and seq_len < DUAL_STREAM_TOKEN_THRESHOLD
            and _gdn_use_alt_stream
        ):
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
            with torch.cuda.stream(self.alt_stream):
                projected_states_ba, _ = self.in_proj_ba(hidden_states)
            current_stream.wait_stream(self.alt_stream)
        elif self._fused_input_proj_cpu_enabled.value:
            projected_states_qkvz, projected_states_ba = (
                torch.ops.sgl_kernel.fused_input_proj_cpu(
                    hidden_states,
                    self.in_proj_qkvz.weight,
                    self.in_proj_ba.weight,
                    True,
                )
            )
        else:
            if (
                self._qkvzba_pack_enabled
                and seq_len <= _gdn_qkvzba_pack_max_m
                and _gdn_qkvzba_packable(self.in_proj_qkvz, self.in_proj_ba)
            ):
                # GDN_QKVZBA_PACK（v1.2）：单次打包 GEMM（stock linear → matmulV2），
                # 输出 [M, N_pack]（N_pack 补到 16 倍数）按行切片成两段视图
                # （行距 = N_pack，列内连续）。仅小 M 打包（M 门控原因见模块顶部
                # _gdn_qkvzba_pack_max_m 注释；graph capture 按固定 bs 烘定
                # 本分支，per-graph 一致）。下游三条消费路径：
                #   1) 方案A Triton split（sgl_kernel_npu fla/utils.py，按 stride(0)
                #      取行距）直接读视图，零拷贝；
                #   2) TP_FUSION 候选1 fused_qkvzba_conv1d v2.1 起原生支持行距视图
                #      输入（host 契约 = 列内连续 + 行距 16 倍数，打包 N 已补齐）——
                #      直读零拷贝（v1.1 的两次 .contiguous() 已删除）；
                #   3) fix_query_key_value_ordering 回退路径本就 split+reshape
                #      拷贝，视图不引入额外拷贝。
                # 注意 ba 段切片必须显式排除 pad 尾列（N_pack 可能 > N_qkvz+N_ba）。
                projected_states_qkvzba = nn.functional.linear(
                    hidden_states, self._get_packed_qkvzba_weight()
                )
                n_qkvz = self.in_proj_qkvz.weight.shape[0]
                n_ba = self.in_proj_ba.weight.shape[0]
                projected_states_qkvz = projected_states_qkvzba[:, :n_qkvz]
                projected_states_ba = projected_states_qkvzba[:, n_qkvz : n_qkvz + n_ba]
            else:
                projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
                projected_states_ba, _ = self.in_proj_ba(hidden_states)
        return projected_states_qkvz, projected_states_ba

    def _forward_input_proj_fused_quant_amd(self, hidden_states):
        """AMD-only variant for the fused AR+RMSNorm+per-group-quant path.

        ``hidden_states`` is a ``(bf16, fp8, scale)`` 3-tuple produced by the
        upstream fused kernel. FP8 ``in_proj_qkvz`` takes ``(fp8, scale)``
        directly; unquantized variants take the bf16 side-output.
        """
        hs_bf16 = hidden_states[0]
        hs_qkvz = _select_fused_ar_input_for_linear(hidden_states, self.in_proj_qkvz)
        seq_len = hs_bf16.shape[0]

        if check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE):
            DUAL_STREAM_TOKEN_THRESHOLD = 0
        else:
            DUAL_STREAM_TOKEN_THRESHOLD = 1024

        if (
            self.alt_stream is not None
            and get_is_capture_mode()
            and seq_len < DUAL_STREAM_TOKEN_THRESHOLD
            and _gdn_use_alt_stream
        ):
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            projected_states_qkvz, _ = self.in_proj_qkvz(hs_qkvz)
            with torch.cuda.stream(self.alt_stream):
                projected_states_ba, _ = self.in_proj_ba(hs_bf16)
            current_stream.wait_stream(self.alt_stream)
        else:
            projected_states_qkvz, _ = self.in_proj_qkvz(hs_qkvz)
            projected_states_ba, _ = self.in_proj_ba(hs_bf16)
        return projected_states_qkvz, projected_states_ba

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        projected_states_qkvz, projected_states_ba = self._forward_input_proj(
            hidden_states
        )

        # TP_FUSION 候选1：decode 且 init 守卫命中时，原始 qkvz/ba 投影以 tuple 直送
        # GDN backend——split+causal_conv1d 在 torch.ops.npu.fused_qkvzba_conv1d 单
        # kernel 内完成（conv_states 原地更新语义不变），z 随返回值回流（供下方
        # self.norm(core_attn_out, z) 使用）。约定：tuple 进 → (core_attn_out, z)
        # 出（配套件 ascend_gdn_backend.py 的 forward_decode tuple 分支；backend
        # 运行期判定不通过时内部回退本地 split+stock conv，仍以 tuple 返回）。
        # 图 capture 时该 Python 分支烘进图（与 A1b 同款）。
        core_attn_out = None
        z = None
        if self._tp_fused_qkvzba_ok and forward_batch.forward_mode.is_decode():
            # GDN_QKVZBA_PACK v1.2：打包路径产出的行距视图**直送** op1——
            # fused_qkvzba_conv1d v2.1 host 契约已放宽为「列内连续 + 行距 16 倍数」
            # （打包 N 补到 16 倍数即满足），v1.1 在此处的两次 .contiguous() 拷贝
            # （35B/TP8 实测合计 ~9µs/层，吞掉打包收益）已删除。
            # 部署耦合：本文件须与本包的 tp_ascendc_fusion_npu.py + csrc host 同批
            # 部署（新守卫放行视图 + 旧 host 仍 TORCH_CHECK 连续 的组合会在 capture
            # 期显式报错而非静默回退）；若运行期守卫不通过（如旧守卫 python 在跑），
            # backend 回退方案A Triton split（stride(0) 版零拷贝消费视图）+ stock
            # conv，正确性不受影响。
            core_attn_out, z = self.attn(
                forward_batch,
                mixed_qkv=(projected_states_qkvz, projected_states_ba),
                a=None,
                b=None,
            )

        if core_attn_out is None:
            # v1 变更（方案A）：NPU 也走融合 split kernel（上方 import 已按平台分派）
            if self.num_v_heads // self.num_k_heads in [1, 2, 4]:
                if _is_cpu:
                    num_k_heads_tp = self.num_k_heads // self.attn_tp_size
                    num_v_heads_tp = self.num_v_heads // self.attn_tp_size
                else:
                    num_k_heads_tp = triton.cdiv(self.num_k_heads, self.attn_tp_size)
                    num_v_heads_tp = triton.cdiv(self.num_v_heads, self.attn_tp_size)
                mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat_contiguous(
                    projected_states_qkvz,
                    projected_states_ba,
                    num_k_heads_tp,
                    num_v_heads_tp,
                    self.head_k_dim,
                    self.head_v_dim,
                )
            else:
                query, key, value, z, b, a = self.fix_query_key_value_ordering(
                    projected_states_qkvz, projected_states_ba
                )
                b = b.contiguous()
                a = a.contiguous()

                query, key, value = map(
                    lambda x: x.reshape(x.shape[0], -1), (query, key, value)
                )
                mixed_qkv = torch.cat((query, key, value), dim=-1)

            core_attn_out = self.attn(
                forward_batch,
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
            )

        z_shape_og = z.shape
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])

        # Add padding for DP-Attn
        if core_attn_out.shape != z.shape:
            core_attn_out_pad = torch.zeros_like(z)
            core_attn_out_pad[: core_attn_out.shape[0], :] = core_attn_out
            core_attn_out = core_attn_out_pad

        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-2], -1)

        output, _ = self.out_proj(core_attn_out)
        return output


class Qwen3_5LinearDecoderLayer(nn.Module):
    """Qwen3.5 Decoder Layer with Linear Attention (GatedDeltaNet)."""

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        linear_attn_quant_config = (
            None
            if quant_config and quant_config.get_name() == "modelopt_fp4"
            else quant_config
        )
        self.linear_attn = Qwen3_5GatedDeltaNet(
            config, layer_id, linear_attn_quant_config, alt_stream, prefix
        )

        # NOTE: Determine the MLP type based on the model type
        # Qwen3.5 use all layers for MLP / Qwen3.5-MoE use sparse MoE blocks
        if config.model_type == "qwen3_5_moe_text":
            self.mlp = Qwen2MoeSparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                alt_stream=(
                    alt_stream
                    if (
                        _is_cuda
                        or _disable_shared_experts_fusion()
                        or envs.SGLANG_NPU_USE_MULTI_STREAM.get()
                    )
                    else None
                ),
                prefix=add_prefix("mlp", prefix.replace(".linear_attn", "")),
                is_nextn=is_nextn,
                support_shared_expert_fusion=not _disable_shared_experts_fusion(),
            )
            is_layer_sparse = True
            is_previous_layer_sparse = True
            is_next_layer_sparse = True
        elif config.model_type == "qwen3_5_text":
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix.replace(".linear_attn", "")),
            )
            is_layer_sparse = False
            is_previous_layer_sparse = False
            is_next_layer_sparse = False
        else:
            raise ValueError(f"Invalid model type: {config.model_type}")

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # GDN layers need both bf16 (for the small in_proj_ba gating
        # projection) and a quantized tuple only when in_proj_qkvz can consume
        # it. Otherwise, stay on the plain AR+RMSNorm path.
        enable_fused_ar_quant = (
            _enable_qwen35_fused_ar_quant()
            and _linear_accepts_fp8_tuple(self.linear_attn.in_proj_qkvz)
        )
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=(layer_id == config.num_hidden_layers - 1),
            enable_fused_ar_quant=enable_fused_ar_quant,
            fused_ar_quant_keep_bf16=enable_fused_ar_quant,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        **kwargs,
    ):
        forward_batch = kwargs.get("forward_batch", None)

        hidden_states, residual = (
            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                hidden_states,
                residual,
                forward_batch,
                captured_last_layer_outputs=kwargs.get(
                    "captured_last_layer_outputs", None
                ),
            )
        )

        # MoE 权重 L2 预取（v1.2）：自目标发射——本层 prepare_attn 之后、
        # attention 之前。EP 的层间通信（RS+AG）在 prepare_attn 内完成、
        # TP 的层尾 AR 在上一层 postprocess_layer 内完成，此点在两条线下
        # 都位于全部层间通信之后。仅 GDN 层；主流不 wait，step 末统一 drain。
        moe_prefetch_emit(self, hidden_states)

        if not forward_batch.forward_mode.is_idle():
            hidden_states = self.linear_attn(
                hidden_states,
                forward_batch,
            )

        # Fully Connected
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        mlp_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        fuse_mlp_allreduce = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )
        with get_forward().scoped(
            fuse_mlp_allreduce=fuse_mlp_allreduce,
            mlp_reduce_scatter=mlp_reduce_scatter,
        ):
            if isinstance(self.mlp, Qwen2MoeSparseMoeBlock):
                hidden_states = self.mlp(
                    hidden_states,
                    forward_batch,
                )
            else:
                hidden_states = self.mlp(hidden_states)
        if fuse_mlp_allreduce:
            hidden_states._sglang_needs_allreduce_fusion = True
        else:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )

        return hidden_states, residual


class Qwen3_5AttentionDecoderLayer(nn.Module):
    """Qwen3.5 Decoder Layer with Full Attention."""

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.attn_tp_rank = get_parallel().attn_tp_rank
        self.attn_tp_size = get_parallel().attn_tp_size
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % self.attn_tp_size == 0
        self.num_heads = self.total_num_heads // self.attn_tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= self.attn_tp_size:
            assert self.total_num_kv_heads % self.attn_tp_size == 0
        else:
            assert self.attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.attn_tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        self.rope_theta, rope_scaling = get_rope_config(config)
        self.partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        self.layer_id = layer_id

        # If rope_scaling doesn't specify a scaling type, treat as no scaling
        if rope_scaling and not ("rope_type" in rope_scaling or "type" in rope_scaling):
            rope_scaling = None

        self.attn_output_gate = getattr(config, "attn_output_gate", True)
        if self.attn_output_gate:
            logger.warning_once("using attn output gate!")

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            rope_scaling=rope_scaling,
            base=self.rope_theta,
            partial_rotary_factor=self.partial_rotary_factor,
            is_neox_style=True,
            dtype=torch.get_default_dtype(),
        )

        attn_quant_config = (
            None
            if quant_config and quant_config.get_name() == "modelopt_fp4"
            else quant_config
        )

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=attn_quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=attn_quant_config,
            reduce_results=False,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("o_proj", prefix),
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=f"{prefix}.attn",
        )

        # Dense MLP for non-MoE variant
        if config.model_type == "qwen3_5_text":
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix.replace(".self_attn", "")),
            )
            is_layer_sparse = False
            is_previous_layer_sparse = False
            is_next_layer_sparse = False
        elif config.model_type == "qwen3_5_moe_text":
            self.mlp = Qwen2MoeSparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                alt_stream=(
                    alt_stream
                    if (
                        _is_cuda
                        or _disable_shared_experts_fusion()
                        or envs.SGLANG_NPU_USE_MULTI_STREAM.get()
                    )
                    else None
                ),
                prefix=add_prefix("mlp", prefix.replace(".self_attn", "")),
                is_nextn=is_nextn,
                support_shared_expert_fusion=not _disable_shared_experts_fusion(),
            )
            is_layer_sparse = True
            is_previous_layer_sparse = True
            is_next_layer_sparse = True
        else:
            raise ValueError(f"Invalid model type: {config.model_type}")

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Standard attention layers benefit from a fused quant epilogue only
        # when qkv_proj can consume the returned quantized tuple.
        enable_fused_ar_quant = (
            _enable_qwen35_fused_ar_quant() and _linear_accepts_fp8_tuple(self.qkv_proj)
        )
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=(layer_id == config.num_hidden_layers - 1),
            enable_fused_ar_quant=enable_fused_ar_quant,
            fused_ar_quant_keep_bf16=False,
        )

        self.alt_stream = alt_stream

        # v1 变更（full_attention A1b）：TP 布局形状守卫在 init 时一次算定
        # （q=2 头/kv=1 头/head_dim=256/rope=64/带 gate 才走自研融合 kernel）
        self._fa_v4_shape_ok = _is_npu and fa_split_qkvgate_scatter_supported(
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            int(self.head_dim * self.partial_rotary_factor),
            self.attn_output_gate,
        )

        # TP_FUSION 候选2/候选3：norm+qkv_proj+KV scatter 三合一与
        # sigmoid_mul+o_proj 融合的 init 期形状守卫烘定（未命中/总开关关闭 →
        # 运行时逐字走 v1 原链路）
        self._tp_fa_norm_qkv_scatter_ok = _is_npu and tp_norm_qkv_scatter_shape_supported(
            self
        )
        self._tp_sigmoid_mul_mm_ok = _is_npu and tp_sigmoid_mul_mm_shape_supported(
            self.o_proj
        )

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Q/K normalization with optional alt_stream overlap."""
        if (
            self.alt_stream is not None
            and get_is_capture_mode()
            and _qknorm_use_alt_stream
        ):
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            q_by_head = q.reshape(-1, self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            with torch.cuda.stream(self.alt_stream):
                k_by_head = k.reshape(-1, self.head_dim)
                k_by_head = self.k_norm(k_by_head)
            current_stream.wait_stream(self.alt_stream)
        elif _is_hip or _is_xpu or _is_cpu:
            q_by_head, k_by_head = fused_qk_gemma_rmsnorm(
                q,
                k,
                self.q_norm.weight.data,
                self.k_norm.weight.data,
                self.q_norm.variance_epsilon,
                self.head_dim,
            )
        else:
            q_by_head = q.reshape(-1, self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            k_by_head = k.reshape(-1, self.head_dim)
            k_by_head = self.k_norm(k_by_head)
        q = q_by_head.view(q.shape)
        k = k_by_head.view(k.shape)
        return q, k

    def forward_prepare_cuda_fused(self, positions, hidden_states):
        """Fused QK GemmaRMSNorm + NeoX RoPE + gate deinterleave."""
        qkv, _ = self.qkv_proj(hidden_states)
        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
        else:
            q_gate, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_out, k_out, gate_out = fused_qk_gemma_rmsnorm_rope_gate(
            q_gate,
            k,
            self.q_norm.weight.data,
            self.k_norm.weight.data,
            self.rotary_emb.cos_sin_cache,
            positions,
            self.q_norm.variance_epsilon,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.rotary_emb.rotary_dim,
            has_gate=self.attn_output_gate,
        )
        seq_len = hidden_states.shape[0]
        q = q_out.view(seq_len, -1)
        k = k_out.view(seq_len, -1)
        gate = gate_out.view(seq_len, -1) if gate_out is not None else None
        return q, k, v, gate

    def forward_prepare_native(self, positions, hidden_states):
        if _use_aiter and isinstance(hidden_states, tuple):
            hidden_states = _select_fused_ar_input_for_linear(
                hidden_states, self.qkv_proj
            )
        qkv, _ = self.qkv_proj(hidden_states)
        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            # gate stays as 3D strided view; fused_sigmoid_mul handles it directly
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            gate = None

        q, k = self._apply_qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v, gate

    def forward_prepare_fused_gate(self, positions, hidden_states):
        if _use_aiter and isinstance(hidden_states, tuple):
            hidden_states = _select_fused_ar_input_for_linear(
                hidden_states, self.qkv_proj
            )
        qkv, _ = self.qkv_proj(hidden_states)
        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            seq_len = q_gate.shape[0]
            q_flat, k_flat, gate_flat = fused_qk_gemma_rmsnorm_with_gate(
                q_gate,
                k,
                self.q_norm.weight.data,
                self.k_norm.weight.data,
                self.q_norm.variance_epsilon,
                self.head_dim,
                self.num_heads,
            )
            q = q_flat.view(seq_len, -1)
            k = k_flat.view(seq_len, -1)
            gate = gate_flat.view(seq_len, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            gate = None
            q, k = self._apply_qk_norm(q, k)

        q, k = self.rotary_emb(positions, q, k)
        return q, k, v, gate

    def forward_prepare_npu(self, positions, hidden_states, forward_batch):
        qkv, _ = self.qkv_proj(hidden_states)
        # Calculate first full attention layer ID based on config
        if self.attn.layer_id == (self.config.full_attention_interval - 1):
            self.rotary_emb.get_cos_sin_with_position(positions)

        # v1 变更（full_attention A1b）：decode 且命中已验证形状时，split+norm+rope
        # 与 KV scatter 合一——k/v 由 kernel 按 out_cache_loc 直接写入 KV cache，
        # 返回 k/v=None，self_attention 据此以 save_kv_cache=False 走 backend
        # （跳过 stock 的两次 npu_scatter_nd_update_）。守卫任一环节未命中则回退
        # stock split；SGLANG_NPU_FULL_ATTN_FUSION_DEBUG=1 可按层打印未命中原因。
        fa_ctx = fa_v4_scatter_context(self, forward_batch, qkv)
        if fa_ctx is not None:
            q, _, _, gate = fa_split_qkvgate_scatter(
                qkv,
                self.rotary_emb.position_sin,
                self.rotary_emb.position_cos,
                self.q_size,
                self.kv_size,
                self.head_dim,
                int(self.head_dim * self.partial_rotary_factor),
                eps=self.q_norm.variance_epsilon,
                q_weight=self.q_norm.weight,
                k_weight=self.k_norm.weight,
                kbuf=fa_ctx[0],
                vbuf=fa_ctx[1],
                loc=fa_ctx[2],
            )
            return q, None, None, gate

        q, k, v, gate = split_qkvgate_gemma_rmsnorm_rope(
            qkv,
            self.rotary_emb.position_sin,
            self.rotary_emb.position_cos,
            self.q_size,
            self.kv_size,
            self.head_dim,
            int(self.head_dim * self.partial_rotary_factor),
            eps=self.q_norm.variance_epsilon,
            q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight,
        )
        return q, k, v, gate

    def self_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Full attention forward pass."""
        if _is_cuda and self.attn_output_gate:
            q, k, v, gate = self.forward_prepare_cuda_fused(
                positions=positions,
                hidden_states=hidden_states,
            )
        elif (_is_hip or _is_xpu or _is_cpu) and self.attn_output_gate:
            q, k, v, gate = self.forward_prepare_fused_gate(
                positions=positions,
                hidden_states=hidden_states,
            )
        elif (
            not _is_npu
            or forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
            or not self.attn_output_gate
        ):
            q, k, v, gate = self.forward_prepare_native(
                positions=positions,
                hidden_states=hidden_states,
            )
        else:
            q, k, v, gate = self.forward_prepare_npu(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        # v1 变更（full_attention A1b）：k is None ⟺ KV 已在融合 kernel 内
        # scatter 进 cache，backend 侧跳过 set_kv_buffer；其余路径 k 恒非 None，
        # save_kv_cache=True 与 stock 默认值一致
        attn_output = self.attn(q, k, v, forward_batch, save_kv_cache=k is not None)

        if self.attn_output_gate:
            if not _is_npu:
                attn_output = fused_sigmoid_mul(attn_output, gate, inplace=True)
            else:
                gate_val = gate.reshape(gate.shape[0], -1) if gate.ndim == 3 else gate
                # TP_FUSION 候选3：sigmoid_mul + o_proj GEMM 单 kernel
                # （torch.ops.npu.fused_sigmoid_mul_mm；输出 [M,2048] partial sum，
                # 无 bias 无 all-reduce，与 o_proj(reduce_results=False) 的返回语义
                # 一致，故直接作为本层输出返回）；守卫未命中 → 回退下方 A2/stock 链
                if self._tp_sigmoid_mul_mm_ok and tp_sigmoid_mul_mm_runtime_ok(
                    attn_output, gate_val
                ):
                    return torch.ops.npu.fused_sigmoid_mul_mm(
                        attn_output, gate_val, self.o_proj.weight
                    )
                # v1 变更（full_attention A2）：sigmoid+mul 融合单 kernel，与 stock
                # 双 op 舍入路径逐位一致；不满足守卫（非 bf16/非连续）回退 stock
                if fa_sigmoid_mul_supported(attn_output, gate_val):
                    attn_output = fa_sigmoid_mul(attn_output, gate_val)
                else:
                    attn_output.mul_(torch.sigmoid(gate_val))

        output, _ = self.o_proj(attn_output)
        return output

    def _self_attention_tp_fused(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        tp_ctx,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """TP_FUSION 候选2 的 FA decode 主干（prepare 的 input_layernorm 并入 kernel）。

        hidden_states/residual 为 prepare 前的原始张量（未做 add+norm）；返回
        (o_proj partial 输出, kernel 产出的新 residual add_out)，供 forward 直接进
        prepare_mlp——与 stock 的 (attn 输出, A3a add_out) 语义逐点一致。
        调用前须过 tp_norm_qkv_scatter_context（tp_ctx 非 None）。
        """
        # 与 forward_prepare_npu 相同：首个全注意力层先刷新 sin/cos 缓存
        if self.attn.layer_id == (self.config.full_attention_interval - 1):
            self.rotary_emb.get_cos_sin_with_position(positions)

        rope_dim = int(self.head_dim * self.partial_rotary_factor)
        # kernel 要求 [M,64] fp32 连续；position_sin/cos 现形 [M,1,1,64]（view 平）。
        # NPU 默认 rope cache 为 bf16（SGLANG_ROPE_CACHE_FP32 未开）→ widening cast
        # 到 fp32（bf16→fp32 精确，取值与 A1b kernel 内 load 后 .to(fp32) 逐位一致）
        sin = self.rotary_emb.position_sin.view(-1, rope_dim)
        cos = self.rotary_emb.position_cos.view(-1, rope_dim)
        if sin.dtype != torch.float32:
            sin = sin.float()
            cos = cos.float()

        kbuf, vbuf, loc = tp_ctx
        add_out, q, gate = torch.ops.npu.fused_norm_qkv_proj_scatter(
            hidden_states,
            residual,
            self.input_layernorm.weight,
            self.qkv_proj.weight,
            self.q_norm.weight,
            self.k_norm.weight,
            sin,
            cos,
            loc,
            kbuf,
            vbuf,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            rope_dim,
            self.input_layernorm.variance_epsilon,
            self.q_norm.variance_epsilon,
        )
        # k/v 已由 kernel 按 out_cache_loc 直写 cache（同 A1b 约定：save_kv_cache=False，
        # backend 侧跳过 set_kv_buffer）
        attn_output = self.attn(q, None, None, forward_batch, save_kv_cache=False)

        # 候选3（与 self_attention 尾部同一套守卫）：命中则 A2+o_proj 一并融合；
        # 未命中回退 A2/stock 尾部（gate 为 kernel 直出的 [M,512] 2D 连续张量）
        if self._tp_sigmoid_mul_mm_ok and tp_sigmoid_mul_mm_runtime_ok(
            attn_output, gate
        ):
            output = torch.ops.npu.fused_sigmoid_mul_mm(
                attn_output, gate, self.o_proj.weight
            )
        elif fa_sigmoid_mul_supported(attn_output, gate):
            attn_output = fa_sigmoid_mul(attn_output, gate)
            output, _ = self.o_proj(attn_output)
        else:
            attn_output.mul_(torch.sigmoid(gate))
            output, _ = self.o_proj(attn_output)
        return output, add_out

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        captured_last_layer_outputs: Optional[list[torch.Tensor]] = None,
        **kwargs,
    ):
        # TP_FUSION 候选2：decode 且运行期守卫全命中时，跳过 prepare 内的
        # input_layernorm——add+gemma rmsnorm 已由 fused_norm_qkv_proj_scatter 并入
        # kernel，原始 hidden_states/residual 直进 kernel，add_out 作为新 residual
        # 进 prepare_mlp。本配置（TP、无 dp-attention、非首层、无 AR fusion pending）
        # 下 prepare_attn 除 norm 外恒为空操作（input_scattered=False、
        # _communicate_simple_fn=trivial、无 qkv_latent），故跳过无遗漏；
        # 任一前提破坏 → tp_ctx=None，走下方 v1 原链路（逐字保留）。
        tp_ctx = (
            tp_norm_qkv_scatter_context(
                self,
                hidden_states,
                residual,
                forward_batch,
                captured_last_layer_outputs,
            )
            if (_is_npu and self._tp_fa_norm_qkv_scatter_ok)
            else None
        )
        if tp_ctx is None:
            hidden_states, residual = (
                self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                    hidden_states,
                    residual,
                    forward_batch,
                    captured_last_layer_outputs=captured_last_layer_outputs,
                )
            )

        # MoE 权重 L2 预取（v1.2）：保持在本层全部层间通信之后、attention 之前。
        # TP 候选2命中时 prepare_attn 被同语义融合路径替代，层间 AR 已由上一层完成。
        moe_prefetch_emit(self, hidden_states)

        if tp_ctx is not None:
            hidden_states, residual = self._self_attention_tp_fused(
                positions, hidden_states, residual, forward_batch, tp_ctx
            )
        else:
            if not forward_batch.forward_mode.is_idle():
                hidden_states = self.self_attention(
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                )

        # Fully Connected
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        mlp_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        fuse_mlp_allreduce = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )
        with get_forward().scoped(
            fuse_mlp_allreduce=fuse_mlp_allreduce,
            mlp_reduce_scatter=mlp_reduce_scatter,
        ):
            if isinstance(self.mlp, Qwen2MoeSparseMoeBlock):
                hidden_states = self.mlp(
                    hidden_states,
                    forward_batch,
                )
            else:
                hidden_states = self.mlp(hidden_states)
        if fuse_mlp_allreduce:
            hidden_states._sglang_needs_allreduce_fusion = True
        else:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )

        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "attention": Qwen3_5AttentionDecoderLayer,
    "linear_attention": Qwen3_5LinearDecoderLayer,
}


class Qwen3_5ForCausalLM(nn.Module):
    """Qwen3.5 Model with support for dense variant."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "out_proj",
        "in_proj_qkvz",
        "gate_up_proj",
        "down_proj",
        "lm_head",
    ]

    def get_hidden_dim(self, module_name: str, layer_idx: int):
        config = self.config
        head_dim = config.head_dim or (config.hidden_size // config.num_attention_heads)

        if module_name == "qkv_proj":
            attn_output_gate = getattr(config, "attn_output_gate", True)
            q_heads = config.num_attention_heads * (2 if attn_output_gate else 1)
            return (
                config.hidden_size,
                head_dim * (q_heads + config.num_key_value_heads * 2),
            )
        elif module_name == "o_proj":
            return config.num_attention_heads * head_dim, config.hidden_size
        elif module_name == "out_proj":
            value_dim = config.linear_value_head_dim * config.linear_num_value_heads
            return value_dim, config.hidden_size
        elif module_name == "in_proj_qkvz":
            key_dim = config.linear_key_head_dim * config.linear_num_key_heads
            value_dim = config.linear_value_head_dim * config.linear_num_value_heads
            return config.hidden_size, key_dim * 2 + value_dim * 2
        elif module_name == "gate_up_proj":
            # MoE: shared expert uses shared_expert_intermediate_size
            # Dense: regular MLP uses intermediate_size
            is_moe = "moe" in getattr(config, "model_type", "")
            if is_moe:
                inter = config.shared_expert_intermediate_size
            else:
                inter = config.intermediate_size
            return config.hidden_size, inter * 2
        elif module_name == "down_proj":
            is_moe = "moe" in getattr(config, "model_type", "")
            if is_moe:
                inter = config.shared_expert_intermediate_size
            else:
                inter = config.intermediate_size
            return inter, config.hidden_size
        elif module_name == "gate_up_proj_moe":
            return config.hidden_size, config.moe_intermediate_size * 2
        elif module_name == "down_proj_moe":
            return config.moe_intermediate_size, config.hidden_size
        elif module_name == "embed_tokens":
            return config.vocab_size, config.hidden_size
        elif module_name == "lm_head":
            return config.hidden_size, config.vocab_size
        else:
            raise NotImplementedError(
                f"get_hidden_dim not implemented for {module_name}"
            )

    def _maybe_autodisable_shared_experts_fusion(self, config, quant_config):
        # Auto-disable fusion when the checkpoint can't fuse (e.g. MXFP4 Qwen3.5)
        # so the model still gets the #25885 multi-streaming path. ROCm-only.
        if (
            config.model_type == "qwen3_5_moe_text"
            and not get_server_args().disable_shared_experts_fusion
            and not can_fuse_shared_expert(config, quant_config)
        ):
            from sglang.srt.arg_groups.overrides import declare_load_time_override

            declare_load_time_override(
                "Qwen3_5ForCausalLM._maybe_autodisable_shared_experts_fusion",
                {"disable_shared_experts_fusion": True},
            )
            logger.info(
                "Qwen3.5: shared-expert fusion not supported for this checkpoint; "
                "auto-disabling (multi-streaming #25885 still applies)."
            )

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        is_nextn: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.pp_group = get_pp_group()

        if _is_hip:
            self._maybe_autodisable_shared_experts_fusion(config, quant_config)

        alt_stream = (
            get_stream("alt")
            if _is_cuda or _hip_use_alt_stream or envs.SGLANG_NPU_USE_MULTI_STREAM.get()
            else None
        )

        # Embedding layer
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                enable_tp=not is_dp_attention_enabled(),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Decoder layers
        def get_layer(idx: int, prefix: str):
            layer_type = config.layers_block_type[idx]
            layer_class = ALL_DECODER_LAYER_TYPES[layer_type]
            if layer_type == "attention":
                prefix = add_prefix("self_attn", prefix)
            else:
                prefix = add_prefix("linear_attn", prefix)
            return layer_class(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
                is_nextn=is_nextn,
            )

        self.layers, self._start_layer, self._end_layer = make_layers(
            config.num_hidden_layers,
            get_layer,
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=f"{prefix}.layers",
        )

        # Final normalization
        if self.pp_group.is_last_rank:
            self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.layers_to_capture = []

        # MoE 权重 L2 预取（v1.2）：注册本模型各层 MoE 权重（layer_id → w13/w2
        # 及 GDN 目标层标记 + 层对象发射标记），并在 capture 外创建专用预取流；
        # is_nextn 的 MTP draft 模型不注册（其层无发射标记，draft capture
        # 内不会误发射）。
        if not is_nextn:
            moe_prefetch_register_model(self)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_dflash_layers_to_capture(self, layers_to_capture: list[int]):
        self.layers_to_capture = layers_to_capture
        for layer_id in self.layers_to_capture:
            setattr(self.layers[layer_id], "_is_layer_to_capture", True)

    @property
    def start_layer(self) -> int:
        return self._start_layer

    @property
    def end_layer(self) -> int:
        return self._end_layer

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        input_deepstack_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        # Initialize hidden states
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        aux_hidden_states = []
        # Pass through decoder layers
        for layer_idx in range(self.start_layer, self.end_layer):
            layer = self.layers[layer_idx]
            with get_global_expert_distribution_recorder().with_current_layer(
                layer_idx
            ):
                hidden_states, residual = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                    forward_batch=forward_batch,
                    captured_last_layer_outputs=(
                        aux_hidden_states
                        if getattr(layer, "_is_layer_to_capture", False)
                        else None
                    ),
                )

            # Process deepstack embeddings if provided
            if (
                input_deepstack_embeds is not None
                and input_deepstack_embeds.numel() > 0
                and layer_idx < 3
            ):
                sep = self.hidden_size * layer_idx
                hidden_states.add_(
                    input_deepstack_embeds[:, sep : sep + self.hidden_size]
                )

        # MoE 权重 L2 预取（v1.2）：step 末 drain——主流 join 预取流
        # （capture 合法性 + 防跨 step 积压）；未启用/本 pass 未发射时为 no-op。
        moe_prefetch_step_drain()

        # Return intermediate tensors for pipeline parallelism
        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )

        # Apply final normalization
        if hidden_states.shape[0] != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            # GDN
            ("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
            ("in_proj_qkvz.", "in_proj_z.", 3),
            ("in_proj_ba.", "in_proj_b.", 0),
            ("in_proj_ba.", "in_proj_a.", 1),
        ]

        loaded_params: Set[str] = set()
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "mtp" in name:
                continue
            if "visual" in name:
                continue
            if "language_model" in name:
                name = name.replace(r"model.language_model.", r"model.")
            if ".self_attn." in name:
                name = name.replace(".self_attn", "")
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self, "start_layer")
                and (layer_id < self.start_layer or layer_id >= self.end_layer)
            ):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                # if is_pp_missing_parameter(name, self):
                #     continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    logger.warning(f"Parameter {name} not found in params_dict")
                    continue
                param = params_dict[name]

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None,
        )


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLM):
    def __init__(
        self,
        config: Qwen3_5TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            # GDN
            ("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
            ("in_proj_qkvz.", "in_proj_z.", 3),
            ("in_proj_ba.", "in_proj_b.", 0),
            ("in_proj_ba.", "in_proj_a.", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
        )

        is_fused_expert = False
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]

        num_experts = self.config.num_experts

        def load_fused_expert_weights(
            name: str,
            params_dict: dict,
            loaded_weight: torch.Tensor,
            shard_id: str,
            num_experts: int,
        ):
            if name not in params_dict:
                return False
            param = params_dict[name]
            weight_loader = param.weight_loader
            # let ep moe layer to gracefully handle expert_ids that do not belong to local moe rank
            for expert_id in range(num_experts):
                curr_expert_weight = loaded_weight[expert_id]
                weight_loader(
                    param,
                    curr_expert_weight,
                    name,
                    shard_id,
                    expert_id,
                )
            return True

        loaded_params: Set[str] = set()
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "mtp" in name:
                continue
            if "visual" in name:
                continue
            if "language_model" in name:
                name = name.replace(r"model.language_model.", r"model.")
            if ".self_attn." in name:
                name = name.replace(".self_attn", "")

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self, "start_layer")
                and (layer_id < self.start_layer or layer_id >= self.end_layer)
            ):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue

                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra parameters for GPTQ/modelopt models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Track if this is an expert weight to enable early skipping
                is_expert_weight = False

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if is_fused_expert:
                        if "experts.gate_up_proj" in name:
                            loaded_weight = loaded_weight.chunk(2, dim=-2)
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[0],
                                "w1",
                                num_experts,
                            )
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[1],
                                "w3",
                                num_experts,
                            )
                        else:
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            )
                    else:
                        # Skip loading extra parameters for GPTQ/modelopt models.
                        if (
                            name_mapped.endswith(ignore_suffixes)
                            and name_mapped not in params_dict
                        ):
                            continue
                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or
                        # not here since otherwise we may skip experts with
                        # # other available replicas.
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    name = name_mapped
                    break
                else:
                    if is_expert_weight:
                        # This is an expert weight but not mapped to this rank, skip all remaining processing
                        continue

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")
            loaded_params.add(name)

        return loaded_params


class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration):
    packed_modules_mapping = Qwen3_5ForCausalLM.packed_modules_mapping
    hf_to_sglang_mapper = None

    supported_lora_modules = Qwen3_5ForCausalLM.supported_lora_modules

    def __init__(
        self,
        config: Qwen3_5Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        language_model_cls=Qwen3_5ForCausalLM,
    ):
        super().__init__(config, quant_config, prefix, language_model_cls)

        rope_config = getattr(self.config, "rope_parameters", None) or getattr(
            self.config, "rope_scaling", {}
        )
        self.is_mrope_enabled = "mrope_section" in rope_config

        self.deepstack_visual_indexes = self.visual.deepstack_visual_indexes

    def get_hidden_dim(self, module_name: str, layer_idx: int):
        return self.model.get_hidden_dim(module_name, layer_idx)

    def should_apply_lora(self, module_name: str) -> bool:
        return module_name.startswith("model.layers.")

    @property
    def start_layer(self) -> int:
        return getattr(getattr(self, "model", None), "start_layer", 0)

    @property
    def end_layer(self) -> int:
        model = getattr(self, "model", None)
        end_layer = getattr(model, "end_layer", None)
        if end_layer is not None:
            return end_layer
        cfg = getattr(model, "config", None)
        return int(getattr(cfg, "num_hidden_layers", 0))

    def get_embed_and_head(self):
        embed = self.model.embed_tokens.weight if self.pp_group.is_first_rank else None
        head = self.lm_head.weight if self.pp_group.is_last_rank else None
        return embed, head

    def set_embed_and_head(self, embed, head):
        if self.pp_group.is_first_rank and embed is not None:
            del self.model.embed_tokens.weight
            self.model.embed_tokens.weight = embed
        if self.pp_group.is_last_rank and head is not None:
            del self.lm_head.weight
            self.lm_head.weight = head
        if _is_xpu:
            torch.xpu.empty_cache()
            torch.xpu.synchronize()
        else:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            # GDN fused projections
            ("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
            ("in_proj_qkvz.", "in_proj_z.", 3),
            ("in_proj_ba.", "in_proj_b.", 0),
            ("in_proj_ba.", "in_proj_a.", 1),
        ]

        loaded_params: Set[str] = set()
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "mtp" in name:
                continue
            if "language_model" in name:
                name = name.replace(r"model.language_model.", r"model.")
            if ".self_attn." in name:
                name = name.replace(".self_attn", "")
            if (
                self.config.tie_word_embeddings
                and self.pp_group.is_last_rank
                and "model.embed_tokens.weight" in name
            ):
                if "lm_head.weight" in params_dict:
                    lm_head_param = params_dict["lm_head.weight"]
                    weight_loader = getattr(
                        lm_head_param, "weight_loader", default_weight_loader
                    )
                    weight_loader(lm_head_param, loaded_weight)
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self, "start_layer")
                and (layer_id < self.start_layer or layer_id >= self.end_layer)
            ):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if "visual" in name or "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                # if is_pp_missing_parameter(name, self):
                #     continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if "visual" in name:
                    # adapt to VisionAttention
                    name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")
                    name = name.replace(r"model.visual.", r"visual.")

                # print(name, loaded_weight.shape)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    logger.warning(f"Parameter {name} not found in params_dict")
                    continue
                param = params_dict[name]

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                if (
                    self.config.tie_word_embeddings
                    and name == "model.embed_tokens.weight"
                    and (_is_cpu and _is_amx_available)
                ):
                    param_lm_head = params_dict["lm_head.weight"]
                    weight_loader = getattr(
                        param_lm_head, "weight_loader", default_weight_loader
                    )
                    weight_loader(param_lm_head, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen3_5MoeForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Qwen3.5 MoE Vision-Language Model."""

    packed_modules_mapping = Qwen3_5ForCausalLM.packed_modules_mapping
    hf_to_sglang_mapper = None

    supported_lora_modules = Qwen3_5ForCausalLM.supported_lora_modules

    def __init__(
        self,
        config: Qwen3_5MoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        language_model_cls=Qwen3_5MoeForCausalLM,
    ) -> None:
        super().__init__(config, quant_config, prefix, language_model_cls)
        rope_config = getattr(self.config, "rope_parameters", None) or getattr(
            self.config, "rope_scaling", {}
        )
        self.is_mrope_enabled = "mrope_section" in rope_config

        self.deepstack_visual_indexes = self.visual.deepstack_visual_indexes
        self.num_fused_shared_experts = 0
        if _use_aiter and not _disable_shared_experts_fusion():
            self.num_fused_shared_experts = self._get_num_fused_shared_experts()

        self.enable_shared_expert_fusion = self.num_fused_shared_experts > 0

    def get_hidden_dim(self, module_name: str, layer_idx: int):
        return self.model.get_hidden_dim(module_name, layer_idx)

    def should_apply_lora(self, module_name: str) -> bool:
        # Accept all language model layer modules (attention, linear_attn, mlp).
        return module_name.startswith("model.layers.")

    def _get_num_fused_shared_experts(self):
        if not (
            hasattr(self.model, "layers")
            and len(self.model.layers) > 0
            and hasattr(self.model.layers[0].mlp, "num_fused_shared_experts")
        ):
            return 0
        return self.model.layers[0].mlp.num_fused_shared_experts

    def get_embed_and_head(self):
        embed = self.model.embed_tokens.weight if self.pp_group.is_first_rank else None
        head = self.lm_head.weight if self.pp_group.is_last_rank else None
        return embed, head

    def set_embed_and_head(self, embed, head):
        if self.pp_group.is_first_rank and embed is not None:
            del self.model.embed_tokens.weight
            self.model.embed_tokens.weight = embed
        if self.pp_group.is_last_rank and head is not None:
            del self.lm_head.weight
            self.lm_head.weight = head
        if _is_xpu:
            torch.xpu.empty_cache()
            torch.xpu.synchronize()
        else:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            # GDN fused projections
            ("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
            ("in_proj_qkvz.", "in_proj_z.", 3),
            ("in_proj_ba.", "in_proj_b.", 0),
            ("in_proj_ba.", "in_proj_a.", 1),
        ]

        num_experts = self.config.num_experts

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=(
                num_experts
                if not self.enable_shared_expert_fusion
                else num_experts + self.num_fused_shared_experts
            ),
        )

        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            "_weight_scale",
            "_input_scale",
        )

        is_fused_expert = False
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]

        if self.enable_shared_expert_fusion:
            """
            When shared experts are fused, we need to map the shared experts to routed experts.

            mlp.share_expert.gate_up_proj.weight  --> experts.512.gate_up_proj.weight -> experts.w13_weight, expert_id = 512
            mlp.share_expert.down_proj.weight  --> experts.512.down_proj.weight -> experts.w2_weight, expert_id = 512
            """
            fused_expert_params_mapping += [
                (
                    "experts.w13_",
                    f"experts.{num_experts}.gate_up_proj.",
                    num_experts,
                    "w1",
                ),
                (
                    "experts.w2_",
                    f"experts.{num_experts}.down_proj.",
                    num_experts,
                    "w2",
                ),
                ## shared experts may contain gate_proj and up_proj instead of gate_up_proj
                (
                    "experts.w13_",
                    f"experts.{num_experts}.gate_proj.",
                    num_experts,
                    "w1",
                ),
                (
                    "experts.w13_",
                    f"experts.{num_experts}.up_proj.",
                    num_experts,
                    "w3",
                ),
            ]

        def load_fused_expert_weights(
            name: str,
            params_dict: dict,
            loaded_weight: torch.Tensor,
            shard_id: str,
            num_experts: int,
        ):
            if name not in params_dict:
                return False
            param = params_dict[name]
            weight_loader = param.weight_loader
            # let ep moe layer to gracefully handle expert_ids that do not belong to local moe rank
            for expert_id in range(num_experts):
                curr_expert_weight = loaded_weight[expert_id]
                weight_loader(
                    param,
                    curr_expert_weight,
                    name,
                    shard_id,
                    expert_id,
                )
            return True

        loaded_params: Set[str] = set()
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "mtp" in name:
                continue
            if "language_model" in name:
                name = name.replace(r"model.language_model.", r"model.")
            if ".self_attn." in name:
                name = name.replace(".self_attn", "")
            if (
                self.config.tie_word_embeddings
                and self.pp_group.is_last_rank
                and "model.embed_tokens.weight" in name
            ):
                if "lm_head.weight" in params_dict:
                    lm_head_param = params_dict["lm_head.weight"]
                    weight_loader = getattr(
                        lm_head_param, "weight_loader", default_weight_loader
                    )
                    weight_loader(lm_head_param, loaded_weight)

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self, "start_layer")
                and (layer_id < self.start_layer or layer_id >= self.end_layer)
            ):
                continue

            if self.enable_shared_expert_fusion:
                if "mlp.shared_expert." in name:
                    # Firstly map mlp.shared_expert.xx_proj to mlp.experts.512.xx_proj
                    name = name.replace(
                        "mlp.shared_expert.",
                        f"mlp.experts.{num_experts}.",
                    )

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if name.endswith("experts.gate_up_proj") or name.endswith(
                    "experts.down_proj"
                ):
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                if "visual" in name:
                    continue

                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra parameters for GPTQ/modelopt models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Track if this is an expert weight to enable early skipping
                is_expert_weight = False

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    if "visual" in name or self.config.encoder_only:
                        continue
                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if is_fused_expert:
                        # is_fused_expert is True, the checkpoint contains gate_up_proj and down_proj for each expert
                        if "experts.gate_up_proj" in name:
                            # experts.gate_up_proj contains all 512 routed experts, excluding shared experts
                            # split into w1 and w3
                            loaded_weight = loaded_weight.chunk(2, dim=-2)
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[0],
                                "w1",
                                num_experts,
                            )
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[1],
                                "w3",
                                num_experts,
                            )
                        elif "experts.down_proj" in name:
                            # experts.down_proj contains all 512 routed experts, excluding shared experts
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            )
                        elif self.enable_shared_expert_fusion:
                            # shared experts should be loaded to experts.w13_weight and experts.w2_weight
                            param = params_dict[name_mapped]
                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            param = params_dict[name_mapped]
                            if f"{num_experts}.gate_up_proj" in name:
                                # split into w1 and w3
                                loaded_weight = loaded_weight.chunk(2, dim=-2)
                                # load to experts.w13_weight, shard_id = w1, expert_id = 512
                                weight_loader(
                                    param,
                                    loaded_weight[0],
                                    name_mapped,
                                    "w1",
                                    expert_id,
                                )
                                # load to experts.w13_weight, shard_id = w3, expert_id = 512
                                weight_loader(
                                    param,
                                    loaded_weight[1],
                                    name_mapped,
                                    "w3",
                                    expert_id,
                                )
                            else:
                                # load down_proj to experts.w2_weight, shard_id = w2, expert_id = 512
                                # Or load gate_proj and up_proj to experts.w13_weight, shard_id = w1/w3, expert_id = 512
                                weight_loader(
                                    param,
                                    loaded_weight,
                                    name_mapped,
                                    shard_id,
                                    expert_id,
                                )
                    else:
                        # Skip loading extra parameters for GPTQ models.
                        if (
                            name_mapped.endswith(ignore_suffixes)
                            and name_mapped not in params_dict
                        ):
                            continue
                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or
                        # not here since otherwise we may skip experts with
                        # # other available replicas.
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    name = name_mapped
                    break
                else:
                    if is_expert_weight:
                        # This is an expert weight but not mapped to this rank, skip all remaining processing
                        continue

                    if "visual" in name:
                        # adapt to VisionAttention
                        name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")
                        name = name.replace(r"model.visual.", r"visual.")

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")
            loaded_params.add(name)

        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.mlp.get_moe_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if isinstance(layer.mlp, Qwen2MoeSparseMoeBlock)
            }
        )

        return loaded_params

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        text_config = getattr(config, "text_config", config)
        return ModelConfigForExpertLocation(
            num_layers=text_config.num_hidden_layers,
            num_logical_experts=text_config.num_experts,
            num_groups=None,
        )


EntryClass = [Qwen3_5MoeForConditionalGeneration, Qwen3_5ForConditionalGeneration]
