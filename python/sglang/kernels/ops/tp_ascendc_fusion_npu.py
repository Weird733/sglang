# Qwen3.5-35B-A3B NPU decode TP 主流线 AscendC 融合 v2（tp_ascendc_fusion_v2 包）开关/守卫模块。
#
# 两个算子（sgl-kernel-npu 主 csrc 注册为 torch.ops.npu.*，算子包与注册清单见
# resource/code/tp_ascendc_fusion_v2/sgl_kernel_npu/csrc/<op>/REGISTRATION.md）：
#   op1 fused_qkvzba_conv1d           GDN decode：qkvzba split + causal_conv1d(UPDATE)
#                                     单 kernel，返回 (y, z, b, a)，conv_states 原地更新。
#                                     【v1 移植】v1 八轮上机唯一精度+性能双过的算子
#                                     （图模式 fused 7.5~18.3µs vs ref 11.2~31.1µs）。
#   op2 fused_sigmoid_gating_recurrent GDN decode：sigmoid gating + delta rule update
#                                     单 kernel（AIV_ONLY），生产 Triton kernel（方案B
#                                     strided 形态）的 drop-in 替代，ssm pool 原地更新。
#                                     【v2 新增】访存模式换代：每 (slot,head) 32KB 全连续
#                                     DMA，摆脱 BV=64 半行模式。
#
# v1 的 fused_norm_qkv_proj_scatter / fused_sigmoid_mul_mm 不随 v2 构建（v1 第八轮
# 止损下线：MIX 单 kernel 固定开销 ~21-23µs，图模式落后 stock 1.2~3.2x 且天花板≈追平）。
# 为兼容服务器上可能存在的 v1 版 qwen3_5.py（其 import 本模块的 5 个守卫函数），
# 这两个算子的守卫函数以「恒停用」桩形式保留（返回 False/None，永不激活）——
# 与 v1 第八轮的默认下线状态一致。
#
# 开关：
#   SGLANG_NPU_TP_ASCENDC_FUSION          总开关（默认 "0"），仅 op1 跟随
#   SGLANG_NPU_TP_ASCENDC_FUSION_QKVZBA   op1 分开关：未设置随总开关，显式 "0" 单关
#   SGLANG_NPU_GDN_RECURRENT_ASCENDC      op2 开关（默认 "0"）；在 gdn_triton.py import
#                                         期判定，须在建图/服务启动前设置
#   SGLANG_NPU_TP_ASCENDC_FUSION_DEBUG=1  守卫未命中按 key 一次性打印
# 图 capture 时守卫判定随 Python 分支烘进图，capture 后改 env 无效。
#
# 部署：本文件新增到 <sglang>/python/sglang/kernels/ops/tp_ascendc_fusion_npu.py
# （沿用 v1 模块名直接覆盖；op1 接口与 v1 逐字一致，op2/3 接口为恒停用桩）。

import logging
import os

import torch

logger = logging.getLogger(__name__)

_TP_FUSION_ENV = "SGLANG_NPU_TP_ASCENDC_FUSION"
_TP_FUSION_QKVZBA_ENV = "SGLANG_NPU_TP_ASCENDC_FUSION_QKVZBA"
_GDN_RECURRENT_ASCENDC_ENV = "SGLANG_NPU_GDN_RECURRENT_ASCENDC"
_TP_DEBUG_ENV = "SGLANG_NPU_TP_ASCENDC_FUSION_DEBUG"
_tp_debug_logged = set()

# op host 侧硬约束（与 csrc host TORCH_CHECK 逐条对应，见各 REGISTRATION.md）：
_TP_C0_ALIGN = 16  # bf16 C0 对齐（op1 行宽须 16 倍数）
_RECURRENT_HEAD_DIM = 128  # fused_sigmoid_gating_recurrent kernel 特化 K/V
_RECURRENT_MAX_HV = 8  # 同上 gating [8] pad 上限
_RECURRENT_MAX_N = 256  # 同上 cu/idx UB 驻留上限（生产 bs<=128）

_TP_OP_CACHE = {}


def tp_fusion_enabled() -> bool:
    """TP 线 AscendC 融合总开关（默认关；="1" 启用，仅 op1 跟随）。"""
    return os.environ.get(_TP_FUSION_ENV, "0") == "1"


def tp_fusion_qkvzba_enabled() -> bool:
    """op1（fused_qkvzba_conv1d）开关：分开关未设置时随总开关，显式 "0" 单独关。
    图模式实证领先（v1 ut_result_4：7.5~18.3 vs 11.2~31.1µs），v2 保留。"""
    return os.environ.get(
        _TP_FUSION_QKVZBA_ENV, os.environ.get(_TP_FUSION_ENV, "0")
    ) == "1"


def gdn_recurrent_ascendc_enabled() -> bool:
    """op2（fused_sigmoid_gating_recurrent，AscendC recurrent）开关：默认 "0"，
    显式 "1" 才开。精度走 re-baseline 验收路径（见包 README「精度闸门」节）。"""
    return os.environ.get(_GDN_RECURRENT_ASCENDC_ENV, "0") == "1"


def tp_debug_log(key, msg):
    """debug 开关打开时按 key 一次性打日志（定位守卫未命中环节用）。"""
    if os.environ.get(_TP_DEBUG_ENV, "0") != "1":
        return
    if key in _tp_debug_logged:
        return
    _tp_debug_logged.add(key)
    logger.warning("[tp_ascendc_fusion_v2] %s", msg)


def tp_op_available(name: str) -> bool:
    """torch.ops.npu 上算子是否已注册（即 sgl-kernel-npu 是否已带本包算子构建）。

    注册发生在 sgl_kernel_npu 扩展加载时（进程启动期），结果进程内静态，故缓存。
    """
    got = _TP_OP_CACHE.get(name)
    if got is None:
        got = hasattr(torch.ops.npu, name)
        _TP_OP_CACHE[name] = got
    return got


# ---------------------------------------------------------------------------
# op1（GDN decode）：fused_qkvzba_conv1d
#   = fused_qkvzba_split_reshape_cat_contiguous + causal_conv1d(run_mode=1)
#   【v1 原样移植，八轮上机验证通过】
# ---------------------------------------------------------------------------
def tp_fused_qkvzba_conv1d_shape_supported(
    num_k_heads_tp, num_v_heads_tp, head_k_dim, head_v_dim, conv_kernel_size
) -> bool:
    """init 期烘定的形状守卫（Qwen3_5GatedDeltaNet 级；头数为 TP 切分后 per-rank 值）。

    镜像算子 host TORCH_CHECK：num_v%num_k==0 且比值 ∈ {1,2,4}（与被替代的
    qwen3_5.py split 分支条件一致）、conv width ∈ [2,4]、qkvWidth 与 qkvz 行宽
    16 对齐（z 拷贝 32B DataCopy 对齐）、z 行字节 ≤ 65535（单块 DataCopy 上限）。
    """
    ok = (
        tp_fusion_qkvzba_enabled()
        and tp_op_available("fused_qkvzba_conv1d")
        and num_k_heads_tp > 0
        and num_v_heads_tp > 0
        and head_k_dim > 0
        and head_v_dim > 0
        and num_v_heads_tp % num_k_heads_tp == 0
        and (num_v_heads_tp // num_k_heads_tp) in (1, 2, 4)
        and 2 <= conv_kernel_size <= 4
    )
    if ok:
        qkv_width = 2 * num_k_heads_tp * head_k_dim + num_v_heads_tp * head_v_dim
        x_row_stride = qkv_width + num_v_heads_tp * head_v_dim
        ok = (
            qkv_width % _TP_C0_ALIGN == 0
            and x_row_stride % _TP_C0_ALIGN == 0
            and num_v_heads_tp * head_v_dim * 2 <= 65535
        )
    if not ok:
        tp_debug_log(
            ("qkvzba_shape",),
            "fused_qkvzba_conv1d init 形状守卫未命中（回退 stock split+causal_conv1d）："
            f"nk_tp={num_k_heads_tp} nv_tp={num_v_heads_tp} dk={head_k_dim} "
            f"dv={head_v_dim} width={conv_kernel_size} "
            f"op_registered={tp_op_available('fused_qkvzba_conv1d')} "
            f"{_TP_FUSION_QKVZBA_ENV}={os.environ.get(_TP_FUSION_QKVZBA_ENV, '<随总开关>')!r} "
            f"{_TP_FUSION_ENV}={os.environ.get(_TP_FUSION_ENV, '0')!r}",
        )
    return ok


def tp_fused_qkvzba_conv1d_inputs_ok(
    qkvz, mixed_ba, num_k_heads_tp, num_v_heads_tp, head_k_dim, head_v_dim
) -> bool:
    """backend 侧运行期轻量判定（tuple 输入的张量属性；形状主集合已在 init 烘定）。

    v2.1（gdn_qkvzba_pack_v1_2）：连续性契约放宽为「行距视图」——列内连续
    （stride(1)==1）、行距 >= 逻辑宽度且 qkvz 行距 16 倍数（镜像 host 对 z 拷贝
    32B 对齐的 TORCH_CHECK；pack 打包 N 已补到 16 倍数）。连续输入天然满足，
    与 v1 行为一致；行距视图（pack 打包 GEMM 输出切片）直读，免两次 .contiguous()。
    注意：本判定须与本包 csrc host 同版本部署（旧 host 仍 TORCH_CHECK is_contiguous）。
    """
    qkv_width = 2 * num_k_heads_tp * head_k_dim + num_v_heads_tp * head_v_dim
    return (
        torch.is_tensor(qkvz)
        and torch.is_tensor(mixed_ba)
        and qkvz.dim() == 2
        and mixed_ba.dim() == 2
        and qkvz.dtype == torch.bfloat16
        and mixed_ba.dtype == torch.bfloat16
        and qkvz.stride(1) == 1
        and qkvz.stride(0) >= qkvz.shape[1]
        and qkvz.stride(0) % _TP_C0_ALIGN == 0
        and mixed_ba.stride(1) == 1
        and mixed_ba.stride(0) >= mixed_ba.shape[1]
        and qkvz.shape[1] == qkv_width + num_v_heads_tp * head_v_dim
        and mixed_ba.shape[0] == qkvz.shape[0]
        and mixed_ba.shape[1] == 2 * num_v_heads_tp
    )


# ---------------------------------------------------------------------------
# op2（GDN decode）：fused_sigmoid_gating_recurrent（AscendC recurrent，v2 新增）
#
# 精度门槛（动工前已与评估文档 §2.2 对齐）：与生产 Triton kernel 逐 bit 不可先验承诺
# （K 维归约加法顺序 + Exp/Ln/Div 指令级差异），验收 = bf16 臂逐 bit 或 1-ulp 有界 +
# fp32 臂误差预算（≤ 现 Triton 误差）+ 长程漂移有界 + e2e A/B，RL 显式签字。
# ---------------------------------------------------------------------------
def _ascendc_recurrent_supported(
    q, k, v, a, b, initial_state_source, initial_state_indices, cu_seqlens,
    A_log, dt_bias,
) -> bool:
    """运行期守卫（镜像 host TORCH_CHECK；未命中 → wrapper 回退 stock Triton）。

    只读张量属性（shape/stride/dtype/连续性），不读 device 数据，graph capture 安全。
    A_log/dt_bias 的 fp32 dtype 不在此卡——wrapper 侧做无损加宽转换（见调用处），
    此处只校验 numel 防 host TORCH_CHECK 崩出。
    """
    ok = (
        tp_op_available("fused_sigmoid_gating_recurrent")
        and torch.is_tensor(q)
        and q.dim() == 4
        and q.size(0) == 1
        and cu_seqlens is not None
        and torch.is_tensor(cu_seqlens)
        and q.size(1) == cu_seqlens.numel() - 1  # T == N：每序列 1 token
        and 1 <= q.size(1) <= _RECURRENT_MAX_N
        and q.size(3) == _RECURRENT_HEAD_DIM
        and v.size(3) == _RECURRENT_HEAD_DIM
        and q.size(2) >= 1
        and v.size(2) >= 1
        and v.size(2) <= _RECURRENT_MAX_HV
        and v.size(2) % q.size(2) == 0
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and a.dtype == torch.bfloat16
        and b.dtype == torch.bfloat16
        and initial_state_source.dtype in (torch.bfloat16, torch.float32)
        and q.stride(3) == 1
        and k.stride(3) == 1
        and v.stride(3) == 1
        and q.stride(2) == q.size(3)
        and k.stride(2) == k.size(3)
        and v.stride(2) == v.size(3)
        and a.is_contiguous()
        and b.is_contiguous()
        and initial_state_source.is_contiguous()
        and initial_state_source.dim() == 4
        and initial_state_source.size(1) == v.size(2)
        and initial_state_source.size(2) == _RECURRENT_HEAD_DIM
        and initial_state_source.size(3) == _RECURRENT_HEAD_DIM
        and initial_state_indices is not None
        and initial_state_indices.numel() >= q.size(1)
        and A_log.numel() == v.size(2)  # host: A_log/dt_bias numel == HV
        and dt_bias.numel() == v.size(2)
    )
    if not ok:
        tp_debug_log(
            ("recurrent_shape",),
            "fused_sigmoid_gating_recurrent 运行期守卫未命中（回退 stock Triton）："
            f"q={tuple(q.shape)}/{q.dtype} v={tuple(v.shape)}/{v.dtype} "
            f"pool={tuple(initial_state_source.shape)}/{initial_state_source.dtype} "
            f"T_vs_N={q.size(1)}/{cu_seqlens.numel() - 1 if torch.is_tensor(cu_seqlens) else None} "
            f"op_registered={tp_op_available('fused_sigmoid_gating_recurrent')}",
        )
    return ok


def fused_sigmoid_gating_delta_rule_update_ascendc(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    initial_state_source,
    initial_state_indices,
    scale=None,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
):
    """生产 Triton wrapper（fused_sigmoid_gating_delta_rule_update_npu，方案B 形态）
    同签名同语义的 AscendC 版 drop-in，仅 decode（T==N）。

    守卫未命中 → 回退 stock Triton wrapper（静默回退，与本包纪律一致；
    SGLANG_NPU_TP_ASCENDC_FUSION_DEBUG=1 时打一次原因）。
    """
    if not _ascendc_recurrent_supported(
        q, k, v, a, b, initial_state_source, initial_state_indices, cu_seqlens,
        A_log, dt_bias,
    ):
        from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent import (
            fused_sigmoid_gating_delta_rule_update_npu,
        )

        return fused_sigmoid_gating_delta_rule_update_npu(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=initial_state_source,
            initial_state_indices=initial_state_indices,
            scale=scale,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
        )
    # scale 默认值与 Triton wrapper 逐字一致（python 侧 K**-0.5，同 double→float 路径）
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    # A_log/dt_bias：host 仅收 fp32，但生产实版 checkpoint 载入后可能为 bf16
    # （stock Triton kernel 本就 tl.load(...).to(tl.float32)，任意 dtype 都能跑）。
    # wrapper 侧做无损加宽（bf16→fp32 值域严格包含，与 Triton 核内转换逐 bit 等价），
    # 并镜像 Triton wrapper 的 .contiguous()；[HV] 小向量，开销可忽略。
    if A_log.dtype != torch.float32:
        A_log = A_log.float()
    if dt_bias.dtype != torch.float32:
        dt_bias = dt_bias.float()
    A_log = A_log.contiguous()
    dt_bias = dt_bias.contiguous()
    return torch.ops.npu.fused_sigmoid_gating_recurrent(
        A_log,
        a,
        dt_bias,
        float(softplus_beta),
        float(softplus_threshold),
        q,
        k,
        v,
        b,
        initial_state_source,
        initial_state_indices,
        float(scale),
        cu_seqlens,
        bool(use_qk_l2norm_in_kernel),
        q.stride(1),
        k.stride(1),
        v.stride(1),
    )


# ---------------------------------------------------------------------------
# v1 遗留桩（恒停用）：fused_norm_qkv_proj_scatter / fused_sigmoid_mul_mm
# v1 第八轮止损下线（图模式落后 stock 1.2~3.2x、天花板≈追平），v2 不构建这两个算子。
# 服务器若存有 v1 版 qwen3_5.py（import 下列 5 个函数），桩保证其可正常运行且
# 两个候选永不激活；kernel 与单测留档于 code/tp_ascendc_fusion/（eager 口径仍
# 3~5x，若有 eager 场景需显式打开，请回退部署 v1 包）。
# ---------------------------------------------------------------------------
def tp_fusion_nqps_enabled() -> bool:
    """v1 候选2 开关桩：v2 恒 False（算子不构建，恒停用）。"""
    return False


def tp_fusion_sigmm_enabled() -> bool:
    """v1 候选3 开关桩：v2 恒 False（算子不构建，恒停用）。"""
    return False


def tp_norm_qkv_scatter_shape_supported(layer) -> bool:
    """v1 候选2 init 守卫桩：v2 恒 False。"""
    return False


def tp_norm_qkv_scatter_context(
    layer, hidden_states, residual, forward_batch, captured_last_layer_outputs
):
    """v1 候选2 运行期守卫桩：v2 恒 None（调用方回退原链路）。"""
    return None


def tp_sigmoid_mul_mm_shape_supported(o_proj) -> bool:
    """v1 候选3 init 守卫桩：v2 恒 False。"""
    return False


def tp_sigmoid_mul_mm_runtime_ok(attn_output, gate) -> bool:
    """v1 候选3 运行期守卫桩：v2 恒 False。"""
    return False
