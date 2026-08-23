# Qwen3.5-35B-A3B NPU decode 全注意力段融合 kernel（full_attention 优化点 A1b/A2/A3a）。
#
# 内容与收益（图内 UT，资产 resource/code/full_attention/，
# 裁决见 docs/practice/npu-qwen35-full-attn-decode-fusion-opt.md）：
#   A1b split_qkvgate_v4_scatter：split q/gate/k/v + q/k gemma rmsnorm + rope
#       单 kernel，且 k/v 按 loc 直接 scatter 进 KV cache（替代 stock
#       split_qkvgate_gemma_rmsnorm_rope 13.5us + 2x npu_scatter_nd_update_ 29us，
#       一体 5.32us）；返回的 k/v 为 None（cache 已写，调用方须跳过 set_kv_buffer）。
#   A2  fa_sigmoid_mul：sigmoid(gate)*attn 融合单 kernel（替代 sigmoid+mul_ 两
#       kernel，3.34→2.29us）；bitwise 模式先舍 bf16 再乘，与 stock 逐位一致。
#   A3a fa_add_gemma_rms_norm_v2：grid=(batch,) 行并行（stock 按核数开），
#       5.30→4.01us；数学与 stock add_gemma_rms_norm 完全一致。
#
# 精度：三个 kernel 在图内 UT 均 bitwise=1.0（bf16 按 int16 逐位对比 stock）。
# 数值约定与 sgl_kernel_npu stock 逐字一致：fp32 中间精度、gemma (w+1)、
# neox rotate_half rope（cat=[-x2,x1]; out=cat*sin+rot*cos）。
#
# 安全网：每个调用点先过守卫（精确命中已验证形状才走自研 kernel，否则回退
# stock）；环境变量 SGLANG_NPU_FULL_ATTN_FUSION=0 可整体关停（图模式下守卫
# 判定在 capture 时烘进图，需在建图前设置）。
#
# 定位日志：SGLANG_NPU_FULL_ATTN_FUSION_DEBUG=1 时，A1b 守卫按层一次性打印
# 「未命中原因」或「已激活」确认（服务器上确认 A1b 是否生效/卡在哪一环用）。
#
# ⚠️ triton-ascend 约束（GMM2 黑名单 + full_attention v2/v3 两轮实证）：
# 不要按 program_id 做运行时段选择、不要用标量条件 mask store、不要 2D masked
# store——v2/v3 均因此误编译。本文件全部 kernel 只用已实证构造子集：
# 纯线性索引 + 每 program 固定职责 + 1D 无掩码 store + constexpr if。

import logging
import os

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

logger = logging.getLogger(__name__)

_FA_FUSION_ENV = "SGLANG_NPU_FULL_ATTN_FUSION"
_FA_DEBUG_ENV = "SGLANG_NPU_FULL_ATTN_FUSION_DEBUG"
_fa_debug_logged = set()


def fa_fusion_enabled() -> bool:
    """full_attention 融合总开关（默认开；=0 时全部回退 stock）。"""
    return os.environ.get(_FA_FUSION_ENV, "1") == "1"


def _fa_debug_log(key, msg):
    """debug 开关打开时按 key 一次性打日志（定位守卫未命中环节用）。"""
    if os.environ.get(_FA_DEBUG_ENV, "0") != "1":
        return
    if key in _fa_debug_logged:
        return
    _fa_debug_logged.add(key)
    logger.warning("[full_attention_fusion] %s", msg)


# ---------------------------------------------------------------------------
# A3a：add_gemma_rms_norm 行并行版（grid=(batch,)，每 program 一整行）
# 数学与 stock add_gemma_rms_norm 完全一致（bf16 加残差→fp32 归约→fp32 (w+1)）。
# P1 已实证 bitwise=1.0、1.32x。
# ---------------------------------------------------------------------------
@triton.jit
def add_gemma_rms_norm_v2_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    add_out_ptr,
    norm_out_ptr,
    eps,
    DIM: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, DIM)
    x = tl.load(x_ptr + row * DIM + cols)
    r = tl.load(res_ptr + row * DIM + cols)
    add = x + r
    tl.store(add_out_ptr + row * DIM + cols, add)
    xf = add.to(tl.float32)
    w = tl.load(w_ptr + cols).to(tl.float32) + 1.0
    var = tl.sum(xf * xf) / DIM
    out = xf * tl.rsqrt(var + eps) * w
    tl.store(norm_out_ptr + row * DIM + cols, out.to(x_ptr.dtype.element_ty))


def fa_add_gemma_rms_norm_v2_supported(x, residual) -> bool:
    """A3a 形状守卫：2D、bf16、连续、hidden 为 2 的幂（UT 验证口径 [32,2048]，
    kernel 对任意 batch 行数逐位等价；hidden 非 2 的幂时 tl.arange 不合法）。"""
    if not fa_fusion_enabled():
        return False
    if residual is None or x.dim() != 2:
        return False
    if x.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16:
        return False
    if not (x.is_contiguous() and residual.is_contiguous()):
        return False
    dim = x.shape[-1]
    return dim > 0 and (dim & (dim - 1)) == 0


def fa_add_gemma_rms_norm_v2(x, weight, residual, eps):
    """返回 (norm_out, add_out)，语义同 sgl_kernel_npu add_gemma_rms_norm。
    调用前须过 fa_add_gemma_rms_norm_v2_supported。"""
    add_out = torch.empty_like(x)
    norm_out = torch.empty_like(x)
    B, D = x.shape
    add_gemma_rms_norm_v2_kernel[(B,)](
        x, residual, weight, add_out, norm_out, eps, DIM=D
    )
    return norm_out, add_out


# ---------------------------------------------------------------------------
# A2：sigmoid(gate) * attn 融合（替代 torch.sigmoid + mul_ 两个 kernel）
# ---------------------------------------------------------------------------
@triton.jit
def sigmoid_mul_kernel(
    attn_ptr,
    gate_ptr,
    out_ptr,
    numel,
    BLOCK: tl.constexpr,
    BITWISE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    a = tl.load(attn_ptr + offs, mask=mask).to(tl.float32)
    g = tl.load(gate_ptr + offs, mask=mask).to(tl.float32)
    s = tl.sigmoid(g)
    if BITWISE:
        # 模拟 stock 两个 op 之间的 bf16 中间舍入：sigmoid 先舍到 bf16 再乘，
        # 与 attn.mul_(torch.sigmoid(gate)) 目标逐位一致（P4 已实证 bitwise=1.0）。
        s = s.to(tl.bfloat16).to(tl.float32)
    out = (a * s).to(tl.bfloat16)
    tl.store(out_ptr + offs, out, mask=mask)


def fa_sigmoid_mul_supported(attn_output, gate) -> bool:
    """A2 形状守卫：等形、bf16、连续（elementwise 融合，无形状特化）。"""
    if not fa_fusion_enabled():
        return False
    if attn_output.dtype != torch.bfloat16 or gate.dtype != torch.bfloat16:
        return False
    if attn_output.shape != gate.shape:
        return False
    return attn_output.is_contiguous() and gate.is_contiguous()


def fa_sigmoid_mul(attn, gate):
    """out = attn * sigmoid(gate)，与 stock 双 op 舍入路径逐位一致。
    调用前须过 fa_sigmoid_mul_supported。"""
    out = torch.empty_like(attn)
    numel = attn.numel()
    BLOCK = 512
    sigmoid_mul_kernel[(triton.cdiv(numel, BLOCK),)](
        attn, gate, out, numel, BLOCK=BLOCK, BITWISE=True
    )
    return out


# ---------------------------------------------------------------------------
# A1b（v4，行并行直线版）：split q/kv/gate + q/k gemma rmsnorm + rope，
# 且 k/v 按 loc 直接 scatter 进 KV cache（免中间张量与两次 scatter_nd_update）。
# grid=(batch,)：一 program 一行，q+gate / k / v 三段在一条直线里顺序做完。
# 设计约束（v2/v3 失败教训）：pid 只做线性索引（不取整除/取模）、段间无运行时
# 选择、store 不带标量条件 mask；SCATTER_KV 为 constexpr 分支，本文件生产
# wrapper 恒 =1，k/v 中间张量写出整段编译期消除。
# TP8 shape 特化：NUM_Q_HEADS=2, NUM_KV_HEADS=1。
# ---------------------------------------------------------------------------
@triton.jit
def split_qkvgate_v4_kernel(
    input_ptr,
    sin_ptr,
    cos_ptr,
    q_ptr,
    gate_ptr,
    k_ptr,
    v_ptr,
    kbuf_ptr,
    vbuf_ptr,
    loc_ptr,
    q_weight_ptr,
    k_weight_ptr,
    q_hidden_size: tl.constexpr,
    kv_hidden_size: tl.constexpr,
    total_hidden_size: tl.constexpr,
    eps: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,  # 2
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HALF_ROPE_DIM: tl.constexpr,
    SCATTER_KV: tl.constexpr,
):
    row = tl.program_id(0)
    base = input_ptr + row * total_hidden_size
    d = tl.arange(0, HEAD_DIM)

    # ---- q+gate：qkv 行前 2*q_hidden_size 连续读入，reshape 成 (头数, 512) 后
    # 逐头切出 [q(256) | gate(256)]（与 stock 的 extract_slice 切法完全一致）
    qg_cols = tl.arange(0, 2 * NUM_Q_HEADS * HEAD_DIM)
    qg = tl.load(base + qg_cols).to(tl.float32).reshape(
        NUM_Q_HEADS, 2 * HEAD_DIM
    )
    q = al.extract_slice(
        qg, offsets=(0, 0), sizes=(NUM_Q_HEADS, HEAD_DIM), strides=(1, 1)
    )
    gate = al.extract_slice(
        qg, offsets=(0, HEAD_DIM), sizes=(NUM_Q_HEADS, HEAD_DIM), strides=(1, 1)
    )

    # q norm（fp32 中间、gemma w+1；运算顺序与 stock 逐字一致）
    qw = tl.load(q_weight_ptr + d).to(tl.float32) + 1.0
    qvar = tl.sum(q * q, axis=1) / HEAD_DIM
    qn = q * tl.rsqrt(qvar + eps)[:, None] * qw[None, :]

    # rope（neox rotate_half，仅前 ROPE_DIM 维；与 v3 已实证块逐字一致）
    sc = row * ROPE_DIM + tl.arange(0, ROPE_DIM)
    sin = tl.load(sin_ptr + sc).to(tl.float32).reshape(1, ROPE_DIM)
    cos = tl.load(cos_ptr + sc).to(tl.float32).reshape(1, ROPE_DIM)
    rot = al.extract_slice(
        qn, offsets=(0, 0), sizes=(NUM_Q_HEADS, ROPE_DIM), strides=(1, 1)
    )
    x1 = al.extract_slice(
        rot, offsets=(0, 0), sizes=(NUM_Q_HEADS, HALF_ROPE_DIM), strides=(1, 1)
    )
    x2 = al.extract_slice(
        rot, offsets=(0, HALF_ROPE_DIM), sizes=(NUM_Q_HEADS, HALF_ROPE_DIM), strides=(1, 1)
    )
    cat = tl.zeros((NUM_Q_HEADS, ROPE_DIM), dtype=tl.float32)
    cat = al.insert_slice(
        cat, -x2, offsets=(0, 0), sizes=(NUM_Q_HEADS, HALF_ROPE_DIM), strides=(1, 1)
    )
    cat = al.insert_slice(
        cat, x1, offsets=(0, HALF_ROPE_DIM), sizes=(NUM_Q_HEADS, HALF_ROPE_DIM), strides=(1, 1)
    )
    roped = cat * sin + rot * cos
    qn = al.insert_slice(
        qn, roped, offsets=(0, 0), sizes=(NUM_Q_HEADS, ROPE_DIM), strides=(1, 1)
    )

    # q/gate 写出：1D 无掩码 store（stock reshape(Q_BLOCK) 同款；2D masked store
    # 是 v3 的嫌疑构造，v4 全避开）
    qg_out = row * q_hidden_size + tl.arange(0, NUM_Q_HEADS * HEAD_DIM)
    tl.store(
        q_ptr + qg_out,
        qn.reshape(NUM_Q_HEADS * HEAD_DIM).to(input_ptr.dtype.element_ty),
    )
    tl.store(
        gate_ptr + qg_out,
        gate.reshape(NUM_Q_HEADS * HEAD_DIM).to(input_ptr.dtype.element_ty),
    )

    # ---- k（1 头）：norm + rope，与 stock k 段的 (1, HEAD_DIM) 形态一致
    kt = tl.load(base + 2 * q_hidden_size + d).to(tl.float32).reshape(1, HEAD_DIM)
    kw = tl.load(k_weight_ptr + d).to(tl.float32) + 1.0
    kvar = tl.sum(kt * kt, axis=1) / HEAD_DIM
    kn = kt * tl.rsqrt(kvar + eps)[:, None] * kw[None, :]
    krot = al.extract_slice(kn, offsets=(0, 0), sizes=(1, ROPE_DIM), strides=(1, 1))
    kx1 = al.extract_slice(
        krot, offsets=(0, 0), sizes=(1, HALF_ROPE_DIM), strides=(1, 1)
    )
    kx2 = al.extract_slice(
        krot, offsets=(0, HALF_ROPE_DIM), sizes=(1, HALF_ROPE_DIM), strides=(1, 1)
    )
    kcat = tl.zeros((1, ROPE_DIM), dtype=tl.float32)
    kcat = al.insert_slice(
        kcat, -kx2, offsets=(0, 0), sizes=(1, HALF_ROPE_DIM), strides=(1, 1)
    )
    kcat = al.insert_slice(
        kcat, kx1, offsets=(0, HALF_ROPE_DIM), sizes=(1, HALF_ROPE_DIM), strides=(1, 1)
    )
    kroped = kcat * sin + krot * cos
    kn = al.insert_slice(kn, kroped, offsets=(0, 0), sizes=(1, ROPE_DIM), strides=(1, 1))

    # ---- v：bf16 原样拷贝（stock v 段不过 fp32）
    vt = tl.load(base + 2 * q_hidden_size + kv_hidden_size + d)

    # constexpr 分支（A2 BITWISE 同族实证）：SCATTER_KV=1 时 k/v 中间张量写出
    # 整段编译期消除，k_ptr/v_ptr 占位指针不会被解引用
    if SCATTER_KV:
        slot = tl.load(loc_ptr + row).to(tl.int64)
        tl.store(
            kbuf_ptr + slot * kv_hidden_size + d,
            kn.reshape(HEAD_DIM).to(input_ptr.dtype.element_ty),
        )
        tl.store(vbuf_ptr + slot * kv_hidden_size + d, vt)
    else:
        tl.store(
            k_ptr + row * kv_hidden_size + d,
            kn.reshape(HEAD_DIM).to(input_ptr.dtype.element_ty),
        )
        tl.store(v_ptr + row * kv_hidden_size + d, vt)


def fa_split_qkvgate_scatter_supported(
    num_heads, num_kv_heads, head_dim, rope_dim, attn_output_gate
) -> bool:
    """A1b 形状守卫（TP 布局特化，UT 验证口径）：每 rank q=2 头、kv=1 头、
    head_dim=256、rope=64、带 attn_output_gate。其余 TP/模型配置回退 stock。"""
    ok = (
        fa_fusion_enabled()
        and num_heads == 2
        and num_kv_heads == 1
        and head_dim == 256
        and rope_dim == 64
        and attn_output_gate
    )
    if not ok:
        _fa_debug_log(
            ("shape_guard",),
            "A1b 形状守卫未命中（回退 stock）："
            f"num_heads={num_heads}(需2) num_kv_heads={num_kv_heads}(需1) "
            f"head_dim={head_dim}(需256) rope_dim={rope_dim}(需64) "
            f"gate={attn_output_gate}(需True) "
            f"{_FA_FUSION_ENV}={os.environ.get(_FA_FUSION_ENV, '1')!r}",
        )
    return ok


def fa_kv_pool_buffers(layer_id):
    """从当前 NPU attention backend 的 KV pool 取该层 k/v buffer。
    仅在 use_fia 布局（[pages*page_size, 1, head_num, head_dim]，与 token 级
    slot=loc 直写口径一致）且非 hybrid SWA 时返回 (kbuf, vbuf)，否则返回 None。

    GDN 混合模型的 pool 是 HybridLinearKVPool 包装：use_fia 判定要读内层
    full_kv_pool（NPUMHATokenToKVPool，外层无此属性）；取 buffer 走外层
    get_key/value_buffer——外层按 full_attention_layer_id_mapping 把全局
    layer_id 翻译成全注意力局部序号后再 delegate（内层 pool 只有全注意力层，
    直接用全局 id 会取错层）。
    """
    from sglang.srt.model_executor.forward_context import get_attn_backend

    backend = get_attn_backend()
    if getattr(backend, "is_hybrid_swa", False):
        _fa_debug_log(
            ("pool_swa", layer_id),
            f"layer {layer_id}: backend is_hybrid_swa=True，A1b 回退 stock",
        )
        return None
    pool = getattr(backend, "token_to_kv_pool", None)
    if pool is None:
        _fa_debug_log(
            ("pool_none", layer_id),
            f"layer {layer_id}: backend {type(backend).__name__} 无 "
            "token_to_kv_pool，A1b 回退 stock",
        )
        return None
    inner = getattr(pool, "full_kv_pool", pool)
    if not getattr(inner, "use_fia", False):
        _fa_debug_log(
            ("pool_fia", layer_id),
            f"layer {layer_id}: pool {type(pool).__name__} 内层 "
            f"{type(inner).__name__}.use_fia=False（ASCEND_USE_FIA 未开？），"
            "A1b 回退 stock",
        )
        return None
    return pool.get_key_buffer(layer_id), pool.get_value_buffer(layer_id)


def fa_v4_scatter_context(layer, forward_batch, qkv):
    """A1b 运行时守卫总入口：全部条件命中返回 (kbuf, vbuf, loc)，任一未命中
    返回 None（调用方回退 stock split + stock set_kv_buffer，行为与 origin
    完全一致）。DEBUG=1 时按层一次性打印未命中原因/激活确认。"""
    layer_id = layer.attn.layer_id
    if not getattr(layer, "_fa_v4_shape_ok", False):
        _fa_debug_log(
            ("shape", layer_id),
            f"layer {layer_id}: _fa_v4_shape_ok=False（见 init 时形状守卫日志），"
            "A1b 回退 stock",
        )
        return None
    if not forward_batch.forward_mode.is_decode():
        _fa_debug_log(
            ("mode", layer_id),
            f"layer {layer_id}: forward_mode={forward_batch.forward_mode} 非 "
            "DECODE（MTP verify / mixed / prefill 均不走 A1b），回退 stock",
        )
        return None
    if not torch.is_tensor(qkv) or qkv.dtype != torch.bfloat16 or not qkv.is_contiguous():
        _fa_debug_log(
            ("qkv", layer_id),
            f"layer {layer_id}: qkv 不满足（tensor={torch.is_tensor(qkv)} "
            f"dtype={getattr(qkv, 'dtype', None)} "
            f"contiguous={qkv.is_contiguous() if torch.is_tensor(qkv) else None}），"
            "A1b 回退 stock",
        )
        return None
    kv_bufs = fa_kv_pool_buffers(layer_id)
    if kv_bufs is None:
        return None  # 具体原因已在 fa_kv_pool_buffers 内打日志
    kbuf, vbuf = kv_bufs
    if kbuf.dtype != qkv.dtype:
        _fa_debug_log(
            ("kvdtype", layer_id),
            f"layer {layer_id}: KV cache dtype={kbuf.dtype} != qkv "
            f"dtype={qkv.dtype}（如 fp8 KV cache），A1b 回退 stock",
        )
        return None
    _fa_debug_log(
        ("active", layer_id),
        f"layer {layer_id}: A1b 融合 split+KV scatter 已激活",
    )
    return kbuf, vbuf, forward_batch.out_cache_loc


def fa_split_qkvgate_scatter(
    input,
    sin,
    cos,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    rope_dim,
    eps,
    q_weight,
    k_weight,
    kbuf,
    vbuf,
    loc,
):
    """stock split_qkvgate_gemma_rmsnorm_rope 的行并行直线版（grid=(batch,)），
    k/v 不写中间张量、按 loc 直接 scatter 进 KV cache。
    返回 (q, None, None, gate)：k/v 为 None 表示 cache 已写，
    调用方须以 save_kv_cache=False 走 attention backend。
    仅支持 q=2 头、kv=1 头（TP8 shape 特化），调用前须过
    fa_v4_scatter_context 守卫。"""
    batch = input.shape[0]
    num_q = q_hidden_size // head_dim
    num_kv = kv_hidden_size // head_dim
    assert num_q == 2 and num_kv == 1, "v4 为 TP8 shape 特化（q=2 头, kv=1 头）"
    assert kbuf.dtype == input.dtype and vbuf.dtype == input.dtype
    q_out = torch.empty(batch, q_hidden_size, device=input.device, dtype=input.dtype)
    gate_out = torch.empty(batch, q_hidden_size, device=input.device, dtype=input.dtype)
    # SCATTER_KV=1：k_ptr/v_ptr 为占位指针（q_out/gate_out），编译期消除不会被写
    split_qkvgate_v4_kernel[(batch,)](
        input,
        sin,
        cos,
        q_out,
        gate_out,
        q_out,
        gate_out,
        kbuf,
        vbuf,
        loc,
        q_weight,
        k_weight,
        q_hidden_size,
        kv_hidden_size,
        q_hidden_size * 2 + kv_hidden_size * 2,
        eps,
        NUM_Q_HEADS=2,
        HEAD_DIM=head_dim,
        ROPE_DIM=rope_dim,
        HALF_ROPE_DIM=rope_dim // 2,
        SCATTER_KV=True,
    )
    return q_out, None, None, gate_out
