"""MoE expert weight L2 prefetch for NPU decode (aclgraph replay).

设计文档：resource/docs/practice/npu-qwen35-moe-w13-prefetch-v1-design.md（v1.4 定稿）
代码包：resource/code/moe_weight_prefetch/（v1.2）

形态（v1.2，相对 v1 的变化由 2026-08-18 服务器首测裁决）：
- **自目标单发射点 = 本层 prepare_attn 完成后、attention 开始前**：
  在该点发射**本层**的权重预取。EP 线的层间通信（reduceScatter + allGather）
  在下一层 prepare_attn 内完成、TP 线的层尾 all_reduce 在上一层
  postprocess_layer 内完成——该点在两条线下都位于**全部层间通信之后**，
  根除了 v1（GMM2 结束后发射，与 AIV AR 抢带宽 +3µs）与 v1.1（层尾发射，
  EP 下压不住 prepare_attn 内的 RS/AG，CMO 与通信同时起跑）两代的通信争用。
  自目标使 layer 0 也获得 attention 窗口（v1/v1.1 无挂点不覆盖）。
- GMM1(w13) 全量预取：发射 w13(j)，仅 GDN 层；默认开。
- GMM2(w2) 全量预取：**并入同一发射点**，w2(j) 块序列接在 w13(j) 之后
  （消费序：w2 的消费截止于 GMM2(j)，比 w13 晚一个 GMM1+SwiGLU）；默认关。
  v1 的同层 GMM1-start 发射已废弃——其窗口账（GMM1+SwiGLU ≈ 63µs）系 TP8 口径，
  ep16 下该窗口仅 ~15-20µs 盖不住 ~56µs 搬运，CMO 与本层 mte2 受限的 GMM 撞车
  （实测 ep16 双开 max 62.8→100.5µs）。
  合并发射门控：w13+w2 ≤ 0.8×L2（ep16 128MiB ✓；TP8 192MiB ✗ 自动关闭 gmm2）。
- best-effort：GMM 前主流不 wait CMO 流（cache 预热无 RAW 冒险），step 末 drain
  （防跨 step 积压 + 侧流经 event 返回主流的 capture 合法性要求）；
- 仅 graph capture 期发射（eager / prefill 不激活）；CMO 是 SDMA 任务，不占 AIV/AIC 核。
- 仅注册过目标模型的层对象会触发发射（`_moe_prefetch_emit` 标记），
  MTP draft 模型的层即使在同名文件内复用本 forward 也不会误发射。

注意：torch_npu 一律在函数内惰性 import，保证本模块在 CUDA 等其他平台可安全 import。
"""

from __future__ import annotations

import ctypes
import glob
import logging
import os
from typing import Dict, Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# ACL_DEV_ATTR_L2_CACHE_SIZE = RT_DEV_ATTR_L2_CACHE_SIZE = 302
_L2_CACHE_ATTR = 302

_VALID_OPS = ("gmm1", "gmm2")
_VALID_MODES = ("auto", "full", "active")


# ---------------------------------------------------------------------------
# L2 容量查询（ctypes，免编译；与探针 code/moe_w13_prefetch_probe 同法）
# ---------------------------------------------------------------------------
def _dlopen_first(candidates):
    for path in candidates:
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    return None


def _try_get_info(lib, func_name) -> Optional[int]:
    fn = getattr(lib, func_name, None)
    if fn is None:
        return None
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
    val = ctypes.c_int64(-1)
    try:
        rc = fn(0, _L2_CACHE_ATTR, ctypes.byref(val))
    except Exception:
        return None
    if rc == 0 and val.value > 0:
        return int(val.value)
    return None


def _query_l2_size_bytes() -> Optional[int]:
    """返回 L2 cache 字节数；查询失败返回 None（调用方据此关闭预取）。"""
    home = os.environ.get("ASCEND_TOOLKIT_HOME") or os.environ.get("ASCEND_HOME_PATH")
    acl_paths = ["libascendcl.so", "libascendcl.so.1"]
    rt_paths = ["libruntime.so"]
    if home:
        acl_paths.append(os.path.join(home, "lib64", "libascendcl.so"))
        rt_paths.append(os.path.join(home, "lib64", "libruntime.so"))
    acl_paths += sorted(
        glob.glob("/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so*")
    )
    rt_paths += sorted(
        glob.glob("/usr/local/Ascend/ascend-toolkit/latest/lib64/libruntime.so*")
    )

    lib = _dlopen_first(acl_paths)
    if lib is not None:
        for name in ("aclrtGetDeviceInfo", "aclrtGetDeviceAttr"):
            size = _try_get_info(lib, name)
            if size is not None:
                return size
    lib = _dlopen_first(rt_paths)
    if lib is not None:
        size = _try_get_info(lib, "rtsDeviceGetInfo")
        if size is not None:
            return size
    return None


def _is_capture_mode() -> bool:
    """惰性 import，避免模块级依赖 model_executor 链路。"""
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        get_is_capture_mode,
    )

    return get_is_capture_mode()


# ---------------------------------------------------------------------------
# 管理器（进程级单例）
# ---------------------------------------------------------------------------
class _MoeWeightPrefetchManager:
    def __init__(self):
        self.configured = False
        self.enabled = False
        self.prefetch_gmm1 = False
        self.prefetch_gmm2 = False
        self.mode = "full"
        self.chunk_bytes = 16 << 20
        self.l2_size: Optional[int] = None
        self.budget_bytes = 0
        self.stream = None
        # layer_id -> {"w13": Tensor, "w2": Tensor, "prefetch_target": bool}
        self.layers: Dict[int, dict] = {}
        self._capacity_checked = False
        self._emitted_in_pass = False

    # ---- 配置（首次注册时解析一次） ----
    def configure(self):
        if self.configured:
            return
        self.configured = True

        if not envs.SGLANG_NPU_MOE_PREFETCH.get():
            return  # enabled 保持 False，后续全部 no-op

        ops_raw = str(envs.SGLANG_NPU_MOE_PREFETCH_OPS.get()).lower()
        ops = [t.strip() for t in ops_raw.split(",") if t.strip()]
        unknown = [t for t in ops if t not in _VALID_OPS]
        if unknown:
            logger.warning(
                f"[MOE_PREFETCH] unknown ops {unknown} ignored, valid: {_VALID_OPS}"
            )
        self.prefetch_gmm1 = "gmm1" in ops
        self.prefetch_gmm2 = "gmm2" in ops

        mode = str(envs.SGLANG_NPU_MOE_PREFETCH_MODE.get()).lower()
        if mode not in _VALID_MODES:
            logger.warning(
                f"[MOE_PREFETCH] unknown mode {mode!r}, fallback to 'auto'"
            )
            mode = "auto"
        if mode == "active":
            # 图内 npu_prefetch 的 offset/max_size 是捕获期常量、节点无条件执行，
            # 运行期激活索引喂不进烘死的 CMO 参数；active 需 copy-kernel +
            # write-allocate 路线（v2）。v1 回退 full。
            logger.warning(
                "[MOE_PREFETCH] mode='active' is not supported in v1 "
                "(in-graph CMO params are capture-time constants); "
                "falling back to 'full'"
            )
            mode = "full"
        self.mode = "full" if mode == "auto" else mode  # v1: auto ≡ full

        chunk_mib = int(envs.SGLANG_NPU_MOE_PREFETCH_CHUNK_MIB.get())
        if chunk_mib <= 0:
            logger.warning(
                f"[MOE_PREFETCH] invalid CHUNK_MIB={chunk_mib}, fallback to 16"
            )
            chunk_mib = 16
        self.chunk_bytes = chunk_mib << 20

        self.l2_size = _query_l2_size_bytes()
        if self.l2_size is None:
            logger.warning(
                "[MOE_PREFETCH] failed to query L2 cache size "
                "(aclrtGetDeviceInfo(302)); prefetch disabled"
            )
            return

        budget_mib = int(envs.SGLANG_NPU_MOE_PREFETCH_BUDGET_MIB.get())
        self.budget_bytes = (
            (budget_mib << 20) if budget_mib > 0 else int(0.8 * self.l2_size)
        )

        self.enabled = self.prefetch_gmm1 or self.prefetch_gmm2
        logger.info(
            f"[MOE_PREFETCH] configured: enabled={self.enabled} "
            f"gmm1={self.prefetch_gmm1} gmm2={self.prefetch_gmm2} "
            f"mode={self.mode} chunk={chunk_mib}MiB "
            f"l2={self.l2_size >> 20}MiB budget={self.budget_bytes >> 20}MiB"
        )

    # ---- 容量判据（首个 MoE 层注册时检查一次；各层同构） ----
    def _check_capacity(self, w13_bytes: int, w2_bytes: int):
        if self._capacity_checked:
            return
        self._capacity_checked = True
        mib = 1 << 20
        if self.prefetch_gmm1 and w13_bytes > self.budget_bytes:
            logger.warning(
                f"[MOE_PREFETCH] w13 size {w13_bytes / mib:.1f}MiB > budget "
                f"{self.budget_bytes / mib:.1f}MiB (0.8*L2 or BUDGET_MIB); "
                f"gmm1 prefetch disabled for this config "
                f"(e.g. TP4 w13=256MiB is a known unsupported case, "
                f"subset prefetch is a v2 candidate)"
            )
            self.prefetch_gmm1 = False
        if self.prefetch_gmm2:
            # v1.1 合并发射门控：w13（若 gmm1 开）+ w2 同驻一个跨层窗口，
            # 合计 ≤ budget 才放行 gmm2。TP8 w13+w2=192MiB=100% L2 不通过
            # （设计文档 R2 容量冲突），ep16 128MiB 通过。
            resident = (w13_bytes if self.prefetch_gmm1 else 0) + w2_bytes
            if resident > self.budget_bytes:
                logger.warning(
                    f"[MOE_PREFETCH] resident w13+w2 size {resident / mib:.1f}MiB "
                    f"> budget {self.budget_bytes / mib:.1f}MiB; "
                    f"merged gmm2 prefetch disabled for this config "
                    f"(v1.1 launches w2(i+1) together with w13(i+1) after the "
                    f"layer-tail all_reduce, so both must fit; "
                    f"TP8 w13+w2=192MiB is a known unsupported case)"
                )
                self.prefetch_gmm2 = False
        self.enabled = self.prefetch_gmm1 or self.prefetch_gmm2
        if self.enabled:
            logger.info(
                f"[MOE_PREFETCH] capacity check passed: w13={w13_bytes / mib:.1f}MiB "
                f"(gmm1={self.prefetch_gmm1}) w2={w2_bytes / mib:.1f}MiB "
                f"(gmm2={self.prefetch_gmm2})"
            )

    # ---- 模型注册（模型 init 期调用，capture 外） ----
    def register_model(self, model):
        self.configure()
        if not self.enabled:
            return

        if self.stream is None:
            import torch_npu  # noqa: F401

            # 专用 CMO 流：capture 外创建（驱动调用），单例复用。
            self.stream = torch.npu.Stream()

        block_types = None
        config = getattr(model, "config", None)
        if config is not None:
            try:
                block_types = config.layers_block_type
            except Exception:
                block_types = None

        n_reg = 0
        layers = getattr(model, "layers", None)
        if layers is None:
            return
        for layer in layers:
            mlp = getattr(layer, "mlp", None)
            experts = getattr(mlp, "experts", None)
            w13 = getattr(experts, "w13_weight", None)
            w2 = getattr(experts, "w2_weight", None)
            layer_id = getattr(layer, "layer_id", None)
            if w13 is None or w2 is None or layer_id is None:
                continue  # 稠密 MLP / PPMissingLayer / 非 MoE 层

            self._check_capacity(
                w13.numel() * w13.element_size(),
                w2.numel() * w2.element_size(),
            )
            # 目标层判定：仅当 j 为 GDN 层时以其为预取目标
            # （FA 层 FIA KV 流自逐出 + 抢带宽）。v1.2 自目标发射后
            # layer 0 也有本层 attention 窗口，不再排除；w13 与 w2 共用目标层集合。
            is_gdn = (
                block_types is not None
                and 0 <= layer_id < len(block_types)
                and block_types[layer_id] == "linear_attention"
            )
            self.layers[layer_id] = {
                "w13": w13,
                "w2": w2,
                "prefetch_target": bool(is_gdn),
            }
            # 发射门禁按层对象标记：未注册的层（如 MTP draft 模型的层，
            # 其 layer_id 可能与注册表碰撞）一律不发射。
            layer._moe_prefetch_emit = True
            n_reg += 1

        if n_reg:
            n_target = sum(1 for v in self.layers.values() if v["prefetch_target"])
            logger.info(
                f"[MOE_PREFETCH] registered {n_reg} MoE layers, "
                f"{n_target} GDN prefetch targets"
            )

    # ---- 发射（仅 capture 期；best-effort，不 wait） ----
    def _emit_chunked(self, weights, anchor: torch.Tensor):
        import torch_npu

        cur = torch.npu.current_stream()
        self.stream.wait_stream(cur)  # fork 边：罚金由预取流支付
        with torch.npu.stream(self.stream):
            for weight in weights:
                nbytes = weight.numel() * weight.element_size()
                offset = 0
                while offset < nbytes:
                    size = min(self.chunk_bytes, nbytes - offset)
                    torch_npu.npu_prefetch(weight, anchor, size, offset)
                    offset += size
        self._emitted_in_pass = True

    def emit(self, layer, anchor: torch.Tensor):
        """本层 attention 开始前（层间通信全部完成后）发射本层 w13(+w2) 预取。

        v1.2 自目标发射：发射点 = decoder layer forward 的 prepare_attn 之后。
        EP 的层间通信（reduceScatter + allGather）在下一层 prepare_attn 内完成、
        TP 的层尾 AR 在上一层 postprocess_layer 内完成，该点在两条线下都位于
        全部层间通信之后。块序：w13 在前（消费截止 GMM1），w2 在后
        （截止 GMM2，多一个 GMM1+SwiGLU 余量）。
        """
        if not self.enabled or not _is_capture_mode():
            return
        if not getattr(layer, "_moe_prefetch_emit", False):
            return  # 未注册的层（MTP draft 等）不发射
        target = self.layers.get(layer.layer_id)
        if target is None or not target["prefetch_target"]:
            return
        weights = []
        if self.prefetch_gmm1:
            weights.append(target["w13"])
        if self.prefetch_gmm2:
            weights.append(target["w2"])
        if not weights:
            return
        self._emit_chunked(weights, anchor)

    # ---- step 末 drain（模型 forward 的 layer 循环后调用） ----
    def drain(self):
        if not self.enabled or not self._emitted_in_pass or not _is_capture_mode():
            return
        torch.npu.current_stream().wait_stream(self.stream)
        self._emitted_in_pass = False


_MANAGER = _MoeWeightPrefetchManager()


# ---------------------------------------------------------------------------
# 对外接口（供 qwen3_5.py 调用）
# ---------------------------------------------------------------------------
def moe_prefetch_register_model(model):
    _MANAGER.register_model(model)


def moe_prefetch_emit(layer, anchor: torch.Tensor):
    _MANAGER.emit(layer, anchor)


def moe_prefetch_step_drain():
    _MANAGER.drain()
