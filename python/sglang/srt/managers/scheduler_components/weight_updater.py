from __future__ import annotations

import hashlib
import logging
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import torch

from sglang.srt.constants import (
    GPU_MEMORY_ALL_TYPES,
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    CheckWeightsReqInput,
    CheckWeightsReqOutput,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
)

logger = logging.getLogger(__name__)

def _log_memory_snapshot(stage: str) -> None:
    """One-shot colocate memory diagnostic: device-level vs torch-allocator view.

    Logged from every TP scheduler around release_memory_occupation. A mismatch
    (device used >> torch reserved after empty_cache) pinpoints memory held
    outside the caching allocator — HCCL buffers, graph pools, driver — which
    neither a TMS pause nor empty_cache can reclaim.
    """
    try:
        device_module = torch.get_device_module()
        free_bytes, total_bytes = device_module.mem_get_info()
        used=total_bytes - free_bytes
        logger.info(
            f"[mem-snapshot:{stage}] device free={free_bytes / 2**30:.2f}GiB"
            f" total={total_bytes / 2**30:.2f}GiB"
            f" used={used / 2**30:.2f}GiB"
            f" torch allocated={device_module.memory_allocated() / 2**30:.2f}GiB"
            f" reserved={device_module.memory_reserved() / 2**30:.2f}GiB"
        )
    except Exception as e:
        logger.warning(f"[mem-snapshot:{stage}] unavailable: {e}")


def _log_segment_breakdown(stage: str) -> None:
    """Aggregate memory_snapshot() segments to classify unreleased cache.

    Splits reserved memory into: fully-free segments (releasable by empty_cache
    — a large value after empty_cache means release itself is failing), free
    space inside partially-active segments (fragmentation pinned by live
    tensors), and a per-pool breakdown (segment_pool_id != 0 marks private /
    graph pools that plain empty_cache never releases).
    """
    try:
        snapshot = torch.get_device_module().memory_snapshot()
        segments = snapshot if isinstance(snapshot, list) else snapshot.get("segments", [])
        reserved = fully_free = frag_free = live = 0
        n_free_segs = 0
        pools: dict[str, list[int]] = {}
        for seg in segments:
            seg_total = seg.get("total_size", 0)
            seg_live = sum(
                b.get("size", 0) for b in seg.get("blocks", []) if str(b.get("state", "")).startswith("active")
            )
            reserved += seg_total
            live += seg_live
            if seg_live == 0:
                fully_free += seg_total
                n_free_segs += 1
            else:
                frag_free += seg_total - seg_live
            pool = str(seg.get("segment_pool_id", "default"))
            stats = pools.setdefault(pool, [0, 0])
            stats[0] += 1
            stats[1] += seg_total
        pool_str = " ".join(f"pool[{p}]={v[0]}segs/{v[1] / 2**30:.2f}G" for p, v in sorted(pools.items()))
        logger.info(
            f"[mem-segments:{stage}] segs={len(segments)} reserved={reserved / 2**30:.2f}G"
            f" live={live / 2**30:.2f}G"
            f" releasable={n_free_segs}segs/{fully_free / 2**30:.2f}G"
            f" frag_free={frag_free / 2**30:.2f}G {pool_str}"
        )
    except Exception as e:
        logger.warning(f"[mem-segments:{stage}] unavailable: {e}")


_census_prev: dict[tuple, int] = {}


def _log_tensor_census(stage: str) -> None:
    """Name the tensor shapes behind live-memory growth between releases.

    _log_segment_breakdown shows where the allocator pool stands; this walks
    gc for device tensors (unique storages, grouped by shape+dtype) and logs
    the delta vs the previous call — a slow leak shows up as a consistently
    growing shape.
    """
    try:
        import gc

        # torch_npu.npu -> "npu", torch.cuda -> "cuda": match storage.device.type
        device_type = torch.get_device_module().__name__.rsplit(".", 1)[-1]
        census: dict[tuple, int] = {}
        seen_storage_ids: set[int] = set()
        for obj in gc.get_objects():
            try:
                if not isinstance(obj, torch.Tensor):
                    continue
                storage = obj.untyped_storage()
                if id(storage) in seen_storage_ids or storage.device.type != device_type:
                    continue
                seen_storage_ids.add(id(storage))
                key = (tuple(obj.shape), str(obj.dtype))
                census[key] = census.get(key, 0) + storage.nbytes()
            except Exception:
                continue
        total = sum(census.values())
        growth = {k: v - _census_prev.get(k, 0) for k, v in census.items() if v > _census_prev.get(k, 0)}
        _census_prev.clear()
        _census_prev.update(census)
        top_size = sorted(census.items(), key=lambda kv: -kv[1])[:6]
        top_growth = sorted(growth.items(), key=lambda kv: -kv[1])[:6]

        def fmt(items):
            return "; ".join(f"{list(k[0])}x{k[1].replace('torch.', '')}={v / 2**20:.0f}M" for k, v in items)

        logger.info(
            f"[tensor-census:{stage}] total={total / 2**30:.2f}G"
            f" top_by_size[{fmt(top_size)}]"
            f" growth_vs_prev[{fmt(top_growth)}]"
        )
    except Exception as e:
        logger.warning(f"[tensor-census:{stage}] unavailable: {e}")

def _get_draft_model_runner(draft_worker):
    # DFlash / FrozenKVMTP workers expose draft_model_runner directly
    runner = getattr(draft_worker, "draft_model_runner", None)
    if runner is not None:
        return runner
    # EAGLEWorkerV2: _draft_worker.draft_runner
    inner = getattr(draft_worker, "_draft_worker", None)
    if inner is not None:
        runner = getattr(inner, "draft_runner", None)
        if runner is not None:
            return runner
    return None


def _merge_checksum_payloads(target: Dict, draft: Dict) -> Dict:
    merged_checksums = dict(target["checksums"])
    for name, chk in draft["checksums"].items():
        merged_checksums[f"draft.{name}"] = chk
    h = hashlib.sha256()
    for name in sorted(merged_checksums):
        h.update(name.encode())
        h.update(merged_checksums[name].encode())
    target["checksums"] = merged_checksums
    target["per_gpu_checksum"] = h.hexdigest()
    return target


@dataclass(kw_only=True, slots=True)
class SchedulerWeightUpdaterManager:
    tp_worker: Any
    draft_worker: Any
    tp_cpu_group: Any
    memory_saver_adapter: Any
    flush_cache: Callable[..., bool]
    is_fully_idle: Callable[..., bool]
    scheduler: Optional[Any] = None
    metrics_collector: Optional[Any] = None
    offload_tags: set = field(default_factory=set)
    stashed_model_static_state: Any = None

    @contextmanager
    def _observe_weight_load(self, source: str) -> Iterator[None]:
        # Edge-trigger weight_load_duration_seconds at the end of each
        # update_weights_from_* call. Engine is paused during the update so
        # the periodic log_stats path can't carry this.
        # `source` distinguishes disk vs distributed vs tensor vs ipc.
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.metrics_collector is not None:
                self.metrics_collector.observe_weight_load(
                    time.perf_counter() - t0, source
                )

    def flush_cache_after_weight_update(self, recv_req) -> None:
        if recv_req.flush_cache:
            flush_cache_success = self.flush_cache(
                empty_cache=recv_req.torch_empty_cache
            )
            assert flush_cache_success, "Cache flush failed after updating weights"

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        """In-place update of the weights from disk."""
        with self._observe_weight_load("disk"):
            success, message = self.tp_worker.update_weights_from_disk(recv_req)
            tp_success = success
            if success and self.draft_worker is not None:
                success, message = self.draft_worker.update_weights_from_disk(recv_req)
            if tp_success:
                self.flush_cache_after_weight_update(recv_req)
            if not success:
                logger.error(message)
            return UpdateWeightFromDiskReqOutput(
                success=success, message=message, num_paused_requests=0
            )

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        """Initialize the online model parameter update group."""
        success, message = self.tp_worker.init_weights_update_group(recv_req)
        return InitWeightsUpdateGroupReqOutput(success=success, message=message)

    def destroy_weights_update_group(
        self,
        recv_req: DestroyWeightsUpdateGroupReqInput,
    ):
        """Destroy the online model parameter update group."""
        success, message = self.tp_worker.destroy_weights_update_group(recv_req)
        return DestroyWeightsUpdateGroupReqOutput(success=success, message=message)

    def update_weights_from_distributed(
        self,
        recv_req: UpdateWeightsFromDistributedReqInput,
    ) -> Tuple[bool, str]:
        """Update the online model parameter."""
        with self._observe_weight_load("distributed"):
            success, message = self.tp_worker.update_weights_from_distributed(recv_req)
            if success:
                self.flush_cache_after_weight_update(recv_req)
            else:
                logger.error(message)
            return UpdateWeightsFromDistributedReqOutput(
                success=success, message=message
            )

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        """Update the online model parameter from tensors."""
        with self._observe_weight_load("tensor"):
            if recv_req.disable_draft_model:
                worker = self.tp_worker
            else:
                worker = self.draft_worker or self.tp_worker
            success, message = worker.update_weights_from_tensor(recv_req)
            if success:
                self.flush_cache_after_weight_update(recv_req)
            else:
                logger.error(message)
            torch.distributed.barrier(group=self.tp_cpu_group)
            return UpdateWeightsFromTensorReqOutput(success=success, message=message)

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """Update the online model parameter from IPC for checkpoint-engine integration."""
        with self._observe_weight_load("ipc"):
            success, message = self.tp_worker.update_weights_from_ipc(recv_req)
            tp_success = success
            if success and self.draft_worker is not None:
                success, message = self.draft_worker.update_weights_from_ipc(recv_req)
            if tp_success:
                self.flush_cache_after_weight_update(recv_req)
            if not success:
                logger.error(message)
            torch.distributed.barrier(group=self.tp_cpu_group)
            return UpdateWeightsFromIPCReqOutput(success=success, message=message)

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self.tp_worker.get_weights_by_name(recv_req)
        return GetWeightsByNameReqOutput(parameter=parameter)

    def release_memory_occupation(self, recv_req: ReleaseMemoryOccupationReqInput):
        assert (
            self.is_fully_idle()
        ), "release_memory_occupation should be called only when server is idle."

        _log_memory_snapshot("release-start")

        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.add(tag)

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            scheduler = self.scheduler
            if scheduler is not None:
                if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
                    for queue_name in (
                        "disagg_decode_transfer_queue",
                        "disagg_decode_prealloc_queue",
                    ):
                        queue = getattr(scheduler, queue_name, None)
                        if queue is not None:
                            queue.release_memory_occupation()
                elif scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
                    queue = getattr(scheduler, "disagg_prefill_bootstrap_queue", None)
                    if queue is not None:
                        queue.release_memory_occupation()
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)
            self.flush_cache()

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self.stashed_model_static_state = _export_static_state(
                self.tp_worker.model_runner.model
            )
            torch.distributed.barrier(self.tp_cpu_group)
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_WEIGHTS)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_CUDA_GRAPH)

        torch.get_device_module().synchronize()

        # The flush_cache above ran before the weights / cuda_graph pauses, so
        # blocks freed by those pauses (and by _export_static_state temporaries)
        # are still cached by the allocator. Empty once more after ALL pauses so
        # the trainer starts its phase with the full headroom. Note
        # current_platform.empty_cache() is a no-op here (no NPU platform
        # plugin overrides it), hence the direct device-module call.

        # torch.get_device_module().empty_cache()
        _log_memory_snapshot("release-end")
        # _log_segment_breakdown("release-end")
        # _log_tensor_census("release-end")

        return ReleaseMemoryOccupationReqOutput()

    def resume_memory_occupation(self, recv_req: ResumeMemoryOccupationReqInput):
        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.remove(tag)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_WEIGHTS)
            torch.distributed.barrier(self.tp_cpu_group)
            _import_static_state(
                self.tp_worker.model_runner.model,
                self.stashed_model_static_state,
            )
            del self.stashed_model_static_state

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)
            scheduler = self.scheduler
            if scheduler is not None:
                if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
                    for queue_name in (
                        "disagg_decode_transfer_queue",
                        "disagg_decode_prealloc_queue",
                    ):
                        queue = getattr(scheduler, queue_name, None)
                        if queue is not None:
                            queue.resume_memory_occupation()
                elif scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
                    queue = getattr(scheduler, "disagg_prefill_bootstrap_queue", None)
                    if queue is not None:
                        queue.resume_memory_occupation()

        return ResumeMemoryOccupationReqOutput()

    def check_weights(self, recv_req: CheckWeightsReqInput):
        try:
            payload = self.tp_worker.model_runner.check_weights(
                action=recv_req.action, allow_quant_error=recv_req.allow_quant_error
            )

            if self.draft_worker is not None:
                draft_runner = _get_draft_model_runner(self.draft_worker)
                if draft_runner is not None:
                    draft_payload = draft_runner.check_weights(
                        action=recv_req.action,
                        allow_quant_error=recv_req.allow_quant_error,
                    )
                    if payload is not None and draft_payload is not None:
                        payload = _merge_checksum_payloads(payload, draft_payload)

            tp_size = torch.distributed.get_world_size(group=self.tp_cpu_group)
            if tp_size > 1 and payload is not None:
                all_payloads = [None] * tp_size
                torch.distributed.all_gather_object(
                    all_payloads, payload, group=self.tp_cpu_group
                )
                payload = all_payloads
            return CheckWeightsReqOutput(
                success=True, message="Success.", payload=payload
            )
        except Exception as e:
            logger.warning(f"check_weights see error: {e}")
            traceback.print_exc()
            return CheckWeightsReqOutput(success=False, message=f"{e}")

    def save_remote_model(self, params):
        url = params["url"]

        self.tp_worker.model_runner.save_remote_model(url)

        if self.draft_worker is not None:
            draft_url = params.get("draft_url", None)
            assert (
                draft_url is not None
            ), "draft_url must be provided when draft model is enabled"
            self.draft_worker.model_runner.save_remote_model(draft_url)

    def save_sharded_model(self, params):
        self.tp_worker.model_runner.save_sharded_model(
            path=params["path"],
            pattern=params["pattern"],
            max_size=params["max_size"],
        )


def _export_static_state(model):
    return dict(
        buffers=[
            (name, buffer.detach().clone()) for name, buffer in model.named_buffers()
        ]
    )


def _import_static_state(model, static_params):
    with torch.inference_mode():
        self_named_buffers = dict(model.named_buffers())
        for name, tensor in static_params["buffers"]:
            self_named_buffers[name][...] = tensor
