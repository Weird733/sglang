import logging
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn

from sglang.kernels.ops.sampling.murmur_hash import murmur_hash32
from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.dp_attention import (
    is_dp_attention_enabled,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.logprob_processor import (
    OutputLogprobProcessor,
)
from sglang.srt.runtime_context import get_parallel, get_server_args
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.utils.async_probe import sanitize_nan_logits
from sglang.srt.utils.common import (
    get_bool_env_var,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
)

if is_cuda():
    from flashinfer.sampling import (
        min_p_sampling_from_probs,
        top_k_top_p_sampling_from_probs,
    )
    from sgl_kernel import (
        top_k_renorm_prob,
        top_p_renorm_prob,
    )

if is_musa():
    from sgl_kernel import (
        min_p_sampling_from_probs,
        top_k_renorm_prob,
        top_k_top_p_sampling_from_probs,
        top_p_renorm_prob,
    )

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()
if _use_aiter:
    from aiter import greedy_sample as _aiter_greedy_sample

# The aiter greedy_sample kernel can return an out-of-range token id (== vocab_size,
# e.g. 151666 for MiniCPM-V) for all-NaN / all -inf logit rows on ROCm, which decodes
# to an empty string and breaks downstream consumers. Set this to 1 to fall back to
# torch.argmax (which always returns a valid index). Default off so behavior is
# unchanged elsewhere.
_disable_aiter_greedy_sample = get_bool_env_var("SGLANG_DISABLE_AITER_GREEDY_SAMPLE")

if is_npu():
    import torch_npu


def _is_ascend_910c() -> bool:
    """Return whether the current NPU is an Ascend 910C-series chip.

    torch.npu.get_device_name() reports names like "Ascend910_9382" on 910C
    (the production server reports exactly this string). The async
    exponential-race path below relies on uniform_() decomposing to
    DSARandomUniform on the DSA_SQE core, which only exists on the 910C
    series; other NPUs route uniform_() to AIV/AICore RNG kernels that would
    contend with the model forward. Fail closed on any query error so
    sampling stays on the stock torch.multinomial path.
    """
    if not is_npu():
        return False
    try:
        return torch.npu.get_device_name().startswith("Ascend910_93")
    except Exception:
        return False


logger = logging.getLogger(__name__)

SYNC_TOKEN_IDS_ACROSS_TP = get_bool_env_var("SYNC_TOKEN_IDS_ACROSS_TP")
SGLANG_RETURN_ORIGINAL_LOGPROB = get_bool_env_var("SGLANG_RETURN_ORIGINAL_LOGPROB")
_CUSTOM_SAMPLER_FACTORIES: Dict[str, Callable[[], "Sampler"]] = {}
_BUILT_IN_SAMPLING_BACKENDS = {"flashinfer", "pytorch", "ascend"}


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.tp_sync_group = get_tp_group().device_group
        if is_dp_attention_enabled():
            self.tp_sync_group = get_parallel().attn_tp_group.device_group

        self.rl_on_policy_target = get_server_args().rl_on_policy_target
        # In RL on-policy mode, deterministic inference is automatically enabled.
        self.enable_deterministic = get_server_args().enable_deterministic_inference
        # In RL on-policy mode, we use log_softmax to compute logprobs to match the trainer.
        self.use_log_softmax_logprob = self.rl_on_policy_target is not None
        self.use_ascend_backend = get_server_args().sampling_backend == "ascend"

        self.output_logprob_processor = OutputLogprobProcessor()

        # Generate uniform noise on a side stream while the model is running, and
        # apply the exponential-race transform at consumption time on the main
        # stream, exactly matching stock aten::exponential_ semantics (NPU
        # op-plugin composite): x = min(1 - u, 1 - eps/2), q = -log(x).
        # uniform_() decomposes to DSARandomUniform plus an async
        # D2D copy, i.e. pure DSA-engine work with no AIV transform kernels, so
        # the side stream cannot contend with the model's vector-core kernels
        # even when it overlaps the next forward (v1 ran the full exponential_()
        # chain here and its AIV tail contended with forward).
        # This is opt-in because it changes the random-number sequence, although it
        # preserves the categorical sampling distribution. The gate is
        # additionally restricted to the Ascend 910C series: the side stream is
        # AIV-free only if uniform_() runs on the DSA_SQE core (910C-only).
        # On any other device the flag stays off and every batch falls back to
        # the stock torch.multinomial path.
        async_exponential_requested = is_npu() and get_bool_env_var(
            "SGLANG_NPU_ASYNC_EXPONENTIAL"
        )
        self.enable_async_exponential = (
            async_exponential_requested and _is_ascend_910c()
        )
        if async_exponential_requested and not self.enable_async_exponential:
            logger.warning(
                "SGLANG_NPU_ASYNC_EXPONENTIAL is set but the current device is "
                "not an Ascend 910C series NPU; async exponential-race sampling "
                "is disabled, falling back to torch.multinomial"
            )
        self._async_exponential_stream = None
        self._async_exponential_event = None
        self._async_exponential_u = None
        self._async_exponential_pending = False
        self._async_exp_min_bound = None

        # post_sample v3: optional fused Triton consumption kernel. When
        # resolved, _sample_with_async_exponential replaces the 5-pass
        # stock-exponential transform + RealDiv + ArgMaxV2 chain (7 full-matrix
        # AIV passes, ~1.8 GB HBM traffic at bs=128 x vocab=248320 fp32) with
        # one kernel that reads probs and u once (~254 MB). Same math, same
        # torch.argmax first-occurrence tie-breaking. Fail-closed: any
        # import/compile/smoke failure keeps the unfused chain.
        self._exp_race_fused = None
        if self.enable_async_exponential and get_bool_env_var(
            "SGLANG_NPU_EXP_RACE_TRITON", "true"
        ):
            self._exp_race_fused = self._init_exp_race_fused()

    @staticmethod
    def _init_exp_race_fused():
        """Resolve the fused kernel and smoke-test it against the unfused chain.

        The smoke shape (4, 10000) deliberately exceeds BLOCK_V=4096 so the
        masked tail path is exercised, and plants boundary u values (0 and
        1-2^-24, the exact cap grid points) plus an exact score tie to check
        first-occurrence tie-breaking. Any exception or mismatch disables the
        fused path (fail-closed).
        """
        try:
            from sglang.srt.layers.exp_race_fused_triton import exp_race_argmax

            device = torch.device("npu")
            u = torch.rand((4, 10000), dtype=torch.float32, device=device)
            u[0, 0] = 0.0
            u[1, 0] = 1.0 - 2.0**-24
            probs = torch.rand((4, 10000), dtype=torch.float32, device=device)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            # Exact score tie at the maximal score: identical (p, u) pair at
            # two columns with u=0 (q pinned at the 5.96e-8 cap); the lower
            # index must win, matching torch.argmax.
            u[:, 50] = 0.0
            u[:, 60] = 0.0
            probs[:, 50] = 0.5
            probs[:, 60] = 0.5
            fused = exp_race_argmax(probs, u)
            q = u.neg_().add_(1.0)
            bound = torch.full(
                (),
                1.0 - torch.finfo(torch.float32).eps / 2.0,
                dtype=torch.float32,
                device=device,
            )
            torch.minimum(q, bound, out=q)
            q.log_().neg_()
            reference = torch.div(probs, q).argmax(dim=-1).view(-1).to(torch.int32)
            if not torch.equal(fused, reference):
                raise RuntimeError(
                    f"fused exp-race smoke test mismatch: {fused} vs {reference}"
                )
            logger.info(
                "Enabled fused Triton exp-race consumption kernel "
                "(SGLANG_NPU_EXP_RACE_TRITON)"
            )
            return exp_race_argmax
        except Exception as e:
            logger.warning(
                "Fused Triton exp-race kernel unavailable (%s); falling back "
                "to the unfused stock-exponential chain",
                e,
            )
            return None

    def can_prepare_async_exponential(
        self, sampling_info: SamplingBatchInfo
    ) -> bool:
        """Return whether this batch can use precomputed uniform noise."""
        return (
            self.enable_async_exponential
            and not sampling_info.is_all_greedy
            and sampling_info.sampling_seed is None
            and not sampling_info.need_top_p_sampling
            and not sampling_info.need_top_k_sampling
            and not sampling_info.need_min_p_sampling
        )

    @torch.no_grad()
    def prepare_async_exponential(
        self,
        batch_size: int,
        vocab_size: int,
        sampling_info: SamplingBatchInfo,
        device: torch.device,
    ) -> bool:
        """Enqueue U(0,1) noise before model forward on a dedicated NPU stream.

        Reusing the buffer is safe because the side stream first waits for all
        previously enqueued work on the current stream. The sampling path later
        inserts a device-side event wait; it never synchronizes the CPU.
        """
        if not self.can_prepare_async_exponential(sampling_info):
            return False

        # A delayed sampler may still own the previous buffer. Do not overwrite it.
        if self._async_exponential_pending:
            logger.warning(
                "Skip async exponential preparation because the previous batch "
                "has not consumed its random buffer"
            )
            return False

        if self._async_exponential_stream is None:
            self._async_exponential_stream = torch.npu.Stream()
            self._async_exponential_event = torch.npu.Event()
            logger.info(
                "Enabled asynchronous NPU exponential-race sampling "
                "(uniform noise on side stream, stock-exponential argmax consumption)"
            )

        current_stream = torch.npu.current_stream()
        self._async_exponential_stream.wait_stream(current_stream)

        expected_shape = (batch_size, vocab_size)
        with torch.npu.stream(self._async_exponential_stream):
            u = self._async_exponential_u
            if (
                u is None
                or tuple(u.shape) != expected_shape
                or u.dtype != torch.float32
                or u.device.type != torch.device(device).type
            ):
                # SGLang converts next-token logits to FP32 before sampling; the
                # noise grid must match CANN's fp32 uniform to preserve the
                # incumbent sampling distribution.
                u = torch.empty(
                    expected_shape,
                    dtype=torch.float32,
                    device=device,
                )
                self._async_exponential_u = u
            u.uniform_()
            self._async_exponential_event.record()

        self._async_exponential_pending = True
        return True

    def _sample_with_async_exponential(
        self, probs: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Consume precomputed U(0,1) noise using the exponential-race identity.

        The full stock exponential_ transform (complement + clamp + -log) runs
        here on the main stream (serial work that the sampling path owns
        anyway), keeping the side stream free of AIV kernels so it cannot
        slow down the model forward.
        """
        if not self._async_exponential_pending:
            return None

        u = self._async_exponential_u
        self._async_exponential_pending = False
        if (
            u is None
            or u.shape != probs.shape
            or u.dtype != probs.dtype
            or u.device != probs.device
        ):
            logger.warning(
                "Async uniform buffer does not match probs; falling back to "
                "torch.multinomial (u=%s/%s/%s, probs=%s/%s/%s)",
                None if u is None else tuple(u.shape),
                None if u is None else u.dtype,
                None if u is None else u.device,
                tuple(probs.shape),
                probs.dtype,
                probs.device,
            )
            return None

        current_stream = torch.npu.current_stream()
        current_stream.wait_event(self._async_exponential_event)
        u.record_stream(current_stream)

        # Do not modify probs in place: the standard backend reuses it for logprobs.
        # argmax over probs / q with q ~ Exp(1): v1's production consumption
        # form, kept because the stock argmax kernel is healthy while argmin at
        # this shape is a slow legacy kernel (0.38ms vs ~0.13ms at bs=128).
        # q is produced by transforming u in place with the exact stock
        # aten::exponential_ values (NPU op-plugin composite, fp32 path):
        #   x = min(1 - u, 1 - eps/2);  q = -log(x),  eps = finfo(dtype).eps
        # The guard is a single torch.minimum against a cached 0-dim bound
        # tensor: min(x, 1-eps/2) is bitwise-identical to stock's
        # ge+masked_fill_ pair (x <= 1 always), and unlike clamp_max_ it stays
        # on the template-grade aclnnMinimum kernel -- clamp_max_ routes to
        # the legacy-family ClipByValueV2 op with two scalar->device uploads
        # per call, which costs more than the unary-template kernels used by
        # the rest of the chain. The cap keeps x below 1 so q can never be 0:
        # u == 0 maps to q ~= 5.96e-8 and scores huge, exactly as stock (v2.3
        # mapped it to +inf/+0.0 instead). For fp32 q stays within
        # [~5.96e-8, ~16.6], finite and strictly positive, so the race has no
        # inf/NaN edge.
        # All transform ops run here on the main stream; the side stream stays
        # pure DSA uniform with zero AIV kernels.
        if self._exp_race_fused is not None:
            # post_sample v3: the same math as the chain below in one Triton
            # kernel (single pass over probs and u, no intermediate [B, V]
            # traffic). Non-destructive on u; returns int32 [B] directly.
            return self._exp_race_fused(probs, u)

        bound = self._async_exp_min_bound
        if bound is None or bound.dtype != u.dtype or bound.device != u.device:
            bound = torch.full(
                (),
                1.0 - torch.finfo(u.dtype).eps / 2.0,
                dtype=u.dtype,
                device=u.device,
            )
            self._async_exp_min_bound = bound
        u.neg_().add_(1.0)
        torch.minimum(u, bound, out=u)
        u.log_().neg_()
        sampled_index = torch.div(probs, u).argmax(dim=-1)
        return sampled_index.view(-1).to(torch.int32)

    def _preprocess_logits(
        self, logits: torch.Tensor, sampling_info: SamplingBatchInfo
    ) -> torch.Tensor:
        """Apply custom logit processors and sanitize non-finite logits."""
        if sampling_info.has_custom_logit_processor:
            apply_custom_logit_processor(logits, sampling_info)
        sanitize_nan_logits(logits, "sampler: next_token_logits")
        return logits

    def forward(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        return_logprob: bool,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
        positions: torch.Tensor,
    ):
        """Run a sampler & compute logprobs and update logits_output accordingly.

        Args:
            logits_output: The logits from the model forward
            sampling_info: Metadata for sampling
            return_logprob: If set, store the output logprob information to
                logits_output
            top_logprobs_nums: Number of top lobprobs per sequence in a batch
            token_ids_logprobs: Per-sequence list of specific token IDs to retrieve
                logprobs for. Each element is a list of token IDs (or None) for one
                sequence in the batch. This is used in speculative decoding.
            positions: The positions of the tokens in the sequence. Used for deterministic sampling
                to get the unique seed for each position.
        """
        logits = logits_output.next_token_logits
        # In the plain probability path, keep the softmax output and apply log
        # only to the values requested by the caller.
        logprobs_are_probs = False

        # Preprocess logits (custom processors and NaN handling)
        logits = self._preprocess_logits(logits, sampling_info)
        return_sampling_mask = any(sampling_info.return_sampling_masks or [])

        if sampling_info.is_all_greedy:
            if _use_aiter and not _disable_aiter_greedy_sample:
                batch_next_token_ids = torch.empty(
                    logits.shape[0], device=logits.device, dtype=torch.int32
                )
                _aiter_greedy_sample(batch_next_token_ids, logits)
            else:
                batch_next_token_ids = torch.argmax(logits, -1)
            if return_sampling_mask:
                self._attach_greedy_sampling_mask_to_output(
                    logits_output, sampling_info, batch_next_token_ids
                )
            if return_logprob:
                original_logprobs = logprobs = torch.nn.functional.log_softmax(
                    logits, dim=-1
                )
        else:
            simple_sampling_case = (
                not sampling_info.need_top_p_sampling
                and not sampling_info.need_top_k_sampling
                and not sampling_info.need_min_p_sampling
            )

            # If requested, cache original logprobs before temperature scaling.
            if return_logprob and SGLANG_RETURN_ORIGINAL_LOGPROB:
                original_logprobs = torch.log_softmax(logits, dim=-1)

            # In RL on-policy mode, we use log_softmax to compute logprobs to match the trainer.
            logprobs_via_logsoftmax_kernel = None
            if self.rl_on_policy_target is not None:
                # TODO: use more inplace ops to save memory
                logits_div_temperature = (
                    logits.bfloat16().div(sampling_info.temperatures).bfloat16()
                )
                logprobs_via_logsoftmax_kernel = torch.log_softmax(
                    logits_div_temperature, dim=-1
                )
                del logits_div_temperature

            if self.use_ascend_backend:
                # Ascend backend: sample from logits directly.
                batch_next_token_ids, logprobs = self._forward_ascend_backend(
                    logits,
                    sampling_info,
                    simple_sampling_case,
                    return_logprob,
                    positions,
                )
            elif (
                self.use_log_softmax_logprob
                and self.enable_deterministic
                and simple_sampling_case
            ):
                # RL on-policy path: sample from logprobs to match the trainer.
                batch_next_token_ids = self._sample_from_logprobs(
                    logprobs_via_logsoftmax_kernel,
                    sampling_info,
                    positions,
                )
                if return_logprob and not SGLANG_RETURN_ORIGINAL_LOGPROB:
                    logprobs = logprobs_via_logsoftmax_kernel
            else:
                # Standard path: do softmax and sample from probs.
                # post_sample v3.2：全 batch temperature==1.0 时 div_ 是 IEEE 逐位恒等
                # （x/1.0==x，不改变任何下游值——softmax/采样/logprob gather 全部 bitwise
                # 不变），直接跳过 RealDiv（生产 --rollout-temperature 1 命中，
                # decode 每 step 省 ~27µs）。标记由 SamplingBatchInfo 维护
                # （host 侧，无 device 同步）；旧版 sampling_batch_info 无此字段时
                # getattr 默认 False = 不跳过，行为逐字一致。
                if not getattr(sampling_info, "temperatures_all_one", False):
                    logits.div_(sampling_info.temperatures)

                # Do not write the softmax output back into logits: the
                # write-back costs a full-matrix TensorMove pass (~0.2 ms on
                # NPU at bs=128 x vocab=248320 fp32). logits (now x/T) is not
                # read again on this path and is overwritten by the next
                # forward anyway. Trade-off: one extra live [batch, vocab]
                # fp32 tensor during sampling. (post_sample v1 的 A1 改动)
                probs = torch.softmax(logits, dim=-1)

                batch_next_token_ids = self._sample_from_probs(
                    probs, sampling_info, positions, simple_sampling_case
                )
                if return_sampling_mask:
                    sampling_mask_data = self._compute_sampling_mask_from_probs(
                        probs, sampling_info
                    )
                    self._attach_sampling_mask_to_output(
                        logits_output,
                        sampling_info,
                        batch_next_token_ids,
                        sampling_mask_data,
                    )
                if return_logprob and not SGLANG_RETURN_ORIGINAL_LOGPROB:
                    logprobs = (
                        logprobs_via_logsoftmax_kernel
                        if logprobs_via_logsoftmax_kernel is not None
                        else probs
                    )
                    logprobs_are_probs = logprobs_via_logsoftmax_kernel is None
                del probs

        if return_logprob:
            if SGLANG_RETURN_ORIGINAL_LOGPROB:
                logprobs = original_logprobs
            logprob_result = self.output_logprob_processor.compute_logprobs(
                logprobs,
                top_logprobs_nums,
                token_ids_logprobs,
                batch_next_token_ids,
                logprobs_are_probs,
            )
            logprob_result.write_output_to(logits_output)

        self._sync_token_ids_across_tp(batch_next_token_ids, sampling_info)

        return batch_next_token_ids

    def _sample_from_probs(
        self,
        probs: torch.Tensor,
        sampling_info: SamplingBatchInfo,
        positions: torch.Tensor,
        simple_sampling_case: bool,
    ) -> torch.Tensor:
        """Sample from probability distribution (after softmax).

        Used for standard sampling with flashinfer/pytorch backends.
        Handles both simple (direct multinomial) and complex (top-k/top-p/min-p) cases.
        """
        if simple_sampling_case:
            batch_next_token_ids = self._sample_with_async_exponential(probs)
            if batch_next_token_ids is None:
                batch_next_token_ids = sampling_from_probs_torch(
                    probs,
                    sampling_seed=sampling_info.sampling_seed,
                    positions=positions,
                )
        else:
            backend = get_server_args().sampling_backend
            if backend == "flashinfer":
                assert (
                    sampling_info.sampling_seed is None
                ), "Sampling seed is not supported for flashinfer backend"
                if sampling_info.need_min_p_sampling:
                    probs = top_k_renorm_prob(probs, sampling_info.top_ks)
                    probs = top_p_renorm_prob(probs, sampling_info.top_ps)
                    batch_next_token_ids = min_p_sampling_from_probs(
                        probs, sampling_info.min_ps
                    )
                else:
                    batch_next_token_ids = top_k_top_p_sampling_from_probs(
                        probs.contiguous(),
                        sampling_info.top_ks,
                        sampling_info.top_ps,
                        filter_apply_order="joint",
                    )
            elif backend == "pytorch":
                # A slower fallback implementation with torch native operations.
                batch_next_token_ids = top_k_top_p_min_p_sampling_from_probs_torch(
                    probs,
                    sampling_info.top_ks,
                    sampling_info.top_ps,
                    sampling_info.min_ps,
                    sampling_info.need_min_p_sampling,
                    sampling_info.sampling_seed,
                    positions,
                )
            else:
                raise ValueError(f"Invalid sampling backend: {backend}")
        return batch_next_token_ids

    def _compute_sampling_mask_from_probs(
        self, probs: torch.Tensor, sampling_info: SamplingBatchInfo
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return sorted token ids, sorted probs, keep mask, and raw probs."""
        vocab_size = probs.shape[-1]
        max_top_k = sampling_info.sampling_mask_max_top_k
        if 0 < max_top_k < vocab_size:
            probs_sort, probs_idx = torch.topk(
                probs,
                k=max_top_k,
                dim=-1,
                largest=True,
                sorted=True,
            )
            positions = torch.arange(max_top_k, device=probs.device).view(1, -1)
        else:
            probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
            positions = torch.arange(vocab_size, device=probs.device).view(1, -1)
        probs_sum = torch.cumsum(probs_sort, dim=-1)

        keep_mask = positions < sampling_info.top_ks.view(-1, 1)
        keep_mask &= (probs_sum - probs_sort) <= sampling_info.top_ps.view(-1, 1)

        if sampling_info.need_min_p_sampling:
            min_p_thresholds = probs_sort[:, 0] * sampling_info.min_ps
            keep_mask &= probs_sort >= min_p_thresholds.view(-1, 1)

        return probs_idx, probs_sort, keep_mask, probs

    def _attach_greedy_sampling_mask_to_output(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        batch_next_token_ids: torch.Tensor,
    ) -> None:
        tokens = batch_next_token_ids.to(torch.int32).cpu().tolist()
        masks = []
        logprobs = []
        for i, should_return in enumerate(sampling_info.return_sampling_masks or []):
            if should_return:
                masks.append([int(tokens[i])])
                logprobs.append(0.0)
            else:
                masks.append(None)
                logprobs.append(None)
        logits_output.next_token_sampling_mask_idx = masks
        logits_output.next_token_sampling_logprobs = logprobs

    def _attach_sampling_mask_to_output(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        batch_next_token_ids: torch.Tensor,
        sampling_mask_data: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> None:
        probs_idx, probs_sort, keep_mask, probs = sampling_mask_data
        return_sampling_masks = sampling_info.return_sampling_masks or []
        if not return_sampling_masks:
            logits_output.next_token_sampling_mask_idx = []
            logits_output.next_token_sampling_logprobs = []
            return

        sampled_tokens = batch_next_token_ids.view(-1, 1)
        sampled_matches_all = probs_idx == sampled_tokens
        sampled_in_idx = sampled_matches_all.any(dim=-1)

        # The sampler is the source of truth for the rollout action space. If a
        # backend/numeric edge chooses a token just outside the reconstructed
        # prefix, include that sampled token so training can replay a support
        # that contained the rollout action.
        effective_keep_mask = keep_mask | sampled_matches_all
        selected_raw_probs = torch.gather(probs, 1, sampled_tokens).squeeze(1)
        support_mass = torch.where(
            effective_keep_mask, probs_sort, torch.zeros_like(probs_sort)
        ).sum(dim=-1)
        support_mass = support_mass + torch.where(
            sampled_in_idx, torch.zeros_like(selected_raw_probs), selected_raw_probs
        )
        selected_logprobs = torch.log(
            selected_raw_probs.float()
            / support_mass.float().clamp_min(torch.finfo(torch.float32).tiny)
        )

        flat_rows, flat_cols = effective_keep_mask.nonzero(as_tuple=True)
        flat_ids = probs_idx[flat_rows, flat_cols].to(torch.int32)
        mask_lengths = effective_keep_mask.sum(dim=-1, dtype=torch.int32)

        flat_ids_cpu = flat_ids.cpu().tolist()
        mask_lengths_cpu = mask_lengths.cpu().tolist()
        sampled_in_idx_cpu = sampled_in_idx.cpu().tolist()
        sampled_tokens_cpu = batch_next_token_ids.to(torch.int32).cpu().tolist()
        selected_logprobs_cpu = selected_logprobs.cpu().tolist()

        masks = []
        logprobs = []
        cursor = 0
        for i, should_return in enumerate(return_sampling_masks):
            mask_len = int(mask_lengths_cpu[i])
            row_ids = flat_ids_cpu[cursor : cursor + mask_len]
            cursor += mask_len
            if not sampled_in_idx_cpu[i]:
                row_ids.append(int(sampled_tokens_cpu[i]))
            if should_return:
                masks.append(row_ids)
                logprobs.append(float(selected_logprobs_cpu[i]))
            else:
                masks.append(None)
                logprobs.append(None)

        logits_output.next_token_sampling_mask_idx = masks
        logits_output.next_token_sampling_logprobs = logprobs

    def _sample_from_logprobs(
        self,
        logprobs: torch.Tensor,
        sampling_info: SamplingBatchInfo,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from log-probabilities using the Gumbel trick.

        Used for deterministic sampling with simple cases (no top-k/top-p/min-p).
        Requires sampling_seed to be set in sampling_info.
        """
        assert (
            sampling_info.sampling_seed is not None
        ), "sampling_seed is required for sampling from logprobs"
        sampled_index = multinomial_with_seed(
            logprobs, sampling_info.sampling_seed, positions
        )
        return sampled_index.view(-1).to(torch.int32)

    def _sample_from_logits(
        self,
        logits: torch.Tensor,
        sampling_info: SamplingBatchInfo,
        simple_sampling_case: bool,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from temperature-scaled logits without softmax.

        Used for the Ascend NPU backend which handles softmax internally.
        """
        if simple_sampling_case:
            probs = torch.softmax(logits, dim=-1)
            if sampling_info.sampling_seed is not None:
                probabilities = probs.to(torch.float64).log_()
                batch_next_token_ids = multinomial_with_seed(
                    probabilities, sampling_info.sampling_seed, positions
                ).view(-1)
            else:
                batch_next_token_ids = self._sample_with_async_exponential(probs)
                if batch_next_token_ids is None:
                    batch_next_token_ids = torch.multinomial(
                        probs, num_samples=1
                    ).view(-1)
            return batch_next_token_ids.to(torch.int32)
        else:
            assert (
                self.use_ascend_backend
            ), "Only ascend backend supports sampling from logits"
            batch_next_token_ids = top_k_top_p_min_p_sampling_from_logits_ascend(
                logits,
                sampling_info.top_ks,
                sampling_info.top_ps,
                sampling_info.min_ps,
                sampling_info.need_min_p_sampling,
                sampling_info.sampling_seed,
                positions,
            )
            return batch_next_token_ids.to(torch.int32)

    def _forward_ascend_backend(
        self,
        logits: torch.Tensor,
        sampling_info: SamplingBatchInfo,
        simple_sampling_case: bool,
        return_logprob: bool,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Handle the full Ascend backend sampling path.

        Ascend backend has fused kernels that handle softmax internally,
        so we sample directly from temperature-scaled logits.

        Returns:
            A tuple of (batch_next_token_ids, logprobs). logprobs is None
            when return_logprob is False or SGLANG_RETURN_ORIGINAL_LOGPROB is set.
        """
        logits.div_(sampling_info.temperatures)
        batch_next_token_ids = self._sample_from_logits(
            logits, sampling_info, simple_sampling_case, positions
        )
        logprobs = None
        if return_logprob and not SGLANG_RETURN_ORIGINAL_LOGPROB:
            logprobs = torch.log_softmax(logits, dim=-1)
        return batch_next_token_ids, logprobs

    def _sync_token_ids_across_tp(
        self, batch_next_token_ids: torch.Tensor, sampling_info: SamplingBatchInfo
    ):
        if SYNC_TOKEN_IDS_ACROSS_TP or sampling_info.grammars:
            # For performance reasons, SGLang does not sync the final token IDs across TP ranks by default.
            # This saves one all-reduce, but the correctness of this approach depends on the determinism of several operators:
            # the last all-reduce, the last lm_head matmul, and all sampling kernels.
            # These kernels are deterministic in most cases, but there are some rare instances where they are not deterministic.
            # In such cases, enable this env variable to prevent hanging due to TP ranks becoming desynchronized.
            # When using xgrammar, this becomes more likely so we also do the sync when grammar is used.

            torch.distributed.all_reduce(
                batch_next_token_ids,
                op=dist.ReduceOp.MIN,
                group=self.tp_sync_group,
            )

    def compute_logprobs_only(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
    ) -> None:
        logprob_result = self.output_logprob_processor.compute_logprobs_only(
            next_token_logits=logits_output.next_token_logits,
            top_logprobs_nums=top_logprobs_nums,
            token_ids_logprobs=token_ids_logprobs,
            preprocess_fn=partial(self._preprocess_logits, sampling_info=sampling_info),
        )
        if logprob_result is not None:
            logprob_result.write_output_to(logits_output)


def register_sampler_backend(backend: str, factory: Callable[[], "Sampler"]) -> None:
    """Register a custom sampler factory for a backend string."""

    if not backend:
        raise ValueError("backend must be a non-empty string")

    from sglang.srt.server_args import SAMPLING_BACKEND_CHOICES

    if backend in _CUSTOM_SAMPLER_FACTORIES:
        logger.warning("Overriding existing sampler factory for backend '%s'", backend)
    SAMPLING_BACKEND_CHOICES.add(backend)
    _CUSTOM_SAMPLER_FACTORIES[backend] = factory


def create_sampler(backend: Optional[str] = None) -> "Sampler":
    """Create a sampler honoring custom backend registrations."""

    server_args = get_server_args()
    backend = backend or (server_args.sampling_backend if server_args else None)

    if backend in _CUSTOM_SAMPLER_FACTORIES:
        sampler = _CUSTOM_SAMPLER_FACTORIES[backend]()
        if not isinstance(sampler, Sampler):
            raise TypeError(
                f"Custom sampler factory for backend '{backend}' must return a Sampler"
            )
        return sampler

    if backend is None or backend in _BUILT_IN_SAMPLING_BACKENDS:
        return Sampler()

    raise ValueError(
        f"Unknown sampling backend '{backend}'. Register it via register_sampler_backend()."
    )


def top_k_top_p_min_p_sampling_from_probs_torch(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_min_p_sampling: bool,
    sampling_seed: Optional[torch.Tensor],
    positions: torch.Tensor,
):
    """
    A top-k, top-p and min-p sampling implementation with native pytorch operations.
    When sampling_seed is not None, deterministic inference will be enabled, it will sample
    with the sampling_seed of each request.
    """
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[
        torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1)
        >= top_ks.view(-1, 1)
    ] = 0.0
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0

    if need_min_p_sampling:
        # TODO: probs_sort should be re-normalized for the use of multinomial_with_seed
        assert (
            sampling_seed is None
        ), "With sampling seed, multinomial_with_seed will provide wrong results"
        min_p_thresholds = probs_sort[:, 0] * min_ps
        probs_sort[probs_sort < min_p_thresholds.view(-1, 1)] = 0.0

    if sampling_seed is None:
        sampled_index = torch.multinomial(probs_sort, num_samples=1)
    else:
        # NOTE: when using top-k/top-p/min-p sampling, we need to modify probs before we
        # apply log to get logprobs. Therefore, we cannot use log_softmax directly.
        # For now, we use log to the modified probs to get logprobs, but for numerical
        # stability, we'd better come up with a solution to use log_softmax.
        logprobs = probs_sort.to(torch.float64)  # Using float64 for numerical stability
        del probs_sort
        logprobs.log_()
        sampled_index = multinomial_with_seed(logprobs, sampling_seed, positions)

    # int32 range is enough to represent the token ids
    probs_idx = probs_idx.to(torch.int32)
    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)
    return batch_next_token_ids


def top_k_top_p_min_p_sampling_from_logits_ascend(
    logits: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_min_p_sampling: bool,
    sampling_seed: Optional[torch.Tensor],
    positions: torch.Tensor,
):
    """A top-k, top-p and min-p sampling implementation for ascend npu with torch_npu interface.

    Takes temperature-scaled logits as input (softmax is applied internally).
    """
    # torch_npu.npu_top_k_top_p requires top_k value range in [1, 1024]
    if hasattr(torch_npu, "npu_top_k_top_p") and torch.all(
        (top_ks <= 1024) & (top_ks >= 1)
    ):
        logits_top_k_top_p = torch_npu.npu_top_k_top_p(logits, top_ps, top_ks)
        probs_top_k_top_p = logits_top_k_top_p.softmax(dim=-1)

        if need_min_p_sampling:
            min_p_thresholds = probs_top_k_top_p.max(dim=-1) * min_ps
            min_p_mask = probs_top_k_top_p < min_p_thresholds.view(-1, 1)
            probs_top_k_top_p.masked_fill_(min_p_mask, 0.0)

        if sampling_seed is None:
            batch_next_token_ids = torch.multinomial(probs_top_k_top_p, num_samples=1)
        else:
            logprobs_top_k_top_p = probs_top_k_top_p.to(
                torch.float64
            )  # Using float64 for numerical stability
            del probs_top_k_top_p
            logprobs_top_k_top_p.log_()
            batch_next_token_ids = multinomial_with_seed(
                logprobs_top_k_top_p, sampling_seed, positions
            )
    else:
        probs = torch.softmax(logits, dim=-1)
        probs_sort, probs_idx = probs.sort(dim=-1, descending=True)

        # when top_k is -1 (in which sglang turns it to TOP_K_ALL), make it explicitly equal to logit's size
        topk_all_mask = top_ks == TOP_K_ALL
        top_ks.masked_fill_(topk_all_mask, probs.shape[1])
        top_k_mask = torch.arange(0, probs.shape[-1], device=probs.device).view(
            1, -1
        ) >= top_ks.view(-1, 1)
        probs_sort.masked_fill_(top_k_mask, 0.0)

        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_p_mask = probs_sum - probs_sort > top_ps.view(-1, 1)
        probs_sort.masked_fill_(top_p_mask, 0.0)

        if need_min_p_sampling:
            min_p_thresholds = probs_sort[:, 0] * min_ps
            min_p_mask = probs_sort < min_p_thresholds.view(-1, 1)
            probs_sort.masked_fill_(min_p_mask, 0.0)

        if sampling_seed is None:
            sampled_index = torch.multinomial(probs_sort, num_samples=1)
        else:
            logprobs = probs_sort.to(
                torch.float64
            )  # Using float64 for numerical stability
            del probs_sort
            logprobs.log_()
            sampled_index = multinomial_with_seed(logprobs, sampling_seed, positions)
        probs_idx = probs_idx.to(torch.int32)
        batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index)

    return batch_next_token_ids.view(-1)


@torch.compile(dynamic=True, disable=is_npu())
def multinomial_with_seed(
    logprobs: torch.Tensor, seed: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """
    Samples n elements from an input tensor `inputs` of shape (n, m) using
    a unique random seed for each row. This is a deterministic batched alternative to
    `torch.multinomial`.

    Args:
        inputs: A float tensor of shape (n, m) representing n categorical
                distributions with m categories each. The values are treated
                as weights and do not need to sum to 1.
        seed:   An integer tensor of shape (n,) containing the random seed
                for each corresponding row in `inputs`.
        positions: The positions of the tokens in the sequence. Used for deterministic sampling
                to get the unique seed for each position.

    Returns:
        A tensor of shape (n,) where the i-th element is an index sampled
        from the distribution in `inputs[i]` using `seed[i]`.
    """
    n, m = logprobs.shape
    seed = seed.to(torch.uint64)
    col_indices = torch.arange(m, device=logprobs.device)
    hashed = murmur_hash32(seed, positions, col_indices)

    # NOTE (sehoon): it is critical to keep gumbel noise calculation in float64 to avoid numerical instability.
    # keeping logprobs in float64 is less critical, but we found it's still safer to keep it in float64.
    x = hashed.to(torch.float64) / torch.iinfo(torch.uint32).max

    # x is a uniform sample in [0, 1]. get gumbel noise from it.
    # which is equivalent to -log(-log(x))
    # keep everything in in-place operations to avoid unnecessary memory allocations.
    x.log_().clamp_(min=torch.finfo(x.dtype).min).neg_()  # -log(x)
    x.log_().neg_()  # -log(-log(x)) == gumbel noise

    # add gumbel noise to logprobs
    x.add_(logprobs.to(torch.float64))

    return torch.argmax(x, dim=1, keepdim=True)


def sampling_from_probs_torch(
    probs: torch.Tensor,
    sampling_seed: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
):
    """A sampling implementation with native pytorch operations, without
    top-k, top-p, or min-p filtering.

    Note: For deterministic sampling from logprobs, use Sampler._sample_from_logprobs instead.
    """
    if sampling_seed is None:
        sampled_index = torch.multinomial(probs, num_samples=1)
    else:
        # Deterministic sampling: convert probs to logprobs and use gumbel trick
        sampled_index = multinomial_with_seed(
            torch.log(probs), sampling_seed, positions
        )
    batch_next_token_ids = sampled_index.view(-1).to(torch.int32)
    return batch_next_token_ids


def top_p_normalize_probs_torch(
    probs: torch.Tensor,
    top_ps: torch.Tensor,
):
    # See also top_k_top_p_min_p_sampling_from_probs_torch
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    return torch.zeros_like(probs_sort).scatter_(-1, probs_idx, probs_sort)


def apply_custom_logit_processor(
    logits: torch.Tensor,
    sampling_batch_info: SamplingBatchInfo,
    num_tokens_in_batch: int = 1,
):
    """Apply custom logit processors to the logits.
    This function will modify the logits in-place.
    num_tokens_in_batch is needed to support spec decoding, where each batch can contain multiple
    tokens. By default, we assume each batch contains only 1 token.
    """

    assert logits.shape[0] == len(sampling_batch_info) * num_tokens_in_batch, (
        f"The batch size of logits ({logits.shape[0]}) does not match the batch size of "
        f"sampling_batch_info ({len(sampling_batch_info)}) x num_tokens_in_batch "
        f"({num_tokens_in_batch})"
    )

    for _, (
        processor,
        batch_mask,
    ) in sampling_batch_info.custom_logit_processor.items():
        # Get the batch indices that need to be processed
        batch_indices = batch_mask.nonzero(as_tuple=True)[0]

        assert batch_mask.shape[0] == len(sampling_batch_info), (
            f"The number of batch mask ({batch_mask.shape[0]}) does not match the number of "
            f"sampling_batch_info ({len(sampling_batch_info)})"
        )
        batch_mask = torch.repeat_interleave(batch_mask, num_tokens_in_batch)

        # Apply the processor to the logits
        logits[batch_mask] = processor(
            logits[batch_mask],
            [sampling_batch_info.custom_params[i] for i in batch_indices],
        )

        logger.debug(
            f"Custom logit processor {processor.__class__.__name__} is applied."
        )
