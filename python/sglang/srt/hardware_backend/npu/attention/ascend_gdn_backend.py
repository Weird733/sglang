from typing import Optional, Tuple, Union

import torch
from sgl_kernel_npu.fla.fused_gdn_gating import (
    fused_gdn_gating_kernel_without_sigmoid,
    fused_gdn_gating_npu,
)

# TP_FUSION: 候选1（fused_qkvzba_conv1d = split+causal_conv1d 单 kernel）的守卫/
# 调试/可用性判定，以及 tuple 输入运行期判定不通过时的本地回退 split kernel
from sgl_kernel_npu.fla.utils import (
    fused_qkvzba_split_reshape_cat_contiguous,
)

from sglang.kernels.ops.tp_ascendc_fusion_npu import (
    tp_debug_log,
    tp_fused_qkvzba_conv1d_inputs_ok,
    tp_fusion_qkvzba_enabled,
    tp_op_available,
)
from sglang.srt.hardware_backend.npu.attention.ascend_hybrid_linear_attn_backend import (
    AscendMambaAttnBackendBase,
)
from sglang.srt.layers.attention.linear.gdn_backend import GDNKernelDispatcher
from sglang.srt.layers.attention.linear.utils import (
    get_linear_attn_decode_backend,
    get_linear_attn_prefill_backend,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput

fused_gdn_gating = fused_gdn_gating_npu


class AscendGDNAttnBackend(AscendMambaAttnBackendBase):

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.conv_states_shape = torch.Size(
            (
                *model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape[
                    :-2
                ],
                model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape[-1],
                model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape[-2],
            )
        )
        decode_backend = get_linear_attn_decode_backend()
        prefill_backend = get_linear_attn_prefill_backend()
        self.kernel_dispatcher = GDNKernelDispatcher(decode_backend, prefill_backend)

    def _prepare_mamba_track_metadata(self, forward_batch: ForwardBatch):
        if self.forward_metadata.has_mamba_track_mask:
            mamba_track_mask_indices = forward_batch.mamba_track_mask.nonzero(
                as_tuple=True
            )[0]
            self.forward_metadata.conv_states_mask_indices = (
                forward_batch.mamba_track_indices[mamba_track_mask_indices]
            )

    def prepare_gdn_inputs(
        self,
        bs: int,
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        cache_indices = self.forward_metadata.mamba_cache_indices
        self.num_accept_tokens = torch.ones(
            [bs], dtype=torch.int32, device=cache_indices.device
        )
        self.actual_seq_lengths = torch.ones(
            [bs], dtype=torch.int32, device=cache_indices.device
        )
        if forward_mode.is_target_verify():
            seq_len = spec_info.draft_token_num
            self.actual_seq_lengths = self.actual_seq_lengths * seq_len
            # indices
            self.ssm_state_indices = torch.arange(
                cache_indices.shape[0] * seq_len,
                dtype=torch.int32,
                device=cache_indices.device,
            )
        else:
            self.ssm_state_indices = cache_indices

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        if forward_batch.forward_mode.is_draft_extend_v2():
            return
        super().init_forward_metadata_out_graph(forward_batch, in_capture=in_capture)
        self.prepare_gdn_inputs(
            forward_batch.batch_size,
            forward_batch.forward_mode,
            forward_batch.spec_info,
        )
        self._prepare_mamba_track_metadata(forward_batch)
        self.graph_mode = True

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_draft_extend_v2():
            return
        # cast_elimination_v2：eager（非图）前向的 metadata 由基类现建 int32，
        # recurrent 无 cast 问题——清掉图流程遗留的 int32 影子，避免误用旧 buffer。
        self._decode_recurrent_shadow_i32 = None
        super().init_forward_metadata(forward_batch)
        self.prepare_gdn_inputs(
            forward_batch.batch_size,
            forward_batch.forward_mode,
            forward_batch.spec_info,
        )
        self._prepare_mamba_track_metadata(forward_batch)
        self.graph_mode = False

    def _get_conv_weights_t(self, layer: RadixLinearAttention) -> torch.Tensor:
        w = getattr(layer, "_conv_weights_t", None)
        if w is None:
            w = layer.conv_weights.transpose(0, 1).contiguous()
            layer._conv_weights_t = w
        return w

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices
        # cast_elimination_v2：recurrent 侧改用 int32 影子 buffer（decode 图流程由
        # _capture/_replay_metadata 图外 eager 同步，内容与 int64 主 buffer 一致）——
        # AscendC recurrent host 的 .to(kInt) 变 no-op，图内不再每层 2 个 cast。
        # conv 侧仍用 int64 主 buffer（host .to(kLong) no-op），两侧契约各取所需。
        # 影子为 None（eager/verify/未部署 v2）时沿用原 metadata，行为与 v1 逐字一致；
        # stock/decode-opt Triton recurrent 对索引 dtype 无差，同受本改动无副作用。
        recurrent_query_start_loc = query_start_loc
        recurrent_cache_indices = cache_indices
        if self._decode_recurrent_shadow_i32 is not None:
            recurrent_query_start_loc, recurrent_cache_indices = (
                self._decode_recurrent_shadow_i32
            )

        # TP_FUSION 候选1：qwen3_5.py 侧守卫命中后以 tuple 传入原始投影
        # (projected_states_qkvz, projected_states_ba)（未经 split）——split +
        # causal_conv1d(run_mode=1) 由 torch.ops.npu.fused_qkvzba_conv1d 单 kernel
        # 完成（conv_states 原地更新语义不变），z 随 (core_attn_out, z) 回流给模型侧
        # （qwen3_5.py 用于 self.norm(core_attn_out, z)）。运行期判定不通过（算子
        # 未注册/张量属性不符）→ 本地跑原 split kernel 再走 stock conv，同样以
        # tuple 返回，调用侧无感（静默回退，图 capture 时分支烘进图）。
        fused_z = None
        if isinstance(mixed_qkv, tuple):
            qkvz, mixed_ba = mixed_qkv
            if tp_fusion_qkvzba_enabled() and tp_op_available(
                "fused_qkvzba_conv1d"
            ) and tp_fused_qkvzba_conv1d_inputs_ok(
                qkvz,
                mixed_ba,
                layer.num_k_heads,
                layer.num_v_heads,
                layer.head_k_dim,
                layer.head_v_dim,
            ):
                mixed_qkv, fused_z, b, a = torch.ops.npu.fused_qkvzba_conv1d(
                    qkvz,
                    self._get_conv_weights_t(layer),
                    conv_states,
                    mixed_ba,
                    layer.num_k_heads,
                    layer.num_v_heads,
                    layer.head_k_dim,
                    layer.head_v_dim,
                    bias=layer.bias,
                    query_start_loc=query_start_loc,
                    cache_indices=cache_indices,
                    activation_mode=1,
                    pad_slot_id=-1,
                )
            else:
                tp_debug_log(
                    ("qkvzba_rt", layer.layer_id),
                    f"layer {layer.layer_id}: fused_qkvzba_conv1d 运行期判定未命中"
                    f"（enabled={tp_fusion_qkvzba_enabled()} "
                    f"op={tp_op_available('fused_qkvzba_conv1d')}），"
                    "本地 split+stock conv 回退",
                )
                mixed_qkv, fused_z, b, a = fused_qkvzba_split_reshape_cat_contiguous(
                    qkvz,
                    mixed_ba,
                    layer.num_k_heads,
                    layer.num_v_heads,
                    layer.head_k_dim,
                    layer.head_v_dim,
                )
                mixed_qkv = torch.ops.npu.causal_conv1d(
                    mixed_qkv,
                    self._get_conv_weights_t(layer),
                    conv_states=conv_states,
                    bias=layer.bias,
                    query_start_loc=query_start_loc,
                    cache_indices=cache_indices,
                    activation_mode=1,
                    pad_slot_id=-1,
                    run_mode=1,
                )
        else:
            assert isinstance(mixed_qkv, torch.Tensor)
            mixed_qkv = torch.ops.npu.causal_conv1d(
                mixed_qkv,
                self._get_conv_weights_t(layer),
                conv_states=conv_states,
                bias=layer.bias,
                query_start_loc=query_start_loc,
                cache_indices=cache_indices,
                activation_mode=1,
                pad_slot_id=-1,
                run_mode=1,
            )

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        bs = forward_batch.batch_size
        query = query.view(1, bs, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, bs, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, bs, layer.num_v_heads, layer.head_v_dim)

        core_attn_out = self.kernel_dispatcher.decode(
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=ssm_states,
            cache_indices=recurrent_cache_indices,
            query_start_loc=recurrent_query_start_loc,
        )

        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices
        )
        # TP_FUSION：tuple 输入路径以 (core_attn_out, z) 回流（与 qwen3_5.py 侧
        # 候选1 分支的解包约定自洽）；tensor 路径返回值与 stock 完全一致
        if fused_z is not None:
            return core_attn_out, fused_z
        return core_attn_out

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        assert isinstance(mixed_qkv, torch.Tensor)
        seq_len = mixed_qkv.shape[0]
        is_target_verify = forward_batch.forward_mode.is_target_verify()
        forward_metadata = self.forward_metadata

        query_start_loc = forward_metadata.query_start_loc
        cache_indices = forward_metadata.mamba_cache_indices
        retrieve_next_token = forward_metadata.retrieve_next_token
        retrieve_next_sibling = forward_metadata.retrieve_next_sibling
        retrieve_parent_token = forward_metadata.retrieve_parent_token

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal
        if is_target_verify:
            assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
            intermediate_state_cache = mamba_cache_params.intermediate_ssm
            intermediate_conv_window_cache = (
                mamba_cache_params.intermediate_conv_window[0]
            )
            has_initial_states = torch.ones(
                seq_len // forward_batch.spec_info.draft_token_num,
                dtype=torch.bool,
                device=forward_batch.input_ids.device,
            )
        else:
            has_initial_states = forward_batch.extend_prefix_lens > 0
        if is_target_verify:
            num_token_padding = mixed_qkv.shape[0]
            if (
                not self.graph_mode
                and forward_batch.num_token_non_padded_cpu != num_token_padding
            ):
                mixed_qkv = mixed_qkv[: forward_batch.num_token_non_padded_cpu]
                a = a[: forward_batch.num_token_non_padded_cpu]
                b = b[: forward_batch.num_token_non_padded_cpu]
                seq_len = forward_batch.num_token_non_padded_cpu

            batch_size = cache_indices.shape[0]
            draft_token_num = forward_batch.spec_info.draft_token_num
            num_accepted_tokens = torch.full(
                (batch_size,),
                draft_token_num,
                dtype=torch.int32,
                device=mixed_qkv.device,
            )
            mixed_qkv = torch.ops.npu.causal_conv1d(
                mixed_qkv,
                self._get_conv_weights_t(layer),
                conv_states=conv_states,
                bias=layer.bias,
                query_start_loc=query_start_loc,
                cache_indices=cache_indices,
                num_accepted_tokens=num_accepted_tokens,
                activation_mode=1,
                pad_slot_id=-1,
                run_mode=1,
            )
        else:
            if forward_metadata.has_mamba_track_mask:
                mixed_qkv_to_track = mixed_qkv[forward_metadata.track_conv_indices]
                conv_states[forward_metadata.conv_states_mask_indices] = (
                    mixed_qkv_to_track
                )
            kernel_size = layer.conv_weights.shape[-1]
            conv_states_for_prefill = conv_states[
                :, -(kernel_size - 1) :, :
            ].contiguous()
            mixed_qkv = torch.ops.npu.causal_conv1d(
                mixed_qkv,
                self._get_conv_weights_t(layer),
                conv_states=conv_states_for_prefill,
                bias=layer.bias,
                query_start_loc=query_start_loc,
                cache_indices=cache_indices,
                has_initial_state=has_initial_states,
                activation_mode=1,
                pad_slot_id=-1,
                run_mode=0,
            )
            conv_states[:, -(kernel_size - 1) :, :] = conv_states_for_prefill
        if is_target_verify:
            g, beta = fused_gdn_gating_kernel_without_sigmoid(
                layer.A_log, a, b, layer.dt_bias
            )
            beta = beta.unsqueeze(0)
            num_heads, head_k_dim = layer.num_q_heads, layer.head_q_dim
            num_value_heads, head_v_dim = layer.num_v_heads, layer.head_v_dim

            mixed_qkv_last_dim = mixed_qkv.shape[-1]

            mixed_qkv = mixed_qkv.view(batch_size, -1, mixed_qkv_last_dim)
            beta = beta.view(batch_size, -1, num_value_heads)
            g = g.view(batch_size, -1, num_value_heads)

            core_attn_out = self.fused_recurrent_gated_delta_rule_update(
                mixed_qkv,
                num_heads,
                num_value_heads,
                head_k_dim,
                head_v_dim,
                recurrent_state=ssm_states,
                beta=beta,
                g=g,
                cache_indices=cache_indices,
                intermediate_state=intermediate_state_cache,
            )
            core_attn_out = core_attn_out.view(-1, num_value_heads, head_v_dim)
            if (not self.graph_mode) and core_attn_out.shape[0] < num_token_padding:
                core_attn_out = torch.cat(
                    [
                        core_attn_out,
                        core_attn_out.new_zeros(
                            num_token_padding - core_attn_out.shape[0],
                            *core_attn_out.shape[1:],
                        ),
                    ],
                    dim=0,
                )
        else:
            query, key, value = torch.split(
                mixed_qkv,
                [layer.q_dim, layer.k_dim, layer.v_dim],
                dim=-1,
            )

            actual_seq_len = query.shape[0]
            query = query.view(1, actual_seq_len, layer.num_q_heads, layer.head_q_dim)
            key = key.view(1, actual_seq_len, layer.num_k_heads, layer.head_k_dim)
            value = value.view(1, actual_seq_len, layer.num_v_heads, layer.head_v_dim)

            g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
            core_attn_out, last_recurrent_state, h = self.kernel_dispatcher.extend(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
            )
            if last_recurrent_state is not None:
                last_recurrent_state = last_recurrent_state.to(
                    ssm_states.dtype, copy=False
                )
                ssm_states[cache_indices] = last_recurrent_state
            if h is not None:
                self._track_mamba_state_extend(
                    forward_batch, h, ssm_states, forward_metadata
                )

        return core_attn_out

    def fused_recurrent_gated_delta_rule_update(
        self,
        mix_qkv: torch.Tensor,
        num_heads,
        num_value_heads,
        head_k_dim,
        head_v_dim,
        recurrent_state: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        cache_indices: torch.Tensor,
        intermediate_state: Optional[torch.Tensor] = None,
    ):
        beta = beta.to(torch.bfloat16)
        g = g.to(torch.float32)
        batch_size = mix_qkv.shape[0]
        seq_len = mix_qkv.shape[1]
        scale = 1 / (head_k_dim**0.5)

        if intermediate_state is not None:
            intermediate_state = intermediate_state.view(
                -1, num_value_heads, head_k_dim, head_v_dim
            )

        if self.graph_mode:
            num_accept_tokens = torch.full(
                [batch_size], 1, dtype=torch.int32, device=cache_indices.device
            )
            actual_seq_lengths = torch.full(
                [batch_size], seq_len, dtype=torch.int32, device=cache_indices.device
            )
            ssm_state_indices = self.forward_metadata.mamba_cache_indices_gdn
        else:
            num_accept_tokens = self.num_accept_tokens
            actual_seq_lengths = self.actual_seq_lengths
            ssm_state_indices = self.ssm_state_indices

        attn_core_out = torch.ops.npu.recurrent_gated_delta_rule(
            mix_qkv,
            recurrent_state,
            beta=beta,
            scale=scale,
            actual_seq_lengths=actual_seq_lengths,
            ssm_state_indices=ssm_state_indices.view(batch_size, seq_len),
            nk=num_heads,
            nv=num_value_heads,
            intermediate_state=intermediate_state,
            cache_indices=cache_indices,
            num_accepted_tokens=num_accept_tokens,
            g=g,
        )

        if intermediate_state is not None:
            intermediate_state = intermediate_state.view(
                -1, seq_len, num_value_heads, head_k_dim, head_v_dim
            )
        return attn_core_out
