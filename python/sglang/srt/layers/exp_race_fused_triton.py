# Fused exponential-race sampling consumption for Ascend NPU (post_sample v3).
#
# Replaces the v2.3.2 main-stream consumption chain
#   u.neg_().add_(1.0); torch.minimum(u, bound, out=u); u.log_().neg_()
#   sampled = torch.div(probs, u).argmax(dim=-1)
# (5 full-matrix AIV passes + RealDiv + ArgMaxV2, ~1.8 GB of HBM traffic at
# bs=128 x vocab=248320 fp32) with fused Triton kernels that read probs and u
# once (~254 MB) and write one int32 index per row.
#
# Math is identical to the unfused chain (stock aten::exponential_ NPU
# composite semantics):
#   x = min(1 - u, 1 - eps/2);  q = -log(x);  out = argmax_rows(probs / q)
# The q transform is BITWISE-identical to the AIV chain on this triton-ascend
# (round-3 UT: full 2^24 u-grid, 0 mismatches, max_ulp=0).
# Tie-breaking matches torch.argmax: first occurrence wins, within a tile
# (lowest lane holding the tile max) and across tiles (partials carry the
# global index; the reducer takes the lowest index among value ties).
#
# Two-kernel "partials" structure: kernel 1 writes one (max, arg) slot per
# tile with NO loop-carried state, kernel 2 reduces the <=64 partials in one
# tile. Kernel 1 deliberately uses only compiler-verified idioms (round-3
# root cause: `t * BLOCK_V + tl.arange` reused in value context got its base
# double-added by triton-ascend -- same family as the gmm2 in-kernel-offsets
# miscompile):
#   - loop form identical to the bitwise-proven q kernel:
#     `for v0 in tl.range(0, V, BLOCK_V)`, loop variable IS the offset;
#   - `lane = tl.arange(0, BLOCK_V)` is the only index vector used as a VALUE
#     (never an arithmetically-derived offs);
#   - global index assembled from scalars only: `v0 + tile_lane`,
#     `v0 // BLOCK_V` (scalar loop-var arithmetic, proven in the gdn kernels).
#
# Fail-closed: if triton is unavailable the module still imports and
# exp_race_argmax_available() returns False; the sampler keeps the unfused
# chain.

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - depends on server env
    _HAS_TRITON = False

# 910_9382 UB per AIV core is 196608 B (192 KB). The compiler places ~8
# BLOCK_V-sized tile buffers (u/p DMA loads duplicated by auto-multi-buffer
# plus the x/q/s/cand intermediates), so BLOCK_V=8192 overflows UB at compile
# time (required 263232 B > 196608 B, observed on the server). 4096 needs
# ~131.6 KB and fits with ~46% headroom. If a future triton-ascend version
# still overflows, drop to 2048.
_BLOCK_V = 4096

# Partials reducer tile width: ceil(248320 / 4096) = 61 partials at the
# production vocab; 64 covers any V <= 262144 with one tile.
_BLOCK_NT = 64

if _HAS_TRITON:

    @triton.jit
    def _exp_race_partial_kernel(
        probs_ptr,
        u_ptr,
        pval_ptr,
        parg_ptr,
        V,
        NT,
        bound,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        base = row * V
        pbase = row * NT
        lane = tl.arange(0, BLOCK_V)

        # No loop-carried state: every iteration writes only its own slot.
        for v0 in tl.range(0, V, BLOCK_V):
            offs = v0 + lane
            mask = offs < V
            # x = min(1 - u, 1 - eps/2): exact in fp32 (1-u is exact on the
            # uniform grid, min is exact). q = -log(x) stays finite and
            # strictly positive, so masked lanes pinned at -inf can never win.
            u = tl.load(u_ptr + base + offs, mask=mask, other=0.5)
            x = 1.0 - u
            x = tl.minimum(x, bound)
            q = -tl.log(x)
            p = tl.load(probs_ptr + base + offs, mask=mask, other=0.0)
            s = p / q
            s = tl.where(mask, s, float("-inf"))

            tile_max = tl.max(s, axis=0)
            # First-occurrence tie-break inside the tile: lowest LANE holding
            # the tile max (pure arange as the only index value; global index
            # reassembled from scalars below). Avoids tl.argmax whose tie
            # semantics on triton-ascend are unverified.
            cand = tl.where(s == tile_max, lane, BLOCK_V)
            tile_lane = tl.min(cand, axis=0)
            tile_arg = v0 + tile_lane

            slot = v0 // BLOCK_V
            tl.store(pval_ptr + pbase + slot, tile_max)
            tl.store(parg_ptr + pbase + slot, tile_arg)

    @triton.jit
    def _exp_race_partial_argmax_kernel(
        probs_ptr,
        u_ptr,
        pval_ptr,
        parg_ptr,
        V,
        NT,
        bound,
        BLOCK_V: tl.constexpr,
    ):
        # Diagnostic arm: identical to _exp_race_partial_kernel but takes the
        # per-tile lane via tl.argmax (tie behavior unverified -- the UT
        # reports which index it returns on exact ties; production ties are
        # measure-zero and any in-range tie pick passes fp64 arbitration with
        # rel diff == 0).
        row = tl.program_id(0).to(tl.int64)
        base = row * V
        pbase = row * NT
        lane = tl.arange(0, BLOCK_V)

        for v0 in tl.range(0, V, BLOCK_V):
            offs = v0 + lane
            mask = offs < V
            u = tl.load(u_ptr + base + offs, mask=mask, other=0.5)
            x = 1.0 - u
            x = tl.minimum(x, bound)
            q = -tl.log(x)
            p = tl.load(probs_ptr + base + offs, mask=mask, other=0.0)
            s = p / q
            s = tl.where(mask, s, float("-inf"))

            tile_max = tl.max(s, axis=0)
            tile_lane = tl.argmax(s, axis=0)
            tile_arg = v0 + tile_lane

            slot = v0 // BLOCK_V
            tl.store(pval_ptr + pbase + slot, tile_max)
            tl.store(parg_ptr + pbase + slot, tile_arg)

    @triton.jit
    def _exp_race_reduce_kernel(
        pval_ptr,
        parg_ptr,
        out_ptr,
        NT,
        BLOCK_NT: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        pbase = row * NT

        offs = tl.arange(0, BLOCK_NT)
        mask = offs < NT
        vals = tl.load(pval_ptr + pbase + offs, mask=mask, other=float("-inf"))
        args = tl.load(parg_ptr + pbase + offs, mask=mask, other=2147483647)

        total = tl.max(vals, axis=0)
        # First occurrence across tiles: lowest global index among value ties
        # (partials store global indices, so min over matches == torch.argmax).
        # Only values loaded from memory are compared -- no in-kernel offsets.
        cand = tl.where(vals == total, args, 2147483647)
        best = tl.min(cand, axis=0)

        tl.store(out_ptr + row, best)

    @triton.jit
    def _exp_race_q_kernel(
        u_ptr,
        q_ptr,
        V,
        bound,
        BLOCK_V: tl.constexpr,
    ):
        # Validation twin (UT only): materializes q = -log(min(1-u, bound))
        # elementwise so the UT can diff it against the unfused torch chain
        # (bitwise / ULP report). Not used in the serving path. This kernel's
        # loop form is the bitwise-proven reference for the partial kernel.
        row = tl.program_id(0).to(tl.int64)
        base = row * V
        for v0 in tl.range(0, V, BLOCK_V):
            offs = v0 + tl.arange(0, BLOCK_V)
            mask = offs < V
            u = tl.load(u_ptr + base + offs, mask=mask, other=0.5)
            x = 1.0 - u
            x = tl.minimum(x, bound)
            q = -tl.log(x)
            tl.store(q_ptr + base + offs, q, mask=mask)


def exp_race_argmax_available() -> bool:
    return _HAS_TRITON


def _bound_for(dtype: torch.dtype) -> float:
    # Same bound as sampler._async_exp_min_bound: 1 - finfo(dtype).eps / 2
    # (fp32 -> 1 - 2^-24, exactly representable; passed as an fp32 kernel arg).
    return 1.0 - torch.finfo(dtype).eps / 2.0


def _check_inputs(probs: torch.Tensor, u: torch.Tensor):
    assert probs.dtype == torch.float32 and u.dtype == torch.float32
    assert probs.shape == u.shape and probs.dim() == 2


def _run_partials(partial_kernel, probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    B, V = probs.shape
    NT = (V + _BLOCK_V - 1) // _BLOCK_V
    assert NT <= _BLOCK_NT, "vocab too large for the partials reducer tile"
    device = probs.device
    pval = torch.empty((B, NT), dtype=torch.float32, device=device)
    parg = torch.empty((B, NT), dtype=torch.int32, device=device)
    bound = _bound_for(probs.dtype)
    partial_kernel[(B,)](
        probs, u, pval, parg, V, NT, bound, BLOCK_V=_BLOCK_V, num_warps=4
    )
    out = torch.empty(B, dtype=torch.int32, device=device)
    _exp_race_reduce_kernel[(B,)](
        pval, parg, out, NT, BLOCK_NT=_BLOCK_NT, num_warps=4
    )
    return out


def exp_race_argmax(probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Fused v2.3.2 consumption: row argmax of probs / q(u), int32 [B].

    Two-kernel partials path (production, exact first-occurrence tie-break).
    Non-destructive: neither probs nor u is modified (the unfused chain
    rewrote u in place; u is overwritten by the next prepare regardless).
    """
    _check_inputs(probs, u)
    probs = probs.contiguous()
    u = u.contiguous()
    return _run_partials(_exp_race_partial_kernel, probs, u)


def exp_race_argmax_argmaxarm(probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Diagnostic arm (UT only): per-tile lane via tl.argmax."""
    _check_inputs(probs, u)
    probs = probs.contiguous()
    u = u.contiguous()
    return _run_partials(_exp_race_partial_argmax_kernel, probs, u)


def exp_race_q_values(u: torch.Tensor) -> torch.Tensor:
    """Validation twin: materialize q for the ULP/bitwise diff (UT only)."""
    assert u.dtype == torch.float32 and u.dim() == 2
    u = u.contiguous()
    B, V = u.shape
    q = torch.empty_like(u)
    _exp_race_q_kernel[(B,)](
        u, q, V, _bound_for(u.dtype), BLOCK_V=_BLOCK_V, num_warps=4
    )
    return q
