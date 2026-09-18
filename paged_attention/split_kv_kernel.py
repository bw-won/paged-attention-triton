"""
Triton Kernel 底层实现：Split-KV 两段式 Attention
"""
import triton
import triton.language as tl

# LSE 哨兵值：替代 -inf，避免 FP32 存储精度丢失导致 reduce 错误
LSE_SENTINEL = -1e30
LSE_VALID_THRESHOLD = -1e29


@triton.jit
def paged_attention_split_kernel(
    Q, K_Cache, V_Cache, Block_Table,
    O_partial, LSE_partial,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kbs, stride_kh, stride_kd,
    stride_vb, stride_vbs, stride_vh, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    HEADS, SEQ_LEN, SPLIT_SIZE, MAX_LOGICAL_BLOCKS,
    DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)

    split_start = pid_s * SPLIT_SIZE
    split_end = tl.minimum(split_start + SPLIT_SIZE, SEQ_LEN)

    if split_start >= SEQ_LEN:
        return

    b = pid_bh // HEADS
    h = pid_bh % HEADS

    d_offs = tl.arange(0, DIM)
    q = tl.load(Q + b * stride_qb + h * stride_qh + d_offs * stride_qd).to(tl.float32)
    scale = 1.0 / tl.sqrt(tl.cast(DIM, tl.float32))

    m_i = tl.full([], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([], dtype=tl.float32)
    acc = tl.zeros([DIM], dtype=tl.float32)

    num_tiles = (split_end - split_start + BLOCK_N - 1) // BLOCK_N
    for tile_idx in range(num_tiles):
        tile_start = split_start + tile_idx * BLOCK_N
        token_pos = tile_start + tl.arange(0, BLOCK_N)
        valid = token_pos < split_end

        logical_idx = token_pos // BLOCK_SIZE
        slot = token_pos % BLOCK_SIZE

        phys_block = tl.load(
            Block_Table + b * MAX_LOGICAL_BLOCKS + logical_idx,
            mask=valid, other=0
        )

        k_offs = (phys_block[:, None] * stride_kb + slot[:, None] * stride_kbs
                  + h * stride_kh + d_offs[None, :] * stride_kd)
        v_offs = (phys_block[:, None] * stride_vb + slot[:, None] * stride_vbs
                  + h * stride_vh + d_offs[None, :] * stride_vd)

        k = tl.load(K_Cache + k_offs, mask=valid[:, None], other=0.0).to(tl.float32)
        v = tl.load(V_Cache + v_offs, mask=valid[:, None], other=0.0).to(tl.float32)

        scores = tl.sum(k * q[None, :], axis=1) * scale
        scores = tl.where(valid, scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)

        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    out_offs = b * stride_ob + h * stride_oh + pid_s * stride_os + d_offs * stride_od
    lse_offs = b * stride_lb + h * stride_lh + pid_s * stride_ls

    l_i_safe = tl.maximum(l_i, 1e-20)
    tl.store(O_partial + out_offs, (acc / l_i_safe).to(tl.float16))
    tl.store(LSE_partial + lse_offs, m_i + tl.log(l_i_safe))


@triton.jit
def paged_attention_reduce_kernel(
    O_partial, LSE_partial, O,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    stride_out_b, stride_out_h, stride_out_d,
    NUM_SPLITS,
    DIM: tl.constexpr,
    LSE_THRESHOLD: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    d_offs = tl.arange(0, DIM)

    g_max = tl.full([], LSE_THRESHOLD, dtype=tl.float32)
    g_sum = tl.zeros([], dtype=tl.float32)
    g_acc = tl.zeros([DIM], dtype=tl.float32)

    for s in range(NUM_SPLITS):
        lse = tl.load(LSE_partial + pid_b * stride_lb + pid_h * stride_lh + s * stride_ls)

        if lse > LSE_THRESHOLD:
            new_max = tl.maximum(g_max, lse)
            alpha_old = tl.exp(g_max - new_max)
            alpha_new = tl.exp(lse - new_max)

            o_part = tl.load(O_partial + pid_b * stride_ob + pid_h * stride_oh
                             + s * stride_os + d_offs * stride_od).to(tl.float32)

            g_acc = g_acc * alpha_old + alpha_new * o_part
            g_sum = g_sum * alpha_old + alpha_new
            g_max = new_max

    out = g_acc / tl.maximum(g_sum, 1e-20)
    out_offs = pid_b * stride_out_b + pid_h * stride_out_h + d_offs * stride_out_d
    tl.store(O + out_offs, out.to(tl.float16))