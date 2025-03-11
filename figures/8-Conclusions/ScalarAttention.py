import triton
import triton.language as tl

@triton.jit
def scalar_attention_kernel(
    output_ptr, q_ptr, k_ptr, v_ptr,
    n_cols, BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    q_start_ptr = q_ptr + row_idx
    k_offsets = tl.arange(0, BLOCK_SIZE)
    mask = k_offsets < n_cols
    q = tl.load(q_start_ptr)
    k = tl.load(k_ptr + k_offsets, mask=mask, other=float('inf'))
    v = tl.load(v_ptr + k_offsets, mask=mask, other=0)
    neg_row = -1 * (k-q).abs()
    row_minus_max = neg_row - tl.max(neg_row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmin_w = numerator / denominator
    outputs = tl.sum(softmin_w * v)
    tl.store(output_ptr + row_idx, outputs)
