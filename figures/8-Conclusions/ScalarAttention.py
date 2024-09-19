import triton
import triton.language as tl

@triton.jit
def softmin_kernel(
    output_ptr,
    x_ptr,
    y_ptr,
    z_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Parallelize across rows
    row_idx = tl.program_id(0)
    x_start_ptr = x_ptr + row_idx
    y_offsets = tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = y_offsets < n_cols
    x = tl.load(x_start_ptr)
    y = tl.load(y_ptr + y_offsets, mask=mask, other=float('inf'))
    z = tl.load(z_ptr + y_offsets, mask=mask, other=0)
    diff_abs = (y-x).abs()
    neg_row = -1 * diff_abs
    row_minus_max = neg_row - tl.max(neg_row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_w = numerator / denominator
    outputs = tl.sum(softmax_w * z)
    tl.store(output_ptr + row_idx, outputs)
