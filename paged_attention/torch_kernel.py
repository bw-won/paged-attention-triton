"""
PyTorch autograd 封装与 Grid 分发逻辑
"""
import torch
from .split_kv_kernel import (
    paged_attention_split_kernel,
    paged_attention_reduce_kernel,
    LSE_SENTINEL,
    LSE_VALID_THRESHOLD
)


class PagedAttentionV2Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K_Cache, V_Cache, Block_Table, seq_len, block_size,
                num_splits=None, block_n=None):
        BATCH, HEADS, DIM = Q.shape

        if num_splits is None:
            sm_count = torch.cuda.get_device_properties(0).multi_processor_count
            target_programs = sm_count * 4
            num_splits = max(1,
                min((seq_len + block_size - 1) // block_size,
                    max(1, target_programs // (BATCH * HEADS)))
            )

        if block_n is None:
            block_n = block_size

        split_size = (seq_len + num_splits - 1) // num_splits

        O_partial = torch.empty(BATCH, HEADS, num_splits, DIM, device=Q.device, dtype=torch.float16)
        LSE_partial = torch.empty(BATCH, HEADS, num_splits, device=Q.device, dtype=torch.float32)
        LSE_partial.fill_(LSE_SENTINEL)
        O = torch.empty_like(Q)

        grid_split = (BATCH * HEADS, num_splits)
        max_logical_blocks = Block_Table.shape[1]

        paged_attention_split_kernel[grid_split](
            Q, K_Cache, V_Cache, Block_Table,
            O_partial, LSE_partial,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K_Cache.stride(0), K_Cache.stride(1), K_Cache.stride(2), K_Cache.stride(3),
            V_Cache.stride(0), V_Cache.stride(1), V_Cache.stride(2), V_Cache.stride(3),
            O_partial.stride(0), O_partial.stride(1), O_partial.stride(2), O_partial.stride(3),
            LSE_partial.stride(0), LSE_partial.stride(1), LSE_partial.stride(2),
            HEADS, seq_len, split_size, max_logical_blocks,
            DIM, block_size, block_n,
            num_warps=4, num_stages=3,
        )

        grid_reduce = (BATCH, HEADS)
        paged_attention_reduce_kernel[grid_reduce](
            O_partial, LSE_partial, O,
            O_partial.stride(0), O_partial.stride(1), O_partial.stride(2), O_partial.stride(3),
            LSE_partial.stride(0), LSE_partial.stride(1), LSE_partial.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            num_splits, DIM,
            LSE_VALID_THRESHOLD,
            num_warps=4,
        )

        return O

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError("PagedAttentionV2 is inference-only.")


def paged_attention(Q, K_Cache, V_Cache, Block_Table, seq_len, block_size, num_splits=None):
    """对外暴露的 PyTorch 接口"""
    return PagedAttentionV2Function.apply(Q, K_Cache, V_Cache, Block_Table, seq_len, block_size, num_splits)