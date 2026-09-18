"""
CUDA Graph 静态 Buffer 管理与 Replay 逻辑
"""
import torch
from .torch_kernel import paged_attention


class PagedAttnGraphRunner:
    def __init__(self, B, H, DIM, seq_len, block_size, num_phys_blocks, device="cuda", dtype=torch.float16):
        self.seq_len = seq_len
        self.block_size = block_size

        # 使用 zeros 避免未初始化内存导致 Kernel 越界
        self.static_q = torch.zeros(B, H, DIM, device=device, dtype=dtype)
        self.static_k = torch.zeros(num_phys_blocks, block_size, H, DIM, device=device, dtype=dtype)
        self.static_v = torch.zeros_like(self.static_k)
        max_blocks = (seq_len + block_size - 1) // block_size
        self.static_bt = torch.zeros(B, max_blocks, device=device, dtype=torch.int32)

        # 独立 stream 预捕获，避免隐式同步开销
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                paged_attention(self.static_q, self.static_k, self.static_v,
                                self.static_bt, seq_len, block_size)
        torch.cuda.current_stream().wait_stream(warmup_stream)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_out = paged_attention(
                self.static_q, self.static_k, self.static_v, self.static_bt, seq_len, block_size)

    def run(self, q, k_cache, v_cache, block_table, copy_kv=False):
        self.static_q.copy_(q)
        self.static_bt.copy_(block_table)
        if copy_kv:
            self.static_k.copy_(k_cache)
            self.static_v.copy_(v_cache)
        self.graph.replay()
        return self.static_out

    def replay_only(self):
        """纯 Graph Replay，不含任何 Host-Side Copy，用于精确测量 GPU 执行延迟"""
        self.graph.replay()