"""
正确性验证逻辑：对比 PyTorch Reference 实现
"""
import math
import torch
from paged_attention import paged_attention, PagedAttnGraphRunner


def check_correctness(device="cuda", dtype=torch.float16, dim=128, block_size=16,
                      batch=1, heads=8, seq_len=4096, num_blocks=4096):
    print("=" * 80)
    print("Correctness Verification")
    print("=" * 80)

    torch.manual_seed(42)

    def ref_attention(q, k, v):
        scale = 1.0 / math.sqrt(dim)
        s = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        return torch.matmul(torch.softmax(s, dim=-1), v.float()).to(dtype)

    max_blk = seq_len // block_size
    Q = torch.randn(batch, heads, dim, device=device, dtype=dtype)
    K_full = torch.randn(batch, heads, seq_len, dim, device=device, dtype=dtype)
    V_full = torch.randn(batch, heads, seq_len, dim, device=device, dtype=dtype)
    K_cache = torch.zeros(num_blocks, block_size, heads, dim, device=device, dtype=dtype)
    V_cache = torch.zeros_like(K_cache)
    bt = torch.zeros(batch, max_blk, device=device, dtype=torch.int32)

    idx = 0
    for i in range(max_blk):
        s, e = i * block_size, (i + 1) * block_size
        K_cache[idx] = K_full[0, :, s:e].permute(1, 0, 2)
        V_cache[idx] = V_full[0, :, s:e].permute(1, 0, 2)
        bt[0, i] = idx
        idx += 1

    O_ref = ref_attention(Q.unsqueeze(2), K_full, V_full).squeeze(2)
    O_v2 = paged_attention(Q, K_cache, V_cache, bt, seq_len, block_size)
    diff = (O_ref.float() - O_v2.float()).abs().max().item()
    ok_split = diff < 2e-2
    print(f"  Split-KV V2 vs Ref: max_diff={diff:.2e} {'✅' if ok_split else '❌'}")

    runner = PagedAttnGraphRunner(batch, heads, dim, seq_len, block_size, num_blocks)
    O_graph = runner.run(Q, K_cache, V_cache, bt, copy_kv=True)
    diff_graph = (O_ref.float() - O_graph.float()).abs().max().item()
    ok_graph = diff_graph < 2e-2
    print(f"  CUDA Graph vs Ref:  max_diff={diff_graph:.2e} {'✅' if ok_graph else '❌'}")
    print("-" * 80)

    del runner
    torch.cuda.empty_cache()
    return ok_split and ok_graph