"""
第4层 Benchmark 逻辑：vLLM Native Baseline 对标
"""
import torch
import triton
from paged_attention import paged_attention


def benchmark_vs_vllm_paged_attention(scenarios, dtype=torch.float16, seed=0):
    """对比自定义 Triton Split-KV vs vLLM 原生 paged attention"""
    try:
        from vllm._custom_ops import paged_attention_v1 as _vllm_pa_v1
        from vllm._custom_ops import paged_attention_v2 as _vllm_pa_v2
    except ImportError:
        print("[SKIP] vllm not installed. Cannot run vLLM baseline comparison.")
        return []

    results = []
    for sc in scenarios:
        torch.manual_seed(seed)
        batch = sc["batch"]
        seq_len = sc["seq_len"]
        num_heads = sc["num_heads"]
        num_kv_heads = sc.get("num_kv_heads", num_heads)
        head_dim = sc["head_dim"]
        block_size = sc["block_size"]
        max_seq_len = seq_len
        scale = head_dim ** -0.5

        if block_size not in (8, 16, 32):
            print(f"[SKIP] vLLM does not support block_size={block_size}. Skipping.")
            continue

        num_blocks_per_seq = (seq_len + block_size - 1) // block_size
        num_blocks = num_blocks_per_seq * batch + 1
        device = "cuda"

        q = torch.randn(batch, num_heads, head_dim, dtype=dtype, device=device)
        elem_bytes = torch.tensor([], dtype=dtype).element_size()
        x = 16 // elem_bytes
        k_cache = torch.randn(num_blocks, num_kv_heads, head_dim // x, x, block_size, dtype=dtype, device=device)
        v_cache = torch.randn(num_blocks, num_kv_heads, head_dim, block_size, dtype=dtype, device=device)
        block_tables = torch.randint(0, num_blocks, (batch, num_blocks_per_seq), dtype=torch.int32, device=device)
        seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
        out_vllm = torch.empty_like(q)

        k_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
        v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

        _PARTITION_SIZE = 512
        use_v2 = (max_seq_len > 8192 and block_size * num_kv_heads > _PARTITION_SIZE)

        if use_v2:
            tmp_out = torch.empty(batch, num_heads, max_seq_len // _PARTITION_SIZE + 1, head_dim, dtype=torch.float32, device=device)
            exp_sums = torch.empty(batch, num_heads, max_seq_len // _PARTITION_SIZE + 1, dtype=torch.float32, device=device)
            max_logits = torch.empty_like(exp_sums)
            def run_vllm():
                _vllm_pa_v2(out_vllm, tmp_out, exp_sums, max_logits, q, k_cache, v_cache,
                            num_kv_heads, scale, block_tables, seq_lens, block_size, max_seq_len,
                            None, kv_cache_dtype="auto", k_scale=k_scale, v_scale=v_scale)
        else:
            def run_vllm():
                _vllm_pa_v1(out_vllm, q, k_cache, v_cache, num_kv_heads, scale,
                            block_tables, seq_lens, block_size, max_seq_len,
                            None, kv_cache_dtype="auto", k_scale=k_scale, v_scale=v_scale)

        # Triton layout
        k_cache_triton = torch.zeros(num_blocks, block_size, num_heads, head_dim, dtype=dtype, device=device)
        v_cache_triton = torch.zeros_like(k_cache_triton)
        bt_triton = torch.zeros(batch, num_blocks_per_seq, dtype=torch.int32, device=device)
        idx = 0
        for b_idx in range(batch):
            for i in range(num_blocks_per_seq):
                bt_triton[b_idx, i] = idx % num_blocks
                idx += 1

        def run_mine():
            paged_attention(q, k_cache_triton, v_cache_triton, bt_triton, seq_len, block_size)

        ms_vllm = triton.testing.do_bench(run_vllm, warmup=25, rep=100)
        ms_mine = triton.testing.do_bench(run_mine, warmup=25, rep=100)
        speedup = ms_vllm / ms_mine if ms_mine > 0 else float('inf')
        results.append((sc, ms_mine, ms_vllm, speedup, use_v2))

        tag = "v2" if use_v2 else "v1"
        print(f"  [Scene] B={batch} S={seq_len} H={num_heads} BLK={block_size} | "
              f"Triton:{ms_mine:.4f}ms | vLLM-{tag}:{ms_vllm:.4f}ms | Speedup:{speedup:.2f}x")

    return results