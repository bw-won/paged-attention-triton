"""
前3层 Benchmark 逻辑：Roofline, System Throughput, CUDA Graph
"""
import torch
from paged_attention import paged_attention, PagedAttnGraphRunner


def run_benchmark_suite(peak_bw_gbs: float):
    DEVICE, DTYPE, DIM, BLOCK_SIZE = "cuda", torch.float16, 128, 128
    NUM_BLOCKS = 4096
    ELEM_BYTES = 2
    HEAD_DIM_FLOPS = 2 * DIM

    L2_EVICTION_SIZE_MB = 64
    l2_evict_buf = torch.empty(L2_EVICTION_SIZE_MB * 1024 * 1024 // ELEM_BYTES, dtype=DTYPE, device=DEVICE)

    def evict_l2():
        l2_evict_buf.copy_(torch.randn_like(l2_evict_buf))
        torch.cuda.synchronize()

    # --- Layer 1: Roofline ---
    print(f"\n{'='*95}")
    print(f"Layer 1: Roofline Performance Model (BLOCK_SIZE={BLOCK_SIZE}, Peak BW={peak_bw_gbs:.1f} GB/s, L2 Evicted)")
    print(f"{'='*95}")

    roofline_configs = [
        (1, 8, 128,   "单请求短上下文"),
        (1, 8, 4096,  "单请求长上下文(Split-KV目标)"),
        (16, 8, 512,  "中Batch稳态"),
        (32, 8, 1024, "满载SM+长序列"),
        (64, 8, 512,  "超大Batch稳态"),
        (128, 8, 256, "极限并发"),
    ]

    for B, H, seq, desc in roofline_configs:
        max_blk = (seq + BLOCK_SIZE - 1) // BLOCK_SIZE
        Q = torch.randn(B, H, DIM, device=DEVICE, dtype=DTYPE)
        K_cache = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, H, DIM, device=DEVICE, dtype=DTYPE)
        V_cache = torch.zeros_like(K_cache)
        bt = torch.zeros(B, max_blk, device=DEVICE, dtype=torch.int32)
        idx = 0
        for b in range(B):
            for i in range(max_blk):
                bt[b, i] = idx % NUM_BLOCKS
                idx += 1

        for _ in range(50):
            paged_attention(Q, K_cache, V_cache, bt, seq, BLOCK_SIZE)
        torch.cuda.synchronize()

        iters = 1000
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            evict_l2()
            start_events[i].record()
            paged_attention(Q, K_cache, V_cache, bt, seq, BLOCK_SIZE)
            end_events[i].record()
        torch.cuda.synchronize()

        latencies_us = [s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)]
        lat_us = sorted(latencies_us)[iters // 2]

        kv_bytes = B * H * seq * DIM * ELEM_BYTES * 2
        qo_bytes = B * H * DIM * ELEM_BYTES * 2
        total_bytes = kv_bytes + qo_bytes
        eff_bw = total_bytes / (lat_us * 1e-6) / 1e9
        flops = B * H * seq * HEAD_DIM_FLOPS * 2
        arith_int = flops / total_bytes
        bw_util = eff_bw / peak_bw_gbs * 100

        print(f"  {desc:28s} | Lat:{lat_us:8.1f}µs | EffBW:{eff_bw:6.1f}GB/s "
              f"| Util:{bw_util:5.1f}% | ArithInt:{arith_int:.3f}")

    # --- Layer 2: System Throughput ---
    print(f"\n{'='*95}")
    print("Layer 2: Paged Architecture Efficiency")
    print(f"{'='*95}")

    B_sys, H_sys, SEQ_SYS = 1, 8, 4096
    kv_mem_gb = B_sys * H_sys * SEQ_SYS * DIM * ELEM_BYTES * 2 / 1e9
    Q_sys = torch.randn(B_sys, H_sys, DIM, device=DEVICE, dtype=DTYPE)
    K_sys = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, H_sys, DIM, device=DEVICE, dtype=DTYPE)
    V_sys = torch.zeros_like(K_sys)
    bt_sys = torch.arange((SEQ_SYS + BLOCK_SIZE - 1) // BLOCK_SIZE, device=DEVICE, dtype=torch.int32).unsqueeze(0)

    for _ in range(50):
        paged_attention(Q_sys, K_sys, V_sys, bt_sys, SEQ_SYS, BLOCK_SIZE)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(1000):
        paged_attention(Q_sys, K_sys, V_sys, bt_sys, SEQ_SYS, BLOCK_SIZE)
    end.record()
    torch.cuda.synchronize()
    sys_lat_us = start.elapsed_time(end) * 1000 / 1000

    tok_s_per_gb = (B_sys / (sys_lat_us * 1e-6)) / kv_mem_gb if kv_mem_gb > 0 else 0
    avail_gb = 8.0
    max_batch = int(avail_gb / kv_mem_gb * B_sys) if kv_mem_gb > 0 else 0

    print(f"  Config: B={B_sys}, H={H_sys}, S={SEQ_SYS}, BLOCK={BLOCK_SIZE}")
    print(f"  KV Cache per Request: {kv_mem_gb*1024:.1f} MB")
    print(f"  Tokens/s/GB:          {tok_s_per_gb:.1f}")
    print(f"  MaxBatch (8GB VRAM):  {max_batch}")

    # --- Layer 3: CUDA Graph ---
    print(f"\n{'='*95}")
    print("Layer 3: CUDA Graph Optimization (Decode Early Stage, B=1)")
    print(f"{'='*95}")

    graph_configs = [
        (1, 8, 128,  "S=128 (SM空闲,launch主导)"),
        (1, 8, 256,  "S=256 (过渡区)"),
        (1, 8, 512,  "S=512 (Split-KV开始生效)"),
    ]

    for B_g, H_g, seq_g, desc in graph_configs:
        torch.cuda.empty_cache()
        runner = PagedAttnGraphRunner(B_g, H_g, DIM, seq_g, BLOCK_SIZE, NUM_BLOCKS)
        Q_g = torch.randn(B_g, H_g, DIM, device=DEVICE, dtype=DTYPE)
        K_g = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, H_g, DIM, device=DEVICE, dtype=DTYPE)
        V_g = torch.zeros_like(K_g)
        bt_g = torch.arange((seq_g + BLOCK_SIZE - 1) // BLOCK_SIZE, device=DEVICE, dtype=torch.int32).unsqueeze(0)

        for _ in range(50):
            paged_attention(Q_g, K_g, V_g, bt_g, seq_g, BLOCK_SIZE)
            runner.replay_only()
        torch.cuda.synchronize()

        start.record()
        for _ in range(1000):
            paged_attention(Q_g, K_g, V_g, bt_g, seq_g, BLOCK_SIZE)
        end.record()
        torch.cuda.synchronize()
        eager_us = start.elapsed_time(end) * 1000 / 1000

        start.record()
        for _ in range(1000):
            runner.replay_only()
        end.record()
        torch.cuda.synchronize()
        graph_us = start.elapsed_time(end) * 1000 / 1000

        speedup = eager_us / graph_us if graph_us > 0 else float('inf')
        print(f"  {desc:30s} | Eager:{eager_us:7.1f}µs | Graph:{graph_us:7.1f}µs | Speedup:{speedup:.2f}x")

        del runner, Q_g, K_g, V_g, bt_g
        torch.cuda.empty_cache()