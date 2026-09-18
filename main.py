"""
main.py
=======
PagedAttention V2 统一执行入口:
  1. 实测硬件 HBM 峰值带宽
  2. 执行正确性校验
  3. 运行 4 层 Benchmark 评估体系
"""
from measure_peak_bw import measure_peak_bw
from tests import check_correctness, run_benchmark_suite, benchmark_vs_vllm_paged_attention


if __name__ == "__main__":
    # Step 0: 实测硬件峰值带宽
    print("🔧 Measuring GPU HBM Peak Bandwidth...")
    peak_bw = measure_peak_bw(size_mb=256, repeat=200)
    print(f"   → Using {peak_bw:.2f} GB/s as Roofline baseline\n")

    # Step 1: 正确性校验
    correct = check_correctness()

    if not correct:
        print("\n❌ 正确性验证失败，请检查 Kernel 实现后再运行 Benchmark。")
        exit(1)

    # Step 2: 前 3 层 Benchmark (Roofline, System, Graph)
    run_benchmark_suite(peak_bw)

    # Step 3: 第 4 层 vLLM 对标 (可选)
    print(f"\n{'='*80}")
    print("Layer 4: vLLM Native PagedAttention Baseline Comparison")
    print("⚠️  Note: vLLM only supports block_size in {8,16,32}. Using 16.")
    print(f"{'='*80}")

    vllm_scenarios = [
        {"batch": 1,   "seq_len": 128,  "num_heads": 8, "head_dim": 128, "block_size": 16},
        {"batch": 1,   "seq_len": 4096, "num_heads": 8, "head_dim": 128, "block_size": 16},
        {"batch": 16,  "seq_len": 512,  "num_heads": 8, "head_dim": 128, "block_size": 16},
        {"batch": 32,  "seq_len": 1024, "num_heads": 8, "head_dim": 128, "block_size": 16},
        {"batch": 64,  "seq_len": 512,  "num_heads": 8, "head_dim": 128, "block_size": 16},
        {"batch": 128, "seq_len": 256,  "num_heads": 8, "head_dim": 128, "block_size": 16},
    ]
    benchmark_vs_vllm_paged_attention(vllm_scenarios)