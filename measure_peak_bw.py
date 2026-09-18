"""
measure_peak_bw.py
==================
通过大缓冲区 Copy Kernel 实测 GPU HBM 有效峰值带宽。
用于为 Roofline 模型提供准确的硬件基准，避免使用理论峰值导致的利用率虚高。

Usage: python measure_peak_bw.py [--size-mb 256] [--repeat 200]
"""
import argparse
import torch
import triton
import triton.language as tl


@triton.jit
def _copy_kernel(src, dst, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(src + offs, mask=mask)
    tl.store(dst + offs, x, mask=mask)


def measure_peak_bw(size_mb: int = 256, repeat: int = 200, block: int = 1024) -> float:
    """返回实测峰值带宽 (GB/s)"""
    n_elems = size_mb * 1024 * 1024 // 2  # FP16 elements
    src = torch.empty(n_elems, dtype=torch.float16, device="cuda")
    dst = torch.empty_like(src)
    grid = (triton.cdiv(n_elems, block),)

    # Warmup
    for _ in range(50):
        _copy_kernel[grid](src, dst, n_elems, block)
    torch.cuda.synchronize()

    # Measure
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(repeat):
        _copy_kernel[grid](src, dst, n_elems, block)
    t1.record()
    torch.cuda.synchronize()

    ms = t0.elapsed_time(t1) / repeat
    bytes_per_copy = n_elems * 2 * 2  # src read + dst write, FP16 = 2B
    peak_bw_gbs = bytes_per_copy / (ms * 1e-3) / 1e9
    return peak_bw_gbs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure GPU HBM peak bandwidth")
    parser.add_argument("--size-mb", type=int, default=256, help="Buffer size in MB (must exceed L2)")
    parser.add_argument("--repeat", type=int, default=200, help="Timing repetitions")
    args = parser.parse_args()

    bw = measure_peak_bw(args.size_mb, args.repeat)
    print(f"   实测 HBM 有效峰值带宽: {bw:.2f} GB/s")
    print(f"   Buffer: {args.size_mb}MB FP16 | Repeat: {args.repeat} | Block: 1024")