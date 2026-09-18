# PagedAttention: Split-KV + CUDA Graph

基于 Triton 实现的高性能 PagedAttention 推理算子，针对 LLM Decode 阶段的 **非连续 KV Cache 访存** 与 **单请求 SM 利用率低** 两大核心瓶颈进行优化。

## ✨ 核心特性

- **Split-KV 两段式 Attention**：将长序列 KV 按 split 维度切分为并行 program，Split 阶段计算局部 Online Softmax，Reduce 阶段通过 LSE 全局合并；`num_splits` 依据 SM 数量动态推导（target = SM×4），解决单请求场景 SM 空闲问题。
- **CUDA Graph 静态 Buffer 封装**：针对轻载 Decode 场景（B=1/S=128）Launch 开销主导的问题，设计静态输入 Buffer + 独立 warmup stream 预捕获方案，将端到端瓶颈从 Launch-bound 转为 Memory-bound。
- **任意 Block Size 支持**：突破 vLLM 原生算子仅支持 {8, 16, 32} 的限制，支持 block_size=128 等大块配置，Block Table 显存占用降至 1/8。
- **四层评估体系**：正确性验证 → Roofline 分析（L2 强制驱逐）→ CUDA Graph 加速比 → vLLM 原生算子对标。

## 📁 项目结构

```text
paged-attention-triton/
├── main.py                    # 统一入口：带宽测试 + 校验 + Benchmark
├── measure_peak_bw.py         # GPU HBM 峰值带宽实测工具
├── paged_attention/           # 核心算子包
│   ├── split_kv_kernel.py     # Triton Kernel 底层实现 (Split & Reduce)
│   ├── torch_kernel.py        # PyTorch autograd 封装与 Grid 分发
│   └── graph_runner.py        # CUDA Graph 静态 Buffer 管理
└── tests/                     # 测试与评估模块
    ├── correctness.py         # 正确性验证 (vs PyTorch Reference)
    ├── benchmark_core.py      # Layer 1-3: Roofline / System / Graph
    └── benchmark_vllm.py      # Layer 4: vLLM Native Baseline 对标

```

## 🚀 快速开始

### 环境要求

- Python >= 3.8
- CUDA >= 11.8
- PyTorch >= 2.1.0
- Triton >= 2.1.0

### 安装与运行

```bash
# 安装依赖
pip install -r requirements.txt

# 一键运行：带宽实测 → 正确性校验 → 4层Benchmark
python main.py

```

## 📊 性能结果

> 以下数据基于 RTX 3050 Laptop (4GB VRAM) 实测，Roofline 基准为实测 HBM 带宽（~179.5 GB/s）而非理论峰值。

### Layer 1: Roofline 分析

| 场景 | Latency | Eff BW | BW Util | Arith Int |
| :--- | :--- | :--- | :--- | :--- |
| 单请求短上下文 (B1 S128) | 313.2µs | 1.7 GB/s | 0.9% | 0.992 |
| 单请求长上下文 (B1 S4096) | 799.5µs | 21.0 GB/s | 11.7% | 1.000 |
| 满载SM+长序列 (B32 S1024) | 872.4µs | 154.0 GB/s | **85.8%** | 0.999 |
| 极限并发 (B128 S256) | 899.1µs | 149.9 GB/s | **83.5%** | 0.996 |

✅ B≥32 稳态下有效带宽达峰值的 **83%~86%**，算术强度 <1.0 确认纯 Memory-Bound 特性。

### Layer 3: CUDA Graph 加速

| 场景 | Eager | Graph | Speedup |
| :--- | :--- | :--- | :--- |
| S=128 (SM空闲, launch主导) | 192.7µs | 45.5µs | **4.24x** |
| S=256 (过渡区) | 173.7µs | 51.6µs | **3.37x** |
| S=512 (Split-KV生效) | 175.8µs | 50.3µs | **3.50x** |

### Layer 4: vs vLLM Native

- **B≥32 高并发稳态**：性能比 **0.89x ~ 0.91x**（接近 CUDA 手写算子水平）
- **额外收益**：支持 block_size=128，Block Table 显存降至 vLLM 的 **1/8**

## 🔧 技术要点

### Split-KV 动态切分策略

```python
num_splits = min(ceil(seq_len / block_size), max(1, SM_count × 4 / (batch × heads)))

```

- **上界**：不超过物理 block 数，避免空 split。
- **下界**：保证每个 SM 至少有 4 个 program 可调度，消除单请求 SM 空闲。

### LSE 哨兵值设计

使用 `-1e30` 替代 `-inf` 作为无效 split 的 LSE 标记，避免 FP32 存储精度丢失导致 Reduce 阶段数值错误。阈值 `-1e29` 用于安全过滤。

### CUDA Graph 预捕获

采用独立 warmup stream 执行 3 次预热后再捕获 Graph，避免首次 replay 时的隐式同步开销与 JIT 编译干扰。

## 📝 License

MIT