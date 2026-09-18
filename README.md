# PagedAttention: Split-KV + CUDA Graph

基于 Triton 实现的高性能 PagedAttention 推理算子，针对 LLM Decode 阶段的 **非连续 KV Cache 访存** 与 **单请求 SM 利用率低** 两大核心瓶颈进行优化。

## ✨ 核心特性

- **Split-KV 两段式 Attention**：将长序列 KV 按 split 维度切分为并行 program，Split 阶段计算局部 Online Softmax，Reduce 阶段通过 LSE 全局合并；`num_splits` 依据 SM 数量动态推导（target = SM×4），解决单请求场景 SM 空闲问题
- **CUDA Graph 静态 Buffer 封装**：针对轻载 Decode 场景（B=1/S=128）Launch 开销主导的问题，设计静态输入 Buffer + 独立 warmup stream 预捕获方案，将端到端瓶颈从 Launch-bound 转为 Memory-bound
- **任意 Block Size 支持**：突破 vLLM 原生算子仅支持 {8, 16, 32} 的限制，支持 block_size=128 等大块配置，Block Table 显存占用降至 1/8
- **四层评估体系**：正确性验证 → Roofline 分析（L2 强制驱逐）→ CUDA Graph 加速比 → vLLM 原生算子对标

## 📁 项目结构

paged-attention-triton/
├── main.py # 统一入口：带宽测试 + 校验 + Benchmark
├── measure_peak_bw.py # GPU HBM 峰值带宽实测工具
├── paged_attention/ # 核心算子包
│ ├── split_kv_kernel.py # Triton Kernel 底层实现 (Split & Reduce)
│ ├── torch_kernel.py # PyTorch autograd 封装与 Grid 分发
│ └── graph_runner.py # CUDA Graph 静态 Buffer 管理
└── tests/ # 测试与评估模块
├── correctness.py # 正确性验证 (vs PyTorch Reference)
├── benchmark_core.py # Layer 1-3: Roofline / System / Graph
└── benchmark_vllm.py # Layer 4: vLLM Native Baseline 对标


## 🚀 快速开始

### 环境要求

- Python >= 3.8
- CUDA >= 11.8
- PyTorch >= 2.1.0
- Triton >= 2.1.0

### 一键运行

# 带宽实测 → 正确性校验 → 4层Benchmark
python main.py