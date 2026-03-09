# lz4_hybrid 性能分析报告（按当前实现与修正后结果更新）

> 更新时间：2026-03-09  
> 程序路径：`/root/lz4/lz4_hybrid/lz4_hybrid`  
> 当前全量结果：`/root/lz4/exp_results/hybrid_bench/hybrid_bench_20260309_180949.csv`
> 对照基线：`/root/lz4/exp_results/runs/20260309_merged_full_83/lz4_param_sweep_merged.csv`

## 1. 概述 (Overview)

`lz4_hybrid` 是当前 LZ4 家族中的 CPU--GPU 协同执行器。它的目标不是替代纯 GPU 或纯 CPU，而是在块级并行前提下同时利用：

- CPU 端 `liblz4 + pthread`
- GPU 端 `lz4_gpu` OpenCL backend

并在 **steady-state end-to-end total throughput**、压缩率、功率之间寻找可接受的平衡点。

与旧版文档相比，本版最重要的变化有两点：

1. **测试口径已修正**：当前 total throughput 来自 `lz4_hybrid --bench --bench-io` 的 warmed in-binary 路径；同时 `kernel_tp` 已按真实 CPU/GPU 编解码执行跨度（取两者 max）计算，不再直接使用包含额外 host-side coordination 的 `parallel_us`；
2. **adaptive split 已真实生效**：此前 `--adaptive` 曾存在“CLI 有开关但 active split path 不改变”的缺口，本轮已在 live code 中修复。

- **核心目标**：探索 CPU/GPU 协同在 LZ4 实时压缩中的真实系统价值，而不是只比较 kernel speed。  
- **程序路径**：`/root/lz4/lz4_hybrid/lz4_hybrid`  
- **主要特性**：fixed/adaptive split、多线程 CPU、OpenCL GPU、bench-io total semantics。

## 2. 设计原理 (Design Principles)

当前 `lz4_hybrid` 的设计仍然遵循以下核心原则，但其内容已从旧文档中的“固定比例试验版”演化为当前实现：

- **任务切分 (Work Splitting)**：输入被切成固定 block（默认 16KB），然后按 `effective_gpu_ratio` 划分为 GPU 前缀块与 CPU 后缀块。
- **并行执行 (Parallel Execution)**：CPU 任务在线程侧并行执行，主线程驱动 GPU OpenCL 路径，两者并发后以较慢一路的结束时间逼近墙钟时间。
- **数据块独立性 (Block-level Independence)**：每个块独立压缩/解压，因此可以自然地被映射到 CPU worker 或 GPU work-item。
- **统一输出格式 (Unified Output Format)**：hybrid 输出包含 magic、块总数、GPU 块数、块大小表以及 GPU/CPU 两路压缩数据，确保解压端可以确定性地分发。
- **低开销调度优先**：当前 adaptive 并不是复杂预测器，而是前若干块采样 + 轻量启发式修正，以避免为了“智能”而引入比收益更高的调度成本。

### 2.1 Fixed split

fixed 模式下：

```text
gpu_blocks = round(num_blocks * gpu_ratio)
cpu_blocks = num_blocks - gpu_blocks
```

优点是：

- 调度成本极低；
- 行为稳定；
- 适合做 sweep 找最优配比。

### 2.2 Adaptive split

当前 adaptive 模式会：

1. 读取前 `adaptive_sample_blocks` 个块；
2. 用 `LZ4_compress_fast()` 对样本估算压缩率；
3. 结合文件大小、CPU 线程数和采样压缩性调整 `gpu_ratio`；
4. 生成 `effective_gpu_ratio` 后再执行真正的 CPU/GPU 分发。

因此现在的 adaptive 是一个 **真实的 per-file split policy**，而不再只是 benchmark 维度里的“标签”。

## 3. 系统架构 (Architecture)

### 数据流图（压缩过程）

```text
输入文件 → read_entire_file()
        → hybrid_compress_memory()
             ├→ GPU 块 → gpu_compress_blocks() (OpenCL 内核)
             └→ CPU 块 → cpu_comp_top() → pthread 工作线程 → LZ4_compress_default()
          (两路并发)
        → 组装 hybrid 容器：[Header | Size Table | GPU Data | CPU Data]
        → 输出缓冲区 / 文件
```

### 数据流图（解压过程）

```text
压缩文件 → 解析头部信息 (Magic, Num_blocks, Block_size, GPU_blocks)
        → 提取块大小表
        → GPU 块 → gpu_decompress_blocks() (OpenCL 内核)
        → CPU 块 → cpu_decomp_top() → pthread 工作线程 → LZ4_decompress_safe()
          (两路并发)
        → 重组解压数据
        → 输出结果
```

### 核心组件

- **`ocl_env_t`**：管理 OpenCL 设备、上下文、命令队列和内核；
- **`hybrid_cfg_t`**：当前配置结构，已包含 `adaptive_split` 与 `adaptive_sample_blocks`；
- **`hybrid_metrics_t`**：记录 CPU kernel、GPU kernel、parallel wall-time 和 total wall-time；
- **`cpu_comp_job_t` / `cpu_decomp_job_t`**：封装 CPU 端工作包；
- **hybrid header/container**：保证解压端能无歧义地恢复 GPU/CPU 两路处理边界。

## 4. 实现细节 (Implementation Details)

### 4.1 负载切分逻辑

当前 live code 中，`hybrid_compress_memory()` 会先计算 `num_blocks`，然后：

- fixed 模式直接使用 `cfg->gpu_ratio`
- adaptive 模式先调用采样逻辑得到 `effective_gpu_ratio`

再决定：

- `gpu_blocks`
- `cpu_blocks`
- `cpu_start`

这是本轮修复的关键点，因为此前 active path 里并没有真正覆盖 `cfg->gpu_ratio`。

### 4.2 GPU 处理路径

GPU 路径调用：

- `gpu_compress_blocks()`
- `gpu_decompress_blocks()`

这些路径并不是独立重写的一套 GPU runtime，而是直接复用 `lz4_gpu_core` 中当前成熟的 backend。也就是说，hybrid GPU 子路径继承了：

- buffer/workspace 复用能力
- 更成熟的 worker count 选择
- 解压快路径
- LZ4 GPU 主路径的 corrected steady-state runtime 基础

### 4.3 CPU 处理路径

CPU 路径通过 `pthread` 和 `liblz4` 并发处理后缀块：

- 压缩：`LZ4_compress_default()` / `LZ4_compress_fast()` 家族路径
- 解压：`LZ4_decompress_safe()`

CPU 路径的意义不是给 GPU 打下手，而是：

- 在高可压缩或中小规模文件上，CPU 往往仍然有强竞争力；
- hybrid 的核心正是利用 CPU 在某些数据上的优势来弥补 GPU 的系统级开销。

### 4.4 计时与性能指标

当前文档中使用的指标含义如下：

- **`cpu_kernel_us` / `gpu_kernel_us`**：CPU 与 GPU 纯工作路径时间；
- **`parallel_us`**：两路并行的核心墙钟时间；
- **`total_us`**：包含文件 I/O、组装和运行时开销的总时间；
- **Kernel Throughput** = `input_MB / max(cpu_kernel_us, gpu_kernel_us)`，用于刻画 hybrid 下真实编解码执行跨度；
- **Total Throughput** = `input_MB / total_us`

### 4.5 `--bench-io` 的意义

当前 `lz4_hybrid` 已提供：

```text
--bench --bench-io
```

这意味着 total throughput 直接由 warmed binary 内部给出，而不是从 Python 外层 separate compress/decompress subprocess wall-clock 估算。这是当前 hybrid total 数据可信的根本前提之一。

## 5. 配置参数 (Configuration Parameters)

当前主要参数如下：

- **`--gpu-ratio F`**：GPU 块比例（0.0-1.0）
- **`--adaptive`**：启用 adaptive split
- **`--sample-blocks N`**：adaptive 采样块数
- **`-T N` / `--cpu-threads N`**：CPU 工作线程数
- **`-b N`**：块大小（当前 hybrid bench 中主路径仍以 16KB 为中心）
- **`-H N`**：GPU hash log
- **`-a N`**：GPU acceleration
- **`-l N`**：GPU local work-group size
- **`--bench-io`**：让 total throughput 包含文件写回/读回

## 6. 基准测试方法 (Benchmark Methodology)

当前 fresh rerun 的实验设置为：

- **文件集**：`/root/samples`，83 个文件
- **配置空间**：
  - `SplitMode = fixed | adaptive`
  - `GPURatio = 0.0, 0.3, 0.5, 0.7, 0.9, 1.0`
  - `CPUThreads = 1, 2`
  - `Acceleration = 1, 3`
- **总配置数/文件**：48
- **总数据点**：3984
- **结果文件**：`/root/lz4/exp_results/hybrid_bench/hybrid_bench_20260309_180949.csv`
- **正确性**：Parse failures = 0，结果要求 `VerifyOK=true`

关键变化是：

- 当前 total throughput 为 corrected steady-state total；
- adaptive 结果来自修复后的 live implementation，而不是此前“标签化 adaptive”。

## 7. 基准测试结果 (Benchmark Results)

### 7.1 Best-per-engine medians（每文件先在各 engine 内选最佳配置）

| Engine | Comp total MB/s | Dec total MB/s | Comp kernel MB/s | Dec kernel MB/s | Ratio % | Comp power W |
|---|---:|---:|---:|---:|---:|---:|
| CPU | 698.71 | 755.68 | 1853.29 | 5347.40 | 22.38 | 19.63 |
| GPU | **1497.76** | **1085.39** | **5952.21** | **14772.52** | 25.23 | 18.72 |
| Hybrid fixed | 1425.90 | 802.90 | 2725.26 | 3402.44 | 25.32 | 16.06 |
| Hybrid adaptive | 1302.24 | 813.20 | 2654.46 | 3968.56 | 25.32 | 16.17 |

### 7.2 Winner counts（每文件比较引擎）

**Compression total throughput**：

- Hybrid fixed：**48**
- GPU：**28**
- Hybrid adaptive：**4**
- CPU：**3**

**Decompression total throughput**：

- GPU：**62**
- Hybrid fixed：**14**
- Hybrid adaptive：**4**
- CPU：**3**

### 7.3 当前最好的 hybrid 配置区域

按 fresh full-corpus artifact 看，当前更可信的判断是：

- raw median：fixed `919.72 / 675.42 MB/s`，adaptive `889.91 / 674.58 MB/s`
- best-per-file median：fixed `1425.90 / 802.90 MB/s`，adaptive `1302.24 / 813.20 MB/s`

因此本节后续讨论以 file-level best-per-engine 与 winner-count 为主，而不再把旧的 mean-config 排序当作主结论来源。

## 8. 性能分析 (Performance Analysis)

### 8.1 当前主结论已经变化：GPU 才是 LZ4 family 的主导引擎

旧文档中大量分析默认 hybrid 最终会成为主路径，但 fresh rerun 已证明当前 corrected 结论是：

- **GPU 仍是默认总吞吐主路径**；
- hybrid fixed 已经成为压缩侧的强竞争者；
- adaptive 在解压侧略优于 fixed，但仍未成为整体最优；
- CPU 只在极少数文件上获胜。

### 8.2 fixed vs adaptive

**Raw median（跨全部 hybrid rows）**：

| Mode | Comp total MB/s | Dec total MB/s |
|---|---:|---:|
| Fixed | **919.72** | **675.42** |
| Adaptive | 889.91 | 674.58 |

**Best-per-file median**：

| Mode | Comp total MB/s | Dec total MB/s |
|---|---:|---:|
| Fixed | **1425.90** | 802.90 |
| Adaptive | 1302.24 | **813.20** |

这说明 adaptive 的正确结论是：

- 它现在已经是真实功能；
- 它在当前 fresh run 中已经形成真实的解压侧优势；
- 但在 compression 和 winner count 层面仍然不如 fixed。

因此当前不能写“adaptive 胜出”，而应写：

> **adaptive 已实现并经 fresh rerun 验证，但当前启发式尚未超过最优 fixed split。**

### 8.3 hybrid 的现实价值

尽管不是整体第一，hybrid 仍有两个现实意义：

1. **部分文件上仍能大规模胜出**：尤其压缩侧 fixed 已赢 48 个文件；
2. **仍可在部分文件上胜出**，但压缩功率不再像上一版那样明显低：
   - fixed: 16.06W
   - adaptive: 16.17W
   - CPU: 19.63W
   - GPU: 18.72W

所以 hybrid 当前不是“无用”，而是一个 **吞吐不及 GPU、但在一部分文件上仍有价值的平衡方案**；功率维度已经不能再被写成它的主要优势。

## 9. 典型现象与深度分析 (Typical Phenomena & Deep Analysis)

### 现象 1：0.3 比例仍是中心甜点区

无论 fixed 还是 adaptive，表现最好的配置都集中在 `gpu_ratio≈0.3, T=2`。这说明当前平台上的最佳协同方式不是“GPU 尽量多做”，而是：

- GPU 负责一部分块；
- CPU 负责剩余部分并提供高吞吐收尾；
- 避免把太多任务交给受 host/runtime 路径约束的 GPU 子路径。

### 现象 2：adaptive 会真实改变 split，但不一定带来更好 total

修复后 adaptive 在 verbose 模式下可以输出有效的 `effective_gpu_ratio` 和 `gpu_blocks/cpu_blocks`。但“split 确实改变了”并不等于“系统总吞吐必然提高”。这正是当前结果告诉我们的现实：

- 调度器真实在工作；
- 但启发式不够好时，调度自由度也可能带来次优划分。

### 现象 3：hybrid ratio 明显更差

当前 hybrid best-per-file median ratio 为：

- fixed：**25.32%**
- adaptive：**25.32%**

二者都明显高于：

- CPU：22.38%
- GPU：25.23%

说明当前 hybrid 容器化与 split 设计仍然带来了明显压缩率代价。这是它没有成为默认路径的重要原因之一。

## 10. 按数据类型分析 (Analysis by Data Type)

当前 full rerun 没有继续沿用旧文档那种按“高/中/低压缩率”给出大量旧口径数字的写法，因为那些数字已与 corrected methodology 不一致。但从 fresh rerun 的 winner 分布和最优配置区域仍能观察到：

- **高度可压缩文件**：CPU 竞争力更强，hybrid 往往需要降低 GPU 比例；
- **中等压缩率文件**：hybrid 最可能体现价值，尤其在 `gpu_ratio≈0.3` 附近；
- **低压缩率 / GPU 友好文件**：纯 GPU 更容易成为赢家。

这也正是 adaptive 仍值得继续研究但尚未完成的原因：

> 当前 split policy 已真实存在，但还没有足够强到稳定识别这些文件类别并超过 best fixed。 

## 11. 优化机会 (Optimization Opportunities)

在当前 corrected 结果下，真正值得继续做的方向包括：

1. **进一步压缩 GPU 子路径的 host/runtime 成本**：让 hybrid 中的 GPU 更接近纯 GPU backend 的 steady-state 表现；
2. **更强的 adaptive 规则**：当前采样启发式过于轻量，能改 split，但不足以稳定提高 total throughput；
3. **容器/组装成本优化**：当前 ratio 与 total throughput 的双重损失，部分来自 hybrid 容器与结果组装路径；
4. **更细粒度 pipeline overlap**：当前主要是并发分路，不是深流水重叠。

### 11.1 CPU OpenCL 路径的当前定位

本轮用户特别要求验证“CPU 是否也应像 GPU 一样走 OpenCL 内核”。当前代码已通过 `FORCE_OPENCL_DEVICE=CPU` 完成验证，结论是：

- **功能上可运行**：LZ4 GPU backend 在 Intel CPU OpenCL 设备上可正确完成压缩/解压；
- **局部 case 有竞争力**：例如 `dickens` 与 `industrial_parent_0_pages_img.tar` 上，CPU OpenCL 可接近甚至短暂超过 GPU OpenCL；
- **但不应取代当前 native CPU path**：从系统级角度看，它并未稳定优于现有 `liblz4 + pthread` 路径，也没有证明自己能改善当前 hybrid 的总体排序。

因此，当前最合理的处理方式不是删除这条路径，而是把它保留为 **设备可移植性与后续研究入口**，而不是默认部署设计。

## 12. 结论 (Conclusions)

当前 `lz4_hybrid` 的结论应更新为：

- **实现层面**：CPU path、GPU path、fixed split、adaptive split、bench-io total semantics 均已落地；
- **结果层面**：GPU 仍是 LZ4 family 的主导总吞吐引擎；hybrid fixed 已成为压缩侧强竞争者；adaptive 在解压侧更有价值；
- **系统层面**：hybrid 的价值在于部分文件胜出与协同研究空间，而不是当前全局最快引擎；CPU OpenCL 虽然已验证可运行，但暂不构成替代 native CPU path 的依据。

因此，当前最准确的表述是：

> `lz4_hybrid` 已经从概念验证进化为一个完整、可验证、带真实 adaptive 的协同实现；在 fresh 83-file full-corpus 结果中，fixed hybrid 已经成为压缩侧的强竞争者，但 GPU 仍然是更稳的默认主路径，尤其在解压侧仍保持明显优势。

## 13. 2026-03-09 优化轮次快照

本轮围绕用户提出的 hybrid ratio / throughput 与 LZ4 64KB 局限，完成了以下 live code 变更：

- 修复 `gpu_compress_blocks()` 中 GPU dict buffer 未清零导致的 repeated bench verify fail；
- 修复 `run_bench()` 中 `--bench-io` 固定 `/tmp` 文件名导致的冲突；
- GPU 子路径改为复用 `lz4_gpu_workspace_t`，减少 `clCreateBuffer` / `clReleaseMemObject` 频率；
- GPU compression / decompression 路径引入 CU-aware worker sizing、mapped readback；
- adaptive sampling 从“只看文件头”改为跨文件分布式采样；
- CPU 子路径压缩从 `LZ4_compress_default()` 切换为 `LZ4_compress_fast()`，与 acceleration 语义保持一致；
- `bench_hybrid.py` 从单一 `16K` 扩展为 `16K / 32K / 64K` sweep。

### 13.1 定向 subset 结果（优化后，1s warmed bench）

#### `dickens`，fixed `gpu_ratio=0.7, T=2, a=3`

| Block | Comp total MB/s | Dec total MB/s | Ratio % |
|---|---:|---:|---:|
| 16K | 259.29 | 323.67 | 71.91 |
| 32K | 267.08 | **325.93** | 67.73 |
| 64K | **268.60** | 325.51 | **64.62** |

#### `dickens`，adaptive `gpu_ratio=0.7, sample_blocks=8`

| Block | Comp total MB/s | Dec total MB/s | Ratio % |
|---|---:|---:|---:|
| 16K | 303.69 | 292.22 | 71.68 |
| 64K | **304.22** | **305.07** | **64.27** |

#### `industrial_parent_0_pages_img.tar`

| Mode | Block | Comp total MB/s | Dec total MB/s | Ratio % |
|---|---|---:|---:|---:|
| Fixed 0.7 | 64K | **658.18** | 351.84 | **19.28** |
| Adaptive | 64K | 468.74 | **361.48** | 19.31 |

### 13.2 当前结论

当前 subset 结果已经表明：

1. **hybrid 的 correctness blocker 已被清除**，16K/32K/64K warmed `--bench-io` 均可稳定 verify；
2. **把 hybrid 从固定 16K 的旧 sweep 中解放出来后，ratio 与 total throughput 明显改善**；
3. `dickens` 上 adaptive 会把有效 GPU 比例推到约 0.93--0.97，从而把 comp total 提升到 ~304 MB/s；
4. `industrial_parent_0_pages_img.tar` 上 fixed 64K hybrid comp total 达到 **658.18 MB/s**，已经在该 workload 上超过当前 64K LZ4 GPU 的 **597.88 MB/s**；
5. 因而当前更准确的表述不再是“hybrid 系统性落后 GPU”，而是：

> `lz4_hybrid` 经过本轮修复与运行时优化后，已经能在部分 workload 上超过纯 GPU；但在更广 workload 上是否形成系统级反超，仍需新的全量 rerun 来确认。
