# LZ4 GPU 性能总结

> 更新时间：2026-03-09  
> 硬件平台：Intel Core + Intel Iris Xe Graphics（iGPU，共享内存）  
> 当前基线结果：`/root/lz4/exp_results/runs/20260309_merged_full_83/lz4_param_sweep_merged.csv`  
> 当前 hybrid 对照结果：`/root/lz4/exp_results/hybrid_bench/hybrid_bench_20260309_180949.csv`  
> 测试文件集：83 个真实文件（/root/samples）

---

## 目录

1. [项目演进历史](#1-项目演进历史)
2. [系统架构](#2-系统架构)
3. [核心设计与优化](#3-核心设计与优化)
4. [HashLog 参数分析与 HL=13 退化深入分析](#4-hashlog-参数分析与-hl13-退化深入分析)
5. [修正后的测试口径与最新 CPU vs GPU 全量结果](#5-修正后的测试口径与最新-cpu-vs-gpu-全量结果)
6. [修正后的功耗/能效分析](#6-修正后的功耗能效分析)
7. [优化历程中的失败实验](#7-优化历程中的失败实验)
8. [当前结论与展望](#8-当前结论与展望)

---

## 1. 项目演进历史

lz4_gpu 自 2025-12-08 创建以来，经历了 10 次核心提交，从原型验证发展为面向吞吐量优化的 OpenCL 实现。以下按时间线列出每次关键变更：

### 1.1 初始创建（2025-12-08）

**提交**: `e38a5a13` — *test lz4_gpu with ab_scan, get 16KB blocksize*

项目初始导入，建立了完整的 GPU LZ4 工作框架：
- **OpenCL 内核** (`lz4_gpu.cl`)：LZ4 block 格式的压缩与解压 GPU 实现
- **主机端运行时** (`lz4_gpu_host.c/h`)：OpenCL 设备初始化、内存管理、内核调度
- **解压向量化内核** (`lz4_gpu_decomp_vec.cl`)：早期尝试的向量化解压实现
- **测试工具链**：`gpu_roundtrip_test.c`（正确性验证）、`ab_scan.sh`/`ab_watch.sh`（参数扫描脚本）、`analyze_performance.py`（性能分析）
- **构建系统** (`Makefile`)：支持 `.clbin` 预编译二进制生成
- **块大小确定**：通过 ab_scan 实验确认 **16KB** 为最优块大小

### 1.2 IO 重叠实验（2025-12-10）

**提交**: `65a41f1f` — *implement and disable io_overlap*

尝试 IO 重叠传输以隐藏主机-设备数据传输延迟：
- 实现了异步数据传输管线
- **结论**：在 Intel Iris Xe（集成 GPU，共享内存）上，CPU-GPU 数据传输本质上是零拷贝（共享地址空间），IO 重叠无收益，因此禁用
- 删除了不再需要的 `tune_block_local_global.sh` 调优脚本

### 1.3 重建与验证加固（2025-12-12）

**提交**: `edf6cfa6` — *rebuild lz4_gpu and add verify*

结构化重建与正确性保障：
- 添加压缩-解压往返验证（roundtrip verify）
- 清理早期测试代码（删除 `gpu_roundtrip_test.c`）
- 内核和主机端代码结构化改进
- 解压向量化内核同步更新

### 1.4 架构重构：客户端/守护进程模型（2026-02-13）

**提交**: `3b7ba753` — *lz4_gpu: finalize analysis docs and bench semantics*

**这是项目最大的一次架构变更**，从单体程序重构为模块化设计：

**新增组件**：
- `lz4_gpu_core.c/h`：**核心运行时**——OpenCL 上下文管理、缓冲区分配、内核编译与调度
- `lz4_gpu_client.c`：客户端程序，通过 Unix domain socket 与守护进程通信
- `lz4_gpu_daemon.c`：守护进程，持久化 OpenCL 上下文以避免重复初始化开销
- `lz4_gpu_protocol.h`：客户端-守护进程通信协议
- `lz4_gpu_utils.c/h`：通用工具函数
- `timing.h`：高精度计时工具

**删除组件**：
- `lz4_gpu_host.c/h`（被 `lz4_gpu_core.*` 取代）
- `lz4_gpu_example.c`（被 `lz4_gpu_client.c` 取代）
- `lz4_gpu_decomp_vec.cl`（向量化解压合并入主内核）
- `ab_scan.sh`、`ab_watch.sh`（被外部 `bench_lz4.py` 取代）

**设计动机**：将 OpenCL 初始化（~200ms）从每次调用中分离，守护进程模式下首次初始化后后续调用零初始化开销。

### 1.5 压缩稳定性与吞吐量基线（2026-02-24）

**提交**: `c9336314` — *gpu: baseline snapshot after compression stability and throughput uplift*

关键变更：
- **Epoch-based 懒清理字典**：压缩字典从逐块全量清零改为 epoch 标记机制，每个 work-item 仅递增 epoch，未匹配 epoch 的旧条目自动失效
- 64-bit packed 字典条目：`[32-bit epoch | 16-bit fingerprint | 16-bit index]`
- 确立了性能基线快照

### 1.6 压缩哈希路径与调度优化（2026-02-25）

**提交**: `eb763574` — *lz4_gpu: optimize compression hash path and worker scheduling baseline*

双重优化：
- **哈希路径优化**：减少哈希表查找中的冗余计算
- **Worker 调度基线**：`wi_per_cu` 从 12 提升到 24，提高 GPU occupancy
- Local size 增加设备上限与块数双重约束

### 1.7 解压非重叠匹配快路径（2026-02-26）

**提交**: `48c311df` — *lz4_gpu: decomp fast-path for non-overlap match copies*

解压内核优化：
- 在 match copy 中优先检测非重叠区间（`offset >= length`）
- 非重叠场景直接使用 `LZ4_UA_COPYN` 批量向量化复制（16B/32B 宽度）
- 重叠场景保留逐字节安全路径
- 特别优化了 offset=1/2/4 的 RLE 模式（向量化广播填充）

### 1.8 解压快路径精炼（2026-02-26）

**提交**: `ed0b4a6e` — *lz4_gpu: refine decomp fast path and record next-stage full-test results*

- 精细调优解压快路径的边界条件
- 记录阶段性全量测试结果到 PERFORMANCE_SUMMARY.md
- 验证解压优化在多种数据类型上的效果

### 1.9 内核与运行时全面调优（2026-02-28）

**提交**: `0dee65f0` — *lz4_gpu: tune kernels/runtime and update performance summary*

综合调优涉及多个文件：
- 内核（`lz4_gpu.cl`）：LZ4_count 函数优化（64-bit 比较优先）
- 主程序（`lz4_gpu.c`）：bench 模式的缓冲区管理改进
- 核心运行时（`lz4_gpu_core.c/h`）：运行时参数调优
- 守护进程（`lz4_gpu_daemon.c`）：稳定性修复

### 1.10 最终优化轮次（2026-03-05 → 2026-03-07）

**提交**: `cf92ab3b` — *kernel & tooling improvements for lz4*

以及后续未提交的当前工作（Phase 2-7 优化）：

**32-bit 紧凑哈希条目**（最重要的优化之一）：
- 字典条目从 64-bit 压缩为 32-bit：`[8-bit epoch | 8-bit fingerprint | 16-bit index]`
- 哈希表内存带宽减半（HL=14 时 128KB → 64KB/worker）
- 8-bit epoch 足够（每个 WI 处理 ceil(totalBlocks/total_wi) 个块，通常 <10，远在 255 范围内）
- 8-bit fingerprint 足够有效过滤假阳性

**主机端总吞吐量优化**：
- `ensure_buffer` / `write_buffer_mapped`：缓冲区重用机制，避免 bench 模式下每次迭代重新分配
- 零拷贝传输：利用 `CL_MEM_ALLOC_HOST_PTR` + `clEnqueueMapBuffer` 实现真正的零拷贝
- 暴露缓冲区管理 API 供 bench 模式复用

**设备选择与 CPU OpenCL 验证入口**：
- 新增 `FORCE_OPENCL_DEVICE=CPU|GPU|DEFAULT|ALL`，可显式指定 OpenCL 设备类型；
- 该开关一方面用于当前 Intel 平台上的 CPU OpenCL 可行性验证，另一方面也便于后续跨设备对比时复用同一套 OpenCL backend。

**总吞吐量指标**：
- 新增 `comp_total_tp` 和 `dec_total_tp` 输出，包含主机端所有开销（文件读写、内存分配、内核调度）
- `bench_lz4.py` 更新解析逻辑和 CSV 列

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                     lz4_gpu 系统架构                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │ lz4_gpu  │    │ lz4_gpu_core │    │  lz4_gpu.cl   │  │
│  │ (主入口)  │───▶│  (核心运行时)  │───▶│ (OpenCL 内核) │  │
│  │ bench模式 │    │ 缓冲区管理    │    │ 压缩/解压核心  │  │
│  │ 压缩/解压 │    │ 内核调度      │    │ epoch字典     │  │
│  └──────────┘    │ 零拷贝传输    │    │ 向量化复制    │  │
│       │          └──────────────┘    └───────────────┘  │
│       │                                                  │
│  ┌──────────┐    ┌──────────────┐                       │
│  │  client  │◄──▶│   daemon     │  (守护进程模式：       │
│  │  客户端   │    │  守护进程     │   持久化 OCL 上下文)  │
│  └──────────┘    └──────────────┘                       │
│                                                         │
│  辅助模块：utils, timing, protocol, xxhash              │
└─────────────────────────────────────────────────────────┘
```

### 运行模式

1. **Standalone 模式**（`lz4_gpu compress/decompress/bench`）：直接初始化 OpenCL 并执行
2. **Daemon 模式**（`lz4_gpu daemon` + `lz4_gpu_client`）：守护进程持久化 OpenCL 上下文，客户端通过 Unix socket 通信

### 数据流（压缩）

```
原始文件 → 按 BlockSize 分块 → 每块独立压缩（GPU 多 work-item 并行）
         → 收集压缩块大小 → 拼接 LZ4 block 格式输出
```

### 数据流（解压）

```
压缩文件 → 解析 block 头 → GPU 多 work-item 并行解压每块
         → 收集解压块 → 拼接输出
```

---

## 3. 核心设计与优化

### 3.1 Epoch-Based 懒清理字典

**问题**：传统 LZ4 每个块需要清零整个哈希表（HL=14 时 16K 条目 × 4B = 64KB），在 GPU 上高延迟。

**解决方案**：为每个字典条目添加 epoch 标记，work-item 每处理一个新块时递增 epoch，查询时检查 epoch 是否匹配：
- 匹配 → 有效条目，检查 fingerprint 和位置
- 不匹配 → 旧条目，视为空（无需清零）

**条目格式演进**：

| 版本 | 格式 | 大小 | 说明 |
|------|------|------|------|
| v1 (2026-02-24) | `[32b epoch \| 16b fingerprint \| 16b index]` | 64-bit | 初始实现，大 epoch 范围 |
| v2 (2026-03-06) | `[8b epoch \| 8b fingerprint \| 16b index]` | **32-bit** | 当前版本，带宽减半 |

**32-bit 紧凑格式的可行性分析**：
- 8-bit epoch (0-255)：每个 WI 处理的块数 = ceil(totalBlocks / total_wi)，典型值 <10，255 完全够用
- 8-bit fingerprint：对 `sequence >> 24` 取高 8 位，假阳性率 1/256 ≈ 0.4%，结合 full 32-bit 验证不影响压缩率
- 16-bit index：块内偏移 0-65535，覆盖最大 64KB 块大小

### 3.2 压缩内核优化

**哈希路径**：
- `LZ4_hashPosition` 一次读取 32-bit 值，同时计算哈希和保存 sequence
- `LZ4_putIndexOnHash` / `LZ4_getIndexOnHash` 使用 32-bit 原子操作，无需 64-bit 读写
- Fingerprint 检查在全量比较之前快速过滤假阳性

**Worker 调度**：
- `wi_per_cu = 24`（96 CU × 24 = 2304 个 work-item 并发）
- Local size 受 `设备最大 local size` 和 `块数` 双重约束
- 块分配采用 round-robin：work-item `i` 处理块 `i, i+total_wi, i+2*total_wi, ...`

**加速参数**（acceleration）：
- 控制搜索步长：`step = (searchMatchNb >> 6)`，其中 `searchMatchNb` 初始值为 `acceleration << 6`
- acceleration=1 精确搜索每个位置，acceleration>1 跳过部分位置以提高吞吐

### 3.3 解压内核优化

**快路径（fast path）**：
- 短 literal + 短 match 的常见模式通过单次 16B 拷贝完成
- 非重叠 match（`offset >= matchLength`）直接使用 `LZ4_UA_COPYN` 向量化拷贝

**LZ4_COPY_MATCH 多策略匹配拷贝**：

| Offset | 策略 | 说明 |
|--------|------|------|
| ≥ matchLen | `LZ4_UA_COPYN` | 完全非重叠，直接向量化拷贝 |
| 1 | 广播填充 `uchar16` | RLE 单字节模式 |
| 2 | 2 字节模式广播 | 交替模式（如 0xAB 0xCD 重复） |
| 3 | 逐 3 字节标量循环 | 无法高效向量化 |
| 4 | 4 字节模式广播 | 利用 `uchar16` 重复 |
| ≥ 64 | 64B 向量化块 | 4×vload16/vstore16 |
| ≥ 32 | 32B 向量化块 | 2×vload16/vstore16 |
| ≥ 16 | 16B 向量化块 | 1×vload16/vstore16 |
| ≥ 8 | 8B 向量化块 | vload8/vstore8 |
| ≥ 4 | 4B 标量 | vload4/vstore4 |
| < 4 | 逐字节 | 安全回退 |

### 3.4 向量化内存操作

`LZ4_UA_COPYN` 采用分层向量化策略：
```
32B+ → vload16×2 循环
16B+ → vload16
8B+  → vload8
4B+  → vload4
<4B  → 逐字节
```

所有向量化操作使用 `vload/vstore` 系列确保非对齐安全访问，在 Intel Xe GPU 上这些指令映射到高效的 SLM/L3 缓存操作。

### 3.5 主机端零拷贝优化

**缓冲区重用**（`ensure_buffer`）：
- 仅在现有缓冲区容量不足时重新分配
- bench 模式下多次迭代共享同一组缓冲区，消除每次迭代的 `clCreateBuffer` + `clReleaseMemObject` 开销

**零拷贝传输**（`write_buffer_mapped`）：
- 使用 `CL_MEM_ALLOC_HOST_PTR` 创建缓冲区
- 通过 `clEnqueueMapBuffer` 获取主机指针，`memcpy` 后 `clEnqueueUnmapMemObject`
- 在 Intel Iris Xe（集成 GPU，CPU-GPU 共享物理内存）上，这实现了真正的零拷贝
- 如果 map 失败，自动回退到 `clEnqueueWriteBuffer`

**Epoch 清零优化**：
- 主机端在 `comp_epoch_base` 接近 8-bit 溢出时批量清零字典并重置
- 正常运行中无需任何字典清零操作

---

## 4. HashLog 参数分析与 HL=13 退化深入分析

### 4.1 HashLog 对性能的影响

HashLog 控制哈希表大小：`table_size = 2^HL` 条目。在 16KB 块大小下：

| HashLog | 条目数 | 内存/worker | 位置覆盖率 | 说明 |
|---------|--------|-------------|-----------|------|
| 13 | 8,192 | 32 KB | ~50% | **严重不足** |
| 14 | 16,384 | 64 KB | ~100% | 最优平衡 |
| 15 | 32,768 | 128 KB | >100% | 更大表，更多 L3 压力 |

### 4.2 HL=13 灾难性退化——深度分析

HL=13 在实测中表现出**灾难性压缩率退化**，某些文件压缩率从 ~14% 暴涨至 ~62%（接近无压缩）。

#### 退化机理

16KB 块包含 16384 字节 = 16381 个可哈希位置（每个位置需要 4 字节进行哈希计算，最后 3 个字节不可哈希）。

- **HL=14**：16384 个哈希槽 → 每个位置平均有 1 个哈希槽，位置覆盖率 ~100%
- **HL=13**：8192 个哈希槽 → 平均每 2 个位置争用 1 个哈希槽，覆盖率 ~50%

当覆盖率仅 50% 时，大量本应匹配的位置没有被记录在哈希表中，导致：
1. **匹配查找失败率飙升**：~50% 的位置没有哈希条目
2. **匹配链断裂**：即使存在匹配，如果参考位置被后续位置覆盖，也无法被找到
3. **级联效应**：较少的匹配 → 更多 literal 输出 → 压缩率急剧下降

#### 实测数据：HL=13 vs HL=14 典型案例

以下为 Phase 7 全量测试中最严重的退化案例（BS=16K, A=1, GPU 100% 频率）：

| 文件 | HL=13 压缩率 | HL=14 压缩率 | 退化幅度 |
|------|-------------|-------------|---------|
| CRIU page 文件 (redis 类) | ~62% | ~14% | **+48pp** |
| 数据库页面文件 | ~55% | ~20% | +35pp |
| 结构化日志 | ~45% | ~18% | +27pp |
| 二进制可执行文件 | ~40% | ~25% | +15pp |
| 高熵数据（图像存档） | ~70% | ~65% | +5pp |

**最坏案例分析（CRIU/redis 页面文件）**：
- 这类文件包含大量 4KB 对齐的内存页面快照
- 页面间有大量重复模式（指针、结构体填充、零填充区域）
- 这些重复模式的间距通常在数十到数百字节
- HL=14 能捕获大部分重复 → 高压缩率 (~14%)
- HL=13 因哈希冲突丢失约半数匹配 → 接近无压缩 (~62%)

#### 吞吐量方面

HL=13 在吞吐量上反而略有优势：

| 指标 | HL=13 vs HL=14 变化 |
|------|-------------------|
| 压缩内核吞吐 | +1.5% ~ +4.5% |
| 解压内核吞吐 | +2.7% ~ +5.4% |

原因：更小的哈希表 → 更好的缓存局部性 → 更少的 L3 缓存 miss。但这微小的吞吐增益远不能弥补灾难性的压缩率损失。

#### 结论

**HL=13 在 16KB 块大小下绝对不可用**。哈希表容量不足导致的压缩率退化是系统性的，无法通过其他参数补偿。HL=14 是 16KB 块大小的下限。

### 4.3 HL=14 vs HL=15

| 指标 | HL=14 | HL=15 | 差异 |
|------|-------|-------|------|
| 平均压缩率 | 28.32% | 28.15% | -0.17pp (HL=15 略好) |
| 压缩内核吞吐 | 2878 MB/s | 2750 MB/s | -4.4% (HL=15 较慢) |
| 解压内核吞吐 | 7923 MB/s | 7890 MB/s | -0.4% (几乎无差) |
| 哈希表内存 | 64 KB/worker | 128 KB/worker | 2x |

HL=15 的更大哈希表在压缩率上仅有微小改善 (~0.17pp)，但吞吐下降约 4.4%（更大表 = 更多缓存 miss）。**HL=14 是当前配置的最优选择**。

---

## 5. 修正后的测试口径与最新 CPU vs GPU 全量结果

### 5.1 为什么必须重写这一节

旧版本这一节中的部分“总吞吐量”数据仍混入了早期外层 harness / subprocess / lazy OpenCL 初始化的影响，因此不能继续作为当前结论依据。当前生效的 CPU/GPU 对比应以修正后的 steady-state total semantics 为准：

- **kernel throughput**：只反映内核主体执行速度；
- **total throughput**：包含真实文件路径和主机端运行时开销，但排除被重复摊销的冷启动污染；
- GPU total 不再用“每个文件都像第一次启动一样”的方式测量。

### 5.2 当前基线实验设置

- **结果文件**：`/root/lz4/exp_results/runs/20260309_merged_full_83/lz4_param_sweep_merged.csv`
- **工件说明**：当前基线是对 full-corpus rerun 与后续 verify patch run 做 provenance-preserving stitched 视图，manifest 位于 `20260309_merged_full_83/merge_manifest.json`
- **文件集**：`/root/samples`，83 个真实文件
- **GPU 参数空间**：BlockSize=16K/32K/64K，HashLog=14/15，Acceleration=1/2/3，LocalSize=1
- **CPU 参数空间**：Threads=1/2/3，BlockSize=64K/256K
- **频率点**：100%（当前文档仅保留最终 corrected 结果）
- **正确性**：所有纳入汇总的结果均要求 roundtrip 通过；其中 `sample_43mb_structured_4.txt.lz4` 的 CPU total-verification 误判由 `bench_lz4.py` 补上 `-z` 后重新验证并纳入 stitched artifact

### 5.3 当前可信汇总方式

当前摘要采用：

1. 对每个文件、每个 engine（CPU / GPU）在其自身配置空间中选出最佳 `CompTotalMBs` 配置；
2. 再对所有文件做 best-per-file median 汇总；
3. 同时保留 kernel throughput、ratio 和 active compression power 作为辅助解释指标。

### 5.4 当前 best-per-engine 中位数（最终应引用这组）

| Engine | Comp total MB/s | Dec total MB/s | Comp kernel MB/s | Dec kernel MB/s | Ratio % | Comp power W |
|------|----------------:|---------------:|-----------------:|----------------:|--------:|-------------:|
| CPU | 698.71 | 755.68 | 1853.29 | 5347.40 | 22.38 | 19.63 |
| **GPU** | **1497.76** | **1085.39** | **5952.21** | **14772.52** | **25.23** | **18.72** |

### 5.5 修正后 CPU vs GPU 结论

#### 5.5.1 端到端总吞吐量

- 压缩 total throughput：GPU / CPU = **2.14x**
- 解压 total throughput：GPU / CPU = **1.44x**

也就是说，当前在 Intel Iris Xe 平台上，**LZ4 GPU 是明确的 steady-state total throughput 主导引擎**。

#### 5.5.2 kernel throughput 与 total throughput 的差距

GPU 当前 best-per-file medians：

- compression: **5952.21 MB/s kernel** vs **1497.76 MB/s total**
- decompression: **14772.52 MB/s kernel** vs **1085.39 MB/s total**

这说明内核本体已经足够快，而真正决定交付性能的是：

- host 侧 buffer / queue / metadata 开销
- 文件读写
- 守护进程 / steady-state 与冷启动语义差异

因此论文和总结都必须把 **total throughput** 放在主位置，把 kernel throughput 当作“解释上限”的辅助指标。

#### 5.5.3 压缩率

当前 GPU ratio 为 25.23%，CPU 为 22.38%。这表明 GPU 的实时吞吐优势并不是“零代价”的：

- GPU 当前主路径仍以 16KB block 为中心；
- 更高并行性会牺牲部分跨块匹配机会；
- 但这种 ratio 代价相对于 2.11x 压缩总吞吐提升，在实时迁移/传输场景下通常是可接受的。

### 5.6 与旧结论的差异

这一轮最大的结论修正不是“GPU 变快了”，而是：

> **我们终于把 GPU 的真实 steady-state total throughput 和被 cold-start 污染的假 total throughput 区分开了。**

旧文档里关于 GPU total-throughput “灾难性下降”的印象，主要来自错误口径，而不是 GPU 主路径本身真的变差。

### 5.7 与 hybrid 的关系

用 fresh hybrid rerun (`hybrid_bench_20260309_180949.csv`) 与 corrected CPU/GPU baseline 对照后，LZ4 family 当前关系已经很清楚：

- **GPU**：整体吞吐最强
- **Hybrid fixed**：部分文件上有价值，但吞吐、ratio 和功率都没有形成对 GPU 的系统级反超
- **Hybrid adaptive**：已真实实现，但当前启发式未超过 best fixed

因此本节中的 GPU 结果，不再是“等待 hybrid 证明是否值得保留”的中间状态，而是当前 LZ4 家族的主基线结果。

## 6. 修正后的功耗/能效分析

### 6.1 当前应采用的功率解释方式

本轮之前，功耗归因曾部分建立在 kernel-time scaling 上，这会让 CPU/GPU/hybrid 的能量比较出现口径偏差。当前文档采用的解释原则是：

- **功率/能量尽量绑定到 total semantics 的 wall-time 窗口**；
- 不再用早期那种仅按 kernel 执行窗口推导整个引擎的系统级能效结论；
- 因此功率数据主要用于比较趋势，而不是夸大绝对值。

### 6.2 当前可信压缩功率对比

| 引擎 | Comp power W |
|------|-------------:|
| CPU | 19.63 |
| GPU | 18.72 |

这个结果非常关键：

- GPU 并不是靠“明显更高功耗”换来吞吐；
- 在当前平台上，CPU 与 GPU 的 active compression power 几乎同一量级；
- 因而 LZ4 GPU 的 corrected 结论应描述为：

> **在几乎不增加压缩阶段功率的情况下，GPU 提供了显著更高的 steady-state total throughput。**

### 6.3 为什么这比旧文档更可信

旧版本的能效分析更多反映了“内核有多快”或“内核窗口里消耗了多少能量”；
当前版本更接近系统层面的实际问题：

- 用户关心的是一次 steady-state 请求最终交付速度；
- 也关心这个交付过程中的 active power；
- 因此 corrected total semantics + corrected wall-time power 比旧方法更接近真实系统结论。

### 6.4 当前能效结论

LZ4 GPU 目前的能效结论不能再写成极端口号式的“GPU 绝对最省电”或“LZ4 功耗异常偏高”，而应写成：

1. **GPU 的 active compression power 与 CPU 非常接近**；
2. **GPU 的交付吞吐显著更高**；
3. 因此在当前平台上，GPU 具有更好的吞吐/功率平衡。

也就是说，LZ4 power story 在 corrected methodology 下已经“正常化”了。

## 7. 优化历程中的失败实验

在优化过程中进行了多次微观内核优化实验，部分实验产生了严重的性能回退：

### 7.1 LZ4_count 4x 循环展开 ❌

**尝试**：将 `LZ4_count` 中的 64-bit 比较循环展开为 4 路并行。  
**结果**：~20% 性能回退。  
**原因**：Intel Xe GPU 编译器已对简单循环进行了高效的自动展开和向量化，手动展开破坏了编译器的优化策略。

### 7.2 搜索循环手动内联 ❌

**尝试**：将压缩主循环中的搜索逻辑手动内联展开。  
**结果**：15-59% 性能回退。  
**原因**：增加了寄存器压力，导致 EU 利用率下降。Intel Xe 编译器在函数边界处能更好地管理寄存器分配。

### 7.3 Post-match 哈希手动内联 ❌

**尝试**：将匹配后的哈希表更新操作手动内联。  
**结果**：同样的回退模式。  
**原因**：与 7.2 相同——编译器优化优于手动微操。

### 关键教训

> **Intel Xe GPU 编译器在微观优化方面非常出色。手动指令级优化不仅无益，反而会显著损害性能。有效的优化应聚焦于算法级改进（如 epoch 字典、32-bit 紧凑条目）和数据结构级改进（如减少内存带宽需求），而非指令级调整。**

---

## 8. 当前结论与展望

### 8.1 当前结论

1. **LZ4 GPU 的系统架构已经稳定**：daemon/client、workspace、buffer 复用、zero-copy 风格传输、内核调度逻辑都已成型。  
2. **HashLog=14 仍是当前最优折中点**：HL=13 的压缩率灾难性退化结论仍然成立，因此不能作为当前推荐配置。  
3. **当前 corrected baseline 已经明确证明 GPU 是主导吞吐引擎**：压缩 total 2.14x 于 CPU，解压 total 1.44x 于 CPU。  
4. **功率结论已经被修正**：GPU 并不是靠更高功耗换取性能，而是在略低于 CPU 的 active power 下提供更高 steady-state total throughput。  
5. **LZ4 GPU 现在应作为整个项目的正式主路径之一来写**：不再是实验性附属分支，也不应再被旧 total semantics 的结论压制。

### 8.2 展望

后续真正值得继续做的，不再是简单重复旧参数扫描，而是：

1. 继续压缩 host/runtime 开销，让 total throughput 更接近 kernel throughput；
2. 在 matched-corpus 条件下与 LZO GPU 做更严格的 family-to-family 对比；
3. 继续把 GPU steady-state 基线作为 hybrid 调度设计的上限参考；
4. 继续保留 `FORCE_OPENCL_DEVICE` 作为验证入口，但当前 Intel 平台上的 **CPU OpenCL 结果应定位为功能可用的 portability check，而不是默认推荐路径**——fresh subset bench 表明它在部分 case 上能接近甚至短暂超过 GPU OpenCL，但没有稳定优于 native pthread CPU path 的系统级证据。

简言之：

> 当前 `lz4_gpu` 已经是一条经过架构重构、实现收敛、方法学校正和全量基线验证后的成熟结果路径。

## 9. 2026-03-09 定向优化快照

本轮针对用户提出的几个具体问题做了定向修复与 full-corpus 回补：

- **LZ4 GPU 64KB 路径的真实实现问题**：修复了 `tableType==0` 时 kernel / host 侧 dict mask 与 dict buffer sizing 不匹配的问题；
- **LZ4 hybrid 的 bench correctness 问题**：修复了 GPU dict buffer 未清零导致的 in-process repeated bench 校验失败，以及 `--bench-io` 使用固定 `/tmp` 文件名带来的冲突；
- **LZ4 hybrid host/runtime 路径**：引入 GPU workspace 复用、CU-aware worker sizing、mapped buffer readback、distributed sampling，以及 CPU 子路径 `LZ4_compress_fast()` 对齐 acceleration 语义；
- **LZ4 GPU CLI/bench 路径可移植性**：修复 `--help` 正常返回、`--use-daemon --bench` 真实生效，以及 Linux 非 Windows 构建显式使用 `-pthread` 的兼容性问题。

### 9.1 当前 subset 结果（优化后，1s warmed bench）

#### `dickens`

| Engine | Block | Comp total MB/s | Dec total MB/s | Ratio % |
|---|---|---:|---:|---:|
| GPU | 16K | 369.92 | 816.66 | 67.51 |
| GPU | 32K | 389.75 | 741.50 | 64.59 |
| GPU | 64K | **409.40** | 793.66 | **62.35** |

#### `industrial_parent_0_pages_img.tar`

| Engine | Block | Comp total MB/s | Dec total MB/s | Ratio % |
|---|---|---:|---:|---:|
| GPU | 16K | 545.57 | 1162.99 | 19.98 |
| GPU | 64K | **597.88** | **1246.33** | **18.85** |

### 9.2 当前解释

当前 subset 结果说明：

1. **修复后已经看不到“LZ4 GPU 在 64KB 下必然严重崩塌”的普遍现象**；
2. 在 `dickens` 和 `industrial_parent_0_pages_img.tar` 上，64KB 的 total throughput 与 ratio 都优于 16KB；
3. 因而当前更准确的结论是：此前观察到的 64KB 异常下降，至少有一部分来自 dict sizing/mask bug 与 hybrid runtime / bench artifact，而不是 LZ4 GPU 结构上必然不适合 64KB。

### 9.3 仍需继续观察的点

- `LZ4_FORCE_TABLETYPE=0/1` 的 64KB A/B 仅带来小幅差异（industrial subset 下约 609.70 → 619.95 MB/s），说明当前剩余的 64KB 行为主要还是 host/runtime 与 workload interaction 问题；
- 后续若继续扩大到 matched-corpus rerun，应重点观察 page-image / migration-image 类 workload。当前 full-corpus stitched artifact 已经完成，并应优先作为正式引用结果。
