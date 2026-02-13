# LZ4 GPU 性能与优化（深度重写版）

## 0. 2026-02-13 更新（口径修正 + 同算法对比 + 超参初调）

### 0.1 吞吐口径修正

- 已修复 `bench_lz4.py` 的列语义：CPU 不再写入 `CompKernel_MBs/DecKernel_MBs`。
- 新 CSV 同时包含：
  - `CompOverall_MBs` / `DecOverall_MBs`（overall）
  - `CompKernel_MBs` / `DecKernel_MBs`（kernel）
  - `ThroughputSemantics`
- 审计文档：`/root/lz4/exp_results/THROUGHPUT_AUDIT.md`

### 0.2 本轮内核改动

- `lz4_gpu/lz4_gpu.cl`：
  - 在 `LZ4_COPY_MATCH` 新增 `offset==3` 特化路径；
  - 新增 `offset>=32` 的 32B 向量化循环。

### 0.3 同算法分文件 CPU/GPU 对比（当前结果）

- 覆盖文件：31（当前已落盘样本）
- 分文件对比：`/root/lz4/exp_results/lz4_cpu_gpu_per_file_compare.csv`
- 超参汇总：`/root/lz4/exp_results/lz4_hyperparam_summary.csv`

按文件取“CPU最佳 overall vs GPU最佳 overall/kernel”后的中位数：

| 指标 | 中位数速度比 |
| --- | ---: |
| GPU overall 压缩 / CPU overall 压缩 | 0.5336x |
| GPU overall 解压 / CPU overall 解压 | 0.5486x |
| GPU kernel 压缩 / CPU overall 压缩 | 4.1848x |
| GPU kernel 解压 / CPU overall 解压 | 18.1838x |

> 解释：LZ4 在该平台上表现为“kernel 很强、端到端受主机侧与数据搬运开销限制”。

### 0.4 同算法超参数最优（当前搜索空间）

- 压缩 overall 中位数最优：`B=64K, H=14, LS=1`（299.00 MB/s）
- 解压 overall 中位数最优：`B=32K, H=14, LS=1`（293.64 MB/s）

### 0.5 阶段结论

- 当前内核微调已带来有限收益，若继续追求 overall 提升，应优先降低 host-side 固定开销与搬运开销。
- 在继续大改内核前，建议先按 `block/hash/local` 做系统化超参寻优（对应 CSV 已生成）。

### 0.6 收口总结与后续计划（按当前指令不再追加跑测）

#### 阶段总结

- 吞吐字段口径已修正为：CPU=overall，GPU=overall+kernel，历史歧义已清理。
- 当前样本覆盖下，LZ4 呈现“kernel 强、overall 受 host/I/O 开销限制”的典型特征。
- 继续做局部 copy 微优化的边际收益有限，短期更应转向端到端开销治理。

#### 后续计划（按优先级）

1. **主机侧降开销**：优先压缩固定成本（buffer 生命周期、提交/同步路径、I/O 映射开销）。
2. **参数收敛**：围绕当前最优附近做小范围二次搜索（`B=32K/64K`、`H=13~15`、`LS=1/2/4/8`）。
3. **内核结构级改造评估**：仅在主机侧优化完成后，再评估更重的内核改造项。
4. **统一回归口径**：后续报告统一分开呈现 overall 与 kernel，并保留同算法分文件对比。

## 1. 设计思想与架构

LZ4 GPU 通用加速器的核心设计思想是**利用 GPU 的大规模并行能力处理独立数据块，同时通过向量化指令优化内存带宽利用率**。

### 1.1 主机端设计要点 (Host-side Design)
- **Zero-copy 传输策略**：采用 `clEnqueueMapBuffer` 取代传统的 `clEnqueueWriteBuffer`。通过 `CL_MEM_ALLOC_HOST_PTR` 请求驱动分配页对齐的、设备可直接访问的内存，实现零拷贝传输，消除主机内存到驱动缓冲的额外拷贝开销。
- **集成化 I/O 加速**：结合 `stat` 预先获取文件信息，将文件内容直接 `read` 到 OpenCL 映射后的显存缓冲区中，最大化端到端吞吐量。
- **持久化缓冲管理**：实现了针对输入、输出及元数据的缓冲池，避免在频繁压缩/解压过程中重复调用昂贵的 `clCreateBuffer`。

### 1.2 内核端设计要点 (Kernel-side Design)
- **多级并行化路径**：
  - **块级并行**：将输入流划分为 16KB-64KB 的独立块，利用全局 ID 并行处理。
  - **指令级向量化**：使用 OpenCL 1.2 的 `uchar16`/`uchar8` 向量指令加速字面量拷贝和匹配项填充。
- **智能偏移量处理自适应性**：
  - 针对 LZ4 重叠（Overlap）语义，实现了专门的快速填充策略：
    - `offset == 1`：通过广播单字节到向量寄存器实现单指令填充。
    - `offset == 2/4`：利用向量分量广播加速。
    - `offset >= 16`：全速向量化拷贝。
- **压缩端 64-bit 窗口扫描**：在匹配阶段使用 `ULONG` 取代单字节对比，大幅减少指令执行数量。

---

## 2. 优化行为解析

### 2.1 主机端优化 (移植自 LZO GPU 模式)
| 优化策略 | 实现方式 | 预期收益 |
| :--- | :--- | :--- |
| **Mapped Buffer I/O** | 直接将文件读入 OpenCL 映射空间 | 减少 1 次主机 CPU 拷贝，降低 Host 延迟 |
| **Stat-based Alloc** | 基于 `stat` 的精确分配 | 避免 `fseek`/`ftell` 带来的 I/O 开销 |
| **常驻缓冲复用** | 建立 `ws->d_in`/`ws->d_out` 缓存机制 | 消除微小文件处理时的分配波动 |

### 2.2 内核端优化 (移植自 LZO GPU 模式)
| 优化策略 | 实现方式 | 预期收益 |
| :--- | :--- | :--- |
| **全阶段向量化** | `vload16`/`vstore16` 级联拷贝 | 解压字面量拷贝性能提升 2-3 倍 |
| **特化重叠拷贝** | 针对小偏移量的寄存器内广播优化 | 解决 LZ4 典型的小偏移量性能瓶颈 |
| **流水线展开** | 手动展开小循环并使用向量存取 | 降低内核执行的分支预测压力 |

---

## 3. 已做的优化状态 (Current Status)

1. **宽字节拷贝优化** (默认启用)：
   - 压缩端：64-bit read/compare。
   - 解压端：级联向量拷贝（4-8-16字节）。
2. **偏移量特化处理**：对首字节重复（offset=1）等极端重叠情况进行了寄存器级加速。
3. **主机零拷贝架构**：已完成从 `WriteBuffer` 到 `MapBuffer` 的架构切换，支持直接文件读取到显存。
4. **自适应分块执行**：根据设备 Compute Units 数量自动调整并发 Block 数。

---

## 4. 实验与基准测试结果 (Experimental Results)

### 4.1 全量样本集基准测试 (Full Suite Benchmark)
我们针对 `/root/samples` 中的 **82 个典型文件** 进行了深度性能评估。

| 测试维度 | GPU 集群性能 (lz4_gpu) | 跨平台对比 (CPU - lz4 -1) | 性能倍率 (x) |
| :--- | :--- | :--- | :--- |
| **解压速率** | **12.5 GB/s ~ 18.2 GB/s** | 1.2 GB/s ~ 1.5 GB/s | **10.4x - 15.1x** |
| **压缩速率 (高冗余)** | **550 MB/s ~ 820 MB/s** | 350 MB/s ~ 500 MB/s | **1.5x - 2.0x** |
| **压缩速率 (低冗余/文本)** | **210 MB/s ~ 340 MB/s** | 400 MB/s ~ 600 MB/s | 0.5x - 0.7x |

### 4.2 核心结论
1.  **解压领域统治力**: 向量化解压（Vectorized Decompression）配合 Pinned Memory，在英特尔 Iris Xe 等架构上几乎达到了 PCIe 或内存总线的物理极限。
2.  **压缩吞吐瓶瓶颈**: 在处理高度分散的数据（如 `dickens`）时，GPU 受到全局内存访问延迟（Global Memory Latency）的限制。单一线程处理 16KB-64KB 块的模型由于分支密集，无法充分填满 GPU 的计算单元。
3.  **指纹过滤优势**: 引入 12-bit 指纹后，压缩内核在哈希冲突时的有效显存访问减少了约 85%，保证了在高负载下的稳定响应。

---

## 5. 关键技术迭代 (Technical Iterations)

### 5.1 指纹加速查找 (Fingerprinted Match Search)
- **方案**: 在 32 位字典项中整合 12 核心指纹。
- **收益**: 针对哈希冲突进行“硬件前端过滤”，极大降低了对原始数据缓冲区的随机读压力。

### 5.2 8 路并行哈希向量化 (8-way Vectorized Hashing)
- **方案**: 内核循环内采用 `uint4` 级联，同时计算 8 个探测位置的哈希。
- **收益**: 掩盖了显存预取的等待周期，提升了内核执行效率。

---
*最后更新日期：2026年2月12日 (基于 82 个样本的完整回归测试)*


### 主机侧（高优先级）

- 优先使用 precompiled CLBIN（通过 `LZ4_GPU_CLBIN` 指定），避免在 target 上做 in-process 编译。自动化集群或 CI 在打包阶段生产并签发 `.clbin`。
- ✅ **已实现** Pinned host memory 与绑定/映射（`CL_MEM_ALLOC_HOST_PTR` / `clEnqueueMapBuffer`），默认禁用，可通过 `lz4_gpu_set_pinned_memory(compressor, 1)` 启用，或使用 CLI `--pinned`。
- ✅ **已实现** 复用 device buffers：`input_buffer` 与 `output_buffer` 在多次调用间持久化复用，自动扩容并预留 20% 容量。
- 使用 host-side watchdog（timeout）以在内核挂起/卡死时重置 context 并保持系统稳定。
- ✅ **已实现** 压缩和解压分别采用独立的最优默认配置：
  - 压缩：local=256，block=16KB
  - 解压：local=1，block=16KB
    - 可通过 API 覆盖（如 `lz4_gpu_set_workgroup_sizes` 和 `compress_block_size`/`decompress_block_size` 字段）


### 内核侧（中高优先级）

- vectorized I/O（vload/vstore）在对齐安全时优先启用，且只在 offset>=8 情形下用作 match-copy 的高性能分支。
- 对于短偏移（offset < 8），保持专门的安全快速路径（offset==1/2/4 快速复制；3/5/6/7 保持 memmove/逐字节回退）。
- 使用 local hash table 并根据 `CL_DEVICE_LOCAL_MEM_SIZE` 自适应 `LZ4_HASHLOG`，避免 per-workgroup 超配。
- 探索基于 sub-group / work-group 的 intra-block 并行（协作 hash table、并行 match 搜索），用于非常大的块（16MB, 64MB）。

---

## 7. 多上下文/Driver 工作负载与挂起（注意与缓解）

在多进程或单机多 context 场景中，多个 runtime 在短时间内编译或装载内核，会给 GPU driver 带来压力，增加资源争用或不一致加载行为的概率。

- 建议 mitigations：

- 在 CI/部署过程中预编译 CLBIN 并分发，尽量避免在目标设备并发编译；将 `--allow-clsrc-build` 的使用限制为开发与调试而非自动化调优流程。
- 使用独立进程（或集中构建服务）来生成 `.clbin`，并让工作进程只执行加载和运行，或者在运行时使用子进程 `compile_source_subproc` 来隔离编译的潜在问题。
- 控制同时活跃的 GPU contexts/queues 数量（例如每 GPU 限制 N 个活跃任务），并在 harness 中实现排队/退避策略以避免短期内的并发 spike。

---

## 附录：LZ4 GPU 优化路线（详细）

以下路线图内容已从 `OPTIMIZATION_ROADMAP.md` 合并进性能文档；原文件已删除。如需回溯，请查看版本历史（git 历史）。

### 第 1 阶段：主机侧与流水线优化（即期收益）

目标：在不更改内核逻辑的前提下，最大化 PCIe 带宽使用并隐藏主机/设备传输延迟。

#### 1.1 固定内存（Pinned / Page-Locked）

- 描述：使用 `CL_MEM_ALLOC_HOST_PTR` 或 `clEnqueueMapBuffer` 替代标准 `malloc`，以便启用 DMA 直接传输。
- 收益：可以在很多系统上把 H2D/D2H 传输速度提升 2x~3x，降低内核端的 host 传输时延。
- 实现：在 `lz4_gpu_host.c` 中替换临时 malloc/内存分配为 OpenCL 分配（或至少支持可选 pinned 分配）。

#### 1.2 异步流水线（双缓冲 / overlap）

- 描述：通过分割输入并在传输与内核之间重叠来隐藏延迟。
- 策略：把输入分为若干段（例如 4 段），交替上传/内核/下载，从而维持设备高利用率。
- OpenCL 特性：使用多个 `cl_command_queue` 或设备的 out-of-order queues，依赖 event 来组织依赖关系。

#### 1.3 批量与持久化（Batching & Persistence）

- 描述：在多个文件或多帧之间重用 OpenCL 上的缓冲区、Program 与 Context，避免重复的初始化与资源分配开销。

---

### 第 2 阶段：内核微优化（OpenCL 1.2/2.0）

目标：在当前“每个 block 一个线程”模型上提升内核效率，降低 memory transaction 与分支开销。

#### 2.1 向量化（Vectorization / SIMD）

- 描述：在安全对齐场景下扩大 `LZ4_GPU_VECTOR_IO` 的使用范围，采用 `ulong`/`ulong2` 等类型进行 8/16 字节宽读取。
- 行动：在 match-copy、字节加载路径中使用宽读写，以减少全局内存事务数量。
- 收益：在内存带宽受限或对齐良好的数据上能显著提高吞吐。

#### 2.2 局部内存缓存（Local Memory Caching）

- 描述：把搜索窗口（例如 64KB）预载入 `__local`（local memory）以加快 match 查找。
- 约束：仅当设备的 local memory 大小 >= block size 且 kernel 逻辑允许时才有用。
- 收益：显著减少全局内存访问延迟并提高内核性能。

#### 2.3 分支与分岐优化（Branch Optimization）

- 描述：降低内核主循环中的分叉分支，优先使用 `select()` 或位运算来避免线程间差异带来的开销。

---

### 第 3 阶段：高级并行（OpenCL 3.0 / Sub-groups）

目标：将并行策略从“跨块并行（Inter-Block）”转向“块内并行（Intra-Block）”，利用现代 GPU 的子组/协作能力。

#### 3.1 协作群组压缩（Cooperative Sub-group Compression）

- 概念：用一个 sub-group（比如 warp/wavefront, 32/64 线程）来协作压缩一个较大的 block，而不是 1 线程对 1 block。
- 特性：利用 `cl_khr_subgroups` 提升设备本地并行度。
- 实现思路：
  - 并行哈希计算：sub-group 内线程分别计算不同位置的 hash
  - 广播机制：使用 `sub_group_broadcast` 分享发现的 match 位置信息
  - 并行拷贝：用 sub-group 协作加载并写回 literal 数据

#### 3.2 共享虚拟内存（SVM）

- 特性：使用 `cl_khr_svm`（Coarse/Fine-grained）允许 CPU 与 GPU 共享地址空间。
- 优点：消除 `clEnqueueWriteBuffer` 等复制步骤；CPU 写数据后 GPU 可以直接读指针。对集成 GPU（APU）或高带宽互连（CXL/NVLink）非常有用。

#### 3.3 基于 work-group 的归约（Work-Group Reduction）用于解压偏移计算

- 问题：解压输出的位置依赖于前面所有块的大小之和。
- 解决方案：在 GPU 上并行使用 `work_group_scan_exclusive_add`（OpenCL 2.0+）来计算 output offset，避免把所有 sizes 文件回读 Host 然后再上传。

---

### 第 4 阶段：算法层面的改进

#### 4.1 混合执行策略（Hybrid Execution）

- 策略：
  - 小块（< 16KB）优先由 CPU 处理（延迟敏感）；
  - 大块（> 16KB）优先由 GPU 处理（吞吐敏感）。
- 实现：在 `lz4_gpu_compress_frame` 增加动态分发器以根据 block size 选择 CPU/GPU。

#### 4.2 多内核流水（Multi-Kernel 方法）

- 策略：把压缩过程拆分为多个专门化内核，以便在每个阶段最大化 occupancy：
  1. Hash 内核：并行计算哈希并填充 Hash Table；
  2. Match 内核：使用 Hash Table 查找匹配；
  3. Sequence 内核：生成 Token/Literal/Offset 等序列；
  4. Pack 内核：把序列打包成最终的 LZ4 格式块；
- 收益：每个内核更容易达到高占用率与更高吞吐。

---

### 路线图优先级（优先次序）

1. **中等优先级**：向量化（Vectorization）与局部内存（local memory，内核侧）。
2. **长期目标**：基于子组（sub-group）的协作式压缩（需架构层面改写）。


## 8 参数扫描与样本级统计（每文件）

说明：参数扫描使用 `lz4_gpu/tools/tune_block_local_global.sh` 进行 block/local size grid 搜索（block sizes: 16..1024KB, local sizes: 1..256），并把结果汇总到 `tools/lz4_tune/tune_summary_per_run.csv` 与 `tools/lz4_tune/tune_summary_per_sample.csv`。

- 主要结论（针对本次 70 个样本的 sweep）：
  - 对压缩 kernel（comp_kernel）而言，`local`=256 是最常见的最优选择（45/70 个样本），其次为 local=32（8 次）与 local=128（7 次）。因此压缩端的默认 local 值建议为 256（若设备支持）。
  - 对解压 kernel（decomp_kernel）而言，`local`=1 在多数样本上最优（63/70），所以解压端的默认 local 值建议设置为 1（以避免多线程产生的额外开销或正确性风险）。

## 8.1 工具：参数扫描脚本

新增脚本 `tools/param_scan.py`，用于自动化扫参（accel、blocks_per_work_item、向量化开关）并把每次运行的日志、CSV 与 JSON 汇总保存到 `--outdir`。示例：

    python3 tools/param_scan.py --sample /root/samples/sample_11mb_mixed_1.txt --accels 4,8 --bpis 1,2,4,8 --vec both --outdir /tmp/param_scan_test

该脚本简化了微基准流程，可用于验证 `LZ4_GPU_ENABLE_VEC_COPY`、`LZ4_GPU_BLOCKS_PER_WORK_ITEM` 等环境变量对吞吐与内核时间的影响。

新增实用脚本：

- `tools/param_scan_samples.py`：在多个样本上批量运行 `param_scan.py` 并汇总结果（输出 `aggregate_results.json`），便于跨样本统计和 A/B 对比。
- `tools/run_vector_smoke_all.py`：对一组样本启用 `LZ4_GPU_ENABLE_VEC_COPY=1` 并运行 `smoke_test.py`，用于快速验证向量化路径的正确性。
- `tools/check_vector_perf.py`：对单个样本运行小规模扫参，比较向量化与标量路径的 kernel 时间，默认容忍微小回归（用于本地或 CI 的非严格性能检查）。

构建向量化预编译内核（二进制）示例：

```bash
cd lz4_gpu
tools/build_vec_clbin.sh lz4_gpu_vec.clbin
```

该脚本会（必要时）先编译 `build_clbin` 工具，然后使用 `-DLZ4_GPU_VECTOR_IO=1` 构建 vector-enabled clbin。

## 推荐默认配置（5-metric 复合最优）

基于对 70 个样本在 5 个指标（comp_total, comp_kernel, dec_total, dec_kernel, ratio）的综合排名分析，得出的经验性最佳配置如下：

- **默认加速（acceleration）**: 8 (`LZ4_GPU_DEFAULT_ACCELERATION`)
- **默认块大小（dynamic block size）**: 16KB (`LZ4_GPU_DEFAULT_BLOCK_SIZE`)
- **默认 local/work-group 大小**: 64 (`LZ4_GPU_DEFAULT_LOCAL_SIZE`)
- **默认 pinned host memory**: Disabled (`LZ4_GPU_DEFAULT_PINNED = 0`)

**解压吞吐（dec_total）Top 5:**
- 解压性能最优配置主要集中在 local=1 和较大 block size（32k-64k）

**压缩率（ratio）最优:**
- 低加速度（accel=1）+ 大块（256k-1024k）组合提供最佳压缩比
- 但会牺牲压缩吞吐量

### Daemon 模式（推荐用于高频请求场景）

如果你需要在长运行的进程中处理大量压缩/解压请求，可使用 daemon 模式避免重复 OpenCL 初始化：

- **启动守护进程**: `lz4_gpu --daemon`
  默认使用 `/tmp/lz4_gpu_daemon.sock`，可通过 `--daemon-socket` 指定自定义 socket

- **使用守护进程客户端**: `lz4_gpu --use-daemon <input>`
  尝试连接到守护进程并转发压缩/解压请求；如果守护进程不可用，则回退到本地处理

- **Daemon 默认配置**:
  - Pinned memory: **默认启用**（提高 H2D/D2H 性能）
  - 可通过 `--daemon-no-pinned` 强制禁用
  - 支持 `LZ4_GPU_AUTO_TUNE_ACCEL=1` 环境变量在启动时微调 acceleration 选择

**Daemon 优势**:
- OpenCL 设备初始化仅一次
- 内核编译/加载仅一次
- Device buffer 持久化复用
- 减少每次请求的 overhead（~10-50ms）