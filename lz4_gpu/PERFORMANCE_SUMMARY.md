# LZ4 GPU 性能与优化（合并版）

目的与范围：

本文档综合了 lz4_gpu 的性能结果、已完成的优化、以及主机/内核层面的工程化建议（含调优策略与长期路线）。文档以正确性为先，仅列出保守默认实现与经过验证的可选优化（不包含调试过程或临时日志）。

---


## 1 概要（要点）

- 默认内核（`lz4_gpu.cl`）以正确性为首要目标：对于可能导致覆盖语义差异的小偏移（例如 offset == 3/5/6/7）采用保守的逐字节实现。
- **宽字节拷贝优化**（已默认启用）：压缩端使用 64-bit read/compare 减少匹配循环；解压端使用 8字节宽拷贝(`LZ4_wildCopy8`等)减少循环迭代。两者本质相同，通过更大数据块(8字节)减少迭代次数，通常带来 10–30% 性能提升。
- 向量化解压变体（`lz4_gpu_decomp_vec.cl`）作为**可选实验路径**：在 offset >= 8 与数据对齐安全的条件下启用 vload/vstore 向量化指令，在部分样本中 kernel-only 解压时间可降低 ~70–80%，但需要额外验证正确性。


---


<!-- Duplicate section removed (merged into the first "## 1 概要（要点）") -->



<!-- Removed misplaced '主机侧' block (duplicates moved to 推荐的工程部分) -->

## 2 已做的优化（实现与行为要点）

1. **宽字节拷贝优化**（默认启用）：
   - 压缩端：使用 64-bit read/compare 减少匹配搜索循环次数
   - 解压端：使用 8字节宽拷贝 (`LZ4_wildCopy8`, `LZ4_match_wildCopy8`) 减少循环迭代
   - 两者本质相同：通过更大数据块(8字节)减少循环次数，显著提高吞吐，效果在大块数据上最明显（10-30%）
2. 解压端（默认）：保守实现（offset==3/5/6/7 逐字节回退），对 offset==1/2/4 采用专门快速复制。
3. 解压端（**可选**实验变体）：`lz4_gpu_decomp_vec.cl` 使用 vload/vstore 向量化指令来加速 match-copy，作为实验变体保留。**未默认启用**。
4. 主机行为：请通过环境变量 `LZ4_GPU_CLBIN`/`LZ4_GPU_CLSRC` 指定 precompiled `.clbin` 或内核源；优先加载 precompiled `.clbin`，并在需要时使用子进程编译源代码（`compile_source_subproc`）。
5. 工具链：增加 `tools/tune_block_local_global.sh` 等脚本以进行 block/local size sweeps 并收集 CSV 供分析。

  新的测试数据生成脚本：`tools/generate-test-data.py`（替代早期非通用脚本），提供 `--suite` 批量生成多种模式（zero/random/repeat/structured/mixed）和不同大小的样本。该脚本默认输出 `samples/`，可使用 `--out-dir` 指定输出路径以配合现有调优 harness（例如 `lz4_gpu/tools/tune_block_local_global.sh`）。


已创建并使用的内核变体（仓库中存在的 clbins）：

- `lz4_gpu_baseline.clbin` — baseline（32-bit，只用于 LZ4_count）。
- `lz4_gpu_optimized.clbin` — 压缩端启用 64-bit 优化（默认变体）。
- `lz4_gpu_decomp_vec.clbin` — 解压侧向量化变体（可选实验变体）。

---

## 3 基准方法与典型结论

使用脚本和测试程序进行 repeatable A/B 测试：每个样本执行 5 次重复测量，收集 Host->Device 上传、压缩 kernel时间、Device->Host 读 blockSizes、解压 kernel 时间、Device->Host 读 解压数据 等分段指标。主要样本集合位于 `lz4_gpu/samples_big`（包含 256KB、1MB、4MB、16MB、64MB 的混合数据）。

重要输出目录（本次测试机）: `/tmp/ab_compare`（早先压缩基线对比），以及 `/tmp/ab_decomp_compare`（解压变体对比），每个目录包含 per-run CSV、per-block CSV 与 summary CSV。

---

## 4. 实验结果（摘要）

注：下面数值为「内核总耗时（compress+decompress）」或「解压 kernel 平均耗时」，单位均为 ms，取 5 次重复的平均值。

### 4.1 压缩端：常见观察

（来源：早期 AB 比较 /tmp/ab_compare/comparison.csv）

样本 | baseline_total_kernel_ms | optimized_total_kernel_ms | 绝对差 | 相对改善
:---|---:|---:|---:|---:
test_16MB_mixed.dat | 80.8904 | 57.7222 | 23.1682 | 28.64% faster
test_1MB_mixed.dat  | 13.6108 | 11.4620 | 2.1488  | 15.79% faster
test_256KB_mixed.dat| 10.3418 | 9.2278  | 1.1140  | 10.77% faster
test_4MB_mixed.dat  | 26.3370 | 20.3470 | 5.9900  | 22.74% faster
test_64MB_mixed.dat |315.9936 |224.3906 |91.6030 | 28.99% faster

说明：64-bit 匹配路径显著减少了压缩阶段的循环迭代，尤其在大输入上收益最明显（≥~25%）。

### 4.2 解压端：常见观察

在对齐和 offset 条件允许的情况下，向量化（vload/vstore）实现的解压 match-copy 路径把解压 kernel 的运行时间显著缩短（在我们的样本集合上 kernel-only 时间下降通常在 70–80% 量级）。但为了保证输出正确性，默认解压内核保留对可能引发 overlap 语义问题的小偏移量（例如 offset 3/5/6/7）的 scalar/preserved-semantics 路径。

---

## 5 结论（高层）

1. 压缩端的 64-bit 比较路径能在大块数据上显著减少匹配循环次数，带来明显的压缩端加速（常见范围 10–30% 优化，因数据集而异）。
2. 解压端的主性能点是 memory copy 路径：向量化可显著降低解压 kernel 时间，但必须使用受控的回退以保护正确性。
3. 对于自动化调优与长期运行，建议将向量化优化作为可选变体并把 precompiled `.clbin` 纳入常规发布流程，保证运行重复性与稳定性。

---

## 6 推荐的工程与优化要点（优先级与建议）

### 优化实现状态

| 优化项 | 预估收益 | 状态 | 默认启用 | 实测效果 |
|--------|---------|------|---------|----------|
| Pinned Memory | 可能提升传输 | ✅ 已实现 | ✔ 是 | ⚠️ 未观察到显著提升 (< 1%) |
| 持久化 Buffer 复用 | 减少分配开销 | ✅ 已实现 | ✔ 是 | ✅ 验证正常，避免重复分配 |
| Local Memory 缓存 | 降低全局内存延迟 | ✅ 部分 | ✔ 是 | ✓ Hash Table 已在 local memory |
| 异步双缓冲 Pipeline | 隐藏传输延迟 | ⬜ 未开始 | - | 需要多队列架构 |
| Sub-group 协作压缩 | 块内并行加速 | ⬜ 未开始 | - | 需要 OpenCL 2.0+ 子组扩展 |

**注**: Pinned Memory 已实现并默认启用，但在实际测试中未观察到显著性能提升。

### 主机侧（高优先级）

- 优先使用 precompiled CLBIN（通过 `LZ4_GPU_CLBIN` 指定），避免在 target 上做 in-process 编译。自动化集群或 CI 在打包阶段生产并签发 `.clbin`。
- ✅ **已实现** Pinned host memory 与绑定/映射（`CL_MEM_ALLOC_HOST_PTR` / `clEnqueueMapBuffer`），默认禁用，可通过 `lz4_gpu_set_pinned_memory(compressor, 1)` 启用，或使用 CLI `--pinned`。
- ✅ **已实现** 复用 device buffers：`input_buffer` 与 `output_buffer` 在多次调用间持久化复用，自动扩容并预留 20% 容量。
- 使用 host-side watchdog（timeout）以在内核挂起/卡死时重置 context 并保持系统稳定。
- ✅ **已实现** 压缩和解压分别采用独立的最优默认配置：
    - 压缩：local=256，block=32KB
    - 解压：local=1，block=32KB
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
  - 对端到端 round-trip（rt_end）而言，高 local 值（128/256）在多数样本上能覆盖主机传输开销，从而提高 E2E 吞吐；但具体最优值依设备与样本而异，建议通过调优脚本验证。

- 示例（top 10 按 rt_end 改善率排序）：

  sample | baseline_rt_end_MBps | best_rt_end_MBps | best_local | 改善
  ---|---:|---:|---:|---:
  sample_400mb_random_5.txt | 62.49 | 1665.80 | 128 | +2565.92%
  sample_226mb_random_4.txt | 62.49 | 1625.57 | 128 | +2501.30%
  sample_8mb_random_3.txt | 62.14 | 1520.33 | 64 | +1912.03%
  sample_354mb_mixed_5.txt | 64.13 | 1069.48 | 256 | +1567.71%
  sample_5mb_random_2.txt | 61.08 | 845.88 | 32 | +1284.77%
  sample_99mb_structured_5.txt | 63.53 | 877.20 | 256 | +1280.81%
  elasticsearch-ycsb__migrate__parent_1__pages-1.img | 71.29 | 982.74 | 256 | +1278.50%
  sample_46mb_mixed_4.txt | 63.73 | 863.74 | 256 | +1255.30%
  redis-video__migrate__parent_6__pages-1.img | 49.69 | 659.62 | 256 | +1227.51%
  elasticsearch-ycsb__migrate__parent_2__pages-1.img | 74.99 | 992.08 | 256 | +1223.01%

- 示例（压缩 kernel 改善 top 10）：

sample | baseline_comp_kernel_MBps | best_comp_kernel_MBps | best_local | 改善
---|---:|---:|---:|---:
sample_132mb_zero_5.txt | 298.20 | 13548.19 | 256 | +4443.32%
  sample_23mb_zero_4.txt | 294.69 | 13142.86 | 128 | +4359.89%
  sample_400mb_random_5.txt | 63.29 | 2628.19 | 128 | +4052.43%
  sample_226mb_random_4.txt | 63.31 | 2529.80 | 128 | +3896.16%
  sample_8mb_random_3.txt | 62.44 | 1966.09 | 128 | +3048.64%
  sample_9mb_zero_3.txt | 295.12 | 9183.67 | 64 | +3011.84%
  sample_354mb_mixed_5.txt | 65.57 | 1880.50 | 256 | +2768.13%
  sample_151mb_repeat_5.txt | 288.02 | 8182.95 | 256 | +2741.09%
  sample_43mb_structured_4.txt | 65.01 | 1826.13 | 256 | +2708.88%
  sample_42mb_repeat_3.txt | 287.95 | 7917.06 | 256 | +2649.48%

- 示例（解压 kernel 改善 top 10）：

  sample | baseline_decomp_kernel_MBps | best_decomp_kernel_MBps | best_local | 改善
  ---|---:|---:|---:|---:
  nginx-nc__migrate__image__pages-1.img | 8.68 | 10.25 | 64 | +18.11%
  sample_23mb_zero_4.txt | 2658.96 | 3117.80 | 8 | +17.26%
  nginx-nc__migrate__parent_3__pages-1.img | 6.52 | 7.49 | 64 | +14.96%
  nginx-nc__migrate__parent_2__pages-1.img | 6.50 | 7.44 | 32 | +14.38%
  elasticsearch-ycsb__migrate__parent_1__pages-1.img | 3370.87 | 3555.86 | 1 | +5.49%
  elasticsearch-ycsb__migrate__parent_2__pages-1.img | 3389.56 | 3526.07 | 256 | +4.03%
  sample_9mb_zero_3.txt | 2940.22 | 3011.04 | 8 | +2.41%
  nginx-nc__migrate__parent_4__pages-1.img | 6.53 | 6.68 | 64 | +2.31%
  sample_61mb_repeat_4.txt | 3384.38 | 3429.86 | 1 | +1.34%
  sample_151mb_repeat_5.txt | 3618.41 | 3663.63 | 1 | +1.25%

- 推荐使用方式（生成样本并运行调优）：

  1. 使用新的数据生成脚本（生成样本到 `lz4_gpu/samples_big`）：

    ```bash
    cd /root/lz4
    python3 tools/generate-test-data.py --suite --out-dir lz4_gpu/samples_big --per-pattern 5 --min-mb 1 --max-mb 512 --seed 1234
    ```

  1. 使用调优脚本运行 sweep（输出到 /tmp/lz4_tune）：

    ```bash
    cd /root/lz4/lz4_gpu
    # 如果你已经在 /root/samples 放置了样本（用户已创建），直接使用调优脚本（脚本默认会使用 /root/samples）
    ./tools/tune_block_local_global.sh 1 /tmp/lz4_tune /root/lz4/lz4_gpu/lz4_gpu.clbin
    # 或者只对单个样本运行，借助 gpu_roundtrip_test 验证默认 local 值：
    ./gpu_roundtrip_test /root/samples/sample_2mb_structured_2.txt --accel=1
    ```

  1. 汇总/处理 CSV（示例）：

    ```bash
    awk -F, 'NR>1 {print $0}' /tmp/lz4_tune/tune_results.csv > tools/lz4_tune/tune_results_raw.csv
    # (已提供脚本或 Jupyter 笔记本对 CSV 聚合成 summary per-run / per-sample 文件)
    ```

- CSV 位置：
  - `tools/lz4_tune/tune_summary_per_run.csv` — 每次 run 的吞吐/时间/比率
  - `tools/lz4_tune/tune_summary_per_sample.csv` — 每个样本的 baseline vs 最优 combo 与改进率

  完整 per-sample CSV 可在 `lz4_gpu/tools/lz4_tune/tune_summary_per_sample.csv` 中找到（70 行）。示例查看命令：

  ```bash
  # 显示 top-20 的 rt_end 改善样本
  awk -F, 'NR>1 {print $1","$29","$30","$32","$33}' lz4_gpu/tools/lz4_tune/tune_summary_per_sample.csv | sort -t',' -k5 -nr | head -n 20
  ```

  更完整的 per-sample 报表可以通过仓库提供的 `csv2md.py` 脚本导出为 Markdown：

  ```bash
  # 在仓库根目录运行（或按需修改路径）
  python3 lz4_gpu/tools/lz4_tune/csv2md.py \
    --input lz4_gpu/tools/lz4_tune/tune_summary_per_sample.csv \
    --output lz4_gpu/tools/lz4_tune/tune_summary_per_sample.md --top 20
  # 然后将生成的 MD 文件（非常详细）打开查看或直接包含到文档中：
  less lz4_gpu/tools/lz4_tune/tune_summary_per_sample.md
  ```

  输出文件（已生成）： `lz4_gpu/tools/lz4_tune/tune_summary_per_sample.md`，它包含 Top-N 和每样本的 summary（前 50 个样本）。若需完整输出，可修改脚本中 `rows[:50]` 限制。

注：上述示例中的路径（如 `/root/lz4` 或 `/root/samples`）可根据本地环境调整；若调优脚本找不到样本，请用 `tools/generate-test-data.py --out-dir` 指向它。


## 9 默认 Work-Group 推荐（主机端默认 local-size）

说明：根据一次针对 70 个样本的 block/local 扫描，压缩端通常在 local=128/256 时获得显著提升，而解压端的 kernel-only 最优 local 往往为 1。为兼顾吞吐与可移植性，本仓库的 host-side 初始化在 `lz4_gpu_initialize()` 中采用如下启发式选择：

- 若设备 local memory 足够（>= 64KB）且 `CL_DEVICE_MAX_WORK_GROUP_SIZE >= 256`，则默认 `local = 256`；否则退化到 128/64/1 等根据 `max_work_group_size` 的值。 该逻辑可通过 API `lz4_gpu_set_workgroup_size()` 覆盖。
- `local` 默认为 1 在某些设备/场景下更稳妥，尤其对于解压 kernel-only 的性能情况。可通过 API 覆盖该默认值。

示例：要在运行时调整默认值，可以使用：

```c
lz4_gpu_set_workgroup_size(compressor, 256);
```

备注：这些默认值基于我们在多样本上的经验与采样统计；不同设备（不同 GPU vendor/driver）可能会有不同的行为，因此推荐在 CI 或生产集群上复现一次调优并将最优 `clbin` 与默认参数部署到系统中。



