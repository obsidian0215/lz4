# LZ4 GPU 性能分析报告（设计实现与结果）

更新时间：2026-02-24

## 1) 当前实现设计

### 1.1 主机侧调度与资源管理（`lz4_gpu_core.c`）

- local/global size 通过 `sanitize_local_size()` 与 `round_up_size()` 自动收敛到设备可执行配置。
- worker 数通过 `choose_worker_count()` 按 `CU × wi_per_cu` 估算，并支持 `LZ4_GPU_WI_PER_CU` 覆盖。
- 字典池按 worker 数分配（而非按 block 数），与内核调度模型一一对应。
- 输入输出缓冲使用 `CL_MEM_ALLOC_HOST_PTR + map/unmap`，保持低开销传输。

### 1.2 压缩内核实现（`lz4_gpu.cl`）

- 压缩 kernel 采用 work-item 网格步进：每个 work-item 负责多个 block。
- 每个 work-item 使用独占字典槽，字典清零由 `dict_clear_single()` 完成。
- 匹配与拷贝路径保留向量化（`vload/vstore`）与小偏移特化分支。

### 1.3 解压内核实现

- 解压仍是 block 级并行解码模型；
- 本轮主要收益集中在压缩侧，解压算法核心逻辑基本保持稳定。

## 2) 测试口径与数据

- 样本目录：`/root/samples`
- 聚合方式：`repeats=3`，`AggMethod=median_mad`
- 矩阵：
   - CPU：`threads=1`, `block=64K,256K,1M`
   - GPU：`block=16K,32K,64K,128K`, `hash=14,15,16`, `local=1`, `accel=1,2,4`
- 数据文件：
   - baseline：`/tmp/ab_compare/base_lz4.csv`
   - modified：`/tmp/ab_compare/mod_lz4_fix.csv`
   - 汇总：`/tmp/ab_compare/ab_summary_submit_check.json`

## 3) 稳定性与性能结果

### 3.1 稳定性

- 行数：`3705 vs 3705`，可比 `3705`
- GPU case：`3420`
- roundtrip：`3420/3420`（100%）

### 3.2 baseline 对比（GPU）

- `CompKernelReported_MBs`：
   - 中位数：**+34.11%**
   - p10：**+8.66%**
- `DecKernelReported_MBs`：
   - 中位数：**-0.07%**
   - p10：**-1.35%**
- `Ratio%` 变化：
   - 中位数：`0.00%`
   - p90：`0.00%`
   - `>1% / >5% / >10%` 回退计数：`0 / 0 / 0`

结论：当前 LZ4 优化效果主要体现在压缩吞吐，解压基本持平，这与实现范围一致。

### 3.3 与 CPU 单核对比（modified）

按 `File` 聚合后对比（pairs=`95`）：

- `Ratio%(GPU/CPU)` 中位数：`0.9968`
- `CompKernel(GPU)/Comp(CPU)` 中位数：`1.0679x`
- `DecKernel(GPU)/Dec(CPU)` 中位数：`1.1113x`

满足“压缩率基本不变，且压缩吞吐至少优于 CPU 单核（中位数）”的提交门槛。

## 4) 瓶颈分析（压缩/解压）

基于 `mod_lz4_fix.csv` 的 kernel 与端到端吞吐差距：

- `CompKernel/CompOverall` 中位数：`1.6698`
   - 对应主机侧开销占比中位：`40.1%`
- `DecKernel/DecOverall` 中位数：`5.1342`
   - 对应主机侧开销占比中位：`80.5%`

说明：

- 压缩端已有内核收益，但仍存在明显 host-side 开销；
- 解压端瓶颈更偏向主机侧链路（传输/调度/写回），不是纯 kernel 算力不足。

## 5) 后续优化与测试计划

### 5.1 压缩端（LZ4）

1. **哈希字典**：评估字典大小/布局/清零成本与 occupancy 的平衡；
2. **匹配查找**：按数据类型分层优化 early-exit 与查找步进；
3. **并行流水线**：扫描 `local_size × wi_per_cu × block_size`，压低 tail case。

### 5.2 解压端（LZ4）

1. 优化 host 下载与输出写回重叠；
2. 评估分块聚合策略，降低小块调度开销；
3. 对比 map/unmap 与 write/read 路径在不同文件规模下的切换阈值。

### 5.3 统一回归标准（LZ4/LZO）

- roundtrip：100%
- 压缩率：相对 CPU 中位保持在 `1.0 ± 5%`
- 压缩吞吐：GPU kernel 中位 > CPU 单核
- 解压吞吐：持续提升，并逐步缩小 `Kernel/Overall` 差距
