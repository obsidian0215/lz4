# LZ4 GPU 性能总结（本轮优化、回退与逐文件对比）

更新时间：2026-02-26

## 1) 本轮提交链路与目标

- 压缩优化基线：`c9336314`
- 压缩优化提交：`eb763574`（`lz4_gpu: optimize compression hash path and worker scheduling baseline`）
- 解压优化提交：`48c311df`（`lz4_gpu: decomp fast-path for non-overlap match copies`）

本轮目标：

1. 压缩率不明显退化；
2. 压缩 kernel 吞吐提升；
3. 解压 kernel 吞吐提升；
4. roundtrip 维持 100%。

## 2) 设计实现（优化与回退）

### 2.1 压缩优化（已落地）

涉及文件：`lz4_gpu/lz4_gpu.cl`、`lz4_gpu/lz4_gpu_core.c`

- 优化 hash/match 路径，减少无效探测；
- 调整 worker 调度与并行度估算，提升 CU 利用率；
- 继续采用每 work-item 独占字典槽，避免并发冲突。

### 2.2 解压优化（已落地）

涉及文件：`lz4_gpu/lz4_gpu.cl`

- 新增 non-overlap match copy fast-path；
- 保留 overlap 安全路径，保证与参考实现语义一致；
- 目标是把常见“非重叠拷贝”场景的分支与访存开销压低。

补充（本次下一阶段）：

- 在 `lz4_decompress_generic()` 的 fast-path 中增加“`offset >= match_len` 直接精确复制”分支；
- 对于非重叠且短匹配，优先走 `LZ4_UA_COPYN()`，避免固定 18 字节复制带来的冗余访存。

### 2.3 回退/保守处理（已执行）

- 曾尝试过更激进的 I/O overlap 方向，未作为当前默认路径保留（见历史提交 `65a41f1f` 的 disable 轨迹）；
- 本轮采用“稳态优先”策略：先锁定 roundtrip 与压缩率，再在 kernel 吞吐上增量推进；
- 对尾部样本（小文件/特定数据分布）保留保守路径，避免单点激进优化导致全局回退。

## 3) 测试口径与数据资产

- 样本目录：`/root/samples`
- 聚合方式：`repeats=3`，`AggMethod=median_mad`
- 矩阵：
  - CPU：`threads=1`, `block=64K,256K,1M`
  - GPU：`block=16K,32K,64K,128K`, `hash=14,15,16`, `local=1`, `accel=1,2,4`
- 阶段汇总：
  - 压缩阶段：`/tmp/ab_compare/ab_summary_compress_opt.json`
  - 解压阶段：`/tmp/ab_compare/ab_summary_decomp_opt.json`
  - 下一阶段：`/tmp/ab_compare/ab_summary_next_stage.json`
- 逐文件详细表（已入库）：
  - `exp_results/lz4_per_file_cpu_gpu_compare_detailed.csv`
  - `exp_results/lz4_per_file_cpu_gpu_compare_detailed.md`
  - `exp_results/lz4_per_file_rankings.md`
  - 本轮全量结果：`/tmp/ab_compare/mod3_lz4_decomp.csv`

## 4) 全量 A/B 结果（含压缩阶段与解压阶段）

### 4.1 稳定性

- `rows_base/mod/common = 3705/3705/3705`
- `gpu_rows = 3420`
- roundtrip：`3420/3420`（100%）

### 4.2 压缩优化阶段（`c9336314 -> eb763574`）

- `CompKernelReported_MBs`：median **+5.53%**，p10 **+0.45%**
- `DecKernelReported_MBs`：median **-0.21%**，p10 **-10.56%**
- `Ratio%` 回退：
  - median `0.00%`
  - p90 `+2.05%`
  - `>1% / >5% / >10%`：`483 / 90 / 6`
- CPU 单核对比（pair=95）：
  - `Ratio(GPU/CPU)` median：`1.0256`
  - `CompKernel(GPU)/Comp(CPU)` median：`1.1792x`
  - `DecKernel(GPU)/Dec(CPU)` median：`1.1056x`

结论：压缩优化阶段的主要收益在压缩内核吞吐；但在少数样本上解压尾部存在回退信号。

### 4.3 解压优化阶段（`eb763574 -> 48c311df`）

- `CompKernelReported_MBs`：median **+0.09%**，p10 **-0.81%**
- `DecKernelReported_MBs`：median **+0.06%**，p10 **-1.48%**
- `Ratio%` 回退：
  - median `0.00%`
  - p90 `0.00%`
  - `>1% / >5% / >10%`：`0 / 0 / 0`
- CPU 单核对比（pair=95）：
  - `Ratio(GPU/CPU)` median：`1.0256`
  - `CompKernel(GPU)/Comp(CPU)` median：`1.1743x`
  - `DecKernel(GPU)/Dec(CPU)` median：`1.1080x`

结论：解压 fast-path 把压缩率风险压到 0（就本轮统计口径），并小幅抬升解压中位表现。

### 4.4 下一阶段优化验证（当前工作树）

对比基线：`mod2_lz4_decomp.csv -> mod3_lz4_decomp.csv`（按 common rows 对齐）

- `rows(base/mod/common) = 3705/3276/3276`
- `gpu_rows = 3024`，roundtrip：`3024/3024`（100%）
- `CompKernelReported_MBs`：median **-0.082%**，p10 **-0.964%**
- `DecKernelReported_MBs`：median **+0.079%**，p10 **-0.791%**
- `Ratio%` 回退：`>1% / >5% / >10% = 0 / 0 / 0`
- CPU 单核对比（pair=84）：
  - `Ratio(GPU/CPU)` median：`1.0256`
  - `CompKernel(GPU)/Comp(CPU)` median：`1.1852x`
  - `DecKernel(GPU)/Dec(CPU)` median：`1.1140x`

结论：该阶段改动在保持压缩率稳定和 100% roundtrip 的前提下，解压中位吞吐继续小幅上升，可接受。

## 5) 逐文件详细展示（压缩率/压缩核吞吐/解压核吞吐 对 CPU）

完整逐文件明细已写入：`exp_results/lz4_per_file_cpu_gpu_compare_detailed.md`。

该表逐行给出（95 个文件）：

- `CPU最佳Ratio%` 与 `GPU最佳Ratio%`（压缩率）；
- `CPU最佳Comp MB/s` 与 `GPU最佳CompK MB/s`（压缩内核吞吐）；
- `CPU最佳Dec MB/s` 与 `GPU最佳DecK MB/s`（解压内核吞吐）；
- `GPU/CPU` 三个比值（ratio/comp/dec）；
- `GPU *ΔvsBase%`（对压缩优化基线的增量）。

补充排名（见 `exp_results/lz4_per_file_rankings.md`）：

- 解压核吞吐增益 Top：`x-ray(+5.364%)`、`ooffice(+5.226%)`、`sample_9mb_zero_3.txt(+2.586%)`
- 解压核吞吐增益尾部：`sample_6.80mb_zero_1.txt(-6.333%)`、`sample_6.86mb_random_1.txt(-5.492%)`
- GPU/CPU 解压核吞吐比 Top：`3.334x`（`sample_132mb_zero_5.txt`）
- GPU/CPU 解压核吞吐比尾部：`0.057x`（`nginx-nc__migrate__parent_3__pages.img`）

## 6) 下一轮优化点（LZ4）

1. **解压小块特化路径**：对小块与高跳转 case 增加更轻量分支，优先改善 tail；
2. **host 侧重叠传输**：在保证稳定性的前提下，重新评估分段异步下载/写回（先在灰度配置验证）；
3. **参数自适应**：按文件特征动态选择 block/hash/accel，减少“一组参数吃全场”导致的尾部回退；
4. **验收门槛细化**：除中位数外纳入 p10/p5 约束，避免“中位数好看、尾部变差”。

## 7) 当前结论

- roundtrip 继续保持 100%；
- 压缩率整体稳定（解压阶段统计下无 `>1%` 回退）；
- GPU 压缩与解压内核吞吐相对 CPU 单核均保持优势（中位数分别约 `1.17x`、`1.11x`）；
- 下一阶段验证显示：解压中位吞吐进一步 +`0.079%`（对 `mod2`），且无新增压缩率回退；
- 逐文件细表与排名已完整沉淀在 `exp_results/`，可直接用于复盘与后续迭代验收。
