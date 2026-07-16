# 频率/功耗扫描与 adaptive 建模计划

更新时间：2026-05-23

## 目标

先暂停复杂 adaptive 建模。当前阶段只做一件事：把单设备和纯 CPU 路径在频率、block、D_BITS、线程/slot 下的压缩率、压缩/解压吞吐、端到端吞吐、功耗增量量化清楚。只有这些曲面可信后，才讨论 `gpu_ratio ∈ [0,1]` 的动态调度。

## 结果目录

105 正式扫描结果固定在：

```text
/root/compress_freq_power_results/
  lz4/
  lzo/
  summary/final/
    combined_config_summary.csv
    combined_per_file_power.csv
    freq_power_analysis_draft.md
```

本机和其他远端可以用于冒烟或补充验证，但不能混入这个目录的正式 105 单平台结果。

## 口径

- `comp_mbs_*` / `dec_mbs_*`：压缩/解压计算阶段吞吐，后续文档中简称压缩/解压吞吐。
- `e2e_comp_mbs_*` / `e2e_dec_mbs_*`：真实压缩/解压端到端吞吐；OpenCL 路径排除 OpenCL init/build。
- `ratio_pct_*`：压缩率，数值越低越好。
- `*_power_increment_w`：workload 平均功率减去运行前 idle baseline。
- CPU 测试只扫 CPU 频率；GPU 测试只扫 GPU 频率；mixed 固定比例以后才扫 CPU×GPU 矩阵。
- 线程/slot 列表当前只允许显式枚举（本轮为 `1,2,4`），暂不加入 `auto`。

## 已完成扫描

| result_set | codec | engine | 配置数 | 文件行 |
| --- | --- | --- | ---: | ---: |
| baseline | LZ4 | native CPU `/usr/bin/lz4` | 12 | 288 |
| baseline | LZ4 | OpenCL GPU | 30 | 720 |
| baseline | LZ4 | hybrid 纯 CPU (`gpu_ratio=0`) | 36 | 864 |
| baseline | LZO | native CPU `lzo_cpu` | 36 | 864 |
| baseline | LZO | OpenCL GPU | 30 | 720 |
| baseline | LZO | hybrid 纯 CPU (`gpu_ratio=0`) | 36 | 864 |
| slots | LZ4 | native CPU `/usr/bin/lz4` | 12 | 288 |
| slots | LZ4 | hybrid 纯 CPU (`gpu_ratio=0`) | 108 | 2592 |
| slots | LZO | native CPU `lzo_cpu` | 108 | 2592 |
| slots | LZO | hybrid 纯 CPU (`gpu_ratio=0`) | 108 | 2592 |

扫描范围：

- CPU 频率：`1200/1800/2400/3000/3600/NA` MHz。
- GPU 频率：`400/700/1000/1300/NA` MHz。
- block：`48KB/64KB`。
- D_BITS：`13/14/15`。
- 线程/slot：baseline 固定 `1`；slots 扫描为 `1/2/4`。
- 样本：`/root/samples` 的 24 个正式样本。

## 关键结果

| codec | engine | 最优压缩吞吐 | 最优端到端压缩 | 最优解压吞吐 | 最优端到端解压 | 典型功耗增量 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LZ4 | native CPU t1 | 1121 MB/s | 621 MB/s | 3989 MB/s | 1076 MB/s | 8.1-9.2 W |
| LZ4 | OpenCL GPU | 1104 MB/s | 553 MB/s | 3772 MB/s | 951 MB/s | 8.2 W @1300MHz |
| LZ4 | hybrid 纯 CPU t4 | 1947 MB/s | 60 MB/s | 4830 MB/s | 61 MB/s | 9.4-9.7 W |
| LZO | native CPU t1/t4 | 1004/3611 MB/s | 609/1222 MB/s | 1044/3187 MB/s | 618/1165 MB/s | 8.1/14.8-15.8 W |
| LZO | OpenCL GPU | 1293 MB/s | 592 MB/s | 2675 MB/s | 901 MB/s | 7.6 W @1300MHz |
| LZO | hybrid 纯 CPU t4 | 1278 MB/s | 730 MB/s | 2736 MB/s | 973 MB/s | 9.5-10.1 W |

结论：

- LZ4 native CPU 单线程已经非常强，端到端压缩仍高于 GPU；GPU 主要价值在可控功耗和与 CPU 并行工作，而不是单设备压倒性吞吐。
- LZO GPU 计算阶段压缩吞吐优于 LZO native CPU 单线程，但端到端与 native CPU 单线程接近；当 native CPU 使用多线程时，CPU 明显更强。
- LZ4 hybrid 纯 CPU 的计算阶段随线程提升明显，但端到端吞吐只有约 60 MB/s，说明调度/进程/文件路径存在严重固定开销；在这个问题修复前不能用于 adaptive。
- LZO hybrid 纯 CPU t4 的端到端吞吐达到约 730/973 MB/s，低于 native CPU t4，但高于 t1；可以作为后续 mixed 的实现基础，但仍需要降低纯设备退化路径开销。
- D_BITS 对压缩率影响符合预期：D_BITS 越大压缩率略好；48KB 和 64KB 差异不大。LZ4 比例约 `41.35%-42.32%`，LZO 比例约 `40.22%-41.61%`。

## 频率影响

- GPU 从 `400MHz` 提升到 `1300MHz`，LZ4/LZO GPU 压缩吞吐提升约 `207%-220%`，端到端压缩提升约 `101%-115%`，主 GPU 功耗增量增加约 `5.4-6.3W`。
- CPU 从 `1200MHz` 提升到 `3600MHz`，单线程 native CPU 压缩吞吐接近线性提升，端到端也明显提升；CPU 功耗增量从约 `0.5W` 增至 `8W+`。
- 多线程 CPU 在 `NA/3600MHz` 下吞吐最高，但功耗增量也最高。能效最优点不一定是最高频，后续模型必须显式纳入功耗目标。

## 线程/slot 影响

| codec | 实现 | block/D_BITS | t1 压缩/端到端 | t2 压缩/端到端 | t4 压缩/端到端 | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| LZ4 | hybrid 纯 CPU | 64KB/D13 | 559 / 54 MB/s | 1091 / 58 MB/s | 1947 / 60 MB/s | 计算吞吐扩展正常，但端到端被固定开销压死。 |
| LZO | native CPU | 64KB/D13 | 1005 / 609 MB/s | 1952 / 916 MB/s | 3576 / 1221 MB/s | 多线程扩展有效，端到端也有效。 |
| LZO | hybrid 纯 CPU | 64KB/D13 | 434 / 337 MB/s | 841 / 550 MB/s | 1257 / 724 MB/s | 有扩展，但明显低于 native CPU。 |

## 当前瓶颈

1. LZ4 hybrid 纯 CPU 的端到端路径有严重额外开销，计算吞吐提升没有转化为真实路径吞吐。
2. LZO hybrid 纯 CPU 比 native CPU 慢，说明 OpenCL CPU 路径或 host 调度仍有固定成本。
3. GPU 对频率敏感，计算阶段收益大于端到端收益，说明读写、分块、输出路径仍会稀释 kernel 提升。
4. adaptive 如果直接同时考虑频率、block、D_BITS、线程、ratio、功耗，会过早复杂化；必须先用单设备曲面做约束。

## Adaptive 建模路线

目标不是选择 `CPU/GPU/HYBRID` 标签，而是选择 `gpu_ratio ∈ [0,1]` 与对应的 CPU/GPU 配置，使吞吐、端到端延迟和功耗在约束下达到 Pareto 最优。

分阶段推进：

1. 修复纯设备退化路径：`gpu_ratio=0` 必须接近最优 OpenCL CPU 或 native CPU 路径；`gpu_ratio=1` 必须接近纯 GPU 路径。
2. 建立单设备预测面：以 `codec/block/D_BITS/thread/frequency/file_size/block_count` 预测压缩率、压缩/解压吞吐、端到端吞吐和功耗增量。
3. 只在固定少量 ratio 上测 mixed 调度开销，确认比例切分不会引入额外串行瓶颈。
4. 使用在线校准的分层模型：先剔除明显劣势设备，再在候选 ratio 上选择满足功耗/吞吐目标的 Pareto 点。
5. 线程列表最后再加 `auto`，而且必须以 `1/2/4` 这类显式扫描结果为依据。

## 拒绝/暂缓项

| 项目 | 状态 | 原因 |
| --- | --- | --- |
| 未校准 adaptive 默认混合 | 暂缓 | 变量太多，且纯设备退化路径仍有开销。 |
| 线程/slot 列表 `auto` | 暂缓 | 先保留显式 `1/2/4`，最后基于扫描结果再实现。 |
| 将测试修正写成优化结论 | 拒绝 | 测试正确性不是性能改进，文档只保留当前正确口径。 |
| 混合默认强制 CPU 参与 | 拒绝作为默认 | 单设备和固定比例曲面未完成前，mixed 可能被慢尾拖累。 |
| 非真实路径的 bench-only 优化 | 拒绝 | 不改善真实压缩/解压端到端路径。 |
