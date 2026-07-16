# lz4_hybrid 优化路线

更新时间：2026-05-21

## 当前定位

`lz4_hybrid` 是 LZ4 的 OpenCL CPU/GPU mixed 实现承载目录。当前重点不是继续堆复杂 adaptive 规则，而是先把 CPU、GPU 和 mixed 的基础能力、频率敏感性、功耗增量、块大小和字典大小关系测清楚。

## 当前实现

- `--gpu-ratio 1`：纯 OpenCL GPU。
- `--gpu-ratio 0`：纯 OpenCL CPU。
- `0 < --gpu-ratio < 1`：CPU/GPU 按 block range 固定比例切分。
- `--adaptive`：保留实验入口；当前不作为默认策略或性能结论。
- `--d-bits 11..15`：控制每个 work-item 的 hash table 位宽，不改变 active block 并发。
- Linux daemon：用于复用 OpenCL context、queue、program、kernel 和主要 buffer；不把 daemon 修复或测试纠错写成性能优化项。

## 现有判断

- GPU-only 仍是当前最可靠基线；mixed 是否有价值必须在相同频率、块大小、字典位宽和真实端到端口径下重新判定。
- OpenCL CPU-only 是 mixed 框架内的 CPU 设备路径，不等同于 native CPU `lz4`。
- 自适应建模暂缓。先建立 `native_cpu`、`opencl_cpu`、`gpu`、固定比例 mixed 的速度/功耗/压缩率数据，再决定 adaptive 的目标函数和可用变量。

## 拒绝项

| 项目 | 判定 | 原因 |
| --- | --- | --- |
| bench-only gather/compaction | 拒绝 | 只改变 bench 计时或中间统计，不改善真实压缩/解压路径。 |
| 用 active lanes 缩字典池 | 拒绝 | 把字典预算和并发度绑死，损害吞吐；字典缩小应通过 `D_BITS` 或表项宽度处理。 |
| 未校准 adaptive 默认混合 | 拒绝作为默认 | 当前证据不足，且固定 mixed 可能被 CPU 慢尾拖累。 |

## 下一步扫描

1. `native_cpu`：系统 `lz4`，固定单线程/多线程列表，扫描块大小、CPU 频率、压缩率、压缩/解压吞吐、端到端吞吐、CPU package/core 功率增量。
2. `opencl_cpu`：`lz4_hybrid --gpu-ratio 0`，同样扫描线程数、块大小、字典位宽和 CPU 频率。
3. `gpu`：`lz4_gpu`/`lz4_hybrid --gpu-ratio 1`，扫描块大小、`D_BITS`、GPU 频率、压缩率、压缩/解压吞吐、端到端吞吐、GPU 功率增量。
4. `mixed`：只在 CPU/GPU 单设备最优实现确认后，再扫描固定比例矩阵；`adaptive` 只消费这些数据，不先手写复杂规则。

## 建模方向

adaptive 的目标不是简单选择 CPU/GPU/HYBRID，而是在给定文件、块划分、设备频率和功耗状态下决定 `gpu_ratio ∈ [0,1]`。候选变量必须来自低开销、可稳定获取的信息：文件大小、block 数、块大小、字典位宽、CPU threads、GPU/CU 信息、近期实测吞吐和设备功耗增量。数据不足前不做规则固化。
