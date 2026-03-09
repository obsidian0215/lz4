# LZ4 GPU 性能总结

更新时间：2026-03-09  
程序路径：`/root/lz4/lz4_gpu/lz4_gpu`

## 1. 当前正式结果工件

- CPU/GPU stitched full-corpus artifact：`/root/lz4/exp_results/runs/20260309_merged_full_83/lz4_param_sweep_merged.csv`
- 工件 provenance：`/root/lz4/exp_results/runs/20260309_merged_full_83/merge_manifest.json`
- 基础 full run：`/root/lz4/exp_results/runs/20260309_160340/`
- 纠正性 patch runs：`/root/lz4/exp_results/runs/20260309_230411/`、`/root/lz4/exp_results/runs/20260309_230842/`
- 跨算法汇总：`/root/analysis/20260309_full_refresh/`

说明：最终 CPU/GPU artifact 不是删除旧结果后“重写历史”，而是在保留原始 run 的前提下，把 83-file `/root/samples` 上的 verified rows 用 manifest 方式 stitch 成一个可追溯视图。`20260309_230842` 同时修正了 `bench_lz4.py` 在 `.lz4` 输入上 CPU total-verification 少传 `-z` 的 harness 问题。

## 2. 测试口径

- 语料：`/root/samples`，83 files
- 本文档只引用 `Roundtrip_OK=yes` 的 verified rows
- CPU/GPU 对比使用 **best-per-file median** 作为主结论口径
- 跨算法比较只在 **matched 83-file corpus** 上成立，不把不同算法的 level / accel 直接视为等价设置

## 3. 83-file full-corpus best-per-file medians

| Engine | Verified files | Comp total MB/s | Dec total MB/s | Comp kernel MB/s | Dec kernel MB/s | Ratio % | Comp power W |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU | 83 | 698.71 | 755.68 | 1853.29 | 5347.40 | 22.38 | 19.63 |
| GPU | 83 | **1497.76** | **1085.39** | **5952.21** | **14772.52** | **25.23** | **18.72** |

相对 CPU：

- GPU compression total：`2.14x`
- GPU decompression total：`1.44x`

## 4. 结果分析

### 4.1 GPU 仍是当前 LZ4 standalone 默认主路径

在 fresh 83-file stitched artifact 上，GPU 同时拿到更高的压缩与解压 best-per-file medians，并且压缩功耗略低于 CPU。这个结论已经不依赖旧 baseline，也不依赖 subset。

### 4.2 kernel 很高，但 total 才决定交付能力

GPU medians：

- compression：`5952.21 MB/s kernel` vs `1497.76 MB/s total`
- decompression：`14772.52 MB/s kernel` vs `1085.39 MB/s total`

这说明当前 LZ4 GPU 的主要限制依然是 runtime / transfer / file-backed path，而不是 kernel 本身不够快。

### 4.3 这次 stitched artifact 的意义

这轮结果和更早 summary 不同，原因不是“指标波动”，而是我们把 full-corpus rerun、failed-key patch run、以及 manifest 化 provenance 合并成了一个更可靠的正式结果视图。文档里今后应该引用 stitched artifact，而不是混用旧 full baseline 与局部补测数字。

## 5. 跨算法位置（只限 matched 83-file corpus）

来自 `cross_family_best_per_file_medians.csv` 的同语料比较：

- `LZ4 GPU` compression total `1497.76 MB/s`，高于 `LZO GPU lzo1x` 的 `1334.63 MB/s` 与 `LZO GPU lzo1y` 的 `1330.73 MB/s`
- `LZ4 GPU` decompression total `1085.39 MB/s`，也高于两条 LZO GPU 路径的约 `808 MB/s`

这个结论只表示：**在当前 83-file matched corpus、当前 bench harness、当前参数矩阵下，LZ4 GPU 是三条算法路径里最强的 standalone total-throughput 结果。**

## 6. 当前结论

> `lz4_gpu` 仍然是当前整套实验中最强的 standalone 高吞吐路径之一；在 fresh 83-file full-corpus stitched artifact 上，它继续显著领先 CPU，并且在 matched cross-algorithm view 中也领先 LZO1X/LZO1Y 的 GPU baseline。
