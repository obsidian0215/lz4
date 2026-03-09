# LZ4 Hybrid 性能总结

更新时间：2026-03-09  
程序路径：`/root/lz4/lz4_hybrid/lz4_hybrid`

## 1. 当前正式结果工件

- Hybrid full-corpus run：`/root/lz4/exp_results/hybrid_bench/hybrid_bench_20260309_180949.csv`
- CPU/GPU stitched baseline：`/root/lz4/exp_results/runs/20260309_merged_full_83/lz4_param_sweep_merged.csv`
- 跨算法汇总：`/root/analysis/20260309_full_refresh/`

这次 hybrid 文档同时保留两种口径：

1. **raw medians**：直接对 full sweep 全部 verified rows 取中位数；
2. **best-per-file medians**：对每个文件先取该模式下的最佳配置，再跨 83 files 取中位数。

不能把两种口径混为一谈。

## 2. 83-file raw medians（全部 verified rows）

| Mode | Verified rows | Comp total MB/s | Dec total MB/s | Comp kernel MB/s | Dec kernel MB/s | Ratio % | Comp power W |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed | 1992 | **919.72** | **675.42** | **3139.23** | 4401.50 | 23.705 | 15.79 |
| Adaptive | 1992 | 889.91 | 674.58 | 2990.15 | **4433.39** | 23.705 | 15.83 |

## 3. 83-file best-per-file medians

| Mode | Verified files | Comp total MB/s | Dec total MB/s | Comp kernel MB/s | Dec kernel MB/s | Ratio % | Comp power W |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed | 83 | **1425.90** | 802.90 | **2725.26** | 3402.44 | 25.32 | **16.06** |
| Adaptive | 83 | 1302.24 | **813.20** | 2654.46 | **3968.56** | 25.32 | 16.17 |

对照 stitched standalone medians：

- `LZ4 CPU`：`698.71 / 755.68`
- `LZ4 GPU`：`1497.76 / 1085.39`

## 4. winner counts（83-file matched corpus）

| Metric | CPU | GPU | Hybrid fixed | Hybrid adaptive |
|---|---:|---:|---:|---:|
| Compression total | 3 | 28 | **48** | 4 |
| Decompression total | 3 | **62** | 14 | 4 |

## 5. 结果分析

### 5.1 Fixed hybrid 已经是 LZ4 compression 的真实竞争者

这轮 fresh full-corpus run 里，`Hybrid fixed` 在文件级 compression winner count 上达到 `48/83`，超过 GPU 的 `28/83`。这说明 fixed hybrid 已经不只是“个别样本有效”，而是在 matched corpus 上具备明显的压缩侧竞争力。

### 5.2 但 GPU 仍是更稳的默认总吞吐主路径

即便 fixed hybrid 在更多文件上拿到 compression win，`LZ4 GPU` 仍然保持更高的 best-per-file compression median（`1497.76` vs `1425.90`），并且在 decompression winner count 上仍明显领先（`62/83`）。因此系统级默认路径仍应写成：

- compression：GPU 与 fixed hybrid 接近，但 GPU 略高；
- decompression：GPU 明显更强。

### 5.3 Adaptive 已经是真实模式，但不是 fixed 的普适替代者

Adaptive 在这次结果里表现为：

- best-per-file compression median 低于 fixed（`1302.24 < 1425.90`）
- best-per-file decompression median 略高于 fixed（`813.20 > 802.90`）
- 文件级 winner count 仍明显少于 fixed

因此当前最准确的表述是：**adaptive 可用、可测、而且在解压侧有一定收益，但 fixed 仍是更强的 hybrid compression 模式。**

## 6. 跨算法位置（matched 83-file corpus）

来自 `cross_family_best_per_file_medians.csv`：

- `LZ4 Hybrid fixed` compression total `1425.90 MB/s`，高于所有 `LZO Hybrid` 模式（最高为 `942.86 MB/s`）
- `LZ4 Hybrid adaptive` decompression total `813.20 MB/s`，低于 `LZO Hybrid adaptive` 的约 `897~901 MB/s`

因此当前 cross-family 结论不是“某一条 hybrid 全面统治”，而是：

- LZ4 hybrid 更强在 compression
- LZO adaptive hybrid 更强在 decompression

## 7. 当前结论

> `lz4_hybrid` 已经进入正式可比较阶段：fixed 模式在 83-file full corpus 上赢下了更多 compression files，但 GPU 仍是更稳的 default engine，尤其在 decompression 上保持明显领先；adaptive 则是一个真实有效、但尚未取代 fixed 的调度模式。
