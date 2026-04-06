# LZ4 Hybrid 优化路线（可读版）

## Wave-0 噪声阈值（G1 门禁，2026-04-06）

测量口径

- 样本目录：`/root/samples_subset`
- 重复次数：`3`
- 单轮时长：`bench_seconds=3.5`
- 代表配置：`HYBRID-only, split_layout=prefix, cpu_threads=2, block=64K, local=1, no-freq-scan`
- 阈值公式：`|Δ| > max(1.5×MAD, P95(|noise_delta|))`
- 判定方向：`CompTotalMBs`、`DecTotalMBs`、`Ratio%` 均按“越大越好”；`Δ <= -threshold_abs` 记为明显回退，`Δ >= threshold_abs` 记为明显提升。
- 阈值工件：`/root/lz4/exp_results/noise_profiles/g1/thresholds/lz4_hybrid_fixed.json`、`/root/lz4/exp_results/noise_profiles/g1/thresholds/lz4_hybrid_adaptive.json`

当前门禁阈值（LZ4 Hybrid fixed）

- `CompTotalMBs(mean)`：`threshold_abs=13.0994 MB/s`
- `CompTotalMBs(median)`：`threshold_abs=46.8400 MB/s`
- `DecTotalMBs(mean)`：`threshold_abs=64.4981 MB/s`
- `DecTotalMBs(median)`：`threshold_abs=38.7880 MB/s`
- `Ratio%(mean)`：`threshold_abs=0.0000 pctpt`
- `Ratio%(median)`：`threshold_abs=0.0000 pctpt`

当前门禁阈值（LZ4 Hybrid adaptive）

- `CompTotalMBs(mean)`：`threshold_abs=25.3601 MB/s`
- `CompTotalMBs(median)`：`threshold_abs=30.4050 MB/s`
- `DecTotalMBs(mean)`：`threshold_abs=25.2616 MB/s`
- `DecTotalMBs(median)`：`threshold_abs=206.0550 MB/s`
- `Ratio%(mean)`：`threshold_abs=0.0000 pctpt`
- `Ratio%(median)`：`threshold_abs=0.0000 pctpt`

说明

- 以上阈值用于当前 Wave-0 的 subset/fullset 采纳门禁；若后续切换配置空间或计时口径，需要重新测量并覆盖本节。
- `Ratio%` 阈值为 `0` 表示噪声测量中几乎无抖动，后续仍按“均值+中位数双判 + 全样本10轮”执行。

## 全集基线结果（当前保留）

- 基线全集目录：`/root/lz4/exp_results/runs/fullset_allcfg_current_lz4/runs/20260403_151152`
- 基线全集主结果：`lz4_param_sweep.csv`
- 主结果哈希：`sha256=41fa025e53b917bb270e93b710cd3a76e6f3a5bdff75074d1e49cb8ddde3cf88`
- 配置汇总哈希：`sha256=976e6bff7acc1fe4f01812e787b5ce2278f305f07786ca4717695002af7dfafa`
- 行数与完整性：`rows=1150`，`Roundtrip_OK=1150/1150`，引擎覆盖 `CPU/GPU/HYBRID`
- HYBRID 引擎聚合：`CompTotal mean=2525.83 MB/s`，`DecTotal mean=4940.69 MB/s`，`Ratio mean=27.9930%`

## 已采纳修改

### 前缀布局直写（免索引构建）

动机

- prefix 模式下索引构建开销偏高，压缩主机路径有固定损耗。

设计

- 压缩路径改为前缀布局直写，避免常规 `gpu/cpu block indices` 构建。

实现

- 文件：`/root/lz4/lz4_hybrid/lz4_hybrid.c`
- 证据：`host_round_20260401_113006_L4H_HOST_R5B_FULLSET_PREADOPT_ab.json`

测试结果

- `Comp +0.3264%/+1.0225%`
- `Dec +1.2066%/+0.7339%`

采纳原因

- 压缩与解压双侧正向，且 fullset 通过。

### adaptive 多目标权重 + ratio 软约束

动机

- adaptive 从单吞吐目标升级到“性能 + 能效 + 压缩率”联合目标。

设计

- 引入动态目标权重与 ratio 软约束。
- 保留解压惩罚项并改为动态输出，提升设备泛化。

实现

- 文件：`/root/lz4/lz4_hybrid/lz4_hybrid.c`
- 证据：`adaptive_round_20260401_154500_L4H_ADAPT_R1_subset_ab.json`、`...FULLSET_PREADOPT_ab.json`

测试结果

- subset：`Comp +0.7150%/+1.3721%`，`Dec +1.7842%/+2.1886%`
- fullset：`Comp +0.5605%/+0.0856%`，`Dec +4.4173%/+1.8628%`
- `Ratio +0.000100/+0.000000 pctpt`

采纳原因

- subset/fullset 双通过，Dec 提升显著。

### ratio refinement 有界搜索

动机

- fullset 复盘显示 adaptive 比率贴近 0，导致 `Comp/Dec` 双回退。

设计

- ratio refinement 改为有界搜索，避免极端值主导。

实现

- 文件：`/root/lz4/lz4_hybrid/lz4_hybrid.c`
- 证据目录：`/root/lz4/exp_results/runs/deep_rework_subset_round2/runs/20260403_121037/`

测试结果

- adaptive 相对 fixed:`R0.5`
- `Comp mean -35.50% -> -5.36%`，`Comp median -32.76% -> -0.94%`
- `Dec mean -16.12% -> -1.78%`，`Dec median -29.36% -> -0.32%`
- `AdaptiveGpuRatio mean 0.0139 -> 0.4765`

采纳原因

- 明确消除近零比率主导问题，结构修复有效。

### neutral floor + adaptive ratio cache

动机

- 继续处理“决策重复求解开销 + 低比率回落”。

设计

- 在动态边界上增加 `neutral_floor`。
- 新增 adaptive ratio cache 命中快路。

实现

- 文件：`/root/lz4/lz4_hybrid/lz4_hybrid.c`
- 证据目录：`/root/lz4/exp_results/runs/lz4_adaptive_deep_r1_prechange_v2/runs/20260403_184920/`

测试结果

- subset：`dComp +22.1505%/+18.2007%`，`dDec +3.7224%/-4.9475%`
- fullset：`dComp +17.2993%/+16.0339%`，`dDec +2.2563%/-3.3987%`
- fullset 压缩文件占比：`42/50` 提升

采纳原因

- 压缩主判显著正向，roundtrip 全通过，作为主线保留。

### 解压 metadata 哈希缓存与条件上传

动机

- 解压路径 `comp_off/comp_sizes` 重复上传造成稳定 host 开销。

设计

- 记录 metadata 哈希与数量，未变化则跳过 metadata 上传。

实现

- 文件：`/root/lz4/lz4_hybrid/lz4_hybrid.c`
- 证据：`/root/lz4/exp_results/runs/hybrid_meta_cache_r1/results/lz4_hybrid_ab_r1.summary.json`

测试结果

- `dComp +0.3160%/+0.7298%`
- `dDec +3.1997%/+2.1240%`
- `dRatio +0.00008/+0.00000 pctpt`

采纳原因

- Comp/Dec 双侧正向且 ratio 稳定。

## 未采纳修改（表格汇总）

| 修改名（实际语义） | 动机 | 设计与实现 | 测试结果 | 拒绝原因 |
| --- | --- | --- | --- | --- |
| 解压 worker 批量 claim | 降低原子调度开销 | 解压路径引入批量 claim（`host_round_...R5A...`） | `Comp -0.5499%/-0.6663%`，`Dec -0.6722%/-0.4466%` | Comp/Dec 双负向 |
| 最小化解压 offsets 构建 | 降低准备开销 | 仅必要分段构建 offsets（`host_round_...R5C...`） | `Comp +0.6209%/+0.1290%`，`Dec -2.3916%/-2.6439%` | Dec 双负向 |
| 压缩 worker 批量 claim | 减少调度热点 | 压缩线程批量领取任务（`host_round_...R6...`） | `Comp +1.7383%/+0.6109%`，`Dec -2.7797%/-2.5088%` | Dec 门禁失败 |
| 解压调度强化（大块场景） | 缩减预处理与调度开销 | 大块场景增强解压路径（`host_round_...R7...`） | `Comp +0.5267%/-1.0244%`，`Dec +0.3901%/-0.2558%` | 中位数未过门禁 |
| 线程感知 ratio guard | 稳定 `T=1/2/4` 自适应分布 | adaptive 中加入线程感知约束（`lz4_threadaware_ratio_guard_20260402_025558`） | `T1: Comp -0.0339%/+0.9857%, Dec +0.6083%/+1.4529%`；`T2: Comp +0.0414%/-0.2809%`；`T4: Comp -0.1432%/+0.1623%, Dec +0.2717%/-0.1711%` | 指标混合，无稳定净收益 |
| 频率模式 + 线程功耗感知模型 | 引入 perf/energy/ratio 联合决策 | 频率感知多目标模型（`freqaware_model_20260402_034500`） | `T1/T2` 有收益，但 `T4` 中位数明显回退（`Comp -2.9969%`, `Dec -3.7062%`） | 高线程档不稳定，被主线替代 |

## 当前代码一致性检查结论

- `lz4_hybrid.c` 保留两类关键结构：adaptive ratio cache 命中快路、metadata 哈希缓存与条件上传。
- 已拒绝项未残留额外入口或新 CLI 参数。
