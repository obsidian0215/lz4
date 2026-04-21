# LZ4 Hybrid 优化路线（可读版）

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
| CPU 压缩批量领取（两轮方案） | 降低压缩侧领取竞争 | 压缩 worker 批量领取候选实现（two-pass） | 收益不稳定，未形成一致正向 | 不满足默认并线门槛 |
| adaptive 默认 `sample-blocks 8->16` | 期望提升 adaptive 采样稳定性 | 调整 adaptive 默认采样块数 | A/B 与 R3 不稳（R3 dec `-0.27%`） | 回滚默认到 `8` |
| 压缩 worker 批量 claim | 减少调度热点 | 压缩线程批量领取任务（`host_round_...R6...`） | `Comp +1.7383%/+0.6109%`，`Dec -2.7797%/-2.5088%` | Dec 门禁失败 |
| 解压调度强化（大块场景） | 缩减预处理与调度开销 | 大块场景增强解压路径（`host_round_...R7...`） | `Comp +0.5267%/-1.0244%`，`Dec +0.3901%/-0.2558%` | 中位数未过门禁 |

## 当前代码一致性检查结论

- `lz4_hybrid.c` 保留两类关键结构：adaptive ratio cache 命中快路、metadata 哈希缓存与条件上传。
- 已拒绝项未残留额外入口或新 CLI 参数。
