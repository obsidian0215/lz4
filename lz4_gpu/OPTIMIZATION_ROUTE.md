# LZ4 GPU 优化路线

## 1. 文档边界

- 本文件用于记录优化路径与实验历史；
- `PERFORMANCE_SUMMARY.md` 仅保留“当前并线实现结果 + 基线比较”；
- 本文件中的实验按两类维护：
  1. 已采纳修改：必须包含动机、设计、实现、测试结果；
  2. 未采纳修改：仅维护未采纳表格。

---

## 2. 已采纳修改（当前主线）

### 2.1 Mapped host copy 默认开启

#### 动机

- Intel/Linux 的 all-off host 组合矩阵表明：`mapped` 是唯一稳定、可解释、值得默认开启的 host 特性；
- `pipeline / overlap / compaction` 长期保留只会增加状态空间，并污染后续 hash/dict 优化的对比基线。

#### 设计

- 统一内存设备默认走 `LZ4_STANDARD_COPY=0`（mapped）路径；
- `LZ4_STANDARD_COPY=1` 只保留为显式 A/B 或 dGPU fallback；
- 不再围绕默认配置保留 `pipeline / overlap / compaction` 开关。

#### 实现

- 文件：`lz4_gpu/lz4_gpu_core.c`
- 关键函数：`lz4_prefers_standard_copy()`、`write_buffer_auto()`、`read_buffer_auto()`
- 关键记录：`lz4_gpu/variant_validation/intel/records/host/intel_host_combo_matrix_mapped_pipeline_overlap_compaction.md`

#### 测试结果

- 组合矩阵最终把 `m1 p0 o0 c0` 作为 host 收敛方向：`mapped` 保留，`pipeline / overlap / compaction` 退休；
- 独立 `compaction-on` rerun（`intel_host_r5_pack_gate_compaction_enable`）给出：`comp_kernel_execution_ms avg -4.9327%`、`comp_inclusive_tp_mbs avg -1.9592%`、`ratio 0`；
- 结论：host 默认路径应当简化为 `mapped`，而不是继续围绕已退休功能做门控调参。

---

### 2.2 压缩并行度按 block 数分段选择

#### 动机

- mapped 写回落地后，压缩并行饱和度仍有提升空间。

#### 设计

- `choose_comp_worker_count()` 改为按 block 数分段取值，替代固定策略。

#### 实现

- 文件：`lz4_gpu/lz4_gpu_core.c`
- 关键函数：`choose_comp_worker_count()`

#### 测试结果

- subset：`Comp +2.2179% / -0.0489%`
- fullset：`Comp +2.2321% / -0.1206%`
- 采纳后复核：`Comp +2.0454% / +0.0360%`

---

### 2.3 回退过激 occupancy，保留 null-sink 快路与按需 offsets

#### 动机

- 需要在压缩收益与解压副作用之间做稳态平衡。

#### 设计

- 回退过激 occupancy 调整；
- 保留 null-sink 快路；
- 保留按需 offsets 构建。

#### 实现

- 文件：`lz4_gpu/lz4_gpu_core.c`

#### 测试结果

- subset：`Comp +2.5761% / +2.5714%`，`Dec -0.4391% / -2.2426%`
- fullset：`Comp +2.7644% / +1.9320%`，`Dec -0.7839% / -0.4742%`
- 压缩文件占比：`48/50` 提升

---

### 2.4 解压 metadata 条件上传（保留 `sizes_out` 读回）

#### 动机

- bench 循环中 metadata 重复上传带来固定耗时。

#### 设计

- metadata 不变时跳过 `comp_off/comp_size` 上传；
- 保留每轮 `sizes_out` 读回，避免稳定性回退。

#### 实现

- 文件：`lz4_gpu/lz4_gpu.c`

#### 测试结果

- `Comp +0.1180% / +0.6098%`
- `Dec +0.8131% / +1.6361%`
- `Dec` 文件占比：`8/8` 提升

---

### 2.5 块内 lazy-match 建模（深改，首轮并线）

#### 动机

- 常规块大小（`64K`）口径下仍存在压缩率尾部；
- 需要“非策略切换、非文件名驱动”的结构级改动，提升块内匹配质量。

#### 设计

- 在 `lz4_compress_core_accelerated()` 中加入一次 look-ahead：
  - 当前匹配点 `ip` 与 `ip+1` 进行候选比较；
  - 当 `nextLen >= curLen + gain` 时，采用 `ip+1` 的匹配（lazy 选择）；
- 默认参数：`enable=1, gain=1, accel_max=2`；
- 通过编译宏暴露可控项：
  - `LZ4_GPU_ENABLE_LAZY`
  - `LZ4_GPU_LAZY_GAIN`
  - `LZ4_GPU_LAZY_ACCEL_MAX`

#### 实现

- 内核文件：`lz4_gpu/lz4_gpu.cl`
  - 新增 lazy 宏默认值；
  - 在主匹配循环插入 look-ahead 决策逻辑。
- 主机端编译参数：`lz4_gpu/lz4_gpu_core.c`
  - `lz4_load_program()` 新增 lazy 宏注入；
  - 仅当 lazy 参数为默认值时，允许加载预编译 `clbin`。
- 预编译产物命名：`lz4_gpu/Makefile`
  - 输出改为 `lz4_gpu_14_model_v2.clbin`、`lz4_gpu_15_model_v2.clbin`。

#### 测试结果（Windows / RTX 4070 Ti SUPER / `bench samples=50`）

- 工件：`lz4/exp_results/baseline/lazy_ab_samples_64k1_rerun_20260408.csv`
- 配置：`BS=64K; ACC=1; LSZ=1; bench=1s`，对比 `lazy=0 -> lazy=1`
- 汇总（`avg/pos/neg/neg_worst_abs`）：
  - `CompTotal = +0.8747% / 29 / 21 / 2.7744%`
  - `DecTotal  = +0.6127% / 27 / 23 / 8.8492%`
  - `RatioX    = +0.0341% / 27 / 1 / 0.0265%`
  - 同时满足 `CompTotal>0` 且 `RatioX>0` 的文件：`13/50`
  - 迭代对比（相对上一版）：`CompTotal avg -0.1759 pct`、`RatioX avg -0.0032 pct`，但 `CompTotal neg_worst_abs` 从 `6.6755%` 收敛到 `2.7744%`（下降约 `58.4%`）

结论：完成口径切换后，64K 官方样本依然保持压缩吞吐与压缩率双正向（均值口径）。

---

## 3. 未采纳修改（仅保留未采纳项）

### 3.1 Host 路径退役：pipeline / overlap / compaction

#### 动机

- 这三条线的维护成本高，而且会显著放大 host 状态空间；
- 在正式 host 稳态口径下，它们没有给出足够稳的默认收益。

#### 设计

- 先用显式 `all-off` 基线跑完 host 组合矩阵；
- 再单独 rerun 可疑分支，确认不是因为解析或阈值口径问题被误判。

#### 实现

- 记录：`lz4_gpu/variant_validation/intel/records/host/intel_host_combo_matrix_mapped_pipeline_overlap_compaction.md`
- 记录：`lz4_gpu/variant_validation/intel/records/host/intel_host_r5_pack_gate_compaction_enable.md`
- 活代码处置：回归最小 host 路径；README / validation docs 同步删除旧入口。

#### 测试结果

- `mapped` 是唯一稳定、可解释、值得保留的 host 特性；
- `pipeline` 在强制触发条件下整体更容易拉低压缩 steady-state；
- `overlap` 只在 `standard-copy + pipeline` 上有局部条件性作用，一旦 `pipeline` 退役即失去价值；
- `compaction` 单独打开主信号不佳，在组合矩阵中也只在部分 pipeline 基座上有条件价值；
- 结论：三条线全部退休，不再进入未来默认候选池。

| 修改主题 | 动机 | 关键实现 | 代表工件 | 结果摘要 | 未采纳原因 |
| --- | --- | --- | --- | --- | --- |
| 稀疏写回改 `writev` 聚合 | 降低 host 写调用开销 | 稀疏块写回改 `writev` | `..._subset_ab.json` | `Comp -0.2671%/+0.0151%`，`Dec -0.1504%/-0.0959%` | 压缩主判未过 |
| 扩大 mapped 稀疏写优先级 | 提升 mapped 路径覆盖 | mapped 优先，失败回退 readback | `..._subset_ab.json` | `Comp -0.3411%/-0.0507%`，`Dec -0.2692%/+0.0797%` | 无稳定收益 |
| pack local size 自适应默认化 | 提升块规模适配 | pack local-size 自适应 | `..._subset_ab.json` | `Comp -0.2279%/+0.0859%`，`Dec -0.0475%/-0.0631%` | 压缩未过门限 |
| standard-copy 直走常规 + fallback `writev` | 降低 mapped 失败分支成本 | standard-copy 不尝试 mapped | `..._subset_ab.json` | `Comp -0.9439%/+0.0904%`，`Dec +0.3475%/+0.0416%` | 压缩均值负向 |
| 压缩路径去 `h_out_offsets` 填充 | 减少固定填充开销 | `offsets==NULL + stride` | `..._subset_ab.json` | `Comp -0.7112%/-0.0602%`，`Dec -0.1420%/-0.0924%` | 双侧无收益 |
| 并行度分段 `64/48/40/32/24` | 冲击中高 block 场景 | 调整 `choose_comp_worker_count()` | `..._subset_ab.json` | `Comp +0.7773%/-0.1259%`，`Dec -0.4262%/+0.0582%` | 压缩主判未过 |
| 并行度分段 `80/64/48/32/24` | 更激进 occupancy | subset + fullset 双检 | `..._subset_fullset_ab.json` | subset 正向但 fullset 未通过 | 全样本未过 |
| 按需 sparse offsets + 并行度微调 | 降低元数据准备开销 | 压缩阶段按需构建 offsets | `..._subset_ab.json` | `Comp +1.0614%/+0.1242%`，`Dec -0.3680%/+0.1644%` | 压缩主判未过 |
| null sink 跳过 payload 回传 | 去掉 bench 固定损耗 | `/dev/null` 仅保留必要统计 | `..._subset_fullset_ab.json` | 压缩提升明显，解压副作用偏大 | 稳态权衡不优 |
| 关闭 dec 调试计数 + 复用 comp_total 元数据 | 继续压缩 host 固定损耗 | hostpath deep v1 | `.../lz4_subset_ab_v2_3s.json` | `Comp -0.2454%/+0.6214%`，`Dec -0.0199%/+0.6875%` | 压缩未跨门限 |
| event 生命周期释放优化 | 降低事件管理开销 | kernel wait 后释放 event | `gpu_hostpath_deep/...` | `Comp +11.7017%/+6.6001%`，`Dec -1.3256%/-0.6825%` | 基线链不一致且 Dec 超门限 |
| metadata cache 二阶段（跳过 `sizes_out`） | 压缩解压 readback 时间 | metadata 不变时跳过 sizes_out 读回 | `gpu_dec_meta_cache/...` | `Comp -0.2840%/-0.1925%`，`Dec +0.1073%/-0.0358%` | 提升不稳定 |
| 执行模型 PoC 分段调度（全样本） | 期望优化大文件吞吐 | PoC 分段 dispatch + chunk 配置 | `gpu_accel_scan_123_poc_execmodel_r3/runs/20260407_112007/` | 相对并线基线出现系统性回退 | 不满足并线门槛 |
| large profile 默认强化策略 | 期望提升大文件路径 | 调整 `lz4_should_use_large_profile` 触发策略 | `gpu_accel_scan_123_poc_execmodel_r3b/runs/20260407_122606/` | 相比上版进一步恶化 | 回退幅度扩大 |
| 解压 chunk 默认放大到 16K | 降低 launch 次数 | 提高 `LZ4_GPU_POC_DEC_CHUNK_BLOCKS` 默认值 | `gpu_accel_scan_123_poc_execmodel_r4_fix/runs/20260407_151705/` | 局部样本有利，全量不稳 | 默认值已回收 |
| PoC 压缩计时口径候选 | 统一统计口径 | 调整 PoC 压缩计时路径 | `gpu_accel_scan_123_poc_execmodel_r4_timing/runs/20260407_161900/` | 相比上一版收敛但仍未优于并线基线 | 不并线 |
| PoC 分段按 chunk 动态 `g_ws` | 减少尾段空转 work-item | 压缩 PoC 路径每个 chunk 重算 dispatch `g_ws` | `gpu_accel_scan_123_poc_execmodel_chunkgws/runs/20260407_174539/` | vs `default_v2`：`CompTotal -11.6608%`，`DecTotal -6.7555%`；`Comp/Dec` 负向文件分别 `150/150`、`147/150` | 系统性回退，已回退代码 |
| 解压并行度固定 `DECOMP_WI_PER_CU=80`（环境候选） | 尝试抬升 `DecKernel` 长尾 | 仅设置环境变量 `LZ4_GPU_DECOMP_WI_PER_CU=80`，不改代码默认值 | `gpu_dec_wi_unset_full/runs/20260407_190333/`、`gpu_dec_wi80_full/runs/20260407_185233/` | **vs 上一版（unset）**：`CompTotal +2.4304%`（`pos/neg=107/43`），`DecTotal +0.0302%`（`91/59`），`DecTotal neg_worst_abs=12.4798%`（`webster`）；**vs 并线基线 `default_v2`**：`CompTotal -11.5349%`（`0/150`），`DecTotal -6.0971%`（`17/131`） | 对并线基线系统性回退，且负向尾部过大；不进入默认实现 |

结论补充：该候选仅在“同轮 unset 控制组”口径下出现边际正向，但在并线基线口径下明显回退，按未采纳处理并保留为环境实验项。

### nvCOMP 同口径修正与默认配置基线（2026-04-08）

#### 动机

- 需满足“同测试方式/同文件/同统计语义”下的硬对位：`kernel + total + ratio + power`。
- 按当前规则，参数调优与非默认块大小扫描不再保留，避免无意义 tune。

#### 实现与工件

- 修脚本：
  - `nvcomp/benchmarks/bench_nvcomp_lz4_python.py`：补齐 `Comp/Dec CPU/GPU Power` 字段；
  - `nvcomp/benchmarks/compare_nvcomp_consistent_vs_lz4_modes.py`：统一 Python-only 对位口径。
- nvCOMP 基线：`nvcomp/exp_results/baseline/nvcomp_lz4_python/runs/20260408_031151/`
- LZ4 仅保留 bench 默认配置工件：`lz4/exp_results/baseline/gpu_only_default/runs/20260408_032757/`
- 对位汇总仅保留：`lz4/exp_results/baseline/compare_nvcomp_consistent_vs_lz4_modes_20260408.{json,csv}`

#### 结果摘要（核心）

- 默认配置口径下已完成 nvCOMP vs LZ4 的统一统计对位；
- 吞吐/压缩率/功耗指标可在同一套默认配置结果中复核；
- 参数调优相关汇总与文档说明已移除。

#### 采纳结论

- 后续仅允许使用 bench 脚本默认配置进行评估与归档；
- 不再新增或保留 tune 类工件与对应说明。

---

## 4. 后续优化方向

1. **压缩长尾收敛**：持续压缩 `CompKernel/CompTotal` 负向尾部。
2. **解压尾部专项**：针对 `DecKernel` 最差样本做定向策略。
3. **采纳门槛**：每轮必须给出 `avg/pos/neg/neg_worst_abs`，并与上一版及并线基线双对比。
4. **并线规则**：任何候选若对并线基线产生系统性回退，不进入默认实现。

---

## 5. 当前主线一致性快照

- `lz4_gpu.c`：保留 metadata 条件上传 + 每轮 `sizes_out` 读回；不再保留 pipeline/compaction bench 分支。
- `lz4_gpu_core.c`：保留 `mapped / standard-copy` 双路径 + 分段并行度 + 保守 occupancy 策略；不再保留 pipeline/compaction 活代码。
- `lz4_gpu.cl`：仅保留压缩/解压 kernel；`pack` kernel 已随 host 路径退休一起删除。
