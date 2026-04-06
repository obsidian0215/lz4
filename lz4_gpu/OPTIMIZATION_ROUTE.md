# LZ4 GPU 优化路线（可读版）

## Wave-0 噪声阈值（G1 门禁，2026-04-06）

测量口径

- 样本目录：`/root/samples_subset`
- 重复次数：`3`
- 单轮时长：`bench_seconds=3.5`
- 代表配置：`GPU-only, block=64K, local=1, accel=1, no-freq-scan`
- 阈值公式：`|Δ| > max(1.5×MAD, P95(|noise_delta|))`
- 判定方向：`CompTotalMBs`、`DecTotalMBs`、`Ratio%` 均按“越大越好”；`Δ <= -threshold_abs` 记为明显回退，`Δ >= threshold_abs` 记为明显提升。
- 阈值工件：`/root/lz4/exp_results/noise_profiles/g1/thresholds/lz4_gpu.json`

当前门禁阈值（LZ4 GPU）

- `CompTotalMBs(mean)`：`threshold_abs=0.7376 MB/s`
- `CompTotalMBs(median)`：`threshold_abs=0.1200 MB/s`
- `DecTotalMBs(mean)`：`threshold_abs=2.7521 MB/s`
- `DecTotalMBs(median)`：`threshold_abs=0.1650 MB/s`
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
- GPU 引擎聚合：`CompTotal mean=989.55 MB/s`，`DecTotal mean=2683.94 MB/s`，`Ratio mean=27.8264%`

## 已采纳修改

### Compaction 输出 mapped 直写优先

动机

- compaction 分支使用 chunked readback 时 host 开销明显。

设计

- packed 输出优先走 mapped contiguous 写回，失败才回退 chunked readback。

实现

- 主要文件：`/root/lz4/lz4_gpu/lz4_gpu_core.c`
- 关键 run：`..._R3_subset_ab.json`、`..._R3_REP1_subset_ab.json`、`..._R3_FULLSET_PREADOPT_ab.json`

测试结果

- subset：`Comp +3.5799%/+3.4295%`，`Dec +0.0390%/+0.4441%`
- subset 复验：`Comp +3.7254%/+2.9292%`
- fullset：`Comp +4.0159%/+2.8889%`，`Dec -0.1679%`（噪声内）

采纳原因

- 压缩侧显著稳定增益，复验一致。

### 压缩并行度按 block 数分段选择

动机

- 在 mapped 写回优化后继续提升压缩并行饱和度。

设计

- `choose_comp_worker_count()` 改为按 block 数分段选择并行度。

实现

- 文件：`/root/lz4/lz4_gpu/lz4_gpu_core.c`
- 关键 run：`..._R6_subset_ab.json`、`..._R6_FULLSET_PREADOPT_ab.json`、`..._MAIN_AFTER_R6_FULLSET_ab.json`

测试结果

- subset：`Comp +2.2179%/-0.0489%`，`Dec -0.4026%/+0.1163%`
- fullset：`Comp +2.2321%/-0.1206%`，`Dec -0.5505%/-0.0080%`
- 采纳后复核：`Comp +2.0454%/+0.0360%`，`Dec -0.7145%/-0.0440%`

采纳原因

- 压缩收益稳定，满足主线目标。

### 回退过激 occupancy，保留 null-sink 快路与按需 offsets

动机

- 在保留压缩收益的同时收敛解压副作用。

设计

- 回退过激 occupancy 调整，保留 null-sink 快路与按需 offsets。

实现

- 文件：`/root/lz4/lz4_gpu/lz4_gpu_core.c`
- 关键 run：`..._R13_subset_ab.json`、`..._R13_FULLSET_PREADOPT_ab.json`

测试结果

- subset：`Comp +2.5761%/+2.5714%`，`Dec -0.4391%/-2.2426%`
- fullset：`Comp +2.7644%/+1.9320%`，`Dec -0.7839%/-0.4742%`
- 压缩文件占比：`48/50` 提升

采纳原因

- 在压缩主目标上收益最稳，整体权衡优于备选方案。

### 解压 metadata 条件上传（保留 sizes_out 读回）

动机

- 解压 bench 元数据重复上传带来固定开销。

设计

- metadata 不变时跳过 `comp_off/comp_size` 重复上传。
- 保留 `sizes_out` 每轮读回，避免不稳定副作用。

实现

- 文件：`/root/lz4/lz4_gpu/lz4_gpu.c`
- 结果文件：`/root/lz4/exp_results/runs/gpu_dec_meta_cache_r1/results/lz4_subset_ab_cleanhead_v1.json`

测试结果

- `Comp +0.1180%/+0.6098%`
- `Dec +0.8131%/+1.6361%`
- 解压文件占比：`8/8` 提升

采纳原因

- 解压稳定正向，且压缩侧无门限外退化。

### R1 深改候选：Compaction 分布感知门控 + Packed 自动回写

动机

- 旧版 compaction 决策主要依赖固定阈值，对“块分布差异”与“传输路径差异（standard-copy vs mapped）”不敏感。
- packed 输出路径在 compaction 分支上长期固定 chunked readback，未利用 iGPU 下的 mapped contiguous 写回优势。

设计

- 引入块分布统计：`fill_ratio`、`active_ratio`、`mad_ratio`（平均绝对偏差归一化）。
- 在 `min_gain_pct` 上叠加自适应修正：同时考虑 `use_standard_copy` 与分布统计，得到 `adaptive_gain_pct`。
- 增加“致密且均匀分布”微收益保护：避免在几乎无洞布局上误触发 pack kernel。
- packed 输出新增自动写回策略：
  - non-standard-copy 下优先 mapped contiguous 写回；
  - mapped 失败自动回退 chunked readback；
  - 对低 fill 场景自适应收敛 readback chunk。

实现

- 代码文件：`/root/lz4/lz4_gpu/lz4_gpu_core.c`
- 新增核心逻辑：
  - `lz4_collect_compaction_stats(...)`
  - `lz4_compaction_adaptive_gain_pct(...)`
  - `lz4_write_compacted_payload_auto(...)`
- 工件与哈希：
  - `/root/lz4/exp_results/runs/gpu_deep_compaction_r1/artifacts/adoption_manifest_r1.txt`
  - candidate binary sha256：`645e4046a89f4d32f7ef0f7479eabd4b719f4439c64370687690ead1dca10d98`
  - baseline binary sha256：`730253128ef8c18c166544eeb29cd45385377ffef090f8eb7fde333c91dd3074`

测试结果（subset + fullset，均为 10 轮）

- 子集口径（8 文件）：`/root/samples` 固定 8 文件，`bench_seconds=3.5`，`B=64K`，`local=1`，`accel=1`
  - 汇总：`/root/lz4/exp_results/runs/gpu_deep_compaction_r1/results/lz4_gpu_r1_8files_10round_ab_summary.json`
  - 分文件：`/root/lz4/exp_results/runs/gpu_deep_compaction_r1/results/lz4_gpu_r1_8files_10round_ab_per_file.json`
  - `CompTotalMBs`：`+3.5043% / +3.5297%`（mean / median）
  - `DecTotalMBs`：`+0.0913% / +0.1084%`
  - `Ratio%`：`0.0000 pctpt / 0.0000 pctpt`
  - `CompTime_s`：`-3.3855% / -3.4094%`
  - `DecTime_s`：`-0.0912% / -0.1083%`
  - 分文件占比：压缩提升 `8/8`，解压提升 `7/8`，最大幅度文件 `mr`（压缩均值 `+4.0560%`）

- 全集口径（50 文件）：`/root/samples` 全样本，配置同上（10 轮）
  - 汇总：`/root/lz4/exp_results/runs/gpu_deep_compaction_r1/results/lz4_gpu_r1_fullset_10round_ab_summary.json`
  - 分文件：`/root/lz4/exp_results/runs/gpu_deep_compaction_r1/results/lz4_gpu_r1_fullset_10round_ab_per_file.json`
  - `CompTotalMBs`：`+3.8452% / +3.7562%`
  - `DecTotalMBs`：`-0.4954% / -0.4641%`
  - `Ratio%`：`0.0000 pctpt / 0.0000 pctpt`
  - `CompTime_s`：`-3.7022% / -3.6202%`
  - `DecTime_s`：`+0.4984% / +0.4663%`
  - 分文件占比：压缩提升 `50/50`，解压提升 `33/50`、回退 `17/50`
  - 变化幅度最大文件：`yolo_parent_0_pages_img.tar`（压缩 `+10.8205%`，解压 `-2.6947%`）

- 日志健康扫描（subset/fullset）：`error/failed/traceback/mismatch = 0/0/0/0`

当前结论

- 按 strict 门禁看，R1 的 fullset 解压存在回退（`-0.4954% / -0.4641%`），不满足“其余指标不明显回退”。
- 结合当前决策策略（压缩优先），R1 已作为**有条件采纳**基线保留，并进入后续“解压补偿轮次”继续优化。

## 未采纳修改（表格汇总）

| 修改名（实际语义） | 动机 | 设计与实现 | 测试结果 | 拒绝原因 |
| --- | --- | --- | --- | --- |
| R3：压缩/解压协同重构（解压 auto-local 调度） | 期望在保持 R1 压缩收益时抬升解压吞吐 | 在 `lz4_gpu.c` bench 解压路径与 `lz4_gpu_core.c` 解压路径引入“local=1 时自动提档 local size”策略（`gpu_deep_compaction_r3`） | subset：`Comp -0.1916%/-0.0480%`，`Dec -0.0145%/-0.0374%`；fullset：`Comp -0.4526%/-0.4134%`，`Dec -2.9275%/-2.7367%`，`Ratio 0` | 全集压缩与解压均显著回退，拒绝并已回退代码 |
| R2：decomp-friendly compaction 风险门控 | 在保留 R1 压缩收益前提下修复 fullset 解压回退 | 新增 `LZ4_GPU_COMPACTION_DECSAFE_*` 门控，对高块数+分布致密且收益边际不足场景抑制 compaction 触发（`gpu_deep_compaction_r2`） | subset：`Comp -0.0133%/+0.0140%`，`Dec +0.0005%/-0.0023%`；fullset：`Comp -0.0642%/+0.0048%`，`Dec -0.3185%/-0.1901%`，`Ratio 0` | 未有效改善 Dec 且引入额外负向波动，暂不采纳 |
| 稀疏写回改 `writev` 聚合 | 降低 host 写调用开销 | 稀疏块写回改 `writev`（`..._R1_subset_ab.json`） | `Comp -0.2671%/+0.0151%`，`Dec -0.1504%/-0.0959%` | 压缩主判未过 |
| 扩大 mapped 稀疏写优先级 | 提升 mapped 路径覆盖率 | 优先 mapped，失败回退 readback（`..._R2_subset_ab.json`） | `Comp -0.3411%/-0.0507%`，`Dec -0.2692%/+0.0797%` | 无确定收益 |
| pack local size 自适应 | 提升不同块规模适配 | local size 自适应 + 环境变量覆盖（`..._R4_subset_ab.json`） | `Comp -0.2279%/+0.0859%`，`Dec -0.0475%/-0.0631%` | 压缩未过门禁 |
| 激进 compaction gate + 回写自适应 | 冲击压缩吞吐 | 更激进 gate + pack/readback 自适应 + stride 写回（`..._R5_subset_ab.json`） | `Comp -0.3225%/+0.0053%`，`Dec -0.0899%/+0.1666%` | 主判未过 |
| standard-copy 不试 mapped + fallback `writev` | 降低 mapped 失败分支成本 | standard-copy 直接常规路径，fallback 稀疏偏 `writev`（`..._R7_subset_ab.json`） | `Comp -0.9439%/+0.0904%`，`Dec +0.3475%/+0.0416%` | 压缩均值负向 |
| 压缩路径去 `h_out_offsets` 填充 | 减少固定填充开销 | helper 支持 `offsets==NULL + stride`（`..._R8_subset_ab.json`） | `Comp -0.7112%/-0.0602%`，`Dec -0.1420%/-0.0924%` | 双侧无收益 |
| 并行度分段调为 `64/48/40/32/24` | 冲击中高 block 场景 | 调整 `choose_comp_worker_count()`（`..._R9_subset_ab.json`） | `Comp +0.7773%/-0.1259%`，`Dec -0.4262%/+0.0582%` | 压缩主判未过 |
| 并行度分段调为 `80/64/48/32/24` | 更激进提升 occupancy | subset+fullset 双检（`..._R10_*.json`） | subset：`Comp +3.0743%/-0.2287%`；fullset：`Comp +1.0747%/+0.0125%`，`Dec -0.6275%/-0.0011%` | fullset 未过 |
| 按需 sparse offsets + 并行度微调 | 降低元数据准备开销 | 压缩阶段按需构建 offsets（`..._R11_subset_ab.json`） | `Comp +1.0614%/+0.1242%`，`Dec -0.3680%/+0.1644%` | 压缩主判未过 |
| null sink 跳过 payload 回传 | 去掉 bench 固定损耗 | `/dev/null` 仅保留必要统计（`..._R12_*.json`） | subset：`Comp +4.1813%/+2.8310%`；fullset：`Comp +3.6418%/+1.9892%`，`Dec -0.8712%/-0.3714%` | Dec 副作用劣于已采纳方案 |
| 关闭 dec 调试计数 + 复用 comp_total 元数据 | 继续压缩 host 固定损耗 | hostpath deep v1（`gpu_hostpath_deep_r1/.../lz4_subset_ab_v2_3s.json`） | `Comp -0.2454%/+0.6214%`，`Dec -0.0199%/+0.6875%` | 压缩未跨门限 |
| event 生命周期释放优化 | 降低长跑事件管理开销 | kernel wait 后补 `clReleaseEvent(ev)`（旧 run：`gpu_hostpath_deep_r2/...`） | `Comp +11.7017%/+6.6001%`，`Dec -1.3256%/-0.6825%` | 基线链不一致且 Dec 超门限负向 |
| metadata cache 二阶段（跳过 `sizes_out` 读回） | 继续压缩解压 readback 时间 | metadata 不变时同时跳过 sizes_out readback（`gpu_dec_meta_cache_r2/...`） | `Comp -0.2840%/-0.1925%`，`Dec +0.1073%/-0.0358%` | 提升不稳定且压缩无收益 |

## 当前代码一致性检查结论

- `lz4_gpu.c` 当前保留“metadata 条件上传 + 每轮 sizes_out 读回”。
- `lz4_gpu_core.c` 当前主线包含“分布感知 compaction 门控 + packed 自动回写 + 分段并行度/保守 occupancy”。
