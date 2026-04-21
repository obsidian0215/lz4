# LZ4 GPU 变体验证标准

更新时间：2026-04-16

## 1. 目标

本标准用于约束 `lz4_gpu/variant_validation/` 下所有 NVIDIA / Intel 平台的变体验证工作，目标是让：

- 目录结构稳定；
- 变体包可迁移；
- 结果判定口径一致；
- Intel Linux 自动验证可以直接复用同一套结构与说明。

## 2. 目录契约

```text
lz4_gpu/variant_validation/
  README.md
  VALIDATION_STANDARD.md
  PROCESS_STAGE_CATALOG.md
  VALUE_GUIDANCE.md
  VARIANT_RECORD_TEMPLATE.md
  KERNEL_VARIANTS.md
  HOST_VARIANTS.md
  nvidia/
    variants/
    scripts/
    results/
    logs/
    work/
  intel/
    variants/
    scripts/
    results/
    logs/
    work/
```

约束如下：

- 根目录只放**标准文档**，可入库；
- `nvidia/`、`intel/` 放平台相关的本地工作区，可忽略入库；
- 新增变体、脚本、结果，默认只允许进入对应 vendor 子目录；
- 验证阶段禁止依赖仓库外的临时散落目录。

### 2.1 根层组件记录文档契约

`lz4_gpu` 根目录必须同时维护：

- `KERNEL_VARIANTS.md`
- `HOST_VARIANTS.md`

要求如下：

- 组件记录按 `vendor -> optimization_object -> stage -> operation` 组织；
- 每个条目至少覆盖：变体名称、动机、设计与实现、整体结果、判定、证据路径；
- 新候选在进入正式 full / recheck 前，必须先写入根层组件账本；
- 正式结果出炉后，必须同时回写：`summary.md`、`records/...`、根层组件账本。

## 3. 样本与环境契约

### 3.1 样本根目录

样本必须来自统一样本仓：

- 当前 Windows/NVIDIA 默认：`C:\Users\Administrator\Documents\git-repo\samples`
- Intel/Linux 自动验证默认：`/root/samples`
- 若 Intel/Linux 环境需要覆盖该路径，仍必须保证它映射到**同一份样本集合**。

统一使用环境变量：

- `SAMPLES_ROOT`：样本根目录（Intel/Linux 默认值为 `/root/samples`）

### 3.2 环境变量

脚本应显式记录并回放以下环境信息：

- `SAMPLES_ROOT`
- `FORCE_OPENCL_DEVICE`（如使用）
- `LZ4_GPU_DIR`（如使用 runtime 内核目录）
- 其他影响验证结果的 vendor/runtime 变量

### 3.3 执行方式

- 所有正式测试都必须**前台阻塞执行**；
- 候选版本只与**上一迭代稳定版本**做配对比较；
- 验证阶段禁止临时切换 build 工具链；
- 内核选择采用“预先准备好的变体目录 + 手动指定”方式完成。

补充约束：

- `smoke` 只用于验证**内核可加载 / roundtrip 正确 / 指标可解析**；
- `smoke` 通过后，其日志与中间结果默认视为临时工件，放在 `work/` 或临时目录中并直接删除，**不再保留 `results/*smoke*` 作为正式结果**；
- 进入正式判定的，只能是 full / recheck / gate 级结果。

### 3.4 正式固定参数

除非某一轴被明确声明为“正在单独扫描的正式变体”，否则 `lz4_gpu` 的正式 full / recheck / gate 结果必须锁定：

- `block = 64KB`
- `localsize = 1`
- `acc = 1`

补充要求：

- `bench` 与 `manual roundtrip` 必须显式传入同一组固定参数，禁止再依赖二进制默认值；
- `run_config.json`、`summary_comparison.*`、manifest 与记录文档都必须回显这组固定参数；
- 若某轮验证要扫描别的维度，必须先在根层组件账本占位，并写明“哪一项固定参数被提升成正式轴”。

### 3.5 组件自动化入口

`lz4_gpu` 的正式自动化入口按组件拆分，至少应维护：

- `nvidia/scripts/kernel_comp/validate_lz4_kernel_comp_stable.ps1`
- `nvidia/scripts/kernel_dec/validate_lz4_kernel_dec_stable.ps1`
- `nvidia/scripts/host/validate_lz4_host_stable.ps1`
- `nvidia/scripts/shared/` 下的共享驱动

对应结果目录也必须按组件落到：`results/kernel_comp/`、`results/kernel_dec/`、`results/host/`。

每次正式运行至少要产出：

- `run_config.json`
- `raw_runs.csv`
- `sample_medians.csv`
- `candidate_comparison.csv`
- `control_comparison.csv`
- `summary_by_role.csv`
- `summary_comparison.csv`
- `summary_comparison.txt`

## 4. 变体包标准

每个变体建议放在：

```text
<nvidia|intel>/variants/<variant_id>/
```

每个变体目录建议至少包含：

- 可执行或其定位说明；
- 对应内核文件（如 `lz4_gpu.cl`、`*.clbin`）；
- `variant.json` 或 `variant.md`：记录变体元信息。

建议元信息字段：

- `id`
- `based_on`
- `stage`（如 `hash`、`search_insert`、`count`）
- `device_vendor`（`nvidia` / `intel`）
- `binary_name`
- `kernel_artifacts`
- `compile_mode`（`prebuilt` / `source`）
- `notes`

对于 `host` 组件，变体目录还必须补充**主机程序/源码记录包**，至少覆盖：

- 当前用于正式验证的可执行程序、启动脚本或运行时目录；
- 关联的主机侧源码文件列表；
- 构建说明、环境变量开关、补丁摘要；
- 若只是环境开关差异，也必须把实际开关值写进 manifest 与结果总结。

补充要求：

- 默认源码应优先保持**最简稳定基线**；
- fingerprint、主备 hash、复杂 copy-match 分支等附加机制，若收益未证实，应作为独立候选进入 `variants/` 与根层组件账本，而不是常驻默认路径。

### 4.1 特殊轴：`hash_table_overhead`

`hash_table_overhead` 是一个**跨 kernel + host 的特殊正式轴**，用于记录：

- kernel 侧 hash table / slot 数量 / entry 布局 / 管理逻辑带来的命中与吞吐影响；
- host 侧对应的 buffer alloc / resize / staging / upload/download 开销与显存占用影响。

凡是触碰该轴的候选，必须同时：

- 在 `KERNEL_VARIANTS.md` 与 `HOST_VARIANTS.md` 建同名或可一一映射的条目；
- 在 manifest 与明细记录中写明 `special_axis_tags = ["hash_table_overhead"]`；
- 同时给出 kernel 指标与 host 分段时间，不能只报其中一边。

## 5. 标准验证流程

### 5.1 准备

1. 选择稳定基线与本轮候选；
2. 固定样本集、顺序、块大小、bench 时长；
3. 在对应 vendor 目录下准备 runtime-ready 变体包；
4. 记录 commit、驱动、设备、环境变量快照。

### 5.2 执行

每个候选至少要产出两类结果：

1. **Track-K**：`--bench` 为主判；
2. **Standalone**：`-v` roundtrip 作为工程映射验证。

最少输出：

- 原始日志；
- 原始 CSV；
- 配对统计 CSV；
- 一份总结 Markdown。

### 5.2.1 稳定性协议（正式结果强制执行）

正式 full / recheck / gate 结果必须使用稳定性协议，而不是单次跑完直接取均值。

内核改动默认执行序列：

1. `control_a`（基线）
2. `candidate`
3. `control_b`（同一基线）

要求如下：

- 固定样本顺序、参数、环境变量；
- 不再单独拆“守门阶段”；
- 每个样本、每个角色统一执行：
  - **`3` 次 `--bench`**（kernel 主判）；
  - **`7` 次手动 standalone roundtrip**（工程映射与 manual 指标）；
- bench 指标按 3 次结果取中位数；manual 指标按 7 次结果取中位数；
- 必须同时产出：
  - `candidate vs baseline_ref` 对比；
  - `control_b vs control_a` 对比（作为 control-vs-control 噪声包络）。

对于 `host` 组件，补充要求：

- 正式结果主协议改为 **7~9 次 manual roundtrip**；
- 必须采集分段时间、总时间、ratio；
- `bench` 只允许用于 smoke，不得作为 `host` 主排序依据。
- 当前正式 host 变体范围只包含：`standard_copy / mapped` 与 `hash_table_overhead`（`buffer_lifecycle` 特殊轴）；
- 一批历史低价值 host 功能已从活代码删除，不再接受新的正式 host 变体 ID；
- `metadata / header` 修补、`telemetry`、bench-only 采样与日志开关，默认只记为**修正 / hygiene**，不作为正式 host 变体 ID 建账。

判读要求：

- 候选的平均收益若未明显越过 control-vs-control 噪声包络，不得直接判为 `adopt`；
- 先看 candidate 信号，再看它是否显著大于 control 自身波动；
- 不能只报 candidate 平均值，不报 control 漂移。

### 5.3 归档

结果路径规范：

```text
<nvidia|intel>/results/<stage>_<variant>_<yyyymmdd>/
```

其中必须包含：

- `raw_results.csv`
- `summary_by_variant.csv`
- `summary_comparison.csv`
- `summary.md`
- 原始 stdout/stderr 日志（如有）

若采用稳定性协议，建议额外包含：

- `raw_runs.csv`
- `sample_medians.csv`
- `control_comparison.csv`
- `candidate_comparison.csv`

## 6. 判定口径

### 6.1 指标方向

必须明确写在报告里：

- `Bench Comp` / `Bench Dec` / `Comp Kernel` / `Dec Kernel`：**越高越好**
- `Comp Total` / `Dec Total`：**越低越好**
- `ratio = compressed_size / original_size`：**越低越好**

### 6.1.1 内核改动与主机改动的判读分离

必须先判断本轮候选属于哪一类：

- **内核改动**：hash、search/insert、matchcopy、count 等内核路径改动；
- **主机改动**：`standard_copy / mapped`、`hash_table_overhead`、queue/event 等 steady-state host/runtime 路径改动。

补充说明：

- `metadata / header` 修补、`telemetry` 与 bench-only 逻辑，若不改变 steady-state 数据路径，默认归为**修正 / hygiene**，不进入 `HOST_VARIANTS.md` 的正式候选池；
- 已退休历史方向若要重启，必须先给出“为何旧结论失效”的新机制说明，不能直接进入正式验证；
- 真正需要主协议验证的 host 候选，应优先回答：它是否改变了 steady-state 的数据搬运、hash table 生命周期或提交同步行为。

判读规则：

- **内核改动**：主要看 `Bench Comp/Bench Dec` 与 `Comp Kernel/Dec Kernel`，`Total` 只作补充参考；
- **主机改动**：主要看分段时间，不能直接拿包含 `OCI Setup/OCL init` 的总吞吐做主判。

换句话说：

> 内核改动，先看 kernel throughput 和 ratio；主机改动，先看分段时间并剔除 `OCI Setup/OCL init` 这一类不稳定大头。

### 6.2 必报字段

每轮至少输出：

- `avg`
- `pos`
- `neg`
- `neg_worst`

如有明显 trade-off，建议补充：

- `pos_best`
- 代表性样本
- `tradeoff_note`

### 6.3 当压缩率与吞吐量方向相反时

这是**强制分析项**，不能只报单指标结论。

#### 情形 A：压缩率改善（`ratio` 下降），吞吐量略退

可接受，但必须同时满足：

- 吞吐退化幅度小且可控；
- `neg_worst` 不越过当前阶段可接受噪声包络；
- 代表性样本不出现系统性崩塌；
- 报告中明确写出“用多少吞吐换来了多少压缩率改善”。

当前默认倾向：

> **如果压缩率更好，而压缩/解压吞吐只出现较小回退，可以作为可接受候选继续推进。**

#### 情形 B：吞吐量改善，压缩率变差（`ratio` 上升）

不得自动判优，必须回答：

1. 吞吐收益是否明显大于压缩率回退？
2. 回退是否集中在少数样本，还是广泛分布？
3. 最差样本是否可接受？
4. 该阶段目标是否允许为了吞吐牺牲压缩率？

只有当以上问题都得到正面答案时，才允许进入“观察候选”或“继续推进”。

### 6.3.1 对 `Total` 指标的使用约束

- 对**内核改动**，`Comp Total/Dec Total` 只用于辅助发现异常尾部，**不作为主排序依据**；
- 对**主机改动**，若要看总吞吐，必须同时给出分段时间，并显式说明是否排除了 `OCI Setup/OCL init` 这类不稳定大头；
- 没有分段时间支撑时，不得只凭 `Total` 给主机改动下结论。

### 6.4 判定标签

建议统一使用三档：

- `adopt`：可以升为下一阶段基线；
- `watch`：有价值，但需要守门样本复核或继续微调；
- `reject`：不继续沿当前方向投入。

### 6.5 阶段组合策略（固定锚点 + 延后组合）

从当前版本起，LZ4 内核阶段优化统一采用：

> **固定锚点基线 + 单阶段筛选 + 延后组合扫描**

具体规则：

1. 在同一轮内核迭代里，所有单阶段候选都先只对比**同一个稳定锚点基线**（当前为 `fp_off`）；
2. 不采用“每过一阶段就立刻叠加成新基线”的逐步滚动方式，以避免路径依赖和跨阶段结果不可比；
3. 每个阶段先筛出 `watch`/`adopt` 候选，再进入组合扫描；
4. 组合扫描以**同一个锚点基线**为参照，统一评估跨阶段叠加效果；
5. 只有组合优胜者经过稳定性复核后，才允许升为下一轮全局基线。

这意味着：

- 单阶段结果之间始终可横向比较；
- 旧的组合结果一旦不符合当前稳定性协议或组合策略，应当清理，而不是继续作为活跃结论引用。

## 7. Intel/Linux 自动化兼容要求

Intel 自动验证脚本必须满足：

- 读取与 NVIDIA 相同的目录语义；
- 默认使用 `SAMPLES_ROOT=/root/samples`，并允许显式覆盖；
- 只依赖相对路径与 manifest，不依赖人工口头约定；
- 输出与 NVIDIA 同名核心工件（`raw_results.csv`、`summary_comparison.csv`、`summary.md`）。

Intel 侧详细执行契约见：`INTEL_AUTORUNNER_CONTRACT.md`。

## 8. 报告最小模板

1. 变体信息：`baseline -> candidate`
2. 样本、参数、设备、环境变量
3. 正确性：roundtrip / verify
4. 主指标：`Bench Comp/Bench Dec`、`Comp Kernel/Dec Kernel`、`ratio`
5. 稳定性：`2 bench + 5 manual` 轮数、中位数口径、control-vs-control 漂移
6. 配对统计：`avg/pos/neg/neg_worst`
7. trade-off 结论：收益是否覆盖代价
8. 最终标签：`adopt/watch/reject`
9. 根层组件账本回写位置：`KERNEL_VARIANTS.md` 或 `HOST_VARIANTS.md`
