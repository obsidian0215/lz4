# LZ4 Hybrid 变体验证标准

更新时间：2026-04-16

## 1. 目标

本标准用于约束 `lz4_hybrid/variant_validation/` 下所有 NVIDIA / Intel 平台的 **host / scheduler** 变体验证，目标是：

- 目录结构统一；
- 组件账本完整；
- 结果判定不再依赖口头解释；
- Intel/Linux 可以消费同一套记录与 manifest。

## 2. 根层文档契约

根目录必须维护：

- `README.md`
- `VALIDATION_STANDARD.md`
- `PROCESS_STAGE_CATALOG.md`
- `VALUE_GUIDANCE.md`
- `VARIANT_RECORD_TEMPLATE.md`
- `INTEL_AUTORUNNER_CONTRACT.md`
- `variant_manifest.template.json`
- `HOST_VARIANTS.md`
- `SCHEDULER_VARIANTS.md`

其中：

- `HOST_VARIANTS.md` 按 `vendor -> stage -> operation` 建账；
- `SCHEDULER_VARIANTS.md` 按 `vendor -> stage -> operation` 建账；
- 每条记录必须至少覆盖：变体名称、动机、设计与实现、整体结果、判定、证据路径。

## 3. 样本与环境契约

- Windows/NVIDIA 默认样本根目录：`C:\Users\Administrator\Documents\git-repo\samples`
- Intel/Linux 默认样本根目录：`/root/samples`
- 统一通过 `SAMPLES_ROOT` 注入样本根目录。

### 3.1 正式固定参数

除非某一轴被明确声明为“正在单独扫描的正式变体”，否则 `lz4_hybrid` 的正式 host / scheduler 结果必须锁定：

- `block = 64KB`
- `localsize = 1`
- `acc = 1`

补充要求：

- host / scheduler 的 manual roundtrip 必须显式传入同一组固定参数，禁止再依赖二进制默认值；
- `run_config.json`、`summary_comparison.*`、manifest 与记录文档都必须回显这组固定参数；
- 若 host / scheduler 要联动扫描 CPU/GPU ratio、fallback 等别的维度，也不能顺带放松这三项固定参数。

### 3.2 自动化入口

`lz4_hybrid` 的正式自动化入口至少应维护：

- `nvidia/scripts/host/validate_lz4_hybrid_host_stable.ps1`
- `nvidia/scripts/scheduler/validate_lz4_hybrid_scheduler_stable.ps1`
- `nvidia/scripts/shared/validate_lz4_hybrid_hostlike_stable.ps1`

每次正式运行至少要产出：

- `run_config.json`
- `raw_runs.csv`
- `sample_medians.csv`
- `candidate_comparison.csv`
- `control_comparison.csv`
- `summary_by_role.csv`
- `summary_comparison.csv`
- `summary_comparison.txt`

## 4. 执行方式

### 4.1 正式结果

- host / scheduler 正式结果必须前台阻塞执行；
- 正式主判据必须来自 **7~9 次 manual roundtrip**；
- 必须记录：分段时间、总时间、ratio、`avg/pos/neg/neg_worst`；
- `bench` 只允许做 smoke，不允许作 host/scheduler 主排序依据。
- `host` 正式候选只覆盖：`standard_copy / mapped` 与 submit path 相关主机路径；
- `scheduler` 正式候选除静态 `fixed ratio / fallback` 外，还必须包含 `adaptive` 一类变体；
- `adaptive` 的正式目标不是“让 GPU 占比尽量高”或“固定某个经验 R”，而是要根据**不同设备性能、运行状态与输入特征**动态决定 CPU/GPU 比例 `R`，尽量保持**最低的并行执行时间**（优先看 `Parallel Span`，再结合总时间与分段时间）；
- 验证 `adaptive` 时，必须把它当成正式 scheduler 变体建账，而不是临时启发式；同时要明确对比的静态 baseline（如固定 `R=0.3/0.5/0.7` 或当前稳定 ratio policy）。
- `metadata / header` 修补、`telemetry` 与 bench-only 逻辑默认记为**修正 / hygiene**，不作为正式 host 变体占位。
- host / scheduler 变体目录必须附带**主机程序/源码记录包**：可执行程序/脚本、源码文件列表、构建说明、环境开关、补丁摘要缺一不可。
- `hash_table_overhead` 作为特殊轴时，虽然 kernel 判据写在 `lz4_gpu` 账本里，但 host / scheduler 侧仍必须同步记录 buffer alloc / resize / upload/download 段的影响。

### 4.2 对比方式

- 候选只与当前稳定基线比较；
- 同一轮必须固定样本集、顺序、参数、环境变量；
- 如需跨平台比较，必须复用相同 `variant_id`，只在 NVIDIA / Intel 分章补结果。

## 5. 判定口径

- 分段时间：越低越好；
- 总时间：越低越好；
- `ratio = compressed_size / original_size`：越低越好；
- 若总时间变好但 `ratio` 变差，必须显式写出 trade-off；
- 没有分段时间支撑时，不得只看总时间下结论。

补充约束：

- `host` 条目默认只分析 steady-state 的搬运与提交同步行为；
- `metadata / telemetry / bench-only` 修正项若不改变 steady-state 路径，不进入 `HOST_VARIANTS.md` 的正式候选池。
- 对 `adaptive` scheduler，优先主判 `Parallel Span` / 并行执行时间是否下降；只有在这一目标成立时，才进一步用总时间、ratio 与分段时间判断 trade-off 是否可接受。

## 6. 必报字段

每轮至少输出：

- `avg`
- `pos`
- `neg`
- `neg_worst`
- 代表性分段时间
- 若为 `adaptive` 变体，必须额外输出 `R` 的选择摘要（至少 `gpu_ratio_mean / min / max` 或等价字段），并说明它相对静态 baseline 的并行执行时间变化
- trade-off 结论
- 最终标签：`pending` / `watch` / `reject` / `adopt`
