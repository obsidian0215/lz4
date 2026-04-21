# LZ4 GPU 变体验证目录

更新时间：2026-04-16

## 1. 目录用途

这里是 `lz4_gpu` 的**阶段化、可建账、可迁移**的变体验证工作区。

从当前版本开始，验证目标不再只是“继续测几个候选”，而是要建立一套可并行、可裁剪、可复用到 Intel/Linux 的流程：

1. 先按 **压缩内核 / 解压内核 / 主机端 / 调度** 划分优化对象；
2. 再按算法阶段/过程分类候选；
3. 先确认 CPU 原始实现与 GPU 当前实现各自做了什么优化；
4. 把当前实现中**无效或存疑**的机制筛掉，收敛出当前最简且最优基线；
5. 然后才进入单阶段候选设计、验证、保留与集成组合。

## 2. 目录结构（改进后）

```text
lz4_gpu/variant_validation/
  README.md
  VALIDATION_STANDARD.md
  PROCESS_STAGE_CATALOG.md
  VALUE_GUIDANCE.md
  VARIANT_RECORD_TEMPLATE.md
  INTEL_AUTORUNNER_CONTRACT.md
  variant_manifest.template.json
  KERNEL_VARIANTS.md
  HOST_VARIANTS.md
  nvidia/
    variants/
      kernel_comp/
      kernel_dec/
      host/
      scheduler/
    scripts/
      kernel_comp/
      kernel_dec/
      host/
      scheduler/
      shared/
    results/
      kernel_comp/
      kernel_dec/
      host/
      scheduler/
    records/
      kernel_comp/
      kernel_dec/
      host/
      scheduler/
    logs/
    work/
  intel/
    ...（与 nvidia 同构）
```

说明：

- `variants/`：按优化对象分类保存 runtime-ready 候选；
- `scripts/`：按优化对象分类保存 harness；
- `results/`：按优化对象分类保存正式结果；
- `records/`：按优化对象与阶段记账，记录尝试过的候选、动机、实现、结果与判定；
- `PROCESS_STAGE_CATALOG.md`：定义阶段名称，避免再出现“知道改了点什么，但说不清属于哪个阶段”的情况；
- `VALUE_GUIDANCE.md`：总结高/低测试价值尝试，减少黑箱乱试。
- `KERNEL_VARIANTS.md` / `HOST_VARIANTS.md`：根层组件账本，按 `vendor -> stage -> operation` 记录每个候选的动机、实现、结果与判定。

## 3. 新流程总览

### 3.1 内核端

1. 按 `kernel_comp` / `kernel_dec` 列阶段；
2. 对照 CPU 原始实现与 GPU 当前实现，列出现有优化点；
3. 逐项验证这些优化是否真的有效；
4. 删除/冻结无效做法，得到当前最优简化基线；
5. 再按阶段设计单一机制候选；
6. 保留 `watch`/`adopt` 候选；
7. 阶段都完成后再做组合测试。

### 3.2 主机端

1. 先把现有功能/配置列出来；
2. 重点扫描主机/设备通信、buffer 生命周期、传输/同步、pack/compaction 等功能；
3. 主机侧**不使用 bench 模式**；
4. 正式结果采用 **7~9 次手动 roundtrip**，采集：
   - 分段时间
   - 总时间
   - 压缩率
5. 把无实际效果或负价值功能优先删掉/冻结。

### 3.3 调度/混合

1. 先以 host/scheduler 为主对象建账；
2. 聚焦 work partition、execution model、fallback policy；
3. 与主机端同样以 manual harness 为主，不用 bench 做主判。

## 4. 当前基线与状态提醒

- 当前默认源码已进一步收敛到**无 fingerprint 的最简 hash 基线**，并移除了 `copy_match` 里部分只增分支深度的非向量化分支；
- `fingerprint` 与 `主备 hash / dual probe` 已从默认路径剥离，统一作为单独候选记录在 `KERNEL_VARIANTS.md` 中；
- 当前 `lz4_gpu` 单阶段稳定锚点仍为：`fp_off`
- 当前高优先保留候选：`count_batch32_first`
- 当前吞吐旁线参考：`search_step_warmup_shift5_cap4`（`throughput-only watch`）
- 当前 overlap-cost 探索参考：`count_overlap_cost_score`（探索性 `watch`，不进主线）

这些状态都应写入 `records/`，而不是只留在临时总结文件里。

## 5. 使用约束

1. 正式结果必须前台阻塞执行；
2. Windows/NVIDIA 样本根目录固定为 `C:\Users\Administrator\Documents\git-repo\samples`；Intel/Linux 默认为 `/root/samples`；
3. 新的正式结果必须进入按对象分类后的 `results/`；
4. 新的候选必须先写记录，再进正式验证；
5. 新的主机端测试必须使用 manual harness，不得用 bench 代替主判；
6. 任何候选的结论都必须同时记录到：

- 结果总结 `summary.md`
- 阶段分类记录 `records/...`
- 根层组件账本（`KERNEL_VARIANTS.md` / `HOST_VARIANTS.md`）

## 6. 迁移说明

目前 `nvidia/results/` 下仍保留历史平铺目录，作为过渡期兼容入口；
但从现在开始：

- 新目录骨架按本 README 执行；
- 新候选优先写入按对象分类后的子目录；
- 历史结果逐步补分类记录，而不是继续扩大平铺历史。

Intel 自动 runner 约束与 manifest 见：`INTEL_AUTORUNNER_CONTRACT.md`、`variant_manifest.template.json`。
