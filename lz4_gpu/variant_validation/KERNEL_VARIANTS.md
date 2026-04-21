# LZ4 GPU 内核变体文档（重写版）

更新时间：2026-04-20

## 1. 文档边界

- 本文档只记录 **内核端** 变体（`kernel_comp` / `kernel_dec`）。
- 主机端变体单独记录在 `HOST_VARIANTS.md`，但本文档的“拒绝总表”会引用主机端结果，便于审计。
- 本文档只保留三类结论：`adopt` / `watch(未采纳)` / `reject`。
- 关于共享表：`R1.18` 当前已重置为“上游 LZ4 单表语义对齐分析”，不再把“所有 lane 共写一张表”的错误原型视为正式候选。

---

## 2. 当前已采纳的内核端修改（仅保留真实名称）

### 2.1 压缩内核：全局预算分片 + 宽哈希表恢复

- 归类：压缩（kernel_comp / hash table）
- 实际修改名称：**全局字典池预算分片，并恢复 64KB 场景宽哈希表路径**
- 核心动机：在保持吞吐提升的同时，修复压缩率回退。
- 设计与实现：
  - host 侧采用预算驱动的 owner 数管理（不再简单按文件线性推导）。
  - `tableType==0` 恢复宽 hash（`HASHLOG+1` 路径）与对应 dict entries。
  - 保留 epoch+offset 打包 entry 语义。
- 验证结果（对 lock）：
  - bench 压缩吞吐：`+53.8151%`
  - bench 解压吞吐：`-0.3817%`
  - ratio：`0.0000%`（不退）
- 结论：`adopt`
- 证据：`intel/hash_optimization_records_20260419.md`

### 2.2 解压内核：相对偏移 Token 解码循环 + 短匹配 fast-direct 组合

- 归类：解压（kernel_dec / token_decode + match_copy）
- 实际修改名称：
  1) **Token 解码相对偏移循环（relative-offset loop）**
  2) **短匹配（<=18B）direct-copy 快路径 helper**
- 核心动机：降低解压 hot loop live-state 压力并加速短匹配主流路径。
- 设计与实现：
  - 在解压 loop 中使用 `op_rel/match_rel` 相对偏移状态组织。
  - 对 `<=18B` direct match 使用专用 helper，减少通用路径分支成本。
- 组合验证结果（全量 25 样本）：
  - bench 解压吞吐：`+21.0230%`
  - manual dec kernel 吞吐：`+18.3843%`
  - manual dec steady：`+9.5449%`
  - ratio：`0.0000%`
- 结论：`adopt`
- 证据：`LZ4_GPU_V1_MIGRATION_ROUTE.md`（M3 章节）

---

## 3. 未采纳内核修改清单（watch / reject）

## 3.1 压缩侧（kernel_comp）

### A. 已拒绝（reject）

1) **固定窄哈希分片（吞吐换压缩率）**
- 动机：统一 hash 宽度并提升吞吐。
- 结果：吞吐强正向，但 ratio 全局回退。
- 判据：ratio 退化，不满足主线约束。
- 结论：`reject`
- 来源：R1（hash 记录）

2) **双候选桶策略（dual-bucket）**
- 动机：提高历史候选质量。
- 结果：压缩吞吐全样本大幅回退（约 -19%~-22%）。
- 判据：主指标崩塌。
- 结论：`reject`
- 来源：M4（迁移记录）

3) **最近密集+旧候选稀疏保留策略**
- 动机：减少被高频覆盖，保留高价值旧候选。
- 结果：仍出现压缩吞吐全样本显著回退。
- 判据：吞吐损失不可接受。
- 结论：`reject`
- 来源：M5（迁移记录）

4) **Lane 自适应加速（跨块信号调节）**
- 动机：利用前块统计信号优化下一块搜索强度。
- 结果：吞吐正向，但 ratio 全样本负向。
- 判据：ratio 退化，不可采纳。
- 结论：`reject`
- 来源：M6（迁移记录）

5) **每 lane 私有四分之一表**
- 动机：进一步把每-lane 表缩到基线 `1/4`，观察更激进减表是否还能保留吞吐收益。
- 结果：bench 压缩吞吐 `+1.62%`，但 ratio `-1.10%`，8/8 文件全部回退。
- 判据：ratio 回退已经超出“受控退化”范围，且文件间波动明显。
- 结论：`reject`
- 来源：`intel/records/kernel_comp/intel_kernel_comp_r1_19_lane_private_quarter_table.md`

6) **直接减少 owner 数 / owner-local 连续 chunk 串行复用**
- 动机：在不回退到“共享单表”错误语义的前提下，测试是否能通过减少 owner 张数来显著压低总字典体积，并进一步用 owner-local 连续 chunk 复用改善局部性。
- 结果：
  - `R1.20 owner 2/3`：bench 压缩吞吐 `avg -34.13%`，manual `comp_kernel_tp_mbs avg -26.70%`，ratio `0.0000%`；
  - `R1.21 owner-local chunk reuse`：bench 压缩吞吐 `avg -37.96%`，manual `comp_kernel_tp_mbs avg -28.52%`，ratio `0.0000%`。
- 判据：压缩主指标在 8/8 文件上全负向，且回退幅度远超噪声；owner-local 连续 chunk 分配也未能挽救主路径。
- 结论：`reject`
- 来源：`intel/records/kernel_comp/intel_kernel_comp_r1_20_owner_two_thirds_widehash.md`、`intel/records/kernel_comp/intel_kernel_comp_r1_21_owner_local_chunk_reuse.md`

7) **lane/table 解耦最小原型：2 lanes 共享 1 物理表槽**
- 动机：验证“owner 数不变、总表更少”是否可行，即保持 active lanes 不变，只减少物理 hash 表槽位。
- 结果：bench 压缩吞吐 `avg +0.85%`，manual `comp_total_no_oci_tp_mbs avg +2.58%`，但 ratio `avg -6.46%`，8/8 文件全部回退，`xml` 最差 `-21.97%`。
- 判据：吞吐未崩，但并发 lane 共用物理表导致匹配质量严重污染，ratio 断崖式退化，远超可接受范围。
- 结论：`reject`
- 来源：`intel/records/kernel_comp/intel_kernel_comp_r1_22_lane_pair_shared_table.md`

8) **packed private-table：每 lane 私有表项压成 16-bit 物理存储**
- 动机：在不减少 active lanes、不破坏每 lane 私有表语义的前提下，只压缩 64K 路径的表项物理存储宽度，测试“总表减半”是否可行。
- 结果：ratio `0.0000%` 完全守住，但 bench 压缩吞吐 `avg -9.98%`，manual `comp_kernel_tp_mbs avg -8.35%`，`comp_total_no_oci_tp_mbs avg -1.63%`。
- 判据：匹配质量未退化，说明私有语义保持成功；但当前实现依赖每 block 清零 packed table，导致压缩主路径显著回退。
- 结论：`reject`
- 来源：`intel/records/kernel_comp/intel_kernel_comp_r1_23_lane_private_packed_u16_table.md`

9) **packed private-table + validity sidecar**
- 动机：延续 `R1.23` 的 packed private-table 方向，尝试用小型 validity sidecar 取代“每 block 清零整张 packed table”，验证能否在保持 ratio 的同时回收吞吐。
- 结果：ratio `0.0000%` 仍完全守住，但 bench 压缩吞吐 `avg -29.18%`，manual `comp_kernel_tp_mbs avg -22.03%`，`comp_total_no_oci_tp_mbs avg -10.15%`，明显比 `R1.23` 更差。
- 判据：说明瓶颈不再是 block 边界清理本身，而是 sidecar 方案在 hash get/put 热路径上引入了额外 metadata 读写，导致主路径访问放大严重。
- 结论：`reject`
- 来源：`intel/records/kernel_comp/intel_kernel_comp_r1_24_packed_u16_validity_sidecar.md`

10) **packed private-table + touched-slot selective clear**
- 动机：延续 packed private-table 路线，但移除 `R1.24` 那种读路径 sidecar 查询；sidecar 只记录本 block 触碰过的 logical slot，块边界只清命中的半槽。
- 结果：quick 子集 ratio `0.0000%`，但 bench 压缩吞吐 `avg -26.42%`，manual `comp_kernel_tp_mbs avg -14.81%`，`comp_total_no_oci_tp_mbs avg -3.85%`，仍稳定负向。
- 判据：说明即便读路径恢复为 packed-data-only，按 touched slot 逐半槽清理的块边界成本仍然过高。
- 结论：`reject`
- 来源：`intel/records/kernel_comp/intel_kernel_comp_r1_25_packed_u16_touched_slot_clear.md`

11) **packed private-table + touched-word clear**
- 动机：在 `R1.25` 基础上继续粗化清理粒度，把 touched sidecar 从 slot-level bitmap 改为 data-word-level bitmap，降低 sidecar 尺寸和块边界扫描复杂度。
- 结果：quick 子集 ratio `0.0000%`，bench 压缩吞吐 `avg -18.81%`，比 `R1.25` 有所回收，但 manual `comp_kernel_tp_mbs avg -22.49%`、`comp_total_no_oci_tp_mbs avg -15.01%` 反而更差。
- 判据：说明“bitmap + 扫描清理”即使粗化到 data word，仍不足以把 packed private-table 方向拉回可接受区间。
- 结论：`reject`
- 来源：`intel/records/kernel_comp/intel_kernel_comp_r1_26_packed_u16_touched_word_clear.md`

12) **packed private-table + sparse touched-list**
- 动机：彻底摆脱固定 bitmap 扫描，只在写路径记录实际触碰过的 packed data word，块边界只清 `touched_list` 中出现过的 word。
- 结果：quick 子集 ratio `0.0000%`，但 bench 压缩吞吐 `avg -33.16%`，manual `comp_kernel_tp_mbs avg -28.09%`，`comp_total_no_oci_tp_mbs avg -18.08%`，比 `R1.26` 进一步恶化。
- 判据：说明 sparse list 的计数、去重和追加维护成本，比固定 bitmap 扫描更重；该分支不值得继续推进 formal gate8。
- 结论：`reject`
- 来源：`intel/records/kernel_comp/intel_kernel_comp_r1_27_packed_u16_sparse_touched_list.md`

13) **packed private-table + single-block once-zero fallback**
- 动机：基于 `R1.23~R1.27` 的失败链，停止继续优化块间复用清理；只在 `blocks_per_owner<=1` 时启用 packed private-table，并在 kernel 启动前一次性 zero，完全移除 per-block clear / validity / touched metadata。若 `blocks_per_owner>1`，则直接回退 baseline 宽表 epoch 路径。
- 结果：quick 子集 ratio `0.0000%`，bench 压缩吞吐 `avg -4.11%`，manual `comp_kernel_tp_mbs avg -11.53%`，`comp_total_no_oci_tp_mbs avg -4.56%`。相比 `R1.23~R1.27` 显著回收了大部分回退，但仍未转正。
- 判据：说明真正的大头瓶颈确实来自 per-block clear / metadata tracking；但就算把这些拿掉，packed 16-bit 热路径本身仍有不可忽视的位拆装成本，暂不足以成为可采纳候选。
- 结论：`reject`
- 来源：`intel/records/kernel_comp/intel_kernel_comp_r1_28_packed_single_block_zero_fallback.md`

### B. 观察中（watch，未采纳）

1) **增大压缩 lane 配额**
- 结果：相对已采纳基线仅微小波动，收益不稳定。
- 结论：`watch`
- 来源：R1.6

2) **提升 hash pool 预算上限**
- 结果：bench 微正向但 manual 内核/端到端未稳定收益。
- 结论：`watch`
- 来源：R1.7

3) **搜索循环 hash 槽预取**
- 结果：bench 主指标偏负，manual 表现混合。
- 结论：`watch`
- 来源：R1.8

4) **搜索阶段轻量化（预计算 mask + masked put/get）**
- 结果：去污染后处于噪声范围，无法形成稳定收益。
- 结论：`watch`
- 来源：C1

5) **短匹配直通编码（压缩侧）**
- 结果：整体偏负，`xml/webster` 有明显回退。
- 结论：`watch`
- 来源：C2

6) **冷路径隔离（压缩侧）**
- 结果：压缩近乎持平但解压侧偏负，整体不成立。
- 结论：`watch`
- 来源：C3

7) **R1.19 私有表缩减系列（`3/4` / `2/3` / `1/2`）**
- 动机：保持 lane 私有表语义不变，只缩小每张表的容量，寻找“表成本下降”与“匹配质量损失”之间的平衡点。
- 结果：
  - `3/4` 表：ratio 回退最小（`avg -0.16%`），manual `comp_kernel_tp_mbs` 回到近乎持平/微正；
  - `2/3` 表：ratio 仍控制在 `avg -0.22%`，但 manual `comp_kernel_tp_mbs avg +2.28%`、`comp_total_no_oci_tp_mbs avg +4.99%`，是当前 throughput / ratio 折中最强的一档；
  - `1/2` 表：端到端收益更强（`comp_total_no_oci_tp_mbs avg +3.05%`），但 kernel 主路径已轻微回退，ratio 回退扩大到 `-0.33%`。
- 判据：三者都不是 adopt，但都属于“有 trade-off 的有效候选”；若 prioritise ratio，则 `3/4` 更均衡；若允许 `~0.22%` 级别受控 ratio 回退以换取更强压缩收益，则 `2/3` 更值得优先保留。
- 结论：`watch`
- 来源：`intel/records/kernel_comp/intel_kernel_comp_r1_19_lane_private_half_table.md`

## 3.2 解压侧（kernel_dec）

### A. 已拒绝（reject）

- 当前 Intel 主线记录中，解压内核主要实验已通过 M3 收敛；无新增强负向 reject 内核项需要单独入主线。

### B. 观察中（watch，未采纳）

- **仅短匹配 helper 单独版本**：单独收益虽明显但存在负向样本，最终以组合 M3 采纳，单独条目保留 `watch`。

---

## 4. 跨文档审计：内核+主机“被拒绝修改”总表

> 本表包含内核端 + 主机端（引用 `HOST_VARIANTS.md`）被拒绝项，便于一次审计。

| 分类 | 修改名称 | 结论 | 拒绝判据（简述） |
| --- | --- | --- | --- |
| 压缩内核 | 固定窄哈希分片 | reject | ratio 全负向退化 |
| 压缩内核 | 双候选桶策略 | reject | comp 吞吐全样本显著回退 |
| 压缩内核 | 最近密集/旧稀疏候选 | reject | comp 吞吐持续大幅负向 |
| 压缩内核 | lane 自适应加速 | reject | ratio 明显负向 |
| 主机端 | direct-final-arena 收尾改写 | reject | 去 OCI 后增益未超噪声 |
| 主机端 | direct-map 解压输出写回强化 | reject | 去 OCI 后净收益不稳定、噪声主导 |

---

## 5. 当前内核基线结论

- **压缩基线（adopt）**：全局预算分片 + 宽哈希恢复（R1.5）
- **解压基线（adopt）**：relative-offset decode + <=18B fast-direct（M3）
- **非采纳方向**：
  - 任何导致 ratio 退化的压缩候选（即便吞吐提升）
  - 任何仅在噪声内波动、无稳定收益的微调
