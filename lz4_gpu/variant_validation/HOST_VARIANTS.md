# LZ4 GPU 主机端变体文档（重写版）

更新时间：2026-04-19

## 1. 文档边界

- 本文档只记录 `host` 组件变体。
- 内核端条目在 `KERNEL_VARIANTS.md`。
- 结论只保留：`adopt` / `watch(未采纳)` / `reject`。

---

## 2. 当前已采纳的主机端修改

### 2.1 统一内存设备默认走 mapped 传输路径

- 阶段：`host_device_transfer`
- 修改名称：**standard copy / mapped 路径收敛，默认 mapped**
- 动机：在统一内存设备上减少不必要拷贝路径切换，稳定 steady-state。
- 设计与实现：
  - 通过 all-off 组合矩阵验证后，将 mapped 作为主默认路径。
  - 保留标准拷贝作为可回退模式，不作为默认。
- 测试效果：
  - 组合矩阵显示 mapped 是唯一稳定、可解释且可保留的 host 特性。
- 结论：`adopt`
- 来源：`HOST_VARIANTS.md` 旧记录 / intel host 组合矩阵结论。

---

## 3. 主机端未采纳修改

## 3.1 观察中（watch，未采纳）

### A. Hash table 生命周期微调（special-axis）

1) **device-managed dict buffer 的 epoch/offset 管理版本**
- 动机：优化 buffer 生命周期与 decode 侧友好性。
- 结果：decode 有局部正向，但 comp 或 total 指标无稳定净收益。
- 结论：`watch`

2) **epoch wrap active-span clear（仅清 active span）**
- 动机：减少 wrap 清零代价。
- 结果：comp/dec 某些段正向，但 dec inclusive / total 仍有尾部负向。
- 结论：`watch`

## 3.2 已拒绝（reject）

### A. direct-final-arena 压缩收尾改写
- 动机：去掉收尾 staging，直接流式写最终输出。
- 测试结果（全量）：去 OCI 后总吞吐收益接近 0，未超噪声。
- 判据：无稳定净收益。
- 结论：`reject`
- 来源：`LZ4_GPU_V1_MIGRATION_ROUTE.md`（M7）

### B. direct-map 解压输出写回强化
- 动机：进一步压缩解压收尾 download/unmap 开销。
- 测试结果（全量）：局部指标有波动，但去 OCI 总口径噪声主导。
- 判据：不构成可复现净收益。
- 结论：`reject`
- 来源：`LZ4_GPU_V1_MIGRATION_ROUTE.md`（M8）

---

## 4. 主机端当前基线结论

- **已采纳主线**：mapped 传输默认路径。
- **保留观察**：hash table 生命周期 special-axis 两条 watch。
- **已明确拒绝**：M7 / M8 两条主机端“收尾写回重排”路线。

---

## 5. 与内核端的联动要求

- 任何 `hash_table_overhead` 变体必须在 `KERNEL_VARIANTS.md` 与本文件双向记账。
- 仅当“主机端改动 + 内核端基线”在同口径下产生稳定净收益，才可升级为 adopt。
