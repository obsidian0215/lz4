# LZ4 Hybrid 主机端变体记录

更新时间：2026-04-16

## 使用约定

- `lz4_hybrid` 的正式 host 变体只聚焦 **steady-state 主机路径**：`standard_copy / mapped` 与提交同步行为；
- `metadata / header` 修补、`telemetry`、bench-only 采样与日志开关，只计为**修正 / hygiene**，不单列为正式 host 变体条目；
- `work_partition` / `fallback policy` 仍归 `SCHEDULER_VARIANTS.md`，不要与 host 路径候选混账。
- host 变体目录必须附带 **program/source record bundle**：可执行程序、源码文件列表、构建说明、环境开关与补丁摘要缺一不可。
- `hash_table_overhead` 作为特殊轴时，必须在本文件建账，并交叉引用 `lz4_gpu/variant_validation/KERNEL_VARIANTS.md` 的同轴条目。

## NVIDIA

| stage | operation | variant_id | 状态 | 说明 |
| --- | --- | --- | --- | --- |
| `buffer_lifecycle` | `hash_table_overhead` | `hash_table_overhead_ab` | `pending` | **[special-axis]** 记录 hybrid host 侧 hash table/dictionary buffer 的 alloc/reuse/resize 与驻留开销；需交叉引用 `lz4_gpu` 内核账本。 |
| `host_device_transfer` | `standard_copy_vs_mapped` | `host_transfer_mode_ab` | `pending` | 对比 hybrid steady-state 主机搬运路径是否走 `standard_copy` / `mapped`，主看 upload/download/readback 段。 |

## Intel

| stage | operation | variant_id | 状态 | 说明 |
| --- | --- | --- | --- | --- |
| `buffer_lifecycle` | `hash_table_overhead` | `hash_table_overhead_ab` | `pending` | **[special-axis]** 待补 Intel/Linux 侧 hash table/dictionary buffer 容量与驻留开销结果，并与 `lz4_gpu` 内核账本对位。 |
| `host_device_transfer` | `standard_copy_vs_mapped` | `host_transfer_mode_ab` | `pending` | 待补 Intel/Linux 侧 `standard_copy / mapped` 对位结果。 |
