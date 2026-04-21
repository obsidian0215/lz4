# LZ4 Hybrid 调度变体记录

更新时间：2026-04-16

## 使用约定

- `scheduler` 条目同样属于 **host program/source 变体**，必须附带 program/source record bundle 与自动化入口。
- 正式结果仍以固定参数 `block=64KB`、`localsize=1`、`acc=1` 的 manual roundtrip 为主；若要扫描调度轴，必须显式说明被提升为正式轴的是哪一项调度参数。

## NVIDIA

| stage | operation | variant_id | 状态 | 说明 |
| --- | --- | --- | --- | --- |
| `work_partition` | `ratio_policy_inventory` | `scheduler_baseline_inventory` | `pending` | 待登记当前 CPU/GPU 切分模型。 |
| `execution_model` | `submit_order_inventory` | `scheduler_submit_model_inventory` | `pending` | 待登记当前 chunk / batch / submit 顺序。 |
| `fallback_policy` | `threshold_inventory` | `scheduler_fallback_inventory` | `pending` | 待登记 fallback 触发条件。 |

## Intel

| stage | operation | variant_id | 状态 | 说明 |
| --- | --- | --- | --- | --- |
| `work_partition` | `ratio_policy_inventory` | `scheduler_baseline_inventory` | `pending` | 待补 Intel/Linux 结果。 |
| `execution_model` | `submit_order_inventory` | `scheduler_submit_model_inventory` | `pending` | 待补 Intel/Linux 结果。 |
| `fallback_policy` | `threshold_inventory` | `scheduler_fallback_inventory` | `pending` | 待补 Intel/Linux 结果。 |
