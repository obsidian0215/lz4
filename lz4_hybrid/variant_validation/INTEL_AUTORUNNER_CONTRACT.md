# LZ4 Hybrid Intel/Linux 自动 Runner 契约

更新时间：2026-04-16

## 1. 固定前提

- 操作系统：Linux
- 样本根目录：`/root/samples`
- Runner 必须默认：`SAMPLES_ROOT=/root/samples`
- `hybrid` 主线结果以 **manual roundtrip** 为主，不以 `bench` 为主判据。

## 2. 目录输入

Runner 以 `lz4_hybrid/variant_validation/intel/` 为工作根目录，消费：

- `variants/host/`
- `variants/scheduler/`
- `results/host/`
- `results/scheduler/`
- `scripts/host/`
- `scripts/scheduler/`

## 3. 必需输入参数

- `--baseline <variant_id>`
- `--candidate <variant_id>`
- `--component <host|scheduler>`
- `--rounds <7|9>`
- `--sample-list <file>` 或 `--sample-glob <glob>`
- `--result-tag <name>`

## 4. Runner 行为

### 4.1 执行

每个样本、每个版本都必须前台阻塞执行，并记录：

- upload / download / dispatch / wait 等分段时间
- total time
- ratio
- 是否 roundtrip 正确

### 4.2 输出

Runner 必须生成：

- `manual_runs.csv`
- `segment_breakdown.csv`
- `summary_comparison.csv`
- `summary.md`

## 5. 判定要求

- 必报 `avg/pos/neg/neg_worst`
- 必报 ratio 变化
- 必报 trade-off 结论
- 不得在缺少分段时间时仅凭总时间给出“通过”结论
