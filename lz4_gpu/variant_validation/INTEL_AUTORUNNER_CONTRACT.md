# LZ4 Intel/Linux 自动 Runner 契约

更新时间：2026-04-15

## 1. 固定前提

- 操作系统：Linux
- 设备目标：Intel OpenCL / Level Zero 兼容设备
- 样本根目录：`/root/samples`
- Runner 必须默认：`SAMPLES_ROOT=/root/samples`

如果 `SAMPLES_ROOT` 被外部覆盖，Runner 必须在日志中明确打印实际值。

## 2. 目录输入

Runner 以 `lz4_gpu/variant_validation/intel/` 为工作根目录，消费：

- `variants/<variant_id>/`
- `scripts/`
- `results/`
- `logs/`
- `work/`

每个变体目录必须提供 manifest，格式见 `variant_manifest.template.json`。

## 3. 必需输入参数

最少需要支持：

- `--baseline <variant_id>`
- `--candidate <variant_id>`
- `--bench-seconds <n>`
- `--block-size <value>`
- `--sample-list <file>` 或 `--sample-glob <glob>`
- `--result-tag <name>`

可选环境变量：

- `SAMPLES_ROOT`（默认 `/root/samples`）
- `FORCE_OPENCL_DEVICE`
- `LZ4_GPU_DIR`

## 4. Runner 行为

### 4.1 准备阶段

1. 读取 baseline / candidate 的 manifest；
2. 校验样本根目录存在；
3. 校验二进制、内核目录、必要环境变量；
4. 生成本轮结果目录：

```text
intel/results/<stage>_<candidate>_<yyyymmdd_hhmmss>/
```

### 4.2 执行阶段

每个样本必须按以下顺序**前台阻塞**执行：

1. baseline bench
2. candidate bench
3. baseline standalone
4. candidate standalone

禁止：

- 后台并发跑样本；
- 未完成就先写结论；
- 混用不同样本顺序。

### 4.3 输出阶段

Runner 必须生成：

- `raw_results.csv`
- `summary_by_variant.csv`
- `summary_comparison.csv`
- `summary.md`
- `logs/<sample>_<variant>_<phase>.log`

## 5. 结果判定要求

Runner 不得只给单指标结论，必须在 `summary.md` 中同时写出：

- 吞吐变化
- `ratio` 变化
- `avg/pos/neg/neg_worst`
- trade-off 判断
- 最终标签：`adopt/watch/reject`

特别要求：

- `ratio = compressed_size / original_size`，**越低越好**；
- 如果吞吐改善但 `ratio` 上升，Runner 必须明确标记“压缩率变差”；
- 如果 `ratio` 下降但吞吐小退，Runner 必须说明该退化是否在可接受范围内。

## 6. 可移植性约束

Runner 必须：

- 使用相对路径或 `SAMPLES_ROOT`；
- 不写死 Windows 路径；
- 不依赖 PowerShell；
- 能被 shell 脚本或 Python 启动；
- 输出与 NVIDIA 同名核心工件，便于横向对照。

## 7. 失败条件

以下任一出现，都必须终止并标红：

- 样本目录不存在；
- baseline / candidate manifest 缺失；
- roundtrip 失败；
- 输出工件缺项；
- 样本执行中断但 summary 仍试图给出“通过”结论。
