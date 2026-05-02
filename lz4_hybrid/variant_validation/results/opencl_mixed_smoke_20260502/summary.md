# OpenCL mixed smoke（2026-05-02）

## 目的

确认 `lz4_hybrid` 迁移到 `lz4_gpu_v2` OpenCL mixed 基础实现后，纯 GPU、纯 OpenCL CPU、固定混合比例和 adaptive 占位路径都能正确 roundtrip。

## 配置

- 平台：本机 Windows / Intel Arc B390。
- 样本：`xml`、`mozilla`、`x-ray`。
- 块大小：`32KB`、`64KB`。
- ratio：`1`、`0`、`0.5`、`adaptive`。
- CPU slots：`--cpu-threads 1`。

## 结果

所有组合压缩/解压 SHA256 校验通过。临时 `.lz4` 和 `.out` 产物已删除，仅保留日志。

## 结论

当前 `lz4_hybrid` 可作为 OpenCL mixed 功能基线。性能结论暂不成立，因为一次性命令仍包含 CPU/GPU 两套 OpenCL context 初始化和 build 开销。
