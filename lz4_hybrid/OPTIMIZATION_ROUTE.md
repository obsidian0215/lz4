# lz4_hybrid 优化路线

更新时间：2026-05-03

## 当前状态

`lz4_hybrid` 已从旧 native CPU pthread split 切换为 `lz4_gpu_v2` 的 OpenCL mixed 基础实现。CPU-only、GPU-only、CPU+GPU mixed 都使用同一套 `lz4_gpu.cl` kernel。

当前迁移已解决“实现路线正确”“功能可用”和 daemon 上下文复用：

- `--gpu-ratio 1`：纯 OpenCL GPU。
- `--gpu-ratio 0`：纯 OpenCL CPU。
- `0 < --gpu-ratio < 1`：CPU/GPU 按 block range 固定比例切分。
- `--adaptive`：占位，当前等价于默认 `0.5` ratio。
- Linux daemon：`--use-daemon` 支持上述 ratio/threads/adaptive 参数；mixed 请求复用 OpenCL CPU/GPU context、queue、program、kernel 和主要 device buffer。

## 已完成验证

smoke：

- 样本：`xml`、`mozilla`、`x-ray`
- 块大小：Windows standalone 覆盖 `32KB`、`64KB`；225 daemon 覆盖 `64KB`
- ratio：`1`、`0`、`0.5`、`adaptive`
- CPU slots：`1`
- 结果：压缩/解压 roundtrip 全部通过。
- daemon 复用：225 上首次 CPU-only/mixed 分别触发设备初始化，后续 mixed 请求 `init_load=0.00ms`。

## 仍不能直接下性能结论的原因

daemon 已解决重复 init/build 问题，但当前只做 smoke，不是全样本多轮性能扫描。是否默认推荐 mixed 仍必须基于 ratio/threads/block size 的正式统计。

## 下一步

1. **比例扫描**：全样本扫描 `gpu_ratio=0/0.25/0.5/0.75/1` 和 `cpu_threads=1/2/...`，记录 kernel、no-ocl-init、真实端到端吞吐。
2. **自适应建模**：在比例扫描后，用文件大小、block 数、CPU slots、GPU/CPU kernel 吞吐和 host 开销建立 ratio 决策。
3. **正式迁移判断**：只有 mixed 在复用上下文口径下有稳定收益，才把它作为默认推荐路径。
