# lz4_hybrid 性能状态

更新时间：2026-05-03

旧版性能数据基于 native CPU pthread + GPU split 实现，已经不适用于当前 `lz4_hybrid`。当前目录已迁移为 OpenCL CPU/GPU/mixed 基础实现，并已接入 Linux daemon 上下文复用；已有结果仍只证明功能和复用链路正确，不能作为最终性能结论。

## 当前可用数据

smoke：

- 样本：`xml`、`mozilla`、`x-ray`
- 块大小：Windows standalone 覆盖 `32KB`、`64KB`；225 daemon 覆盖 `64KB`
- ratio：`1`、`0`、`0.5`、`adaptive`
- CPU slots：`1`
- 结果：压缩/解压 roundtrip 全部通过。
- daemon 复用：225 上后续同 daemon 进程内 mixed 请求 `init_load=0.00ms`。

## 暂不写性能结论

daemon 已消除重复 init/build 的主要干扰。正式性能数据仍需要重新生成，因为当前只做 smoke，没有全样本、多 ratio、多线程、多轮统计。

后续正式表格至少应包含：

- 压缩/解压 kernel 吞吐。
- 压缩/解压 no-ocl-init 吞吐。
- 真实端到端吞吐。
- 压缩率。
- `gpu_ratio`、`cpu_threads`、block size。
