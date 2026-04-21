# LZ4 Hybrid 变体验证目录

更新时间：2026-04-16

## 1. 目录用途

这里是 `lz4_hybrid` 的**主机端 / 任务调度** 变体验证工作区。

与 `lz4_gpu` 不同，这里不重复把 GPU 内核优化再建一遍账；`hybrid` 目录只关注：

1. `host`：文件、buffer、上传下载、同步、缓存/回退等主机机制；
2. `scheduler`：任务分配、CPU/GPU 比例、fallback policy、执行模型；
3. 手动 roundtrip 的分段时间与总时间；
4. Intel/Linux 与 Windows/NVIDIA 的同名变体横向对照。

## 2. 根层文档

```text
lz4_hybrid/variant_validation/
  README.md
  VALIDATION_STANDARD.md
  PROCESS_STAGE_CATALOG.md
  VALUE_GUIDANCE.md
  VARIANT_RECORD_TEMPLATE.md
  INTEL_AUTORUNNER_CONTRACT.md
  variant_manifest.template.json
  HOST_VARIANTS.md
  SCHEDULER_VARIANTS.md
  nvidia/
  intel/
```

## 3. 记录原则

- `hybrid` 目录只维护 `HOST_VARIANTS.md` 与 `SCHEDULER_VARIANTS.md` 两本账；
- 主机/调度正式结果必须以 **7~9 次 manual roundtrip** 为主；
- `bench` 只允许用于 smoke，不允许作为 host/scheduler 主判据；
- 若某个改动同时涉及内核与调度，必须拆成两条记录，分别归到 `lz4_gpu` 与 `lz4_hybrid`。

## 4. 当前状态

- 目录骨架已就位；
- 当前优先事项是：补主机端 manual harness、登记当前调度模型、对无效开关做删减扫描；
- GPU 内核条目请看：`lz4_gpu/variant_validation/KERNEL_VARIANTS.md`。
