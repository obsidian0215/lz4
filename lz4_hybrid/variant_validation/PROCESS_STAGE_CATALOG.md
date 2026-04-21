# LZ4 Hybrid 过程阶段目录

更新时间：2026-04-16

## 1. Host 阶段（`host`）

### 1.1 `file_io`

- 文件读写
- metadata / header 处理

> `metadata / header` 修补默认记为**修正 / hygiene**，除非它直接改变 steady-state 数据路径，否则不作为正式 host 变体主线。

### 1.2 `buffer_lifecycle`

- host/device buffer alloc/free
- staging / map / unmap / pin
- hash table / dictionary staging buffer 的 alloc / resize / reuse

### 1.3 `host_device_transfer`

- upload / download / readback / writeback
- `standard_copy` / `mapped` 传输模式
- packed / sparse 相关搬运

### 1.4 `dispatch_sync`

- queue / event / wait / flush / completion
- host 与 GPU 间的同步策略
- steady-state submit order

### 1.5 `host_feature`

- 其他会改变 steady-state 主机路径的功能开关

> `telemetry`、bench-only 采样/日志、纯修补型 metadata 开关不作为正式 host 变体，统一归为修正记录。
>
> 若 host 侧改动了 hash table / dictionary 的容量、驻留方式或 buffer 生命周期，必须额外打上 `special_axis_tags = ["hash_table_overhead"]`，并交叉引用 `lz4_gpu` 的对应内核条目。

## 2. Scheduler 阶段（`scheduler`）

### 2.1 `work_partition`

- CPU/GPU 任务切分
- block mapping / ratio / threshold

### 2.2 `execution_model`

- staged execution / submit order
- chunk / batch 执行模型

### 2.3 `fallback_policy`

- CPU fallback / no-trace fallback
- heuristic / threshold / mode switch

## 3. 分类要求

每个新条目都必须标注：

- `component`: `host` / `scheduler`
- `stage`
- `operation`
- `baseline`
- `motivation`
- `special_axis_tags`
- `program/source record`：host / scheduler 条目必须能回溯到实际程序工件与源码记录
