# lz4_hybrid（完整实现说明）

`lz4_hybrid` 是本仓库中面向 CPU + GPU 协同执行的 LZ4 主实现。该程序不是简单包装器，而是包含完整任务切分、并发调度、OpenCL 运行时管理、容器封装/回放、计时与验证的独立执行引擎。

本文档采用“可直接用于开发与复现实验”的完整说明方式，覆盖构建、运行、参数、内部结构、性能语义、验证规范和问题排查。

## 1. 模块定位与设计目标

`lz4_hybrid` 的目标是：

1. 在块级并行基础上，同时利用 `liblz4 + pthread`（CPU）与 `lz4_gpu` OpenCL 后端（GPU）；
2. 在统一容器格式下保证压缩/解压双向可回放；
3. 提供固定比例与自适应比例两套分工策略；
4. 以进程内预热基准（`--bench`）输出稳定可复现的吞吐数据；
5. 在保持正确性的前提下持续迭代执行路径开销。

## 2. 构建说明

### 2.1 Linux

先构建 GPU 后端，再构建 Hybrid：

```bash
make -C ../lz4_gpu
make
```

依赖要求：

- C 编译器（`gcc`/`clang`）
- OpenCL 头文件与加载器（如 Debian/Ubuntu 的 `opencl-headers` + `ocl-icd-opencl-dev`）
- `pthread`

### 2.2 Windows（MSYS2 / MinGW-w64）

Windows 受支持构建路径为 **MSYS2 MinGW-w64**。先构建 `lz4_gpu`，再构建 `lz4_hybrid`：

```bash
make -C ../lz4_gpu \
  OS=Windows_NT CC=gcc \
  OPENCL_INCLUDE_DIR="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include" \
  OPENCL_LIB_DIR="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/lib/x64"

make OS=Windows_NT CC=gcc \
  OPENCL_INCLUDE_DIR="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include" \
  OPENCL_LIB_DIR="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/lib/x64"
```

补充说明：

- 如导入库命名不同，可通过 `OPENCL_LIB_NAME` 覆盖；
- 当前 Windows 支持重点是 standalone 与 benchmark 路径；
- 守护进程/客户端等 Unix 机制仅在 Linux 路径下有效。

## 3. 运行与参数

### 3.1 基本命令

```bash
./lz4_hybrid input.bin -o out.lz4h
./lz4_hybrid -d out.lz4h -o restored.bin
```

### 3.2 关键参数（完整）

- `-b`, `--block-size`：块大小（`16K`、`32K`、`64K` 等，当前默认 `64K`）
- `-T`, `--cpu-threads`：CPU 工作线程数（默认自动探测可用核数）
- `--gpu-ratio`：固定 GPU 块比例（`0.0~1.0`）
- `--adaptive`：启用自适应分配
- `--sample-blocks`：自适应采样块数
- `-a`, `--acceleration`：LZ4 acceleration 参数
- `-l`, `--local`：OpenCL local work-group size
- `--bench N`：在同一进程内重复运行 `N` 轮（预热后取稳定统计）

### 3.3 基准示例

```bash
./lz4_hybrid --bench 3 -b 64K -T 2 -a 3 --gpu-ratio 0.7 /path/to/file
./lz4_hybrid --bench 3 -b 64K -T 2 -a 3 --adaptive --sample-blocks 8 /path/to/file
```

## 4. 容器格式与数据路径

压缩输出容器包含：

1. magic 与版本信息；
2. 总块数、块大小、GPU 块数等调度元信息；
3. 每块压缩长度表；
4. GPU/CPU 两路压缩数据段。

解压阶段严格依据头部与长度表回放，保证跨轮次、跨参数的一致可恢复性。

## 5. 执行架构（实现级）

### 5.1 CPU 路径

- 基于 `liblz4 + pthread`；
- 线程按块区间并行处理；
- 压缩调用链对应 `LZ4_compress_*` 系列，解压采用安全边界检查路径；
- 线程数可通过 `-T` 显式约束，也可自动探测。

### 5.2 GPU 路径

- 基于 `lz4_gpu` OpenCL 后端；
- 维护可复用 workspace，避免循环内高频创建/释放缓冲；
- 内核参数与缓冲重用策略用于降低 steady-state dispatch 成本；
- 统一内存平台可获得更低传输路径开销。

### 5.3 协同路径

- 任务在块级分配到 CPU 与 GPU；
- 两路并发执行，整体时延由较慢路径主导；
- 合并阶段写回统一容器，供后续解压一致回放。

## 6. 自适应调度模型（完整描述）

`--adaptive` 的调度逻辑包含吞吐、负载与能耗三个维度。

1. **设备基线校准**：启动阶段估计 CPU/GPU 的单位吞吐与固定启动开销；
2. **数据特征修正**：通过采样估计当前输入的可压缩性影响；
3. **负载修正**：结合系统负载估计有效可用算力；
4. **比例求解**：在总工期近似最小化目标下生成 `effective_gpu_ratio`；
5. **保护逻辑**：小输入时优先回退 CPU 以避免 GPU 固定开销主导。

该模型默认采用确定性策略，优先保证验证空间稳定和跨轮次复现。

## 7. 计时语义与结果解读

### 7.1 Kernel Throughput

Kernel 吞吐依据 CPU 与 GPU 的实际执行跨度计算，反映编解码核心负载能力。

### 7.2 Total Throughput

Total 吞吐采用进程内预热后的整体 wall-time，覆盖运行时与协调成本，适合做工程对比与回归门禁。

### 7.3 正确性优先级

所有性能结论默认建立在 roundtrip 验证通过的前提下；若正确性失败，性能数据视为无效。

## 8. 验证与回归规范

推荐最小闭环：

1. 压缩与解压各执行多轮 `--bench`；
2. 每轮校验解压结果与原文件一致；
3. 统计 comp/dec 吞吐均值、中位、分位；
4. 对比 ratio 变化；
5. 保存 artifact 路径与二进制哈希。

## 9. 常见问题与排查

1. **OpenCL 初始化失败**：确认驱动、平台与设备可见；
2. **吞吐波动过大**：检查是否混入冷启动与后台负载；
3. **多线程收益异常**：核对 `-T` 与系统 CPU 绑定策略；
4. **结果不可复现**：固定输入集、二进制哈希和运行顺序。

## 10. 当前版本结论

- `lz4_hybrid` 已具备完整 CPU/GPU 协同能力；
- 固定比例与自适应比例均可用于正式验证；
- 默认执行路径以可复现、可回归、可解释为优先目标；
- 文档与实现保持一致，后续迭代以“正确性不退化 + 口径不漂移”为硬约束。
