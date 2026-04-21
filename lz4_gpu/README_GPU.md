# lz4_gpu

`lz4_gpu` is the OpenCL-based GPU execution path for LZ4 in this repository.

It includes:

- standalone compression/decompression
- benchmark mode with kernel and total throughput reporting
- optional daemon/client execution on Linux
- reusable OpenCL runtime and kernel workspace management

## Build

### Linux

```bash
make
```

Dependencies:

- C compiler (`gcc` or `clang`)
- OpenCL headers and ICD loader
- pthread support

### Windows (MSYS2 / MinGW-w64)

Supported Windows build target is **MSYS2 MinGW-w64**.

Example with NVIDIA CUDA-provided OpenCL headers/libs:

```bash
make OS=Windows_NT CC=gcc \
  OPENCL_INCLUDE_DIR="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include" \
  OPENCL_LIB_DIR="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/lib/x64"
```

If your import library name differs, also pass:

```bash
OPENCL_LIB_NAME=OpenCL
```

Notes:

- CUDA on Windows ships OpenCL headers and import libraries; the runtime `OpenCL.dll` is supplied by the installed GPU driver.
- Windows builds currently support **standalone** and **benchmark** modes.
- `--daemon`, `--use-daemon`, and `--stop-daemon` are Linux-only because the current implementation uses Unix-domain sockets and POSIX process management.

## Usage

```bash
./lz4_gpu input.bin -o out.lz4
./lz4_gpu -d out.lz4 -o restored.bin
./lz4_gpu --bench 3 -b 64K /path/to/file
```

### Important options

- `-b`, `--block-size`: GPU block size (`16K`, `32K`, `64K`, ...)
- `-a`, `--acceleration`: compression acceleration
- `--local`: local work-group size
- `--bench [N]`: warmed stable benchmark loop (N seconds, default 3)

### Device selection

```bash
FORCE_OPENCL_DEVICE=GPU ./lz4_gpu --bench 3 -b 64K file
FORCE_OPENCL_DEVICE=CPU ./lz4_gpu --bench 3 -b 64K file
```

Accepted values: `GPU`, `CPU`, `DEFAULT`, `ALL`.

### Host-memory copy mode

```bash
LZ4_STANDARD_COPY=0 ./lz4_gpu --bench 3 -b 64K file
LZ4_STANDARD_COPY=1 ./lz4_gpu --bench 3 -b 64K file
```

- `0`: map/zero-copy 优先（统一内存设备常用）
- `1`: standard host->device copy

当前正式 host 结论（2026-04-16 Intel all-off 组合矩阵）：

- `mapped` 是唯一稳定、可解释、值得保留的 host 特性；
- 若需要做 host A/B，对比口径只保留 `LZ4_STANDARD_COPY=0/1`。

### Environment variables（当前保留）

| 变量 | 取值 / 默认 | 作用 |
| --- | --- | --- |
| `FORCE_OPENCL_DEVICE` | `GPU`(默认) / `CPU` / `DEFAULT` / `ALL` | 指定 OpenCL 设备优先级 |
| `LZ4_STANDARD_COPY` | `auto`(默认) / `0` / `1` | host 与 device 之间的数据读写方式 |
| `LZ4_GPU_COMP_WI_PER_CU` | 正整数（默认自动） | 覆盖压缩 worker 并发目标 |
| `LZ4_GPU_DECOMP_WI_PER_CU` | 正整数（默认自动） | 覆盖解压 worker 并发目标 |
| `LZ4_GPU_WI_PER_CU` | 正整数（默认空） | 压缩/解压共享的 fallback 并发目标 |
| `LZ4_GPU_DISABLE_DECOMP_MAPPED_WRITE` | `0/1`（默认 `0`） | 关闭解压输出的 mapped 直写路径 |
| `LZ4_GPU_BENCH_WARMUP_ROUNDS` | 正整数（默认 `1`） | 控制 bench 计时前的 warmup 轮数 |

建议：实验报告至少记录 `FORCE_OPENCL_DEVICE`、`LZ4_STANDARD_COPY` 与所有显式设置的 `WI_PER_CU` 开关。

## Current implementation notes

- Recent fixes corrected the 64KB `tableType==0` dict sizing/mask mismatch.
- The runtime uses reusable OpenCL buffers and steady-state bench loops to separate kernel throughput from delivered total throughput.
- Host side is now intentionally simpler: keep `mapped` as the default unified-memory path, and avoid reintroducing retired host branches without new evidence.
- On current verified subset workloads, 64KB is no longer showing a universal throughput collapse after the latest fixes.
