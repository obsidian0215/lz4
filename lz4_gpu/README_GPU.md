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
- `-H`, `--hash`: hash log bits
- `-l`, `--local`: local work-group size
- `--bench [SECONDS]`: warmed stable benchmark loop

### Device selection

```bash
FORCE_OPENCL_DEVICE=GPU ./lz4_gpu --bench 3 -b 64K file
FORCE_OPENCL_DEVICE=CPU ./lz4_gpu --bench 3 -b 64K file
```

Accepted values: `GPU`, `CPU`, `DEFAULT`, `ALL`.

## Current implementation notes

- Recent fixes corrected the 64KB `tableType==0` dict sizing/mask mismatch.
- The runtime uses reusable OpenCL buffers and steady-state bench loops to separate kernel throughput from delivered total throughput.
- On current verified subset workloads, 64KB is no longer showing a universal throughput collapse after the latest fixes.
