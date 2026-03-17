# lz4_hybrid

`lz4_hybrid` is the CPU+GPU collaborative LZ4 path in this repo. It combines:

- CPU compression/decompression via `liblz4 + pthread`
- GPU compression/decompression via the `lz4_gpu` OpenCL backend
- fixed or adaptive CPU/GPU block splitting
- warmed in-process benchmarking via `--bench` and `--bench-io`

## Build

### Linux

```bash
make -C ../lz4_gpu
make
```

Dependencies:

- C compiler (`gcc` or `clang`)
- OpenCL development headers and loader (`opencl-headers`, `ocl-icd-opencl-dev` on Debian/Ubuntu)
- pthread support

### Windows (MSYS2 / MinGW-w64)

Supported Windows build target is **MSYS2 MinGW-w64**.

Install a MinGW shell and compiler first, then point the build at OpenCL:

```bash
make -C ../lz4_gpu \
  OS=Windows_NT CC=gcc \
  OPENCL_INCLUDE_DIR="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include" \
  OPENCL_LIB_DIR="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/lib/x64"

make OS=Windows_NT CC=gcc \
  OPENCL_INCLUDE_DIR="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include" \
  OPENCL_LIB_DIR="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/lib/x64"
```

Notes:

- NVIDIA CUDA installs OpenCL headers and `OpenCL.lib`; MinGW can link if the import library is provided in a compatible path/name.
- If your environment uses a MinGW import library with a different name, override `OPENCL_LIB_NAME`.
- Windows builds target the standalone and benchmark paths. Unix daemon/client features are Linux-only.

## Usage

```bash
./lz4_hybrid input.bin -o out.lz4h
./lz4_hybrid -d out.lz4h -o restored.bin
```

### Common options

- `-b`, `--block-size`: block size such as `16K`, `32K`, `64K`
- `-T`, `--cpu-threads`: CPU worker count (default: auto = all cores via `sysconf`)
- `--gpu-ratio`: fixed GPU block fraction
- `--adaptive`: enable adaptive split selection
- `--sample-blocks`: adaptive sample count
- `-a`, `--acceleration`: LZ4 acceleration factor
- `-l`, `--local`: OpenCL local work-group size

### Benchmarks

```bash
./lz4_hybrid --bench 3 --bench-io -b 64K -T 2 -a 3 --gpu-ratio 0.7 /path/to/file
./lz4_hybrid --bench 3 --bench-io -b 64K -T 2 -a 3 --adaptive --sample-blocks 8 /path/to/file
```

`--bench-io` measures warmed in-process end-to-end throughput including file write/read paths.

## Current implementation notes

- The current file format records the total block count, block size, GPU block count, and per-block compressed sizes.
- The current LZ4 hybrid container still assumes a GPU-first front segment in the compressed payload layout, so arbitrary GPU block permutations are not yet a format-compatible optimization.
- Recent fixes include repeated-bench correctness, GPU workspace reuse, distributed adaptive sampling, and broader 16K/32K/64K benchmark coverage.
- Bench loop optimization: GPU kernel args cached across iterations; input upload skipped on repeated iterations.
- Thread auto-detection: defaults to all available cores via `sysconf(_SC_NPROCESSORS_ONLN)`, overridable via `-T`.

## Adaptive scheduling model

The adaptive split model (`--adaptive`) is energy-aware, load-aware, and compute-resource-aware.

- **Throughput model**: Calibrates per-byte CPU throughput (Pc0) and GPU throughput (Pg0) at startup.
- **Compression-ratio gain** (gC/gG): Adjusts for actual vs reference compression ratio.
- **Load awareness**: Reads `/proc/stat` for CPU idle fraction, scales by thread count vs total cores.
- **GPU availability**: Monitors GPU utilization.
- **Compute-resource awareness**: CPU capacity = `Pc0 * threads * cpu_availability`; thread count is the CPU capacity bound.
- **Energy-aware correction**: Measures per-byte energy via RAPL core domain (`intel-rapl:0:0`) for CPU and uncore domain (`intel-rapl:0:1`) for GPU during calibration. Final ratio = 70% throughput-optimal + 30% energy-optimal blend.
- **Small-input guard**: Routes to CPU-only if input smaller than GPU overhead.
- **Degenerate fallback**: Returns 0.5 when both effective throughputs are zero (decoupled from user-specified gpu_ratio).
