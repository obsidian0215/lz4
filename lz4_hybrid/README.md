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
- `-T`, `--cpu-threads`: CPU worker count
- `--gpu-ratio`: fixed GPU block fraction
- `--adaptive`: enable adaptive split selection
- `--sample-blocks`: adaptive sample count
- `-a`, `--acceleration`: LZ4 acceleration factor
- `-H`, `--hash`: GPU hash log
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

## 2026-03 benchmark refresh

The latest full-corpus hybrid result is:

- `exp_results/hybrid_bench/hybrid_bench_20260309_180949.csv`

Recommended companion baseline for endpoint comparison:

- `../exp_results/runs/20260309_merged_full_83/lz4_param_sweep_merged.csv`

Current matched 83-file highlights:

- raw medians: fixed `919.72 / 675.42 MB/s`, adaptive `889.91 / 674.58 MB/s`
- best-per-file medians: fixed `1425.90 / 802.90 MB/s`, adaptive `1302.24 / 813.20 MB/s`
- winner counts: fixed wins `48/83` files on compression total, while GPU still dominates decompression with `62/83`

Interpretation: fixed hybrid is now a real compression competitor on the full corpus, but GPU remains the safer default endpoint, especially for decompression.
