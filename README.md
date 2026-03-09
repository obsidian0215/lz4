LZ4 - Extremely fast compression
================================

LZ4 is lossless compression algorithm,
providing compression speed > 500 MB/s per core,
scalable with multi-cores CPU.
It features an extremely fast decoder,
with speed in multiple GB/s per core,
typically reaching RAM speed limits on multi-core systems.

Speed can be tuned dynamically, selecting an "acceleration" factor
which trades compression ratio for faster speed.
On the other end, a high compression derivative, LZ4_HC, is also provided,
trading CPU time for improved compression ratio.
All versions feature the same decompression speed.

LZ4 is also compatible with [dictionary compression](https://github.com/facebook/zstd#the-case-for-small-data-compression),
both at [API](https://github.com/lz4/lz4/blob/v1.8.3/lib/lz4frame.h#L481) and [CLI](https://github.com/lz4/lz4/blob/v1.8.3/programs/lz4.1.md#operation-modifiers) levels.
It can ingest any input file as dictionary, though only the final 64KB are used.
This capability can be combined with the [Zstandard Dictionary Builder](https://github.com/facebook/zstd/blob/v1.3.5/programs/zstd.1.md#dictionary-builder),
in order to drastically improve compression performance on small files.


LZ4 library is provided as open-source software using BSD 2-Clause license.


|Branch      |Status   |
|------------|---------|
|dev         | [![Build status][AppveyorDevBadge]][AppveyorLink]  |

[AppveyorDevBadge]: https://ci.appveyor.com/api/projects/status/github/lz4/lz4?branch=dev&svg=true "Windows test suite"
[AppveyorLink]: https://ci.appveyor.com/project/YannCollet/lz4-1lndh


Benchmarks
-------------------------

The benchmark uses [lzbench], from @inikep
compiled with GCC v8.2.0 on Linux 64-bits (Ubuntu 4.18.0-17).
The reference system uses a Core i7-9700K CPU @ 4.9GHz (w/ turbo boost).
Benchmark evaluates the compression of reference [Silesia Corpus]
in single-thread mode.

[lzbench]: https://github.com/inikep/lzbench
[Silesia Corpus]: http://sun.aei.polsl.pl/~sdeor/index.php?page=silesia

|  Compressor             | Factor  | Compression | Decompression |
|  ----------             | -----   | ----------- | ------------- |
|  memcpy                 |  1.000  | 13700 MB/s  |  13700 MB/s   |
|**LZ4 default (v1.9.0)** |**2.101**| **780 MB/s**| **4970 MB/s** |
|  LZO 2.09               |  2.108  |   670 MB/s  |    860 MB/s   |
|  QuickLZ 1.5.0          |  2.238  |   575 MB/s  |    780 MB/s   |
|  Snappy 1.1.4           |  2.091  |   565 MB/s  |   1950 MB/s   |
| [Zstandard] 1.4.0 -1    |  2.883  |   515 MB/s  |   1380 MB/s   |
|  LZF v3.6               |  2.073  |   415 MB/s  |    910 MB/s   |
| [zlib] deflate 1.2.11 -1|  2.730  |   100 MB/s  |    415 MB/s   |
|**LZ4 HC -9 (v1.9.0)**   |**2.721**|    41 MB/s  | **4900 MB/s** |
| [zlib] deflate 1.2.11 -6|  3.099  |    36 MB/s  |    445 MB/s   |

[zlib]: http://www.zlib.net/
[Zstandard]: http://www.zstd.net/


Installation
-------------------------

```
make
make install     # this command may require root permissions
```

LZ4's `Makefile` supports standard [Makefile conventions],
including [staged installs], [redirection], or [command redefinition].
It is compatible with parallel builds (`-j#`).

[Makefile conventions]: https://www.gnu.org/prep/standards/html_node/Makefile-Conventions.html
[staged installs]: https://www.gnu.org/prep/standards/html_node/DESTDIR.html
[redirection]: https://www.gnu.org/prep/standards/html_node/Directory-Variables.html
[command redefinition]: https://www.gnu.org/prep/standards/html_node/Utilities-in-Makefiles.html

### Building LZ4 - Using vcpkg

You can download and install LZ4 using the [vcpkg](https://github.com/Microsoft/vcpkg) dependency manager:

    git clone https://github.com/Microsoft/vcpkg.git
    cd vcpkg
    ./bootstrap-vcpkg.sh
    ./vcpkg integrate install
    ./vcpkg.exe install lz4

The LZ4 port in vcpkg is kept up to date by Microsoft team members and community contributors. If the version is out of date, please [create an issue or pull request](https://github.com/Microsoft/vcpkg) on the vcpkg repository.

Documentation
-------------------------

The raw LZ4 block compression format is detailed within [lz4_Block_format].

Arbitrarily long files or data streams are compressed using multiple blocks,
for streaming requirements. These blocks are organized into a frame,
defined into [lz4_Frame_format].
Interoperable versions of LZ4 must also respect the frame format.

[lz4_Block_format]: doc/lz4_Block_format.md
[lz4_Frame_format]: doc/lz4_Frame_format.md


Other source versions
-------------------------

Beyond the C reference source,
many contributors have created versions of lz4 in multiple languages
(Java, C#, Python, Perl, Ruby, etc.).
A list of known source ports is maintained on the [LZ4 Homepage].

[LZ4 Homepage]: http://www.lz4.org

### Packaging status

Most distributions are bundled with a package manager
which allows easy installation of both the `liblz4` library
and the `lz4` command line interface.

[![Packaging status](https://repology.org/badge/vertical-allrepos/lz4.svg?columns=4&exclude_unsupported=1)](https://repology.org/project/lz4/versions)


### Special Thanks

- Takayuki Matsuoka, aka @t-mat, for exceptional first-class support throughout the lifetime of this project

# lz4 repository with GPU and hybrid extensions

This tree contains the upstream LZ4 code plus local GPU and CPU+GPU hybrid implementations used for heterogeneous compression experiments.

## Main subprojects

- `programs/`, `lib/`: upstream CPU LZ4 implementation
- `lz4_gpu/`: OpenCL GPU execution path
- `lz4_hybrid/`: CPU+GPU collaborative path
- `tools/`: benchmark drivers and analysis helpers

## Build

### Linux

CPU LZ4:

```bash
make -C programs lz4
```

GPU path:

```bash
make -C lz4_gpu
```

Hybrid path:

```bash
make -C lz4_hybrid
```

Typical Linux packages:

- Debian/Ubuntu: `build-essential opencl-headers ocl-icd-opencl-dev`
- Fedora: `gcc make opencl-headers ocl-icd-devel`

### Windows

Supported Windows build target is **MSYS2 / MinGW-w64**.

This is the supported path because:

- `lz4_hybrid` depends on pthreads
- the benchmark and runtime code are GCC/Make oriented
- it provides the least invasive portability story for the current codebase

Install MSYS2 packages first:

```bash
pacman -S --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-make mingw-w64-ucrt-x86_64-winpthreads
```

Then build with CUDA-provided OpenCL headers/libs (example path):

```bash
make -C lz4_gpu OS=Windows_NT CC=gcc \
  OPENCL_INCLUDE_DIR="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include" \
  OPENCL_LIB_DIR="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/lib/x64"

make -C lz4_hybrid OS=Windows_NT CC=gcc \
  OPENCL_INCLUDE_DIR="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include" \
  OPENCL_LIB_DIR="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/lib/x64"
```

Notes:

- NVIDIA CUDA on Windows includes OpenCL headers and import libraries.
- The runtime `OpenCL.dll` is supplied by the installed GPU driver.
- If you use a different SDK/vendor, set `OPENCL_INCLUDE_DIR`, `OPENCL_LIB_DIR`, and optionally `OPENCL_LIB_NAME` accordingly.
- Daemon/client mode in `lz4_gpu` is Linux-only today.

## Benchmarks

Representative runners:

- `tools/bench_lz4.py`: CPU vs GPU benchmark driver
- `tools/bench_hybrid.py`: hybrid parameter sweeps across 16K/32K/64K

Examples:

```bash
python3 tools/bench_lz4.py --samples-dir /root/samples_subset --gpu-only
python3 tools/bench_hybrid.py --samples-dir /root/samples_subset --bench-seconds 1
```

## Current status

- `lz4_gpu` is the default throughput leader in the current full verified corpus.
- `lz4_hybrid` now has corrected benchmark semantics, fixed repeated-bench validation, wider block-size coverage, and improved runtime reuse.
- On targeted workloads, `lz4_hybrid` can now exceed `lz4_gpu`, but a new full rerun is still needed before claiming a system-wide ranking change.

## 2026-03 full-corpus refresh

Fresh artifacts now exist for the full 83-file `/root/samples` corpus and should be preferred over older summaries:

- CPU/GPU stitched artifact: `exp_results/runs/20260309_merged_full_83/lz4_param_sweep_merged.csv`
- Hybrid full sweep: `exp_results/hybrid_bench/hybrid_bench_20260309_180949.csv`
- Cross-family analysis bundle: `/root/analysis/20260309_full_refresh/`

Current matched-corpus best-per-file medians:

- `LZ4 CPU`: `698.71 MB/s` compress total, `755.68 MB/s` decompress total
- `LZ4 GPU`: `1497.76 MB/s` compress total, `1085.39 MB/s` decompress total
- `LZ4 Hybrid fixed`: `1425.90 MB/s` compress total, `802.90 MB/s` decompress total
- `LZ4 Hybrid adaptive`: `1302.24 MB/s` compress total, `813.20 MB/s` decompress total

Important scope note: cross-algorithm claims in this repository now refer to the **matched 83-file corpus** and the stated parameter matrices only; they do not imply that LZ4 settings are semantically equivalent to LZO settings.
