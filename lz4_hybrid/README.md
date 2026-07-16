# lz4_hybrid

`lz4_hybrid` 现在使用 `lz4_gpu_v2` 的 OpenCL mixed 基础实现：CPU-only、GPU-only、CPU+GPU mixed 都运行同一套 `lz4_gpu.cl` kernel。旧版 native CPU pthread + GPU split 实现已经废弃，不再作为后续路线。

当前路线里，字典规模与并发度是分开的两个维度：缩小字典不通过减少 work-item / slots 实现。`lz4_hybrid` 已接入 `--d-bits 11..15`，该参数只改变每个 OpenCL CPU/GPU lane 的 hash table 宽度；GPU block 数、CPU block 数和 CPU slots 仍由 `--gpu-ratio` / `--cpu-threads` 控制。后续自适应需要继续把“设备比例、设备块大小、字典位宽”分开建模。自适应块大小不应只是选择一个全局固定块大小，而应允许 CPU 与 GPU 使用不同块大小。不要再把 slot pool 当作默认缩字典方案。

## 构建

```bash
make
```

Windows/MSYS2 示例：

```bash
mingw32-make
```

默认目标生成：

- `lz4_hybrid` / `lz4_hybrid.exe`
- `build_clbin` / `build_clbin.exe`

`make precompile` 可显式生成 `lz4_gpu_14.clbin`。默认运行时仍可从源码构建 kernel；调试或跨设备测试时应记录是否存在 `.clbin`。

## 运行

GPU-only：

```bash
./lz4_hybrid -c -B 64K --gpu-ratio 1 -o out.lz4 input
./lz4_hybrid -d -o restored out.lz4
```

OpenCL CPU-only：

```bash
./lz4_hybrid -c -B 64K --gpu-ratio 0 --cpu-threads 1 -o out.lz4 input
./lz4_hybrid -d --gpu-ratio 0 --cpu-threads 1 -o restored out.lz4
```

字典位宽扫描：

```bash
./lz4_hybrid -c -B 64K --gpu-ratio 1 --d-bits 13 -o out.lz4 input
./lz4_hybrid -c -B 64K --gpu-ratio 0 --cpu-threads 1 --d-bits 15 -o out.lz4 input
```

固定比例 mixed：

```bash
./lz4_hybrid -c -B 64K --gpu-ratio 0.5 --cpu-threads 1 -o out.lz4 input
./lz4_hybrid -d --gpu-ratio 0.5 --cpu-threads 1 -o restored out.lz4
```

`--adaptive` / `--gpu-ratio adaptive` 会自动选择 `gpu_ratio` 与分设备块大小。当前保守策略是默认 `64KB`，大文件才考虑 GPU `32KB`、CPU `64KB`；中小文件不应自动切到更细分块，因为 105 上它会带来明显压缩率回退。压缩输出在 split v2 容器头中记录 GPU/CPU block 数和各自 block size，解压时可自动识别 split 容器，不需要再次指定 `--gpu-ratio`。

Linux daemon 路径：

```bash
./lz4_hybrid --daemon
./lz4_hybrid --use-daemon -c -B 64K --gpu-ratio 0.5 --cpu-threads 1 -o out.lz4 input
./lz4_hybrid --use-daemon -d --gpu-ratio 0.5 --cpu-threads 1 -o restored out.lz4
./lz4_hybrid --stop-daemon
```

daemon mixed 请求复用 OpenCL CPU/GPU context、queue、program、kernel 和主要 device buffer；mixed 请求在 daemon 内串行化执行，避免共享 kernel 参数竞争。

## 当前实现边界

- `--gpu-ratio 1`：全部 block 交给 OpenCL GPU。
- `--gpu-ratio 0`：全部 block 交给 OpenCL CPU。
- `0 < --gpu-ratio < 1`：按连续 byte range 切分到 GPU/CPU，再由两侧各自 block size 分块。
- `--cpu-threads N`：限制 OpenCL CPU device 的 worker/slot 数。
- `--d-bits N`：限制每个 lane 的 hash table 宽度，不改变 GPU/CPU 的并发上限。
- split v2 容器记录 `gpu_blocks/cpu_blocks/gpu_block_size/cpu_block_size`，压缩输出可由同实现自动识别并解压回放。

## 已验证

已做 smoke：

- 样本：`xml`、`mozilla`、`x-ray`
- 块大小：Windows standalone 覆盖 `32KB`、`64KB`；225 daemon 覆盖 `64KB`
- 模式：`--gpu-ratio 1`、`0`、`0.5`、`adaptive`
- CPU slots：`--cpu-threads 1`
- 结果：压缩/解压 roundtrip 全部通过，临时压缩产物已删除。
- 225 daemon 复用确认：首次 CPU-only 初始化 CPU context/kernel；首次 mixed 初始化 GPU context/kernel；后续同 daemon 进程内 `init_load=0.00ms`。
- 105 `--d-bits` smoke：`ooffice, 64KB, D13/D15, gpu/cpu/mixed` roundtrip 全通过；输出中 `d_bits=` 与命令一致，说明字典位宽已进入 split 路径。
- 105 adaptive smoke：`ooffice` roundtrip 通过；修正后的保守策略在该文件上保持 `64KB/64KB`，避免早先 `32KB/64KB` 引起压缩率明显变差。

## 后续工作

1. 做 CPU/GPU ratio 扫描，分别统计 kernel、no-ocl-init、真实端到端吞吐。
2. 根据扫描结果把 adaptive 改成“选择最优比例”，允许选择 GPU-only；当前 105 daemon 全样本扫描显示强行 mixed 不如 GPU-only。
3. 保留分设备块大小结构，但默认仍以 `64KB` 为主；只有在明确的 profile 条件下才考虑 GPU `32KB`、CPU `64KB`。
4. 用 105 全样本验证 adaptive 是否同时不低于 GPU-only 和 CPU-only/1-thread。
