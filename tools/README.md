# tools

本目录放 LZ4 测试、功耗/频率 wrapper、硬件遥测和控制脚本。主测试入口是 `bench_lz4.py`；功耗和频率扫描由外部 wrapper 调用。

## 脚本职责

| 脚本 | 职责 | 输出 |
| --- | --- | --- |
| `bench_lz4.py` | 压缩/解压性能测试；默认 `1` 轮 bench + `6` 轮真实压缩/解压 | `raw.csv`、`per_file_summary.csv`、`aggregate.csv` |
| `bench_lz4_power_wrapper.py` | 外部频率点循环、功耗/频率采样、调用 `bench_lz4.py` | `power_frequency_summary.csv`、`per_file_power_summary.csv` |
| `hw_telemetry.py` | CPU/GPU/DRAM 功耗与频率采样接口 | 被 wrapper 调用 |
| `cpu_control.sh` | Linux CPU 频率/核心/模式控制和 reset | 被 wrapper 调用 |
| `gpu_control.sh` | Linux GPU 频率/功耗/模式控制和 reset | 被 wrapper 调用 |

## `bench_lz4.py`

默认样本目录是仓库同级的 `samples`。本机 GPU 路径不稳定时不要运行 GPU/daemon 测试，转到远端 Linux 机器执行。

常用调用：

```bash
python3 tools/bench_lz4.py \
  --platform-id linux_xe \
  --samples /root/samples \
  --engines gpu,native_cpu \
  --gpu-block-sizes 64K \
  --gpu-hashlogs 14 \
  --bench-seconds 5 \
  --manual-rounds 6
```

输出字段：

- `comp_mbs_*` / `dec_mbs_*`：压缩/解压主吞吐；优先来自真实 manual 阶段的 kernel/span 字段，缺失时回退到 bench 阶段。
- `e2e_comp_mbs_*` / `e2e_dec_mbs_*`：端到端吞吐；真实压缩/解压路径，OpenCL 路径排除 OpenCL init/build。
- `ratio_pct_*`：压缩率，数值越低表示压缩率越好。
- `verify_ok` / `verify_all`：解压 hash 校验结果。

`raw.csv` 保留底层解析字段，例如 `bench_comp_kernel_mbs`、`manual_comp_kernel_mbs` 和 `manual_comp_no_ocl_mbs`，用于诊断；结论和汇总表优先使用 `per_file_summary.csv` / `aggregate.csv` 的精简字段。

## `bench_lz4_power_wrapper.py`

该脚本负责频率点和功耗采样，不把功耗字段写回 `bench_lz4.py` 的三张核心表。

示例：

```bash
python3 tools/bench_lz4_power_wrapper.py \
  --platform-id linux_xe \
  --cpu-frequencies 2100,3400,NA \
  --gpu-frequencies 1000,NA \
  --output-dir exp_results/power_freq_runs/linux_xe_scan \
  -- \
  --engines gpu,native_cpu \
  --samples /root/samples \
  --bench-seconds 5 \
  --manual-rounds 6
```

频点规则：

- `NA` / `NONE` / `-`：不限制该设备，并先 reset。
- 数字或 `MHz`：固定绝对 MHz。
- 百分比：相对设备可读最大频率换算。
- GPU-only 只扫描 GPU 频率，CPU-only 只扫描 CPU 频率，hybrid mixed 才扫描 CPU×GPU 矩阵。
