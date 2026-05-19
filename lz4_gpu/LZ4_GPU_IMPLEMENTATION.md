# LZ4 GPU 当前实现说明

更新时间：2026-05-19

本文只描述 `lz4_gpu` 当前主线已经启用的实现，不展开被拒绝的历史变体。后续优化记录仍放在 `lz4_experimental`，只有经过全样本真实路径验证且明确采纳的改动才迁移到这里。

## 1. 当前主线范围

当前实现覆盖三条路径：

- `lz4_gpu` standalone：命令行直接压缩、解压、bench。
- `lz4_gpu_daemon`：常驻 OpenCL context/queue/program/kernel/workspace，并通过 Unix socket 接收请求。
- `lz4_gpu_client`：向 daemon 发送路径请求或 `--raw-buffer` 标准输入/输出请求。

当前主线已迁移的第一轮有效改动：

| 类别 | 已采纳改动 | 作用 |
|---|---|---|
| 压缩字典 | `block_size <= 64KB` 默认 `clear16` | 16-bit block-local offset 足够表示候选位置，字典 entry 减半 |
| 字典清零 | `uint4` 向量化清表 | 降低 clear16 每 block 清表的写出循环开销 |
| hash 路径 | 固定 `HASHLOG=14`，删除旧 `HASHLOG+1/tableType` 宽表分支 | 统一 GPU 分块字典口径，避免旧 CPU 单流表宽逻辑污染 |
| hash 输入 | `LZ4_hashPosition()` 固定写回 sequence，不保留 nullable 分支 | 去掉 hot path 上无意义的运行时判断 |
| literal copy | 删除 `LZ4_GPU_DISABLE_VEC_COPY` scalar fallback | 主线只保留向量化 copy，避免代码路径分裂 |
| 编译路径 | standalone/bench/daemon 统一调用 `lz4_load_program()` | daemon 与普通路径使用同一套 build flag 和 dict mode |
| daemon 协议 | 增加 request magic/version 与 `RAW_BUFFER` | 防止新旧 client/server 混用，并支持 raw-buffer 桥接 |

没有迁移的项目：

- `HASHLOG=13` 默认化：全样本有轻微吞吐收益，但压缩率增加且文件差异明显，只保留为未来 profile/adaptive 候选。
- `sig8/hsig8` 候选过滤：没有形成稳定整体收益。
- 解压 small-offset 内联 copy、copy 分支收缩：子集或全样本未通过。
- gather/compaction/pipeline：不进入主线；bench-only 或真实路径无收益的 host 优化不采纳。

## 2. 压缩内核

### 2.1 调度模型

压缩入口是 `lz4_compress_block`。每个 active work-item 绑定一个私有 hash table slice，并通过跨步循环处理 block：

```c
for (uint b = wi; b < totalBlocks; b += total_wi, ++epoch) {
    ...
}
```

host 侧用 `lz4_build_comp_plan()` 生成执行计划：

```text
raw_worker_count   = min(num_blocks, max(local_size, CU * 24))
dict_owner_count   = min(raw_worker_count, dict_pool_budget / dict_bytes_per_owner)
active_lane_count  = dict_owner_count
launched_wi_count  = round_up(active_lane_count, local_size)
blocks_per_owner   = ceil(num_blocks / active_lane_count)
```

`dict_pool_budget` 默认取 `global_mem / 32`，并限制在 `16MB..512MB`。这个预算只控制字典池规模，不改变输出 block 顺序。

### 2.2 字典 entry 模式

`lz4_effective_dict_mode_for_block()` 根据 block size 选择字典模式：

```text
block_size <= 64KB: clear16，entry=ushort，dict_entries=1<<14，约 32KB/table
block_size >  64KB: epoch32，entry=uint，dict_entries=1<<14，约 64KB/table
```

`clear16` 的语义：

- 每个 block 开始前调用 `LZ4_clearDictEntries()` 清空当前 work-item 的表；
- entry 存 `idx + 1`，`0` 表示 empty；
- 16-bit entry 只在 `block_size <= 64KB` 安全，因为 block-local offset 不溢出；
- 压缩率不变，因为候选表达能力对 64KB 及以下 block 没有丢失。

`epoch32` 的语义：

- entry 使用 `[12-bit epoch | 20-bit position]`；
- 多 block 复用同一张表时靠 epoch 淘汰旧 entry；
- 当 epoch 低 12 bit 接近回绕时 host 清零整张 dict buffer。

### 2.3 hash 路径

当前只保留一个 GPU 分块 hash 口径：

```c
LZ4_HASHLOG = 14
dict_entries = 1 << LZ4_HASHLOG
hash = (sequence * 2654435761U) >> (32 - LZ4_HASHLOG)
```

旧实现中的 `tableType==0`、`HASHLOG+1`、`32768 entry / 128KB table` 不再是当前主线。当前默认 64KB block 在 `clear16` 下是 `16384 entry / 32KB table`。

### 2.4 match search 与输出

压缩核心仍保持 LZ4 语义：

1. 读 4B sequence；
2. 查 hash table 得到单候选；
3. 做 distance 和 4B equality 校验；
4. 用 `LZ4_count()` 扩展 match；
5. 写 literal、offset、match length 和 last literals。

`LZ4_count()` 当前采用 `16B -> 8B -> 4B -> 2B -> 1B` 的比较结构。宽比较不是无限加宽：继续扩大到 32/64B 会增加越界保护、临时向量和分支压力，只有诊断证明长 match 占主导时才允许作为变体验证。

## 3. 解压内核

解压入口是 `lz4_decompress_blocks`。每个 work-item 解一个 block，不需要压缩侧的大字典池。

host 侧并发上限：

```text
num_blocks >= 4096: CU * 48
otherwise:          CU * 96
```

解压 copy 当前已有：

- literal copy 使用 `LZ4_UA_COPYN()`；
- 非重叠 match (`offset >= len`) 走向量 copy；
- `offset == 1/2/4` 有重复模式专门路径；
- 普通 match copy 有 `64/32/16/8/4B` 分层。

注意：`offset < len` 是重叠 match，不能机械套用普通宽 copy。已有诊断显示问题不是 copy 宽度不够，而是分支链和不同 offset 分布造成的代码路径复杂度；未验证通过前不再增加更宽 copy 分支。

## 4. Host 路径

### 4.1 memory copy 策略

`lz4_prefers_standard_copy()` 根据设备类型选择默认路径：

- Intel/统一内存平台优先 mapped/zero-copy；
- 离散 GPU 可走 standard copy；
- `write_buffer_auto()` 和 `read_buffer_auto()` 封装 mapped 与 explicit copy。

真实路径指标必须区分：

- kernel throughput：只看 OpenCL kernel；
- end-to-end throughput：排除 OpenCL init 后的真实文件压缩/解压路径；
- ratio：以实际压缩输出大小计算。

### 4.2 输出组装

压缩 kernel 输出仍是固定槽位布局：

- `out_buf`：每 block 一个 `single_block_max_out` 槽；
- `output_size_buf`：每 block 实际压缩长度；
- host 侧读取 length table 后按 block 顺序组装有效 payload。

这不是 gather/compaction kernel。主线不采纳 bench-only 的 gather/compaction，因为它不能证明真实文件路径收益。

### 4.3 daemon

daemon 当前做的是真实有效的复用：

- 进程常驻；
- OpenCL context 常驻；
- worker 持有 queue、kernel、workspace；
- `program_comp[clear16/epoch32]` 按 block size lazily build；
- 请求级 `t.ocl_setup_us=0`，因为 init/build 已被 daemon 生命周期吸收。

daemon 与 standalone 使用同一个 `lz4_load_program(context, device, 14, block_size)`，因此 clear16/epoch32 的 build flag 保持一致。

`--raw-buffer` 协议用于 socket/pipe bridge：

1. client 从 stdin 读完整 payload；
2. request 设置 `LZ4_DAEMON_FLAG_RAW_BUFFER` 和 `input_size`；
3. daemon 接收 payload 后用 `memfd` 暴露为 `/proc/self/fd/N` 路径；
4. 复用现有 `lz4_compress_core()` / `lz4_decompress_core()`；
5. daemon 把输出长度和输出 payload 回传给 client。

当前 raw-buffer 仍通过 memfd 复用文件路径核心，优点是风险低、与 standalone 代码一致；后续如果要进一步降低小消息开销，应实现真正的 buffer-core 接口，而不是继续堆路径协议分支。

## 5. bench 脚本口径

`tools/bench_lz4.py` 默认生成：

- `raw.csv`：每轮原始数据；
- `per_file_summary.csv`：按文件/配置聚合；
- `aggregate.csv`：跨文件聚合。

默认轮次：

```text
bench:  1 轮，每轮 5 秒
manual: 6 轮真实压缩/解压
```

manual 产物写入统一临时目录，轮次结束后删除，不污染样本目录。功耗和频率扫描不写进 bench 主脚本，由外部 wrapper 生成独立 CSV。

## 6. 105 远端验证结果

测试平台：`192.168.2.105`，Intel Iris Xe OpenCL GPU，样本目录 `/root/samples`，共 26 个文件。

测试口径：

- 当前实现：`/root/lz4/lz4_gpu/lz4_gpu`
- 对照基线：`/root/lz4/lz4_gpu_baseline_/lz4_gpu`
- 配置：`64K` block，`HASHLOG=14`，`accel=1`，`local=1`
- 轮次：每配置 `1` 轮 bench，每文件 `6` 轮真实压缩/解压 manual
- 结果归档：`exp_results/remote105_gpu_compare_full6`

总体结果：

| 实现 | ratio 中位数 | 压缩 kernel 中位 | 解压 kernel 中位 | 端到端压缩中位 | 端到端解压中位 |
|---|---:|---:|---:|---:|---:|
| baseline | 34.775% | 828.00 MB/s | 3487.51 MB/s | 470.41 MB/s | 1038.58 MB/s |
| current | 34.838% | 914.47 MB/s | 3545.99 MB/s | 536.13 MB/s | 1037.34 MB/s |

文件级相对基线统计：

- 压缩 kernel：中位 `+10.74%`，平均 `+10.81%`，所有文件均为正向，最小 `+2.89%`、最大 `+23.23%`。
- 端到端压缩：中位 `+14.35%`，平均 `+14.92%`，所有文件均为正向，说明字典缩减和 hot path 简化在真实路径也有效。
- 解压 kernel：中位 `+1.42%`，平均 `+1.86%`，轻微正向。
- 端到端解压：中位 `-1.08%`，平均 `-1.12%`，Linux 真实输出路径没有因当前改动受益，且 chunked 默认关闭。
- ratio 数值：中位相对变化 `+0.264%`，平均 `+0.307%`，压缩率轻微变差，主要来自当前 LZ4 迁移后输出口径与候选路径差异，需要继续关注典型文本/结构化文件。

结论：LZ4 的 `clear16 + uint4 清表 + hot path 简化` 在 105 上同时提升压缩 kernel 和真实压缩路径，可以作为当前主线优化；解压端当前没有真实路径收益，不应继续追加未验证的 copy 分支。

最终 GPU 基线已在同一机器重新收集，结果归档到 `exp_results/gpu_baseline_105_final`。当前 64K/HASHLOG=14 基线为：

| ratio 中位数 | 压缩 kernel 中位 | 解压 kernel 中位 | 端到端压缩中位 | 端到端解压中位 |
|---:|---:|---:|---:|---:|
| 34.838% | 914.61 MB/s | 3545.35 MB/s | 540.35 MB/s | 1043.90 MB/s |

## 7. 后续优化边界

下一轮优化优先级：

1. 字典/state 与并发解耦：继续研究 table owner 数、dict footprint、occupancy 的稳定折中；
2. 压缩 probe 热路径：必须证明减少随机 table load、candidate 读取、failed compare 或 store 次数；
3. 解压分支简化：必须证明减少真实高频路径分支，而不是增加更宽 copy；
4. daemon buffer-core：只有在 raw-buffer/memfd 成为端到端瓶颈时推进。

禁止重新发散的方向：

- 文件特征 gate；
- bench-only gather/compaction；
- pipeline/overlap；
- 未经分布诊断的更宽 compare/copy；
- 只改代码形状、不减少真实操作数的 hot path 微调。
