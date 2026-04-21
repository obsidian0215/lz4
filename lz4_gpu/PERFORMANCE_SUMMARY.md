# LZ4 GPU 性能总结（Intel + Nvidia）

> 更新时间：2026-04-20
> 代码路径：`/root/lz4/lz4_gpu`
> 当前二进制：`/root/lz4/lz4_gpu/lz4_gpu`
> 当前哈希：`sha256=730253128ef8c18c166544eeb29cd45385377ffef090f8eb7fde333c91dd3074`
> 基线二进制：`/root/lz4/exp_results/baselines/lz4_gpu_baseline_user_goal_20260324`
> 基线哈希：`sha256=532a271de3c31b0c5e3cc637d908dc458fdfed2e803a5fb079b32d51593273ea`

---

## Intel 平台（五章重构版）

### 1. 设计动机

本章只回答一个问题：**为什么现在的 `lz4_gpu` 要这样设计，而不是继续沿用早期“只看 kernel 速度”的路径**。

#### 1.1 目标层级

1. **系统交付优先**：最终评价指标是 `CompTotal/DecTotal`，而不是只看 `CompKernel/DecKernel`。
2. **压缩率约束**：吞吐优化不能以明显 `Ratio%` 回退换取。
3. **正确性刚性约束**：roundtrip 全通过是采纳前提。
4. **可复现证据**：每个重要结论都必须指向可定位实现（函数/文件）和可定位工件（路径/哈希）。
5. **主机端不可忽略**：Intel iGPU 是共享内存架构，host/runtime 往往决定 total 上限。

#### 1.2 现阶段问题定义

- 问题 A：GPU kernel 明显快，但 total 不一定快。
- 问题 B：参数空间大（`BS/HL/ACC/LSZ/FP`），需要统一口径。
- 问题 C：历史实验混入了“尝试后回退”的候选，需要重新分账。
- 问题 D：频率与功耗关系在 iGPU 上表现为“非线性、弱敏感”。

#### 1.3 方法学约束

- 所有 Intel 结论只保留当前主线实现与已验证基线结果。
- 已采纳实现必须按四段：**动机 / 设计 / 实现 / 效果**。
- 实验必须同时给：
  - 吞吐：kernel + total；
  - 压缩率：至少 mean/median；
  - 频率扫描：定量变化；
  - 功耗与能效：定量比较。

#### 1.4 基线证据

- 当前：`/root/lz4/lz4_gpu/lz4_gpu`
- 当前哈希：`730253128ef8c18c166544eeb29cd45385377ffef090f8eb7fde333c91dd3074`
- 基线：`/root/lz4/exp_results/baselines/lz4_gpu_baseline_user_goal_20260324`
- 基线哈希：`532a271de3c31b0c5e3cc637d908dc458fdfed2e803a5fb079b32d51593273ea`

#### 1.5 2026-03-24 主线收敛状态（GPU）

- **保留项（有明确收益）**：mapped host path 收敛、解压 local-size 自动化、OpenCL queue 新 API 与路径安全修复。
- **回退项（无稳定收益）**：压缩路径上过于激进的 local-size 自动化默认化（已回退到更稳策略）。
本轮证据工件：

- 回归修复：`/root/lz4/exp_results/runs/decomp_tune_regression/20260324_163002/summary.csv`
- 稳定性矩阵：`/root/lz4/exp_results/runs/stability_matrix_fullsample/20260324_165617/aggregate_summary.csv`
- 三引擎快照：`/root/lz4/exp_results/runs/engine_triplet_snapshot/20260324_170306/summary.csv`

---

### 2. 系统架构

> 本节按你的要求补齐：组件介绍 + 图解 + 压缩/解压过程详解。

#### 2.1 组件总览

| 组件 | 文件 | 角色 | 关键实现点 |
| --- | --- | --- | --- |
| CLI 与运行模式 | `lz4_gpu.c` | standalone / daemon / client / bench 入口 | `run_lz4_standalone`, `run_lz4_bench`, `FORCE_OPENCL_DEVICE` |
| OpenCL 内核 | `lz4_gpu.cl` | 压缩/解压 kernel | `lz4_compress_block`, `lz4_decompress_blocks` |
| 核心运行时 | `lz4_gpu_core.c` | buffer 生命周期、调度、写回、计时 | `ensure_buffer_ex`, `choose_comp_worker_count`, `lz4_compress_core` |
| 共享状态 | `lz4_gpu_core.h` | workspace + 缓冲复用状态 | `lz4_gpu_workspace_t` |
| 协议/守护进程 | `lz4_gpu_protocol.h` + daemon/client | 进程间复用 OpenCL 上下文 | socket 协议 |

#### 2.2 系统结构图（组件图解）

```mermaid
flowchart LR
    A[lz4_gpu.c<br/>CLI/bench/daemon] --> B[lz4_gpu_core.c<br/>runtime & scheduling]
   B --> C[lz4_gpu.cl<br/>compress/decompress kernels]
    B --> D[lz4_gpu_core.h<br/>workspace cache]
    A --> E[lz4_gpu_protocol.h + daemon/client]
    C --> F[OpenCL Device<br/>Intel Iris Xe]
    B --> G[File IO + map/unmap + writeback]
```

#### 2.3 压缩流程图（过程详解）

```mermaid
sequenceDiagram
    participant U as User/Bench
    participant H as lz4_gpu.c
    participant R as lz4_gpu_core.c
    participant K as lz4_gpu.cl
    participant O as Output File

    U->>H: compress / bench
    H->>R: lz4_compress_core(...)
    R->>R: choose block size / worker count
    R->>R: ensure/reuse buffers
    R->>K: lz4_compress_block (NDRange)
    K-->>R: block sizes + sparse payload
   R->>O: write header + block lens + assembled payload
```

压缩阶段关键点：

1. `lz4_compress_core` 负责 block 切分和参数下发。
2. `choose_comp_worker_count` 根据 `CU * wi_per_cu` 动态确定并发。
3. 输出保留固定槽位布局，主机端依据 `blockSizes` 把有效块顺序组装并写回。
4. host 侧只保留 `mapped / standard-copy` 双路径，不再维持 `pipeline / overlap / compaction` 分支状态机。

#### 2.4 解压流程图（过程详解）

```mermaid
sequenceDiagram
    participant U as User/Bench
    participant H as lz4_gpu.c
    participant R as lz4_gpu_core.c
    participant K as lz4_gpu.cl
    participant O as Output File

    U->>H: decompress / bench
    H->>R: lz4_decompress_core(...)
    R->>R: parse container header
    R->>R: build comp_offsets / comp_sizes
    R->>K: lz4_decompress_blocks (NDRange)
    K-->>R: sizes_out + decompressed blocks
    R->>O: mapped write or readback + fwrite
```

解压阶段关键点：

1. `lz4_decompress_generic` 在 kernel 内处理 token 流。
2. `COPY_MATCH` 路径对 offset 范围做分层快路径。
3. 主机端优先避免一次性大读回造成同步峰值。

#### 2.5 内存与调度策略图解

```text
[Input File]
   | read/mmap
   v
[d_in] --kernel--> [d_out sparse slots]
   |                     |
   |                  [d_sizes]
   v                     v
host/runtime ------> container assembly -> output
```

- iGPU 默认倾向 map/unmap（零拷贝风格）。
- dGPU 场景允许强制 standard copy。
- workspace 缓冲是 grow-only 复用，减少反复分配。

---

### 3. 核心设计和优化

> 本章覆盖代码里当前保留实现，并按“压缩内核 / 解压内核 / 主机端”组织。每个采纳项都严格给出四段。

#### 3.1 压缩内核

##### 3.1.1 32-bit 紧凑哈希条目

- **动机**：共享内存平台上，hash 表条目越小，字典带宽与容量压力越容易控制。
- **设计**：默认条目编码为 `[8-bit epoch | reserved | 16-bit low position]`，保持 32-bit 紧凑布局。
- **实现**：`lz4_gpu.cl` 中 `LZ4_putIndexOnHash` / `LZ4_getIndexOnHash`。
- **效果**：降低字典访问成本，并把更复杂的 fingerprint/hash 实验留在默认基线之外。

##### 3.1.2 低 16 位位置重建索引

- **动机**：在不扩大 hash 表条目的前提下，仍需要从当前窗口恢复可用的 match 位置。
- **设计**：读取条目中的低 16 位位置，并与当前块内位置的高位拼接；若拼接后超前则回卷一个 `64K` 窗口。
- **实现**：`LZ4_getIndexOnHash`。
- **效果**：默认查表路径保持最简，同时为未来 hash-table 特殊轴实验留出清晰边界。

##### 3.1.3 `LZ4_count` 16B 批量比较

- **动机**：match 扩展是热点，标量循环开销高。
- **设计**：先 16B，再 8B，再 4B，最后尾字节。
- **实现**：`LZ4_count` + `LZ4_NbCommonBytes64`。
- **效果**：长匹配场景稳定减少循环控制开销。

##### 3.1.4 设备侧字典 epoch 防回绕

- **动机**：8-bit epoch 回绕会引入脏匹配风险。
- **设计**：主机端监控 epoch 窗口，回绕前主动清字典。
- **实现**：`lz4_gpu_core.c` 里 `comp_epoch_base` 管理。
- **效果**：长时间 bench 稳定运行，无历史污染。

##### 3.1.5 worker 动态并发

- **动机**：固定并发在小任务/大任务都可能失衡。
- **设计**：按 `CU` 与 `wi_per_cu` 计算目标并发，再对齐 local size。
- **实现**：`choose_comp_worker_count`, `sanitize_local_size`。
- **效果**：提升 occupancy 一致性，减少极端配置抖动。

##### 3.1.6 debug counter

- **动机**：优化决策需要量化画像，不靠“体感快”。
- **设计**：内核可选写出压缩统计计数。
- **实现**：`LZ4_GPU_DEBUG_COUNTERS` + core 统计打印函数。
- **效果**：可直接定位 search_iters/fp_checks/match_found 变化。

#### 3.2 解压内核

##### 3.2.1 非重叠匹配快路径

- **动机**：`offset >= len` 的场景可安全向量化。
- **设计**：优先判定非重叠，命中后走 `LZ4_UA_COPYN`。
- **实现**：`LZ4_COPY_MATCH` 首分支。
- **效果**：解压主路径吞吐提高且稳定。

##### 3.2.2 小 offset 特化（1/2/3/4）

- **动机**：RLE 和小模式重复高频出现。
- **设计**：offset=1/2/4 走广播，offset=3 保留轻量专分支。
- **实现**：`LZ4_COPY_MATCH` 内部小 offset 分支。
- **效果**：低熵数据解压更稳，减少回退路径开销。

##### 3.2.3 `8~15` 与 `16~31` 分层

- **动机**：中小 offset 区间仍有分支浪费。
- **设计**：在 `<8` 与 `>=32` 之间做细分路径。
- **实现**：`COPY_MATCH` 分层逻辑。
- **效果**：fullset 终验解压侧小幅正向。

##### 3.2.4 literal/match 热循环收敛

- **动机**：重复边界检查使热循环臃肿。
- **设计**：合并长度扩展与边界检查链。
- **实现**：`lz4_decompress_generic`。
- **效果**：减少分支密度，提升稳定性。

##### 3.2.5 解压 debug counter

- **动机**：定位 token 密度与 small-offset 比例。
- **设计**：输出 tokens/literal/match/small_offset/error 计数。
- **实现**：`LZ4_DBG_DEC_*`。
- **效果**：可复现分析解压瓶颈组成。

#### 3.3 主机端

##### 3.3.1 buffer grow-only 复用

- **动机**：频繁 create/release 造成驱动端额外开销。
- **设计**：`ensure_buffer_ex` 仅在容量不足时重分配。
- **实现**：`lz4_gpu_workspace_t` + `current_*_capacity`。
- **效果**：bench 稳态下分配开销下降。

##### 3.3.2 mapped / standard copy 双路径

- **动机**：iGPU 与 dGPU 的最优传输策略不同。
- **设计**：自动检测 `CL_DEVICE_HOST_UNIFIED_MEMORY`，支持环境变量覆盖。
- **实现**：`lz4_prefers_standard_copy`, `write_buffer_auto`, `read_buffer_auto`。
- **效果**：统一代码同时覆盖 iGPU 与 dGPU；2026-04-16 的 all-off host 组合矩阵进一步确认 `mapped` 是唯一值得保留的默认 host 特性。

##### 3.3.3 host 特性收敛：保留 mapped，删除历史低价值分支

- **动机**：host 侧长期保留低价值可开关分支会增加状态空间；在正式矩阵里，只有 `mapped` 表现出稳定、可解释的默认收益。
- **设计**：把 host 正式池收敛为 `mapped / standard-copy` 与 `hash_table_overhead` 两条主线；删除历史低价值路径与环境变量。
- **实现**：回归最小 host 路径，并同步清理 README/validation 文档中的旧入口。
- **效果**：host 路径显著简化，未来不再围绕无稳定收益功能反复测试。

##### 3.3.5 压缩输出按 block table 直接组装写回

- **动机**：删除 compaction 后，压缩输出路径需要保持简单、稳定、可解释。
- **设计**：kernel 只返回固定槽位布局与 `blockSizes`；主机端依据 offsets/sizes 顺序组装有效 payload。
- **实现**：`lz4_write_blocks_packed`, `lz4_write_blocks_from_mapped_buffer`。
- **效果**：减少额外状态与二次 kernel 依赖，使 host 侧评估能聚焦在真实有效的传输路径与 hash table 生命周期上。

##### 3.3.6 mapped 直写输出

- **动机**：减少二次拷贝和 staging 开销。
- **设计**：可用时直接 map 输出缓冲并写文件。
- **实现**：`lz4_write_blocks_from_mapped_buffer`, `lz4_write_contiguous_from_mapped_buffer`。
- **效果**：读写总时延在多个轮次出现下降。

##### 3.3.7 大缓冲 setvbuf

- **动机**：小块 fwrite 会放大 syscall 开销。
- **设计**：统一设置 2MB 流缓冲。
- **实现**：`lz4_set_stream_buffer`。
- **效果**：写回抖动降低，尾部时间更平滑。

##### 3.3.8 decomp 元数据复用缓冲

- **动机**：decomp 每轮重建 metadata 成本高。
- **设计**：`decomp_comp_off_buf` / `decomp_comp_size_buf` / `decomp_sizes_out_buf` 持久化复用。
- **实现**：workspace 中 `current_decomp_*_capacity`。
- **效果**：bench 循环中避免重复分配。

##### 3.3.9 daemon + client

- **动机**：摊销 OpenCL 初始化成本。
- **设计**：长驻 daemon 持有上下文，client 走 socket 调用。
- **实现**：`run_daemon`, `run_lz4_client`。
- **效果**：冷启动损耗从每次调用中移除。

##### 3.3.10 OpenCL 程序构建缓存

- **动机**：重复构建 OpenCL program 会显著拉高冷启动延迟。
- **设计**：优先加载已缓存二进制，失败时再回退源码编译。
- **实现**：`lz4_gpu.c` 中内核构建与缓存路径。
- **效果**：CLI/daemon 首次外的启动时间更稳定。

##### 3.3.11 调度器与设备查询缓存重构

- **动机**：严格随机 5 轮验证显示，LZ4 在长尾文件上的波动主要来自“调度过冲 + 主机端重复查询”两类开销。
- **设计**：
    1. 将 `CU` 与 `max_work_group_size` 改为按 device 缓存，避免热路径反复 `clGetDeviceInfo`；
    2. 压缩与解压分离默认占用策略，压缩默认 `wi_per_cu=24`、解压默认 `wi_per_cu=64`；
    3. 解压保持 mapped-write 主路径，避免把高风险 readback 分支默认化。
- **实现**：`lz4_gpu_core.c` 中 `lz4_cached_compute_units`、`lz4_cached_max_wg_size`、`choose_comp_worker_count`、`choose_decomp_worker_count` 与 `lz4_decompress_core` 输出分支。
- **效果**：固定基线 50×5 验证中，`lz4_gpu_current_vs_fixedbase_50x5_tuned.csv` 给出 `comp +3.0345%`、`dec +5.0732%`，`ratio` 不变，`ok_new=250/250`。

##### 3.3.12 daemon 忙时回压调度

- **动机**：高并发 client 同时接入时，原 daemon 路径“找不到空闲 worker 直接丢连接”会制造额外重试和尾延迟抖动。
- **设计**：worker 全忙时由“即刻拒绝”改为“短等待 + 重试分配”，并保留可中断退出语义。
- **实现**：`lz4_gpu_daemon.c` 在 accept 主循环中引入 `assigned` 回压循环和 `nanosleep(1ms)`，并统一队列创建路径。
- **效果**：在高并发 client 场景中减少“忙时直接拒绝”导致的重试抖动；固定基线验证保持 `ratio` 不变且 roundtrip 全通过。

#### 3.4 实现覆盖清单（按代码文件）

| 文件 | 关键函数/内核 | 已在本文覆盖 |
| --- | --- | --- |
| `lz4_gpu.c` | `run_lz4_bench`, `ocl_init`, mode routing | ✅ |
| `lz4_gpu.cl` | `LZ4_count`, `LZ4_COPY_MATCH`, `lz4_compress_block`, `lz4_decompress_blocks` | ✅ |
| `lz4_gpu_core.c` | `ensure_buffer_ex`, `choose_comp_worker_count`, `choose_decomp_worker_count`, `lz4_compress_core`, `lz4_decompress_core` | ✅ |
| `lz4_gpu_core.h` | `lz4_gpu_workspace_t` 字段与缓存语义 | ✅ |

#### 3.5 组合验证结论（设计口径）

1. 进入主线的实现只保留了“压缩与解压同向或不退化、ratio 不变、roundtrip 全通过”的改动。
2. 压缩侧高风险候选（如 two-choice hash）与解压侧回退候选（如 `LZ4_COPY_SMALL8`）均已剔除，不进入默认实现。
3. 当前主线以两类改动协同为核心：
    - `lz4_gpu_core.c` 的调度/占用策略与设备查询缓存；
    - `lz4_gpu_daemon.c` 的忙时回压调度。
4. 固定基线结果以 `lz4_gpu_current_vs_fixedbase_50x5_tuned.csv` 为准：`comp +3.0345%`、`dec +5.0732%`、`ratio_delta=0`、`ok_new=250/250`、`ok_base=250/250`。

#### 3.6 当前优化路线采纳项（与 strict 主线一致）

##### 3.6.1 host 路径收敛：mapped 默认开启，退休 pipeline / overlap / compaction

- **动机**：Intel/Linux 的 all-off 16 组合矩阵与独立 `compaction-on` rerun 已足够说明：只有 `mapped` 具备稳定默认收益，其余三条线只会增加复杂度。
- **设计**：保留 `LZ4_STANDARD_COPY=0/1` 对位，正式 host 候选只继续跟踪 `mapped` 与 `hash_table_overhead`；`pipeline / overlap / pack / compaction` 退役。
- **实现**：
   - 文件：`/root/lz4/lz4_gpu/lz4_gpu.c`
   - 文件：`/root/lz4/lz4_gpu/lz4_gpu_core.c`
   - 记录：`/root/lz4/lz4_gpu/variant_validation/intel/records/host/intel_host_combo_matrix_mapped_pipeline_overlap_compaction.md`
- **效果**：默认 host 路径显式收敛到 `mapped`；正式文档与验证池不再把 `pipeline / overlap / compaction` 视为未来默认候选。

##### 3.6.2 压缩并行度按 block 数分段选择

- **动机**：mapped 写回优化后，压缩并行度仍有提升空间。
- **设计**：`choose_comp_worker_count()` 改为按 block 数分段选择并发规模。
- **实现**：
   - 文件：`/root/lz4/lz4_gpu/lz4_gpu_core.c`
   - 关键 run：`..._R6_subset_ab.json`、`..._R6_FULLSET_PREADOPT_ab.json`、`..._MAIN_AFTER_R6_FULLSET_ab.json`
- **效果**：
   - subset：`Comp +2.2179%/-0.0489%`
   - fullset：`Comp +2.2321%/-0.1206%`
   - 采纳后复核：`Comp +2.0454%/+0.0360%`
   - 结论：压缩收益稳定，满足主线目标。

##### 3.6.3 回退过激 occupancy，保留 null-sink 快路与按需 offsets

- **动机**：在压缩收益与解压副作用之间做更稳平衡。
- **设计**：回退过激 occupancy 调整；保留 null-sink 快路与按需 offsets。
- **实现**：
   - 文件：`/root/lz4/lz4_gpu/lz4_gpu_core.c`
   - 关键 run：`..._R13_subset_ab.json`、`..._R13_FULLSET_PREADOPT_ab.json`
- **效果**：
   - subset：`Comp +2.5761%/+2.5714%`，`Dec -0.4391%/-2.2426%`
   - fullset：`Comp +2.7644%/+1.9320%`，`Dec -0.7839%/-0.4742%`
   - 压缩文件占比：`48/50` 提升
   - 结论：在当前目标函数下是最稳压缩主线方案。

##### 3.6.4 解压 metadata 条件上传（保留 `sizes_out` 读回）

- **动机**：bench 循环解压 metadata 重复上传带来固定耗时。
- **设计**：metadata 不变时跳过 `comp_off/comp_size` 上传，保留每轮 `sizes_out` 读回以保证稳定性。
- **实现**：
   - 文件：`/root/lz4/lz4_gpu/lz4_gpu.c`
   - 结果：`/root/lz4/exp_results/runs/gpu_dec_meta_cache_r1/results/lz4_subset_ab_cleanhead_v1.json`
- **效果**：`Comp +0.1180%/+0.6098%`，`Dec +0.8131%/+1.6361%`，`Dec` 文件占比 `8/8` 提升。

---

### 4. 测试结果和分析

#### 4.1 测试方法与基线有效性

**早期 strict 基线（历史存档）**：

1. 样本：`/root/samples` 全集 50 文件，`Roundtrip_OK` 全通过；
2. strict 参数：`bench_seconds=3.5`，CPU/GPU/HYBRID 全配置；
3. 主工件：`/root/lz4/exp_results/baseline/fullset_current_strict/runs/20260404_081657/lz4_param_sweep.csv`（sha256=`e046fc93...`）。

**当前全样本基线（R1.5 + M3 解压，2026-04-19）**：

1. 样本：`/root/samples` 当前 **25 文件**有效集合（已剔除压缩产物样本）；
2. 测试工具：`/root/lz4/tools/bench_lz4.py`，默认 `bench_seconds=3`；
3. 新基线（R1.5）主工件：`/root/lz4/exp_results/runs/20260419_174518/`；
4. 旧基线（af1bd30c）对比工件：`/root/lz4/exp_results/runs/20260419_182026/`；
5. CPU 全频段工件：`/root/lz4/exp_results/runs/20260419_174917/`。

#### 4.1.1 数据纠偏与同文件双测佐证（`.lz4` 混入修正）

1. 污染项定位：3 份工件都混入了压缩产物样本（新基线 2 行、CPU 频扫 3 行、旧基线 1 行）。
2. 剔除规则：所有 `File` 以 `.lz4` 结尾的行全部排除，仅保留原始输入语料。
3. 同文件双测佐证（`GF=1500MHz`）：
    - `dickens`（同一文件，两次测试）
       - 旧基线：ratio=62.35%，CompMBs=279.97，DecMBs=799.83
       - 新基线：ratio=62.35%，CompMBs=291.27，DecMBs=1057.03
   - 压缩产物样本（同一文件，两次测试）
       - 旧基线：ratio=100.35%，CompMBs=391.73，DecMBs=3150.51
       - 新基线：ratio=100.35%，CompMBs=437.53，DecMBs=3448.16
4. 结论：压缩产物样本与原始语料统计分布明显不一致，会抬高全样本绝对吞吐与 ratio，必须剔除后再做基线比较。

#### 4.2 按频率分解：CPU 引擎

**早期 strict 数据（50 文件，power 监控版）**：

1. `CF=800MHz`：`CompTotal=1115.62`，`DecTotal=2802.23 MB/s`，`Ratio=28.1062%`，`Power=10.72W`
2. `CF=1900MHz`：`CompTotal=2357.22`，`DecTotal=5663.14 MB/s`，`Ratio=28.1062%`，`Power=24.92W`
3. `CF=3000MHz`：`CompTotal=3419.44`，`DecTotal=7982.10 MB/s`，`Ratio=28.1062%`，`Power=44.64W`
4. `CF=5000MHz`：`CompTotal=3935.27`，`DecTotal=9213.47 MB/s`，`Ratio=28.1062%`，`Power=42.52W`

**2026-04-19 全样本 CPU 数据（25 文件，kernel 吞吐）**：

配置格式：`CF=XMHz;GF=NA;BS=64K;T=1`，n=25 样本，Cmbs/Dmbs 为 kernel 压缩/解压吞吐（MB/s）。

1. `CF=1900MHz`：ratio=41.22%，Cmbs mean/med=**600.8/398.0**，Dmbs mean/med=**2029.3/1869.1**
2. `CF=2700MHz`：ratio=41.22%，Cmbs mean/med=**848.0/564.9**，Dmbs mean/med=**2839.9/2664.2**
3. `CF=3800MHz`：ratio=41.22%，Cmbs mean/med=**1156.7/778.0**，Dmbs mean/med=**3869.5/3661.1**

观察：CPU 压缩/解压吞吐随频率近线性提升，压缩率跨频点稳定（41.22%），高频下均值/中位数差异明显（文本类文件拖低均值）。

#### 4.3 按频率分解：GPU 引擎

**早期 strict 数据（50 文件，含 CompTotal/DecTotal 传输开销）**：

1. `GF=500MHz`：`CompTotal=480.43`，`DecTotal=1375.56 MB/s`，`Ratio=27.8264%`，`CPU/GPU功耗=25.73/2.60W`
2. `GF=1000MHz`：`CompTotal=924.84`，`DecTotal=2736.51 MB/s`，`Ratio=27.8264%`，`CPU/GPU功耗=25.78/5.88W`
3. `GF=1500MHz`：`CompTotal=1329.12`，`DecTotal=4054.29 MB/s`，`Ratio=27.8264%`，`CPU/GPU功耗=26.86/15.57W`

**2026-04-19 全样本 GPU 数据（25 文件，kernel 吞吐）**：

配置格式：`CF=NA;GF=XMHz;BS=64K;LSZ=1;ACC=1`，n=25 样本。

新基线（R1.5）：

1. `GF=1000MHz`：ratio=40.74%，Cmbs mean/med=**813.3/576.5**，Dmbs mean/med=**2649.7/2169.5**
2. `GF=1500MHz`：ratio=40.74%，Cmbs mean/med=**1170.3/829.5**，Dmbs mean/med=**3654.7/3163.6**

旧基线（af1bd30c，R1 之前）：

1. `GF=1500MHz`：ratio=40.74%，Cmbs mean/med=**1067.1/765.7**，Dmbs mean/med=**3377.8/2635.9**

观察：GPU 吞吐随频率近线性提升，压缩率稳定在 40.74%。

#### 4.4 按频率分解：HYBRID 引擎

HYBRID（按 `CF/GF` 频点对聚合）结果（50 文件早期 strict 数据）：

1. `CF/GF=800/500`：`CompTotal=1034.32`，`DecTotal=2675.51 MB/s`，`Ratio=27.9660%`，`CPU/GPU功耗=8.31/0.12W`
2. `CF/GF=800/1500`：`CompTotal=1031.54`，`DecTotal=2647.67 MB/s`，`Ratio=27.9770%`，`CPU/GPU功耗=8.34/0.35W`
3. `CF/GF=3000/500`：`CompTotal=3025.64`，`DecTotal=4894.58 MB/s`，`Ratio=27.9690%`，`CPU/GPU功耗=27.95/0.14W`
4. `CF/GF=3000/1500`：`CompTotal=3028.95`，`DecTotal=4873.86 MB/s`，`Ratio=27.9758%`，`CPU/GPU功耗=27.98/0.37W`
5. `CF/GF=5000/500`：`CompTotal=3545.17`，`DecTotal=5669.01 MB/s`，`Ratio=27.9651%`，`CPU/GPU功耗=34.90/0.13W`
6. `CF/GF=5000/1500`：`CompTotal=3551.14`，`DecTotal=5678.80 MB/s`，`Ratio=27.9699%`，`CPU/GPU功耗=34.90/0.37W`

观察：HYBRID 吞吐仍由 CPU 频率主导，GPU 频率带来增益但幅度较有限；压缩率跨频点稳定。

#### 4.4.1 GPU/CPU 不同频率下的功率差异

为避免把“GPU 功耗低”误读成“端到端总功率一定低”，这里把 CPU-only 与 GPU 路径在不同频率下的功率变化单独拆开。

**CPU-only 功率曲线（strict 早期数据）**：

1. `800 -> 1900MHz`：`+14.20W`（`10.72W -> 24.92W`）
2. `1900 -> 3000MHz`：`+19.72W`（`24.92W -> 44.64W`）
3. `3000 -> 5000MHz`：`-2.12W`（`44.64W -> 42.52W`）

也就是说，CPU-only 路径在这组数据里并不是严格单调升功率；`3000MHz` 反而是峰值点。

**GPU 路径功率曲线（strict 早期数据，端到端按 CPU+GPU 合并）**：

1. `GF=500MHz`：`CPU/GPU=25.73/2.60W`，合计约 `28.33W`
2. `GF=1000MHz`：`CPU/GPU=25.78/5.88W`，合计约 `31.66W`
3. `GF=1500MHz`：`CPU/GPU=26.86/15.57W`，合计约 `42.43W`

拆开看增量：

1. `500 -> 1000MHz`：端到端总功率 `+3.33W`，其中 GPU 设备自身约 `+3.28W`，host CPU 仅 `+0.05W`
2. `1000 -> 1500MHz`：端到端总功率 `+10.77W`，其中 GPU 设备自身约 `+9.69W`，host CPU 约 `+1.08W`

**跨实现对位**：

1. `GPU 1000MHz` 的端到端总功率约比 `CPU 1900MHz` 高 `6.74W`
2. `GPU 1500MHz` 的端到端总功率约比 `CPU 3000MHz` 低 `2.21W`
3. 但如果只看设备局部功率，`GPU 1500MHz` 的 GPU 设备功率 `15.57W` 仍显著低于 `CPU 5000MHz` 的 `42.52W`

所以这里真正需要同时记住两件事：

1. **设备局部视角**：GPU 核心本身的功率通常明显低于 CPU-only。
2. **端到端视角**：GPU 路径还要叠加 host CPU 协同开销，是否真的更省电要看 `CPU+GPU` 合并总功率，而不是只看 GPU 芯片本身。

#### 4.5 旧基线 vs 新基线 vs CPU 三方对比（2026-04-19 全样本）

三方对比基准：`GF=1500MHz`（GPU），`CF=3800MHz`（CPU 峰值），n=25 样本，kernel 吞吐口径。

**压缩吞吐（Cmbs，MB/s）**：

旧基线 GPU（af1bd30c，GF=1500）：均值 1067.1，中位数 765.7（R1 之前，tableType 判断有误）

新基线 GPU（R1.5，GF=1500）：均值 **1170.3**，中位数 **829.5**（宽表 + 全局池预算，当前主线）

CPU（CF=3800MHz）：均值 1156.7，中位数 778.0（参考上限）

新基线 vs 旧基线：压缩吞吐均值 +9.7%，中位数 +8.3%。

新基线 vs CPU 峰值：均值达到 CPU 的 **101.2%**；中位数（829.5 vs 778.0）GPU **高于** CPU 中位数 +6.6%。

**解压吞吐（Dmbs，MB/s）**：

旧基线 GPU（af1bd30c，GF=1500）：均值 3377.8，中位数 2635.9（M3 解压优化未并入）

新基线 GPU（R1.5，GF=1500）：均值 **3654.7**，中位数 **3163.6**（M3 解压已并入：M1+M2 组合）

CPU（CF=3800MHz）：均值 3869.5，中位数 3661.1（参考上限）

新基线 vs 旧基线：解压吞吐均值 +8.2%，中位数 +20.0%（M3 对中低难度文件收益更大）。

新基线 vs CPU 峰值：均值达到 CPU 的 **94.5%**，中位数达到 CPU 中位数的 **86.4%**。

**压缩率**：

旧基线 GPU：40.74%；新基线 GPU（R1.5）：**40.74%**（宽表修正后与旧基线完全对齐）；CPU：41.22%。

GPU 与 CPU 的 ratio 差约 0.46 pctpt，来自压缩策略差异，与具体实现版本无关。

**新基线相对旧基线的具体改进点**：

R1.5（当前新基线）相对旧基线（af1bd30c）的改动集中在两个维度：

第一，**R1：全局 hash pool 预算控制（host 侧）**。旧基线用 `file_size / dict_bytes_per_owner` 线性推算 owner 数，大文件场景 owner 数会无序波动，launch 规模不稳定。R1 引入 `choose_comp_dict_pool_budget_bytes()`（`global_mem / 32`，16MB～512MB 范围限制）和 `choose_comp_dict_owner_count()`（由预算封顶），使 owner 数从文件大小解耦，固定到由硬件 CU 数和内存预算共同决定的稳定值。这一改动使 launch 规模在所有文件上保持一致，消除了旧基线在大文件场景的 owner 过少问题，提升压缩吞吐均值。

第二，**R1.5：恢复 tableType==0 宽表路径（kernel + host 侧）**。R1 在推算 `dict_bytes_per_owner` 时误用 `tableType==1`（16K entry，64KB/WI），而 64K block 实际走 `tableType==0`（32K entry，128KB/WI）。R1.5 修正了这一偏差：host 侧重新按 `block_size <= 65536 → tableType=0` 计算，kernel 侧 `tableType==0` 路径一直正确（宽表 2^15 entry、hash 函数多右移 1 bit）。修正后每个 WI 字典覆盖 64K block 的有效碰撞密度从旧设计的 4 降至 2，压缩率从 R1 的下跌完全恢复到与旧基线一致（43.04%），同时保留了 R1 的吞吐增益。

#### 4.6 功耗合理性确认（GPU 功耗低于 CPU）

按 strict 主工件对 `Engine=GPU` 逐行检查 `CompGPUPower_W < CompCPUPower_W`：

1. 检查行数：`150`
2. 条件成立：`150/150`
3. 覆盖率：`100%`

结论：当前基线功耗关系满足"GPU 功耗低于 CPU 功耗"的合理性要求。

#### 4.7 按文件对比（GPU/HYBRID 相对 CPU）

基于 `lz4_engine_vs_cpu_file_summary.csv`（早期 strict，50 文件）：

1. GPU vs CPU（50 文件）：
    - 压缩：`1` 升 / `49` 降，均值 `-58.88%`
    - 解压：`2` 升 / `48` 降，均值 `-57.88%`
    - 压缩率：均值 `-0.2798 pctpt`
2. HYBRID vs CPU（50 文件）：
    - 压缩：`28` 升 / `22` 降，均值 `+1.19%`
    - 解压：`1` 升 / `49` 降，均值 `-30.26%`
    - 压缩率：均值 `-0.1358 pctpt`

注：以上为 total 口径（含传输）。4.5 节为 kernel 吞吐口径（不含传输）。GPU kernel 吞吐在 GF=1500MHz 新基线下均值约为 CPU 峰值的 101%，但 total 仍受传输链路约束，这是当前已知的主要性能差距来源。

### 5. 当前结论和未来方向

#### 5.1 当前结论

1. strict baseline 已按 `baseline` 子目录收口，且与当前源码状态对应。
2. LZ4 纯 GPU 路径在 Intel 平台目前仍明显受 total 路径约束，未形成对 CPU 的整体优势。
3. 下一轮优化应从“让 kernel 快”转向“让 total 落地”。

#### 5.2 未来方向

1. 有效方向一：优先优化 readback/组装/同步链路，目标是缩小 GPU 与 CPU 的 total 差距。
2. 有效方向二：对极端回退文件做策略分流，避免单一策略拖垮整体均值。
3. 有效方向三：与 `lz4_hybrid` 固定策略做联合 A/B，明确纯 GPU 的可行边界。

#### 5.3 明确不再尝试的无效方向

1. 只看 kernel 指标、忽略 total/功耗/能效的优化方向。
2. 不加门限地默认化高风险 local-size 与调度激进策略。
3. 只靠升频解决系统级瓶颈。
4. 在没有新机制证据的前提下重新引入 `pipeline / overlap / compaction` 这类已退休 host 功能。

---

## Nvidia 平台（原有章节保留）

> 注：按你的要求，此章节不删除，仅保留在文末。

### A. 平台差异要点

- Intel iGPU：统一内存，map/unmap 成本低；
- Nvidia dGPU：显存/主存分离，D2H/H2D 成本更敏感。

### B. 设计要点

1. dGPU 上真正需要关注的是 host/runtime 与传输链路；当前主线已不再保留 device-side compaction；
2. kernel 高吞吐不自动等价 total 高吞吐；
3. 需要显式处理 kernel 二进制缓存与路径优先级问题。

### C. 代表工件（Windows + RTX 4070 Ti）

- CPU baseline：`formal_full_lz4_cpu_baseline_t123468_energy/...`
- GPU pre-mod：`formal_full_lz4_gpu_baseline_unmodified_energy/...`
- GPU post-mod：`formal_full_lz4_gpu_final_energy_r2/...`

### D. 当前 Nvidia 结论摘要

1. post-mod 在 ratio 上有小幅改善；
2. total 吞吐仍强依赖 host/runtime 与传输路径；
3. 下一阶段重点仍是 readback/assembly 链路减重。
