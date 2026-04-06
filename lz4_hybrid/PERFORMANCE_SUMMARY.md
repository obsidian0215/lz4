# LZ4 Hybrid 性能总结（Intel + Nvidia）

> 代码路径：`/root/lz4/lz4_hybrid`
> 当前二进制：`/root/lz4/lz4_hybrid/lz4_hybrid`
> 当前哈希：`sha256=90c6cb1485eab590df133c0314fcaa4369881d6ae4a0945c71e1f193a1bf81b1`
> 基线二进制：`/root/lz4/exp_results/baselines/lz4_hybrid_baseline_round22e`
> 基线哈希：`sha256=4d29a66ac7fbe5ee24eec7455288ac82e8cb2c5c87f70e6f8e195e4d36ab270e`

---

## Intel 平台（五章结构）

### 1. 设计动机

#### 1.1 目标边界

`lz4_hybrid` 的核心目标是把 CPU 与 GPU 放到同一个容器语义内进行协同执行，保证以下四件事同时成立：

1. `CompTotal / DecTotal` 作为主评估口径；
2. `Ratio` 不出现结构性偏移；
3. roundtrip 正确性稳定通过；
4. 每条结论都能回链到函数与工件。

#### 1.2 当前实现收敛点

当前实现已经收敛为单一路径：

- 分区只走 `partition_blocks_prefix()`；
- 容器头字段 `gblk` 仅表达“GPU 前缀块数”；
- 解压端按同一前缀边界复原；
- CLI 仅保留 `--split-prefix` 语义。

这意味着文档不再描述其它分区口径，避免“文档多口径、代码单口径”的偏差。

#### 1.3 约束原则

1. 文档描述必须可在 `lz4_hybrid.c` 直接检索；
2. 固定比例与自适应比例都必须落到同一分区函数；
3. 历史实验仅用于支撑趋势，不覆盖当前实现边界；
4. 对比关系要同时给出 current/baseline 路径与哈希。

#### 1.4 保留 Hybrid 路径的理由

1. 能直接研究 `gpu_ratio`、`cpu_threads`、`block_size` 的联动曲线；
2. 能拆分并行窗口与整段阶段值，定位瓶颈位于 kernel 还是 host；
3. 能复用 GPU 主干能力，同时保留 CPU 回退路径。

---

### 2. 系统架构

#### 2.1 组件分层

| 层级 | 文件 | 责任 |
| --- | --- | --- |
| 入口层 | `lz4_hybrid.c` | 参数解析、压缩/解压切换、bench 驱动 |
| 调度层 | `lz4_hybrid.c` | 分块、分区、并行窗口编排 |
| GPU 执行层 | `lz4_hybrid.c` + `../lz4_gpu/*` | OpenCL kernel 发射、buffer 复用、回读 |
| CPU 执行层 | `lz4_hybrid.c` | pthread worker + LZ4 编解码 |
| 容器层 | `lz4_hybrid.c` | 头部、长度表、payload 组织与解析 |

#### 2.2 压缩主流程

1. 根据输入大小与 `block_size` 计算 `num_blocks`；
2. 计算 `effective_gpu_ratio`（fixed 或 adaptive）；
3. 由 `partition_blocks_prefix()` 生成 GPU 前缀块与 CPU 后缀块；
4. CPU 路径启动 `cpu_comp_top()`；
5. GPU 路径调用 `gpu_compress_blocks()`；
6. 汇总 `all_sizes[]`；
7. 组装容器：`magic + nblk + bsz + gblk + sizes + payload`；
8. 输出到文件或 bench 内存缓冲。

#### 2.3 解压主流程

1. 校验 `HYBRID_MAGIC`；
2. 读取 `num_blocks / block_size / gpu_blocks`；
3. 解析长度表并构建全局偏移；
4. GPU 解前缀，CPU 解后缀；
5. 汇总输出并写回目标文件。

#### 2.4 容器字段定义

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `magic` | `uint32_t` | 容器魔数 |
| `num_blocks` | `uint32_t` | 总块数 |
| `block_size` | `uint32_t` | 块大小 |
| `gpu_blocks` | `uint32_t` | GPU 前缀块数 |
| `sizes[]` | `uint32_t[]` | 每块压缩长度 |
| `payload` | bytes | 先 GPU 前缀块，再 CPU 后缀块 |

#### 2.5 关键调用链

- 压缩入口：`hybrid_compress_memory()`
- 解压入口：`hybrid_decompress_memory()`
- CPU 压缩 worker：`cpu_comp_worker()`
- CPU 解压 worker：`cpu_decomp_worker()`
- GPU 压缩：`gpu_compress_blocks()`
- GPU 解压：`gpu_decompress_blocks()`
- bench 入口：`run_bench()`

---

### 3. 核心设计和优化

#### 3.1 分区与调度

##### 3.1.1 固定比例

- 参数来源：`--gpu-ratio`；
- 计算方式：`gpu_blocks = round(num_blocks * gpu_ratio)`；
- 分区函数：`partition_blocks_prefix()`。

##### 3.1.2 自适应比例

- 采样函数：`collect_lz4_sample_stats()`；
- 决策函数：`choose_adaptive_gpu_ratio()`；
- 块数修正：`lz4_adaptive_adjust_gpu_blocks()`；
- 分区落点仍是 `partition_blocks_prefix()`。

##### 3.1.3 自适应模型（代码实装）

在 `choose_adaptive_gpu_ratio()` 中，分配比例由三组因子共同给出：

1. 设备能力：`Pc0 / Pg0 / t0`；
2. 数据特征：`gC / gG`；
3. 运行态：`sC / sG`。

核心比值先由 makespan 口径计算，再叠加能效权重与解压侧 host 协调修正项，最终落在 $[0,1]$。

#### 3.2 GPU 路径实现细节

1. `ensure_buffer()` 实现 grow-only 复用；
2. `choose_comp_worker_count()` 与 `choose_decomp_worker_count()` 按设备并发估算规模；
3. `comp_epoch_base` 管理字典 epoch 回绕；
4. `lz4_should_use_device_compaction()` 控制 pack 启停；
5. `hybrid_write_buffer_auto()` / `hybrid_read_buffer_auto()` 按设备特性分流传输路径。

#### 3.3 CPU 路径实现细节

1. 压缩 worker：`LZ4_compress_fast`；
2. 解压 worker：`LZ4_decompress_safe`；
3. 任务领取：`atomic_fetch_add`；
4. 线程编排：`cpu_comp_top()` / `cpu_decomp_top()`。

#### 3.4 阶段指标拆分

`hybrid_metrics_t` 提供四组阶段值：

- `cpu_kernel_us`
- `gpu_kernel_us`
- `parallel_us`
- `total_us`

用于区分“并行窗口变化”与“host 组装变化”。

#### 3.5 前缀快路径与 OCL 跳过路径

1. `adaptive_should_skip_ocl()` 在小输入条件下可直走 CPU 路径；
2. 解压端当 `gpu_blocks == 0` 时允许不初始化 OCL；
3. 前缀快路径策略是确定性开关，不引入额外口径分叉。

#### 3.6 实现映射表

| 主题 | 函数/符号 |
| --- | --- |
| 前缀分区 | `partition_blocks_prefix` |
| 自适应比例 | `choose_adaptive_gpu_ratio` |
| 自适应块数修正 | `lz4_adaptive_adjust_gpu_blocks` |
| 压缩主流程 | `hybrid_compress_memory` |
| 解压主流程 | `hybrid_decompress_memory` |
| CPU 压缩线程顶层 | `cpu_comp_top` |
| CPU 解压线程顶层 | `cpu_decomp_top` |
| bench 主流程 | `run_bench` |

#### 3.7 一致性核查位

1. `gpu_blocks + cpu_blocks == num_blocks` 恒等；
2. `sizes[]` 求和必须与 payload 对齐；
3. 压缩写入顺序与解压读取顺序必须一致；
4. fixed 与 adaptive 必须共用同一容器解释。

#### 3.8 adaptive 调度（实现级全展开）

这一节专门回答你点名的问题：`hybrid` 的 adaptive 在 LZ4 线里到底做了什么、如何做、边界在哪里。

##### 3.8.1 adaptive 在调用链中的真实位置

在当前实现里，adaptive 只发生在压缩路径：

- 入口：`hybrid_compress_memory()`；
- 条件：`cfg->adaptive_split == 1` 且 `ocl != NULL`；
- 调度函数：`choose_adaptive_gpu_ratio()`；
- 离散化函数：`lz4_adaptive_adjust_gpu_blocks()`；
- 落地分区：`partition_blocks_prefix()`。

这意味着“解压不重算 adaptive”。解压直接读取容器头 `gpu_blocks`，按写入时边界复原。

##### 3.8.2 adaptive 的前置短路

压缩阶段有三类短路：

1. `cpu_threads <= 0`：直接 `effective_gpu_ratio = 1.0`；
2. adaptive 开启但 `ocl == NULL`：立即失败，防止误走半残路径；
3. `adaptive_should_skip_ocl()` 命中时，运行配置被重写为 CPU-only：
   - `run_cfg.adaptive_split = 0`；
   - `run_cfg.gpu_ratio = 0.0`。

第 3 条是你关注的“输入太小不要硬上 GPU”的工程化实现。

##### 3.8.3 adaptive skip-ocl 门限

`adaptive_should_skip_ocl()` 走 `adaptive_skip_ocl_threshold_bytes()`：

- 默认阈值：`12 MB`；
- 若 `FORCE_OPENCL_DEVICE=CPU`：阈值改为 `4 MB`。

当 `input_size < threshold` 且自适应开启时，压缩会被主动切到 CPU-only。

这一步并不改变容器定义，只改变本次分配策略。

##### 3.8.4 设备画像（`calibrate_device_profile`）

adaptive 的设备能力来自一次性校准缓存：`g_dev_profile`。

CPU 校准：

1. 2MB 样本；
2. `LZ4_compress_fast` 连跑 3 次；
3. 得到 `cpu_throughput`；
4. 若能读 RAPL，计算 `cpu_energy_per_byte`。

GPU 校准：

1. 同一 2MB 样本走 `gpu_compress_blocks()`；
2. 记录总时间与 kernel 时间；
3. 估算 `gpu_throughput` 与 `gpu_overhead_s`；
4. 若可用，读取 GPU 域 RAPL 估算 `gpu_energy_per_byte`。

兜底值：

- CPU 吞吐 `500e6`；
- GPU 吞吐 `2000e6`；
- GPU 固定开销 `0.0005s`。

##### 3.8.5 数据特征采样（`collect_lz4_sample_stats`）

LZ4 线不是熵驱动，而是“压缩率采样 + 样本 CPU 吞吐”驱动。

采样要点：

1. 样本块数：`cfg->adaptive_sample_blocks`（默认 8）；
2. 索引分布：`sampled_block_index()` 均匀落点；
3. 去重：`prev_block` 防重复采样；
4. 每样本执行 `LZ4_compress_fast`，获得 `comp_sz`。

统计量：

- `mean_ratio_pct = sample_comp_bytes / sample_bytes * 100`；
- `sample_cpu_throughput`（样本阶段字节/秒）；
- `low_ratio_blocks`（块压缩率 < 35%）；
- `high_ratio_blocks`（块压缩率 > 70%）。

其中 35%/70% 是采样标签阈值，用于分布观测，不直接作为硬门限裁决。

##### 3.8.6 `gC` 与 `gG` 的计算

`gC`（CPU 数据因子）：

$$
g_C = \text{clamp}\left(\frac{\text{sample\_cpu\_throughput}}{P_{c0}},\ 0.3,\ 3.0\right)
$$

`gG`（GPU 数据因子）基于样本压缩率 $R$：

$$
R = \frac{\text{mean\_ratio\_pct}}{100},\quad R\in[0.05,1.0]
$$

$$
g_G = \frac{1+m+R_{ref}}{1+m+R},\quad m=2.0,\ R_{ref}=0.50,\ g_G\in[0.5,2.0]
$$

直觉解释：压缩率会影响写回字节量，从而影响 GPU 实际吞吐表现。

##### 3.8.7 运行态因子 `sC/sG`

`sC` 来自 `/proc/stat` 空闲占比，`sG` 来自 `gpu_busy_percent`。

此外有线程缩放：

- 当 `cpu_threads < total_cores`，`sC *= total_cores / cpu_threads`；
- 再上限到 `1.0`。

这是为了避免“线程数设小”被误判成“CPU很忙”。

##### 3.8.8 makespan 主公式

有效吞吐：

$$
P_{c,eff}=P_{c0}\cdot g_C\cdot s_C\cdot \text{thread\_count}
$$

$$
P_{g,eff}=P_{g0}\cdot g_G\cdot s_G
$$

基础分配：

$$
r_{base}=\frac{P_{g,eff}}{P_{c,eff}+P_{g,eff}}
$$

开销修正：

$$
r^*=r_{base}-\frac{t_0\cdot P_{c,eff}\cdot P_{g,eff}}{B\cdot(P_{c,eff}+P_{g,eff})}
$$

其中 $B=\text{input\_size}$。

##### 3.8.9 小输入保护

若满足：

$$
B \le t_0\cdot P_{g,eff}
$$

则直接返回 `0.0`（CPU-only）。

这不是性能回归，而是“固定开销未被摊薄”的主动保护。

##### 3.8.10 能效纠偏（70/30）

默认权重：

- 性能权重 `70`；
- 能效权重 `30`。

当 `eC/eG` 可用时，计算能效比例：

$$
r_{energy}=\frac{P_{g,eff}\cdot e_C}{P_{c,eff}\cdot e_G + P_{g,eff}\cdot e_C}
$$

再与性能比例做加权融合。

##### 3.8.11 ratio soft objective 的当前状态

代码里预留了“压缩率软目标”路径：

- 函数：`lz4_adaptive_ratio_scale()`；
- 参数：`target_ratio_pct=45.0`、`max_penalty=0.40`、`span_pct=40.0`、`min_scale=0.20`。

但在当前确定性配置中：`ratio_weight_pct = 0.0`，因此该分支默认不生效。

这点非常关键：文档必须写明“功能存在但默认关闭”，避免误会当前结果来自该机制。

##### 3.8.12 `dec_host_penalty_pct` 的语义说明

当前实现在 `choose_adaptive_gpu_ratio()` 里固定：

- `dec_host_penalty_pct = 15.0`；
- 最终 `r_star *= 0.85`。

变量名带 `dec`，但该函数实际用于压缩路径。可以理解为历史命名沿用的“主机协调保守项”。

##### 3.8.13 连续比例到离散块

`lz4_adaptive_adjust_gpu_blocks()` 常量：

- `min_mixed = 8`；
- `quantum = 4`；
- `collapse_small = 1`。

行为拆解：

1. 先四舍五入得到 `gpu_blocks`；
2. 小块场景（`num_blocks <= 16`）可塌缩到单侧；
3. 大块场景保证混合两侧都不少于 8 块；
4. 中间值按 4 块量化；
5. 防止量化结果越界到 0 或 `num_blocks`。

这一步是稳定性核心，直接抑制边界抖动。

##### 3.8.14 adaptive 输出如何进入 prefix 容器

最终 `gpu_blocks` 进入 `partition_blocks_prefix()`，形成：

- GPU 前缀 `[0, gpu_blocks)`；
- CPU 后缀 `[gpu_blocks, num_blocks)`。

容器头把 `gpu_blocks` 直接写入 `gblk` 字段。解压只需读头，不需要重算策略。

#### 3.9 GPU pack kernel（实现级全展开）

##### 3.9.1 sparse 与 packed 的定义

GPU 压缩先写入稀疏槽位：

- 槽位宽度：`single_block_max_out`；
- 总稀疏体积：`sparse_total = num_blocks * single_block_max_out`。

随后根据 `h_sizes[]` 求和得到：

- `packed_total = sum(h_sizes[i])`；
- `h_packed_offsets[i]` 由前缀和得到。

##### 3.9.2 `single_block_max_out` 的真实计算

代码不是固定 `compressBound`，而是：

1. 先取 `1.1 * block_size + 64`；
2. 再与 `LZ4_compressBound(block_size)` 取较大值。

这样做的目的是兼顾 GPU kernel 写入布局和最坏情况安全边界。

##### 3.9.3 启动 pack 的门限

`lz4_should_use_device_compaction()` 条件必须全部满足：

1. `kpack` 可用；
2. `packed_total > 0` 且 `< sparse_total`；
3. `num_blocks >= 8`；
4. 节省比例至少 `5%`。

公式：

$$
\frac{sparse\_bytes - packed\_bytes}{sparse\_bytes} \ge 5\%
$$

##### 3.9.4 pack Host 端缓冲准备

若开启 pack，Host 分配并更新：

1. `packed_offsets_buf`：偏移表；
2. `packed_out_buf`：紧凑输出区。

偏移通过 `hybrid_write_buffer_auto()` 上传；数据通道可依据设备特性选择 map/unmap 或标准 copy。

##### 3.9.5 pack kernel 参数绑定顺序

`lz4_pack_blocks` 参数按下列顺序绑定：

1. `out_buf`（稀疏输入）
2. `packed_out_buf`
3. `packed_offsets_buf`
4. `output_size_buf`（每块长度）
5. `singleBlockMaxOut`
6. `totalBlocks`

参数顺序与 kernel 声明严格一致，避免错位导致的数据破坏。

##### 3.9.6 pack 启动几何

`pack_global = round_up_size(num_blocks, lsz)`，`local_size` 复用主压缩路径的 `lsz`（已过 `sanitize_local_size`）。

语义上是“每个 work-group 处理一个块”，因为 kernel 用 `get_group_id(0)` 作为块索引。

##### 3.9.7 kernel 内部小块路径

当 `sz <= 32`：

- 仅 `lane==0` 处理；
- 先搬 `uchar16`，再搬 `uchar8`，最后逐字节。

原因：小块场景并行拆分收益低，单线程路径更省调度开销。

##### 3.9.8 kernel 内部大块路径（三段式）

当 `sz > 32`：

1. `vec32` 段：每轮两次 `vload16/vstore16`；
2. `vec16` 段：处理 16 对齐尾段；
3. scalar 段：处理最终字节尾巴。

分工方式为 `lane` 条带化，步长 `lanes * chunk`。

##### 3.9.9 回读与回填

pack 开启：

- 直接回读 `packed_out_buf` 到 `packed_out`；
- `packed_out` 本身已是连续 payload。

pack 关闭：

- 回读 `out_buf` 得到稀疏布局；
- 按 `h_packed_offsets + h_sizes` 在 Host 端重排成连续 payload。

两条路径最终都输出相同语义的连续压缩数据。

##### 3.9.10 pack 对统计窗口的影响

pack kernel 时间会累加到 `gpu_kernel_us`；
回读与上传分别计入 I/O 窗口（由外围阶段吸收）。

因此评估 pack 不能只看“节省字节”，还要看“新增 kernel + 同步”的总成本。

##### 3.9.11 pack 不适合的典型场景

1. 块数过少（接近 `min_blocks`）；
2. `packed_total` 与 `sparse_total` 差距很小；
3. 小文件短任务，额外 kernel 发射成本占比太高。

这也是 8 块与 5% 双门限存在的直接原因。

#### 3.10 压缩状态机（逐状态）

##### 3.10.1 状态定义

1. `C0` 读取输入；
2. `C1` 计算 `num_blocks`；
3. `C2` fixed/adaptive 得到 `gpu_blocks`；
4. `C3` prefix 分区；
5. `C4` 启动 CPU 线程；
6. `C5` 启动 GPU 压缩；
7. `C6` 汇总 `all_sizes`；
8. `C7` 写容器头与长度表；
9. `C8` 先写 GPU payload；
10. `C9` 再写 CPU payload；
11. `C10` 收尾并返回。

##### 3.10.2 并行窗口边界

并行窗口是 `C4~C5` 的重叠区。

- CPU 侧由 `cpu_comp_top()` 管理；
- GPU 侧由 `gpu_compress_blocks()` 管理；
- `parallel_us` 取两侧完成后的窗口值，并至少不小于两侧 kernel 最大值。

##### 3.10.3 失败回滚

任一阶段失败时统一释放：

- `all_sizes`
- `gpu_block_indices / cpu_block_indices`
- `gpu_sizes/gpu_offsets/gpu_slots`
- `cpu_job.out_slots / cpu_job.out_sizes`

若 CPU 线程已启动，会先 `pthread_join` 再清理。

#### 3.11 解压状态机（逐状态）

##### 3.11.1 状态定义

1. `D0` 校验 `HYBRID_MAGIC`；
2. `D1` 读取 `num_blocks/block_size/gpu_blocks`；
3. `D2` 读取 `sizes[]`，构建 `global_offsets`；
4. `D3` 分离 GPU 前缀与 CPU 后缀视图；
5. `D4` CPU 线程与 GPU kernel 并行解压；
6. `D5` 统计输出总长度；
7. `D6` 写目标文件。

##### 3.11.2 解压分工与压缩对偶

压缩是“按分区写”，解压是“按同分区读”。

- GPU 解 `sizes[0..gpu_blocks-1]`；
- CPU 解 `sizes[gpu_blocks..num_blocks-1]`。

这就是 `gblk` 字段的核心价值：把分工信息固定到容器头，避免口径漂移。

##### 3.11.3 GPU 解压核参数

`gpu_decompress_blocks()` 绑定：

1. `in_buf`
2. `out_buf`
3. `decomp_comp_off_buf`
4. `decomp_comp_size_buf`
5. `decomp_sizes_out_buf`
6. `block_size`
7. `totalBlocks`

返回后验证 `out_sizes[i] != 0xFFFFFFFFU`，作为错误哨兵检查。

#### 3.12 adaptive + pack 联合时序（你关心的主线）

##### 3.12.1 串联顺序

1. 采样统计；
2. adaptive 比例；
3. 块数修正；
4. prefix 分区；
5. CPU/GPU 并行压缩；
6. GPU 长度回读；
7. pack 判定；
8. pack 或 host 重排；
9. 容器落盘。

##### 3.12.2 作用域划分

- adaptive 决定“谁处理”；
- pack 决定“GPU结果如何压紧回收”；
- 二者串联但职责分离。

##### 3.12.3 为什么不能把二者混为一谈

如果把 adaptive 和 pack 混在一个结论里，会出现误判：

1. adaptive 变好但 pack 变差，整体看起来“无变化”；
2. pack 节省了回读字节，但分配不均拖慢并行窗口；
3. 只看总时延无法分辨主因。

所以必须同时看：`cpu_kernel_us / gpu_kernel_us / parallel_us / total_us`。

#### 3.13 代码映射（扩展版）

##### 3.13.1 adaptive 相关

| 概念 | 代码符号 |
| --- | --- |
| adaptive 入口 | `choose_adaptive_gpu_ratio` |
| 采样统计 | `collect_lz4_sample_stats` |
| 样本索引 | `sampled_block_index` |
| 小输入跳过 | `adaptive_should_skip_ocl` |
| 设备校准 | `calibrate_device_profile` |
| 块数修正 | `lz4_adaptive_adjust_gpu_blocks` |
| 分区落地 | `partition_blocks_prefix` |

##### 3.13.2 pack 相关

| 概念 | 代码符号 |
| --- | --- |
| 启停判定 | `lz4_should_use_device_compaction` |
| 稀疏输出缓冲 | `ocl->ws.out_buf` |
| 长度缓冲 | `ocl->ws.output_size_buf` |
| 偏移缓冲 | `ocl->ws.packed_offsets_buf` |
| 紧凑输出缓冲 | `ocl->ws.packed_out_buf` |
| pack kernel | `lz4_pack_blocks` |
| 回读策略 | `hybrid_read_buffer_auto` |

##### 3.13.3 常量与门限

| 常量 | 值 | 用途 |
| --- | ---: | --- |
| `min_mixed` | 8 | 混合分配最小块数 |
| `quantum` | 4 | 块数量化步长 |
| `min_blocks` | 8 | pack 最小块门限 |
| `min_gain_pct` | 5 | pack 最小节省比例 |
| `perf_weight_pct` | 70 | adaptive 性能权重 |
| `energy_weight_pct` | 30 | adaptive 能效权重 |
| `dec_host_penalty_pct` | 15 | 保守修正项 |

##### 3.13.4 你审代码时建议盯住的段落

1. `choose_adaptive_gpu_ratio`：确认公式与权重是否被改；
2. `lz4_adaptive_adjust_gpu_blocks`：确认门限与量化是否被改；
3. `gpu_compress_blocks` 中 pack 判定分支：确认阈值是否被改；
4. `lz4_gpu.cl` 的 `lz4_pack_blocks`：确认小块路径和向量路径是否被改。

#### 3.14 常见误解与校正

##### 3.14.1 “adaptive 会在解压重算”

当前实现不会。解压按头字段 `gpu_blocks` 回放分工。

##### 3.14.2 “pack 开启一定更快”

不成立。pack 追求的是减少回读字节，不是无条件降低总时延。

##### 3.14.3 “设置了 `--gpu-ratio` 就不会触发 adaptive”

如果启用了 `--adaptive`，最终仍以 adaptive 结果为准；`--gpu-ratio` 更像初始偏好或非 adaptive 模式值。

##### 3.14.4 “CPU/GPU 各自吞吐高，总体就一定高”

不成立。还要看并行窗口重叠、host 组装和同步成本。

#### 3.15 面向调参的操作化建议（实现对齐）

##### 3.15.1 先看是否命中 skip-ocl

如果输入普遍小于门限，adaptive 可能经常被重写成 CPU-only，先确认日志中是否出现该提示。

##### 3.15.2 再看 `gpu_blocks` 是否稳定

重点看 `lz4_adaptive_adjust_gpu_blocks` 后的块数是否在相邻样本间剧烈跳动。

##### 3.15.3 再看 pack 门限命中率

若 `packed_total` 常常仅略小于 `sparse_total`，pack 可能反复开关。可先观察门限附近分布，再决定是否调参。

##### 3.15.4 最后看总窗口拆解

保持同输入、同参数，比较：

- `cpu_kernel_us`
- `gpu_kernel_us`
- `parallel_us`
- `total_us`

若仅 kernel 改善而 total 不动，优先排查 host 侧搬运与组装。

#### 3.16 语义不变量（发布前强校验）

1. `gblk` 与实际 GPU 前缀块数一致；
2. `sizes[]` 是 payload 的唯一分割依据；
3. GPU payload 必须在 CPU payload 之前写入；
4. 解压 offsets 必须严格前缀和构造；
5. 任意轮次都要 roundtrip 一致。

这五条若任意一条破坏，性能结论都应判无效。

#### 3.17 `gpu_compress_blocks()` 函数级拆解

这一节专门把 `gpu_compress_blocks()` 拆到“变量与分支”粒度，便于你对照源码逐段审查。

##### 3.17.1 输入与模式判定

函数最先判定两类模式：

1. 普通前缀压缩（`mapped_block_indices == NULL`）；
2. 映射压缩（`mapped_block_indices != NULL`）。

当前 `hybrid_compress_memory()` 主线走前缀模式，所以 `mapped_block_indices` 为 `NULL`，`num_blocks` 直接来自输入总块数。

##### 3.17.2 输出槽位宽度计算

`single_block_max_out` 先按经验值估算，再被 `LZ4_compressBound` 托底：

1. `single_block_max_out = 1.1 * block_size + 64`；
2. 若小于 `LZ4_compressBound(block_size)`，则提升到 compressBound。

这保证了 GPU kernel 每块输出槽位不会因输入特征变化而越界。

##### 3.17.3 工作组规模与并发规模

该函数分两层规模控制：

1. `lsz = sanitize_local_size(...)`：确保本地组大小合法；
2. `gsz = round_up_size(choose_comp_worker_count(...), lsz)`：
   - 基于 CU 数和 `LZ4_HYBRID_COMP_WI_PER_CU_DEFAULT` 估计并发；
   - 再做 `lsz` 对齐。

这种“先估计后对齐”的方式保证 launch 参数稳定且不越设备上限。

##### 3.17.4 设备缓冲准备（grow-only）

关键缓冲都通过 `ensure_buffer` 做 grow-only：

1. `comp_in_buf`：输入缓冲；
2. `out_buf`：稀疏槽位输出；
3. `output_size_buf`：每块长度；
4. `dict_buf`：字典区；
5. 映射模式下还有 `block_info_buf`。

好处是多轮 bench 不反复创建释放对象，减小抖动。

##### 3.17.5 字典 epoch 管理

代码里维护 `comp_epoch_base`，并在可能回绕时主动清零字典区：

1. 估算本轮 `epochs_needed`；
2. 检测低位回绕或 `UINT32_MAX` 风险；
3. 触发则 `zero_cl_buffer(dict_buf)` 并重置 epoch。

这是为了避免历史字典残留污染当前块匹配。

##### 3.17.6 输入上传策略

上传路径由 `hybrid_write_buffer_auto()` 决定：

- 设备偏好标准 copy 时走 `clEnqueueWriteBuffer`；
- 否则尝试 map/unmap；
- map 失败再回退到标准 copy。

该逻辑同样用于 offsets/size 表的上传，保证路径一致性。

##### 3.17.7 kernel 参数绑定

普通路径使用 `kcomp`，映射路径使用 `kcomp_mapped`。

普通路径关键参数：

1. 输入/输出/长度缓冲；
2. `totalBlocks/inputSize/block_size`；
3. `single_block_max_out`；
4. `tableType/acceleration/globalIndexBase`；
5. `dict_buf` 与 `epoch_base`。

其中缓冲参数会尽量复用缓存，减少重复 `clSetKernelArg`。

##### 3.17.8 kernel 发射与长度回读

发射后 `clFinish()`，随后回读 `output_size_buf` 到 `h_sizes`。

回读后有一个重要校验：

- 若任意 `h_sizes[i] == 0xFFFFFFFFU`，视为 kernel 失败哨兵，整轮失败。

##### 3.17.9 packed 偏移构建

`h_packed_offsets` 构建规则：

1. `offset[0] = 0`；
2. 每块累加其 `h_sizes[i]`；
3. 最终得到 `packed_total`。

这个数组既用于 pack kernel，也用于非 pack Host 重排路径。

##### 3.17.10 pack 与非 pack 收敛

两条分支最后都输出：

- `out_sizes = h_sizes`；
- `out_offsets = h_packed_offsets`；
- `out_slots = packed_out`（连续 payload）；
- `out_slot_size = packed_total`。

因此上层容器组装完全不关心“是否走过 pack”。

#### 3.18 `lz4_pack_blocks` 内核级拆解

##### 3.18.1 块映射方式

kernel 使用 `blk = get_group_id(0)`，即每个 work-group 负责一个压缩块。

组内线程（lane）共同搬运该块，避免跨块同步。

##### 3.18.2 地址计算

每块地址由两条公式给出：

1. 稀疏源地址：`src = sparse_output + blk * singleBlockMaxOut`；
2. 紧凑目标地址：`dst = packed_output + packed_offsets[blk]`。

只要偏移表合法，就可保证块间目标区间不重叠。

##### 3.18.3 小块优化路径

`sz <= 32` 时只让 `lane==0` 执行。

执行序：

1. 16 字节向量搬运；
2. 8 字节向量搬运；
3. 尾部逐字节。

目标是在极短块上减少组内同步和控制开销。

##### 3.18.4 大块向量路径

`sz > 32` 时走三段：

1. `vec32_end = sz & ~31`，每轮 32 字节；
2. `vec16_end = sz & ~15`，补齐 16 对齐区；
3. 最后 scalar 尾部。

每段都按 lane 条带化，步长分别是 `lanes * 32`、`lanes * 16`、`lanes`。

##### 3.18.5 安全边界

kernel 内部并不检查 `sz > singleBlockMaxOut`。因此必须依赖 Host 端保证：

1. `h_sizes[i]` 来自同一压缩 kernel 的可信输出；
2. 输出槽位容量由 `singleBlockMaxOut` 托底；
3. `packed_offsets` 是严格前缀和。

#### 3.19 内存对象生命周期（压缩主线）

##### 3.19.1 长驻对象

在 bench 多轮场景，以下对象通常长驻并扩容复用：

- `comp_in_buf`
- `out_buf`
- `output_size_buf`
- `dict_buf`
- `packed_offsets_buf`
- `packed_out_buf`

##### 3.19.2 短命主机对象

每轮主机侧会重新分配的典型对象：

- `h_sizes`
- `h_packed_offsets`
- `packed_out`
- 非 pack 路径下的 `h_out`

它们都在成功路径转移或失败路径统一释放。

##### 3.19.3 生命周期与抖动关系

如果把长驻对象误改成每轮销毁重建，会直接导致：

1. `parallel_us` 抖动扩大；
2. `gpu_kernel_us` 周边噪声上升；
3. bench 中位值变差。

因此 grow-only 策略属于性能稳定性的基础设施，不是可随意删减的“优化点缀”。

#### 3.20 CPU 路径细粒度说明

##### 3.20.1 压缩 worker

`cpu_comp_worker()` 使用 `atomic_fetch_add` 领取任务；每个任务：

1. 计算块真实输入长度；
2. 调 `LZ4_compress_fast`；
3. 写回 `out_sizes`。

失败（返回 `<=0`）会把 `job->err` 置位。

##### 3.20.2 解压 worker

`cpu_decomp_worker()` 同样用原子领取任务，执行：

1. 计算压缩源地址 `comp_data + comp_offsets[i]`；
2. 调 `LZ4_decompress_safe`；
3. 写回解压长度。

任何负返回值都直接判定失败。

##### 3.20.3 顶层线程器

`cpu_comp_top()` / `cpu_decomp_top()` 负责：

1. 根据块数裁剪线程数；
2. 线程对象栈/堆分配；
3. join 后取最大 worker 耗时作为 CPU kernel 窗口。

#### 3.21 指标解释补强（避免误读）

##### 3.21.1 `gpu_kernel_us` 的口径

它是 GPU kernel 窗口，不等于 GPU 全部阶段（上传、回读也会占时间）。

##### 3.21.2 `parallel_us` 的口径

它表示 CPU/GPU 并行阶段窗口，且被约束至少不小于两侧 kernel 最大值。

##### 3.21.3 `total_us` 的口径

`total_us` 覆盖整段流程，包含线程调度、数据搬运、容器组装和收尾。

所以出现“kernel 提升但 total 变化有限”是合理现象，不代表优化无效。

#### 3.22 代码审阅清单（按优先级）

1. 先看 `choose_adaptive_gpu_ratio` 是否改了权重或惩罚；
2. 再看 `lz4_adaptive_adjust_gpu_blocks` 是否改了 `min_mixed/quantum`；
3. 再看 `lz4_should_use_device_compaction` 的 `min_blocks/min_gain_pct`；
4. 再看 `lz4_pack_blocks` 小块与向量路径是否被改；
5. 最后看容器写入与读取顺序是否保持对偶。

按这个顺序审，可以最快定位“策略漂移”和“语义漂移”。

#### 3.23 你关心的三连问（是什么/怎么做/目标）

##### 3.23.1 adaptive 是什么

它是一个基于设备画像、数据采样和运行态的动态分配器，输出 GPU 前缀块数。

##### 3.23.2 adaptive 怎么做

按“采样 -> 比例公式 -> 能效纠偏 -> 块数离散化 -> prefix 分区”执行。

##### 3.23.3 adaptive 目标是什么

在不破坏容器语义前提下，降低总阶段时间并稳定多轮波动。

##### 3.23.4 pack kernel 是什么

它是把 GPU 稀疏槽位压缩结果搬运成连续 payload 的 OpenCL 内核。

##### 3.23.5 pack kernel 怎么做

按“偏移前缀和 -> work-group 按块搬运 -> 向量化 + 尾部处理”执行。

##### 3.23.6 pack kernel 目标是什么

降低无效回读字节，减少 Host 重排成本，前提是节省比例达到门限。

#### 3.24 结语（本章）

到这一层粒度，`adaptive` 与 `pack` 已经可以从“概念词”直接落到“函数、变量、门限、调用顺序”。

后续如果你要我继续加深，我可以再往下拆到：

1. 每个 kernel 参数与输出字段的一一对应矩阵；
2. 每条失败分支的资源回收路径图；
3. 按输入规模分桶的 adaptive 行为样例表。

#### 3.25 当前采纳优化（与 strict 主线一致）

##### 3.25.1 前缀布局直写（免索引构建）

- **动机**：prefix 模式下常规索引构建存在固定主机开销。
- **设计**：压缩路径使用前缀布局直写，避免 `gpu/cpu block indices` 的常规构建。
- **实现**：文件 `lz4_hybrid.c`；工件 `host_round_20260401_113006_L4H_HOST_R5B_FULLSET_PREADOPT_ab.json`。
- **效果**：`Comp +0.3264%/+1.0225%`，`Dec +1.2066%/+0.7339%`（fullset 通过）。

##### 3.25.2 adaptive 多目标权重 + ratio 软约束

- **动机**：adaptive 从单吞吐目标升级为“性能 + 能效 + 压缩率”联合目标。
- **设计**：引入动态目标权重与 ratio 软约束，保留解压惩罚项并改为动态输出。
- **实现**：
  - 文件：`/root/lz4/lz4_hybrid/lz4_hybrid.c`
  - 工件：`adaptive_round_20260401_154500_L4H_ADAPT_R1_subset_ab.json`、`...FULLSET_PREADOPT_ab.json`
- **效果**：
  - subset：`Comp +0.7150%/+1.3721%`，`Dec +1.7842%/+2.1886%`
  - fullset：`Comp +0.5605%/+0.0856%`，`Dec +4.4173%/+1.8628%`
  - `Ratio +0.000100/+0.000000 pctpt`

##### 3.25.3 ratio refinement 有界搜索

- **动机**：adaptive 比率贴近 0 会触发 `Comp/Dec` 双回退。
- **设计**：把 ratio refinement 改成有界搜索，避免极端值主导。
- **实现**：
  - 文件：`/root/lz4/lz4_hybrid/lz4_hybrid.c`
  - 目录：`/root/lz4/exp_results/runs/deep_rework_subset_round2/runs/20260403_121037/`
- **效果**（adaptive 相对 fixed `R=0.5`）：
  - `Comp mean -35.50% -> -5.36%`，`Comp median -32.76% -> -0.94%`
  - `Dec mean -16.12% -> -1.78%`，`Dec median -29.36% -> -0.32%`
  - `AdaptiveGpuRatio mean 0.0139 -> 0.4765`

##### 3.25.4 neutral floor + adaptive ratio cache

- **动机**：继续抑制“重复求解开销 + 低比率回落”导致的抖动。
- **设计**：在动态边界增加 `neutral_floor`，并加入 adaptive ratio cache 命中快路。
- **实现**：
  - 文件：`/root/lz4/lz4_hybrid/lz4_hybrid.c`
  - 目录：`/root/lz4/exp_results/runs/lz4_adaptive_deep_r1_prechange_v2/runs/20260403_184920/`
- **效果**：
  - subset：`dComp +22.1505%/+18.2007%`，`dDec +3.7224%/-4.9475%`
  - fullset：`dComp +17.2993%/+16.0339%`，`dDec +2.2563%/-3.3987%`
  - fullset 压缩文件占比：`42/50` 提升

##### 3.25.5 解压 metadata 哈希缓存与条件上传

- **动机**：解压路径 `comp_off/comp_sizes` 的重复上传产生稳定 host 固定开销。
- **设计**：记录 metadata 哈希与数量，未变化时跳过 metadata 上传。
- **实现**：文件 `lz4_hybrid.c`；工件 `/root/lz4/exp_results/runs/hybrid_meta_cache_r1/results/lz4_hybrid_ab_r1.summary.json`。
- **效果**：`dComp +0.3160%/+0.7298%`，`dDec +3.1997%/+2.1240%`，`dRatio +0.00008/+0.00000 pctpt`。

---

### 4. 测试结果和分析

#### 4.1 测试方法与基线有效性

1. 样本固定为 `/root/samples` 全集 50 文件，`Roundtrip_OK` 全通过。
2. strict 参数：`bench_seconds=3.5`，覆盖 CPU/GPU/HYBRID 全配置。
3. 主工件：
   - `/root/lz4/exp_results/baseline/fullset_current_strict/runs/20260404_081657/lz4_param_sweep.csv`
   - `sha256=e046fc93b44b9782ccd418029773740b653d2980979ddba65defdf94a78eab83`
4. 实现一致性：strict CSV 后 `.c/.h/.cl` 新修改为 0，当前实现与基线一致。
5. 结论口径：该 strict 工件是当前最新且主线最优（按当前采纳实现集合）的评估锚点。

#### 4.2 按频率分解：CPU 引擎

CPU（按 `CF` 聚合）结果：

1. `CF=800MHz`：`CompTotal=1115.62`，`DecTotal=2802.23 MB/s`，`Ratio=28.1062%`，`Power=10.72W`
2. `CF=1900MHz`：`CompTotal=2357.22`，`DecTotal=5663.14 MB/s`，`Ratio=28.1062%`，`Power=24.92W`
3. `CF=3000MHz`：`CompTotal=3419.44`，`DecTotal=7982.10 MB/s`，`Ratio=28.1062%`，`Power=44.64W`
4. `CF=5000MHz`：`CompTotal=3935.27`，`DecTotal=9213.47 MB/s`，`Ratio=28.1062%`，`Power=42.52W`

#### 4.3 按频率分解：GPU 引擎

GPU（按 `GF` 聚合）结果：

1. `GF=500MHz`：`CompTotal=480.43`，`DecTotal=1375.56 MB/s`，`Ratio=27.8264%`，`CPU/GPU功耗=25.73/2.60W`
2. `GF=1000MHz`：`CompTotal=924.84`，`DecTotal=2736.51 MB/s`，`Ratio=27.8264%`，`CPU/GPU功耗=25.78/5.88W`
3. `GF=1500MHz`：`CompTotal=1329.12`，`DecTotal=4054.29 MB/s`，`Ratio=27.8264%`，`CPU/GPU功耗=26.86/15.57W`

#### 4.4 按频率分解：HYBRID 引擎

HYBRID（按 `CF/GF` 频点对聚合）结果：

1. `CF/GF=800/500`：`CompTotal=1034.32`，`DecTotal=2675.51 MB/s`，`Ratio=27.9660%`，`CPU/GPU功耗=8.31/0.12W`
2. `CF/GF=800/1500`：`CompTotal=1031.54`，`DecTotal=2647.67 MB/s`，`Ratio=27.9770%`，`CPU/GPU功耗=8.34/0.35W`
3. `CF/GF=3000/500`：`CompTotal=3025.64`，`DecTotal=4894.58 MB/s`，`Ratio=27.9690%`，`CPU/GPU功耗=27.95/0.14W`
4. `CF/GF=3000/1500`：`CompTotal=3028.95`，`DecTotal=4873.86 MB/s`，`Ratio=27.9758%`，`CPU/GPU功耗=27.98/0.37W`
5. `CF/GF=5000/500`：`CompTotal=3545.17`，`DecTotal=5669.01 MB/s`，`Ratio=27.9651%`，`CPU/GPU功耗=34.90/0.13W`
6. `CF/GF=5000/1500`：`CompTotal=3551.14`，`DecTotal=5678.80 MB/s`，`Ratio=27.9699%`，`CPU/GPU功耗=34.90/0.37W`

结论：Hybrid 在压缩侧接近 CPU 高频组合，但解压均值仍低于 CPU；压缩率跨频点稳定。

#### 4.5 功耗合理性确认（GPU 功耗低于 CPU）

按 strict 主工件逐行检查 `Engine=GPU` 的 `CompGPUPower_W < CompCPUPower_W`：

1. 检查行数：`150`
2. 条件成立：`150/150`
3. 覆盖率：`100%`

结论：当前数据满足“GPU 功耗低于 CPU 功耗”的合理性要求。

#### 4.6 按文件分析（CPU/GPU/HYBRID）

基于 `lz4_engine_vs_cpu_file_summary.csv`：

1. HYBRID vs CPU：
   - 压缩：`28` 升 / `22` 降，均值 `+1.19%`
   - 解压：`1` 升 / `49` 降，均值 `-30.26%`
   - 压缩率：均值 `-0.1358 pctpt`
2. GPU vs CPU（横向参考）：
   - 压缩均值 `-58.88%`
   - 解压均值 `-57.88%`

#### 4.7 Hybrid 内部：adaptive vs fixed(R=0.5)

按文件/频点/线程配对，共 `600` 对：

1. 总体：`dComp mean=-0.54%`，`median=+0.59%`；`dDec mean=+0.88%`，`median=-0.05%`；`dRatio mean=+0.0400 pctpt`
2. 胜场：`Comp 348/600`，`Dec 296/600`
3. 分线程：`T1(dComp=-2.58%，dDec=+0.72%)`，`T2(dComp=+1.49%，dDec=+1.04%)`

结论：adaptive 已接近可用，但方差与长尾回退仍需约束，不宜直接全局默认。

#### 4.8 基线判定

1. strict 工件已满足“全量 + 全配置 + 可追溯哈希”要求，可作为当前基线。
2. 当前发布默认仍建议 `fixed(R=0.5,prefix)`；adaptive 作为受控策略继续优化。

---

### 5. 当前结论和后续方向

#### 5.1 当前结论

1. `lz4_hybrid` 仍保持 prefix 单路径与一致容器语义。
2. strict 下 Hybrid 在压缩均值上可与 CPU 接近，但解压仍是主短板。
3. adaptive 相比 fixed 已接近可用，但稳定性不足，不宜直接全局默认。

#### 5.2 后续方向

1. 有效方向一：以 `fixed(R=0.5,prefix)` 作为默认基线继续迭代，保证稳定交付。
2. 有效方向二：对 adaptive 增加“长尾回退保护”（按文件/频点阈值降级）。
3. 有效方向三：优先优化解压 total 路径（回读与主机组装），缩小对 CPU 的差距。

#### 5.3 明确不再走的无效方向

1. 在未做方差约束前把 adaptive 直接全局默认。
2. 引入额外分区语义破坏 prefix 单路径可解释性。
3. 仅凭少量频点或单批样本做策略推广。

#### 5.4 发布前核查清单

1. `partition_blocks_prefix` 是否仍为唯一路径；
2. 容器头字段解释是否未漂移；
3. `gpu_blocks==0` 分支是否仍可无 OCL 解压；
4. 文档中的函数名是否可直接检索；
5. current/baseline 哈希与二进制路径是否同步。

---

## Nvidia 平台（保留章节）

Nvidia 平台沿用同一容器定义与前缀分区语义。跨平台比较时需固定输入集、参数集与统计字段，避免把平台差异与口径差异叠加在同一结论中。
