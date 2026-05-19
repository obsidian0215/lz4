# LZ4 GPU / Hybrid：相关工作对比、现状与后续重构路线

本文基于 `C:\Users\obsid\Desktop\博士毕设\edgecompress\relate_works.md`、当前 `lz4_gpu/lz4_hybrid` 实现，以及 LZO/LZ4 GPU 近期实验结论，整理 LZ4 GPU/hybrid 的技术位置和后续路线。

当前优先级固定为：

1. **hash table/state-slot 与并发结构**；
2. **适配 table 结构的压缩 kernel**；
3. **解压 kernel 与输出排水**；
4. **hybrid CPU/GPU 比例调度**；
5. **daemon / 边缘信息流验证**。

结构化流和边缘链路是最终系统验证层，不应优先于 table 资源、kernel hot path 和 hybrid 基线。

---

## 1. 相关工作与参数对比

### 1.1 GPU LZSS / GPULZ

`GPULZ` 对 LZ 类 GPU 并行化的参数很直接：

- 输入先分 block，再分 chunk；
- chunk 映射到 GPU thread block；
- match 不跨 chunk；
- 典型扫描：
  - chunk size：`2048 / 4096 / 8192 / 16384`；
  - sliding window：`32 / 64 / 128 / 255`；
  - symbol length：`1 / 2 / 4`；
- 默认配置偏小窗口和小 chunk，例如 `chunk=2048`、`window=128`、`symbol=2`。

这说明 GPU LZ 路线常用“缩短依赖范围 + 增加并行 chunk”换吞吐。但 `lz4_gpu` 仍要保持 LZ4 格式语义和接近原始压缩率，不能直接把窗口降到几百字节。

### 1.2 LZ4 FPGA/ASIC 工作

近年 LZ4 硬件工作常见做法：

- 限制每个并行窗口内的 match；
- 提前比较候选；
- 改进 hash table 端口或减少表复杂度；
- 通过牺牲部分压缩率换高吞吐和低资源。

这些工作对 `lz4_gpu` 的启发是：

- table 资源预算必须先于 kernel 微调；
- match search 是核心瓶颈；
- 缩表或限制候选必须量化压缩率代价，不能只看 kernel 吞吐。

### 1.3 nvCOMP / batched LZ4

GPU LZ4 库通常采用 batched/chunk 模型，常见 chunk 粒度围绕 `64KB`。这与 LZ4 格式最大 distance `65535` 对齐。

含义：

- LZ4 的自然 block 上限是 `64KB`；
- 小于 `64KB` 可以提高并发，但会增加边界损失；
- 大于 `64KB` 不会增加 LZ4 单 match distance，只会让一个 block 内有更多完整窗口区域，但并发下降。

### 1.4 当前 LZ4 GPU 参数

当前 `lz4_gpu`：

- `LZ4_DISTANCE_MAX = 65535`；
- `LZ4_HASHLOG = 14`；
- 当前 GPU 分块字典按 `1 << 14` 个槽组织；32-bit epoch entry 下约 `64KB/table`，16-bit clear entry 下约 `32KB/table`；
- 旧的“翻倍 table”口径属于 CPU 单流路径遗留，不再作为 GPU 分块字典预算口径；
- block size 常用 `32/48/64KB`。

当前核心问题不是 `HASHLOG=14` 本身错误，而是 table owner 与 worker/work-item 绑定后，table 总数随并发放大。

---

## 2. 块增多造成压缩率损失的来源

### 2.1 跨 block history 丢失

LZ4 格式允许最大 match distance：

```text
W = 65535 bytes
```

GPU 分块后，每个 block 不能引用前一 block 的历史。block 内偏移 `p` 处可用历史是 `p`，原始串行路径最多可用 `W`。

```text
available_history(p) = min(p, W)
lost_history(p) = max(0, W - p)
```

当 `B <= W` 时，整个 block 都处于 warm-up 状态；平均历史覆盖约：

```text
avg_history_ratio ≈ B / (2W)
```

量级：

- `B=32KB`：平均覆盖约 `25%`；
- `B=48KB`：平均覆盖约 `37.5%`；
- `B=64KB`：平均覆盖约 `50%`。

这不是直接压缩率损失，但说明 LZ4 小 block 会明显降低长距离重复利用能力。

### 2.2 字典/table 冷启动

每个 block 开头 hash table 为空或逻辑 epoch 重置。冷启动影响：

- block 前部 candidate 少；
- 跨 block 重复无法匹配；
- 文本、日志、JSON/XML、内存镜像等结构化数据更敏感；
- x-ray 等难压缩数据影响较小，不应作为主要判断样本。

### 2.3 边界 match 被切断

如果原始串行 LZ4 会在 block 边界后引用边界前的内容，GPU block 独立后只能输出 literal 或更短 match。

可用定量指标：

```text
boundary_loss_bytes =
  reference match 中 source < block_start 且 match_pos >= block_start 的覆盖字节数
```

该值除以输入大小，可作为跨 block history 禁止导致的潜在压缩率损失上界。

### 2.4 block header / size table 开销

LZ4 block 越多，元数据越多。但该项通常不是主因。

如果每 block 元数据按 4 字节估算：

```text
overhead_ratio ≈ 4 / B
```

量级：

- `32KB`：约 `0.012%`；
- `48KB`：约 `0.008%`；
- `64KB`：约 `0.006%`。

所以压缩率下降主要来自 history 和冷启动，不是 header。

### 2.5 hash table 缩小导致 collision 增加

如果 `HASHLOG=14` 降到 `13`：

- entry 数减半；
- table 从 `64KB` 降到 `32KB`；
- collision 增加；
- 有效候选更容易被覆盖；
- 压缩率可能下降；
- failed compare 可能上升，吞吐也可能下降。

缩 table 只有在 table/cache 压力明显超过 candidate 质量损失时才值得采纳。

---

## 3. 当前 `lz4_gpu/hybrid` 实现现状

### 3.1 GPU 压缩

当前压缩路径：

- block 独立；
- hash table per worker/state；
- `HASHLOG=14`；
- mapped host copy 默认化；
- 压缩 worker 分段选择；
- lazy-match 已并入，用一次 look-ahead 改善部分压缩率/吞吐折中。

主要问题：

- table 数量与并发绑定；
- table 预算没有作为一等调度对象；
- `sig8/hsig8` 类候选过滤没有稳定收益；
- 压缩 kernel 仍受 hash lookup、candidate validation、match extension 约束。

### 3.2 GPU 解压

当前解压路径：

- block 独立解压；
- metadata 条件上传；
- chunked readback/write；
- 可继续优化 literal copy、match copy 和 token fast path。

解压优化的可靠性高于压缩 table/search 改造，但优先级低于 table/state-slot。

### 3.3 Hybrid

`lz4_hybrid` 当前：

- 使用 OpenCL CPU/GPU mixed 基础实现；
- `gpu_ratio=1` 是 GPU-only；
- `gpu_ratio=0` 是 OpenCL CPU-only；
- `0<gpu_ratio<1` 是 CPU/GPU range split；
- `--cpu-threads` 限制 OpenCL CPU slots；
- daemon 可复用 OpenCL context、queue、program、kernel 和主要 buffer。

主要问题：

- adaptive 仍未形成正式策略；
- mixed 性能依赖 GPU-only/CPU-only 稳定性；
- 不能用 hybrid 策略掩盖 GPU table/kernel 本体问题。

---

## 4. 后续改进阶段

### 4.0 OpenCL kernel 优化硬约束

后续所有 kernel 改动必须先按 OpenCL 并行执行模型审查，不能只按普通 C 串行代码思路做微调。每个候选改动至少说明它影响下面哪一类成本：

1. **向量数据类型与访存宽度**
   - 优先考虑 `uchar4/uchar8/uchar16`、`uint2/uint4`、`ulong` 等类型对 load/store 合并的影响；
   - 字典清零、literal copy、match copy、token 写出都要先确认是否能用自然对齐的 64/128-bit 访问表达；
   - 向量化不是无条件更快：如果引入额外对齐修正、tail 分支或寄存器压力，必须通过全样本验证。

2. **global/local/private 地址空间**
   - 当前 hash table 主体在 `global`，优化重点是减少 entry 宽度、减少随机 load、减少 table 总 footprint；
   - `local` 只在 work-group 内共享，不能直接承载每个 work-item 独立的大字典，否则会压低 occupancy；
   - private 缓存只适合短生命周期标量，例如 hash、candidate、token 状态，不能把大数组搬进 private。

3. **work-item/work-group/occupancy**
   - 改动必须检查 register pressure、work-group size、global size、CU occupancy 和每 work-item 串行处理量；
   - 不能只减少 work-item 数来降低字典占用；如果 occupancy 或延迟隐藏下降，kernel 吞吐可能变差；
   - slots/table pool 属于资源预算问题，不再用无依据 gate 处理。

4. **分支、发散与编译期路径**
   - GPU/CPU 差异、字典模式、block size 安全范围优先用 OpenCL build flag 固化；
   - hot path 中新增运行时分支必须证明减少了更多 load/store 或 copy 操作；
   - 对只改变代码形状、不减少真实操作数的改动，最多作为清理项，不能当作性能优化。

5. **OpenCL 内建函数和编译选项**
   - `clz`、向量 load/store、位操作、显式 inline 是优先工具；
   - `-cl-fast-relaxed-math` 对当前整数 LZ4 kernel 基本无直接价值，不作为默认优化；
   - `-cl-std=CL1.2` 足够覆盖当前需要的向量类型和整数内建，除非明确需要更新特性，否则不把 OpenCL 版本升级当作优化项。

6. **验证口径**
   - kernel 吞吐用于判断内核本体；
   - 端到端吞吐用于判断真实文件路径；
   - 字典、copy、readback/write 这类非纯计算改动必须同时看 kernel 与端到端；
   - 任何采纳项必须有动机、设计、实现、结果和原因分析；被拒绝项只进入拒绝表，不在路线里膨胀。

### 4.1 P0：table slot pool 与并发解耦

目标：让 table 数量由资源预算决定，而不是由 active worker/work-item 数隐式决定。

步骤：

1. **引入 table slot**
   - table slot 持有 hash table；
   - 第一版 `global_size == table_slots`；
   - 保持输出 block id 不变。

2. **静态 range 映射**
   - 每个 table slot 处理连续 block range；
   - 记录 `blocks/table_slot`；
   - 不先引入动态 task queue。

3. **table slot 预算**
   - `table_slots = min(nblk, CU * factor, mem_budget / table_bytes)`；
   - factor 只测少量候选，例如 `4/8/12/16`；
   - 同时统计 table bytes。

4. **验证**
   - block size：`32/48/64KB`；
   - `HASHLOG=14` 固定；
   - 全样本多轮；
   - 指标：`CompKernel`、no-ocl-init `CompTotal`、压缩率、table bytes、table slots、blocks/slot。

可靠性：**中高**。它针对 table 膨胀根因，但 LZ4 table 访问比 LZO 轻，收益可能小于 LZO。

当前状态（2026-05-20）：

- `lz4_gpu` 与 `lz4_experimental` 已统一到 `HASHLOG=14` 默认路径；
- standalone/bench/daemon 已统一调用 `lz4_load_program()`，编译参数由 block size 和字典模式决定；
- `clear16` 已迁移到 `lz4_gpu` 主线，`block_size <= 64KB` 时 entry 使用 16-bit block-local offset；
- table slot pool 只保留为诊断/平台适配开关，不作为默认策略；后续不再围绕 slot gate 发散。

实现前提：

- experimental 二进制必须**优先加载自身目录下的 `lz4_gpu.cl`**，不能误用兄弟目录原始 `../lz4_gpu/lz4_gpu.cl`。
- 否则会出现 host/kernel ABI 不一致，表现为 `clSetKernelArg(...)= -51` 之类的假失败，结果不能用于 P0 判定。

### 4.2 P1：HASHLOG 与 table slot 联动

目标：确认是否需要减小单 table。`HASHLOG` 不是单纯的压缩率开关，而是 table footprint、cache 访问、candidate 质量和真实端到端时间之间的权衡。

步骤：

1. 在 `32/48/64KB` block size 下扫描 `HASHLOG=13/14`，必要时加入 `15` 作为对照；
2. 固定其他变量，避免把 slot pool、block size、accel 和 hashlog 混成不可解释的三元扫描；
3. 同时统计压缩主吞吐、no-ocl-init 端到端压缩吞吐、压缩率、压缩输出大小、table bytes；
4. debug 轮次再统计 candidate hit、failed compare、match length，用于解释为什么某些文件吞吐或压缩率变化；
5. 分文件看文本/XML/内存镜像、日志类、二进制类、难压缩类数据差异。

判定：

- `HASHLOG=13` 允许压缩率有小幅损失；不能因为压缩率数值变大就直接拒绝；
- 是否采纳取决于：table footprint 减半是否带来压缩主吞吐或 no-ocl-init 端到端压缩吞吐的稳定收益，以及该收益能否覆盖输出变大带来的写出/传输代价；
- 如果压缩率损失集中在少数高度可压缩文件，而吞吐收益在大多数文件稳定存在，可以保留为显式 profile 或后续 adaptive 候选；
- 如果 kernel 吞吐提升但 no-ocl-init 端到端吞吐没有提升，说明收益被输出变大、host copy 或文件写入抵消，不采纳为默认；
- 如果 kernel 吞吐和端到端吞吐都没有稳定提升，则拒绝。

可靠性：**中**。缩表资源收益确定，压缩率风险明确。

当前状态（2026-05-20）：

- `HASHLOG=13` 曾显示 table/cache 受益，但压缩率与文件差异风险仍在；
- 第一轮主线迁移不采纳 `HASHLOG=13` 默认化；
- `HASHLOG=14` 继续作为 `lz4_gpu` 主线默认；
- 后续若重做 `HASHLOG`，必须同时看 kernel、端到端、输出大小和典型文件，不再只凭子集或单指标判断。

### 4.2.1 P4：字典 entry、epoch 与清表对照

目标：确认 hash table 真实瓶颈到底来自 table 大小、entry 宽度、epoch tag 还是清表。

当前实现现状：

- 当前 GPU 分块字典 entry 数为 `1 << HASHLOG`，不是 `1 << (HASHLOG + 1)`；
- `HASHLOG=14` 时每个 table 有 `16384` 个 entry；
- 当前默认 entry 为 32-bit `[12-bit epoch | 20-bit block-local position]`；
- 对 `block_size <= 64KB`，20-bit position 过宽，16-bit block-local offset 已足够。

已完成对照并迁移到主线：

- `epoch32`：32-bit epoch entry，作为大于 64KB block 的回退模式；
- `clear32`：32-bit position-only，每 block 清表，未采纳；
- `clear16`：16-bit position-only，每 block 清表，已采纳；
- 105 远端最终测试显示，当前主线相对基线在 64K/HASHLOG=14 下压缩 kernel 中位 `+10.74%`，端到端压缩中位 `+14.35%`，ratio 中位相对变化 `+0.264%`。

下一步：

- 继续比较 `clear16 × HASHLOG=13/14`，但默认仍保持 `HASHLOG=14`；
- 后续压缩/解压执行流优化必须以 `clear16` 字典结构为新默认基线；
- 只有在能降低总 load 或清表成本时，才考虑 slot-local epoch / side-table，避免重新引入双 load 热路径。

### 4.3 P2：适配 table 结构的压缩 kernel

步骤：

1. **match extension 宽比较**
   - 用 4/8 字节比较替代逐字节扩展；
   - 统计 extension steps。

2. **候选初筛固定路径**
   - hash lookup 后先做固定最小比较；
   - 失败立即走 literal。

3. **lazy-match 保守化**
   - 保留已有 lazy；
   - 只在 look-ahead 命中率和收益足够时扩展。

4. **输出 token 路径整理**
   - 减少 literal/match 状态更新；
   - 不作为主收益来源。

拒绝方向：

- 继续 `sig8/hsig8`；
- pipeline/compaction/gather；
- 文件特征 gate；
- 过宽 block size × worker × acceleration 扫描。

可靠性：**中**。match extension 是明确热点，但必须依赖 P0/P1 后的新 table 结构。

当前状态（2026-05-20）：

- `LZ4_count()` 已采用 `16B -> 8B -> 4B -> 2B -> 1B` 批量比较，不重复做扩大比较宽度；
- `sig8/hsig8` 类候选过滤已拒绝，不再作为 P2 方向；
- `uint4` 清表、`LZ4_hashPosition()` 非空路径、删除 vector copy fallback 已迁移到 `lz4_gpu`；
- 下一轮 P2 只保留能减少真实操作数的改动：随机 table load、candidate read32、failed compare、store、token 写出或分支层级。

### 4.4 P3：解压优化

步骤：

1. literal copy 宽复制；
2. match copy short fast path；
3. 非重叠 match 8 字节复制；
4. 小 token fall-through；
5. metadata cache 保持；
6. chunked readback/write 平台化默认。

验证：

- `DecKernel`；
- no-ocl-init `DecTotal`；
- 真实文件解压时间；
- hash mismatch / missing / out-of-order。

可靠性：**中高**。

当前状态（2026-05-20）：

- 解压端本轮没有采纳新的 copy 分支；
- 105 最终测试中解压 kernel 中位 `+1.42%`，端到端解压中位 `-1.08%`，说明当前收益主要来自压缩侧；
- P3 暂不继续增加 small-offset 分支、更宽 copy 或 copy 分支收缩；后续重点是真实路径的 readback/write、daemon context reuse 和批处理队列。

### 4.5 P4：Hybrid ratio 自适应

步骤：

1. 固定 GPU-only 与 OpenCL CPU-only 基线；
2. 扫描 `gpu_ratio` 和 `cpu_threads`；
3. 离线建立 `(block_size, input_size, cpu_threads, gpu_ratio)` 表；
4. daemon 中按最近 N 个 batch 的 CPU/GPU 吞吐修正；
5. 小 batch 避免 mixed，大 batch 才启用 mixed。

可靠性：**中**。

### 4.6 P5：Daemon / 边缘信息流验证

步骤：

1. `raw-buffer-session` 长连接；
2. byte/time window batch；
3. 输出 queue/build/codec/readback/publish 分段；
4. structured-stream 中比较 `none/lz4/lzo_cpu/lz4_gpu/lzo_gpu`。

可靠性：**系统层中高、核心压缩低**。只用于端到端验证，不作为核心 kernel 优化证据。

---

## 5. 论文表述建议

`lz4_gpu/hybrid` 不应写成单纯“GPU 加速 LZ4”。更准确的是：

> 面向大块数据压缩和异构设备执行的 OpenCL LZ4 后端，通过 table state 预算、mapped host path、块级 GPU 并行、解压输出排水、lazy-match 压缩建模和 CPU/GPU range split，把传统 LZ4 压缩器扩展为可在真实链路中进一步验证的异构压缩组件。

## 6. 当前最终对比

最终对比结果见 `exp_results/remote105_gpu_compare_full6`。该结果来自 `192.168.2.105` 的 `/root/samples` 全量 26 文件，配置为 `64K block / HASHLOG=14 / accel=1 / local=1`，每文件 `1` 轮 bench + `6` 轮真实压缩/解压。

相对 `lz4_gpu_baseline_`：

- 压缩 kernel：中位 `+10.74%`，平均 `+10.81%`；
- 端到端压缩：中位 `+14.35%`，平均 `+14.92%`；
- 解压 kernel：中位 `+1.42%`，平均 `+1.86%`；
- 端到端解压：中位 `-1.08%`，平均 `-1.12%`；
- ratio 数值：中位相对变化 `+0.264%`，平均 `+0.307%`，压缩率轻微变差。

当前已迁移到 `lz4_gpu` 的有效项：编译路径统一、`clear16` 字典、`uint4` 清表、`LZ4_hashPosition()` 非空路径、删除 vector copy fallback、daemon raw-buffer 协议与 context/program/kernel 复用。未迁移项：`HASHLOG=13` 默认、slot factor 默认化、sig8/hsig8、small-offset fast path、解压 copy 分支收缩、bench-only gather/compaction。

提交前重新收集的 105 当前 GPU 基线见 `exp_results/gpu_baseline_105_final`，用于后续改进轮次作为新基线。
