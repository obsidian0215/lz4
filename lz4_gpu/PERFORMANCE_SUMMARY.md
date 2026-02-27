# LZ4 GPU 性能总结（本轮实现与全量结果）

更新时间：2026-02-27

## 1. 范围与口径

- 对比区间：`mod3_lz4_decomp.csv` → `mod6_lz4_full.csv`
- 对齐方式：按 `File + BlockSize + HashLog + Acceleration + Threads_LSZ` 对齐 GPU 行
- 对齐样本：`3024` 组
- Roundtrip：`3024/3024` 通过（GPU 行失败 `0`）
- CPU 对照口径：采用 `mod3` 中同文件 CPU 最佳 `CompMBs/DecMBs` 作为参考（`mod6` 为 GPU-only 跑测）

## 2. 本轮设计与实现（不含 roundtrip 故障修正细节）

涉及文件：`lz4_gpu/lz4_gpu.cl`、`lz4_gpu/lz4_gpu_core.c`、`lz4_gpu/lz4_gpu.c`、`lz4_gpu/lz4_gpu_daemon.c`

1. **压缩字典“按 epoch 懒清理”**
   - 词典槽从 32-bit entry 扩展为 64-bit packed（高 32 位为 epoch tag，低 32 位为 entry）。
   - 每个 work-item 仅前进 epoch，不再每块全量清表，减少了词典初始化开销。
   - host 侧在 epoch 接近回卷时触发一次性清零并重置基线。

2. **压缩调度并行度上调**
   - `wi_per_cu` 默认值由 12 提升到 24（可被环境变量覆盖）。
   - local size 做设备上限与 block 数双重约束，稳定 occupancy。

3. **解压 copy 路径的非重叠快路强化**
   - 在 match copy 中优先识别非重叠区间，走 `LZ4_UA_COPYN` 批量复制。
   - 重叠场景仍保留安全路径，保持语义一致。

4. **OpenCL 初始化鲁棒性增强**
   - 设备选择从“仅 GPU”改为 `GPU → DEFAULT → ALL` 逐级回退。
   - standalone 与 daemon 均补充关键创建/编译失败检查与报错。

## 3. 全量结果（mod6 vs mod3）

### 3.1 聚合指标

- `CompKernelReported_MBs`：median **+217.52%**，p10 **+73.13%**
- `DecKernelReported_MBs`：median **+111.24%**，p10 **-18.30%**
- `Ratio%`：median **0.00**，p90 **0.00**，`>1% / >5% / >10% = 0 / 0 / 0`
- 解压提升覆盖率：`2530/3024 = 83.66%`

### 3.2 与 CPU 的中位对照（逐文件）

- `GPU CompKernel / CPU Comp`：**3.09x**（文件级中位）
- `GPU DecKernel / CPU Dec`：**1.71x**（文件级中位）

## 4. 提升较多/较少文件（逐文件中位）

> 说明：表中 `old/new` 分别对应 `mod3/mod6`；CPU 吞吐为 `mod3` 同文件 CPU 最佳值；压缩率变化均为 0（本轮无压缩率回退）。

### 4.1 提升较多（按解压 kernel 提升排序）

| 文件 | CompK old→new (MB/s) | DecK old→new (MB/s) | DecK变化 | CPU Comp / Dec (MB/s) | GPU/CPU(CompK, DecK) | Ratio 变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `transportation_parent_0_pages_img.tar` | 257.28 → 4017.66 | 590.49 → 7799.43 | **+1220.84%** | 1677.31 / 5536.50 | 2.40x / 1.41x | 0.00 |
| `sample_2mb_structured_2.txt` | 267.09 → 4439.95 | 664.07 → 5181.49 | **+680.26%** | 1621.40 / 7331.70 | 2.74x / 0.71x | 0.00 |
| `ooffice` | 261.06 → 1892.45 | 735.03 → 3751.40 | **+410.37%** | 606.40 / 3604.20 | 3.12x / 1.04x | 0.00 |

### 4.2 提升较少（含回退）

| 文件 | CompK old→new (MB/s) | DecK old→new (MB/s) | DecK变化 | CPU Comp / Dec (MB/s) | GPU/CPU(CompK, DecK) | Ratio 变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sample_6.80mb_zero_1.txt` | 2902.50 → 4176.06 | 9736.16 → 2751.70 | **-71.74%** | 27621.64 / 5627.00 | 0.15x / 0.49x | 0.00 |
| `sample_9mb_zero_3.txt` | 3400.10 → 5023.73 | 11357.01 → 3230.74 | **-71.55%** | 27486.81 / 5643.00 | 0.18x / 0.57x | 0.00 |
| `sample_42mb_repeat_3.txt` | 4540.06 → 9325.06 | 15032.87 → 7742.07 | **-48.50%** | 14098.01 / 17401.90 | 0.66x / 0.44x | 0.00 |

## 5. 结论

1. 本轮改动在压缩与解压 kernel 吞吐上都实现了显著中位提升。
2. 压缩率保持稳定（统计口径下无正向回退样本）。
3. 存在少量“高重复/全零类”尾部文件，解压 kernel 仍有明显回退，需要专项治理。

## 6. 下一步优化方向

1. **主机侧链路优化（优先）**：当前部分场景 kernel 已快于端到端链路，后续将重点压缩 `upload/download/write` 占比，推进分段异步与更稳健 pipeline。
2. **尾部文件分型优化**：针对全零/高重复数据，增加更轻量的解压分流策略，改善 p10/p5。
3. **参数自适应**：基于文件特征动态选择 `block/hash/accel/local`，避免单配置拖累尾部。
4. **验收门槛升级**：除 median 外，固定纳入 p10 与“正向覆盖率”双指标。
