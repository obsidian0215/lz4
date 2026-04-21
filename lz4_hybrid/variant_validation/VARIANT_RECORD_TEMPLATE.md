# 变体记录模板

> 适用于：`lz4_gpu`、`lzo_gpu`、`lz4_hybrid`、`lzo_hybrid`

## 1. 根层组件记录文档要求

### 1.1 GPU 引擎

`gpu` 目录必须维护：

- `KERNEL_VARIANTS.md`
- `HOST_VARIANTS.md`

### 1.2 Hybrid 引擎

`hybrid` 目录必须维护：

- `HOST_VARIANTS.md`
- `SCHEDULER_VARIANTS.md`

### 1.3 组件记录文档组织方式

每个组件文档都必须按以下层级组织：

1. `vendor`（至少 `NVIDIA` / `Intel` 两章）
2. `optimization_object`
3. `stage`
4. `operation`
5. 具体变体条目

### 1.4 组件记录文档中的每个条目必填字段

- `variant_id`
- 显示名称
- `engine`
- `component`
- `optimization_object`
- `stage`
- `operation`
- `baseline`
- `status`：`pending` / `draft` / `tested` / `watch` / `throughput-only watch` / `reject` / `adopt` / `backfill-pending`
- `formal_fixed_params`
- `special_axis_tags`（如 `hash_table_overhead`）
- `program_artifacts / source_record / build_notes`（`host` / `scheduler` 条目强制）
- 动机
- 设计思路与实现
- 整体结果
- 判定
- 证据路径

## 2. 组件记录条目模板

### `variant_id`

- 显示名称：
- `engine`:
- `component`:
- `optimization_object`:
- `stage`:
- `operation`:
- `baseline`:
- `status`:
- `formal_fixed_params`:
- `special_axis_tags`:
- `program_artifacts`:
- `source_record`:
- 动机：
- 设计思路与实现：
- 整体结果：
- 判定：
- 证据路径：

## 3. 单变体明细记录模板（可选，但推荐）

## 3.1 基本信息

- `variant_id`:
- `engine`:
- `component`:
- `optimization_object`:
- `stage`:
- `operation`:
- `baseline`:
- `status`:
- `formal_fixed_params`:
- `special_axis_tags`:
- `program_artifacts`:
- `source_record`:
- `owner`:
- `date`:

## 3.2 动机

- 当前实现与 CPU 原始实现 / 现有 GPU/Hybrid 实现的差异：
- 为什么当前值得测：
- 为什么它比历史失败版更轻/更稳：

## 3.3 设计与实现

- 核心思路：
- 具体改动点：
- 涉及文件：
- 程序/二进制工件：
- 源码/补丁记录：
- 构建说明：
- 预期收益指标：
- 主要风险指标：

## 3.4 测试设置

- 样本根目录：
- 样本数量：
- 参数：
- 固定参数：
- 运行协议：
- 自动化入口：
- 对比基线：
- 结果目录：

## 3.5 结果摘要

- 正确性：
- 关键指标：
- `avg / pos / neg / neg_worst`：
- `control-vs-control` 噪声包络：

## 3.6 判定

- 最终标签：
- trade-off 判读：
- 是否保留进下一轮：
- 若 reject，是否进入黑名单/低优先级名单：

## 3.7 后续动作

- 下一步建议：
- 需要组合验证的关联候选：
- 需要删除/冻结的相关功能（如适用）：
