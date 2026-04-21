# LZ4 Hybrid 变体测试价值指引

更新时间：2026-04-16

## 1. 高测试价值方向

### 1.1 先盘点当前调度与主机功能，再做删减

优先级最高的是：

- 当前 CPU/GPU 分配策略建账；
- queue / event / wait 路径建账；
- 没有稳定收益、却增加状态空间的 host/scheduler 开关优先冻结或删除。

### 1.2 单阶段 host / scheduler 候选

高价值候选示例：

- `host_device_transfer / segmented_time_capture`
- `dispatch_sync / event_wait_policy`
- `work_partition / block_ratio_policy`
- `fallback_policy / threshold_tune`

这类条目收益归因最清楚。

## 2. 低测试价值方向

### 2.1 同时改内核、主机、调度

这会把问题重新搅成一锅粥，不允许作为 `lz4_hybrid` 正式条目。

### 2.2 只看 bench，不记 manual 分段时间

对 `hybrid` 来说，这种结果不可审计，测试价值接近零。

### 2.3 同时改 partition 与 fallback

如果两个决策门一起动，几乎无法判断到底是谁造成收益/回退。
