# Chunk 实现验收记录（2026-09-23）

## 结论与范围

实现、真实环境短测和阶段边界续训检查均通过。**没有启动 32 条正式长训**，
也没有把短测分数解释为算法效果或稳定性证据。

- 基础：`global-mean-field-control` / `31e4b77defc97f16b6cc6229c906790b16cbfc44`。
- 新分支：`global-mean-field-control-chunk`。
- 16 组完整短测使用同一实现提交 `271245b`，各自 manifest 中的 Git 状态均为空。
- 此后的交付整理补充测试断言、正式尺寸测试和本文档；汇总 CSV 去除了冗长的内部配置签名列。
  Actor、critic、N、数据采样、训练及 checkpoint 实现未在短测后改变。
- 原训练入口及基础 agent、数据、评估文件保持原样；原有源码完整性测试通过。
- 运行机器：`new_server_4090`；没有重新安装环境或修改既有实验仓库。

代码：`/data/lrh/AM-MF-global-mean-field-control-chunk`。
Python：`/home/lrh/AM-MF-4090/.venv/bin/python`。
所有本轮运行产物位于：
`/data/lrh/am-mf-experiments/chunk-size/validation-20260923/`。

## 自动测试

常规测试共 **26 项通过**：原有与新增的 agent/data/runtime 等 24 项完整回归通过，
另新增的矩阵与汇总测试 2 项通过。增强后的 TD target 数值断言及非平凡动态 alpha
续训断言也分别重跑通过。正式尺寸测试为额外的 opt-in 检查，单独执行通过；
默认测试发现时跳过它，避免常规 CPU 测试承担大模型编译开销。

| 检查 | 验收内容 | 结果 |
| --- | --- | --- |
| 原有回归 | 原源码完整性、N 数学与原行为测试 | 通过 |
| 完整 H 步窗口 | H=1 等价，H=2/5/10 动作顺序、奖励和下一状态 | 通过 |
| 边界 | 真实终止、超时、短 episode、环形覆盖，对照暴力索引 | 通过 |
| 归一化 | 原始有效离线数据统计，H/容量改变不影响统计量 | 通过 |
| Shared critic | 同 H/seed 的 B0/N 参数相同，显式 LR 生效，TD target 数值一致 | 通过 |
| 候选选择 | JIT 下候选数 1/2/5 对照显式计算，ensemble 轴固定 | 通过 |
| N 生命周期 | 冻结 prior 不漂移、EMA 更新正确、chunk Q 梯度非零且有限 | 通过 |
| 执行预算 | 队列动作顺序、结束清空、H 不改变真实步数与 update 数 | 通过 |
| 评估独立性 | 不推进训练 RNG，不计入训练交互预算 | 通过 |
| Checkpoint | 模型、优化器、prior/EMA、alpha/history、RNG、归一化恢复；下一次更新一致 | 通过 |
| 恢复拒绝 | 任务或 H 不匹配时拒绝加载 | 通过 |
| 实验与汇总 | 正式矩阵 32 条、短测 16 条；梯形 AUC、配对差值、重复 seed 检查 | 通过 |

复查常规测试：

```bash
cd /data/lrh/AM-MF-global-mean-field-control-chunk
JAX_PLATFORMS=cpu bash scripts/chunk_env.sh -m unittest discover -s tests -v
```

## 真实环境矩阵：16/16 通过

每组使用 **完整离线数据 + 真实 D4RL 环境**，seed=1；每组执行
40 次 offline 更新、20 个 online 环境步及对应的 20 次 learner update。
Offline 初始/终点、online 起点/终点均有独立评估记录，每次 2 个 episode。

| 任务 | 方法 | H=1 | H=2 | H=5 | H=10 |
| --- | --- | --- | --- | --- | --- |
| door-cloned-v1 | B0 | 通过 | 通过 | 通过 | 通过 |
| door-cloned-v1 | N | 通过 | 通过 | 通过 | 通过 |
| pen-cloned-v1 | B0 | 通过 | 通过 | 通过 | 通过 |
| pen-cloned-v1 | N | 通过 | 通过 | 通过 | 通过 |

所有 16 组进一步核验：

- 总 learner updates=60，online env steps=online updates=20；评估未计入预算。
- 执行动作候选数=5、TD target 候选数=5、critic LR=3e-4。
- 策略输出维度确为 `H * env_action_dim`。
- 每个 N 短测 warmup=4，实际进行了 **56 次 guided update**，不是仅跑 warmup。
- N 日志包含有限的 adjoint 总范数、每坐标 RMS、每个动作位置范数。
- 同任务的 8 组归一化 fingerprint 和评估种子列表分别完全一致。
- Offline 终点与 online 起点的同种子评估值完全一致。
- 这些短 episode 未在 20 步内结束，H=1/2/5/10 分别调用策略 20/10/4/2 次；
  每组仍执行 20 次 learner update。提前结束清队列另由单元测试覆盖。

短测显式缩小了模型和算量：actor 32×1、critic 32×2、batch=16、time_steps=8、
teacher_steps=2、teacher_batch_size=4、Jacobian batch=2/interval=4。
这些覆盖仅用于工程验收，**不代表正式超参数下的学习表现**。

原始目录：`matrix/<task>/<b0|n>/H<1|2|5|10>/seed1/`。
每组保留 `manifest.json`、`requested_config.json`、`normalization.npz`、
`train.jsonl`、`eval.jsonl`、`COMPLETED.json` 和 checkpoints。
`matrix/matrix_status.json` 记录各条运行的状态与 stdout 路径。

## 正式尺寸数值检查

额外检查 **B0 和 N 的 H=10**：door 观测维度 39、环境动作维度 28，策略输出 280；
actor 256×3、critic 512×4、batch=256、time_steps=50、候选数 5/5。
N 使用正式 teacher_steps=8、teacher_batch_size=32、Jacobian batch=4。
只将 N warmup 临时缩短为 2，以便第三次更新实际进入调控路径。

使用合成 batch 检查编译、更新、整段动作采样与 adjoint 指标；两种方法均通过，
没有 NaN/Inf。此检查不是完整环境长训，也不能证明长期数值稳定性。

```bash
# GPU ID 是示例；复查前先确认空闲。
CUDA_VISIBLE_DEVICES=0 AM_MF_FULL_MODEL_TEST=1 \
  bash scripts/chunk_env.sh -m unittest discover \
  -s tests -p 'test_chunk_production_shape.py' -v
```

## 真实 offline → online 恢复一致性

使用 `pilot-door-n-H10` 的 `offline_complete.pkl`，重新启动独立进程生成
`resume-door-n-H10`，继续 20 个 online 步。与 pilot 中直接连续运行的结果比较：

- 最终序列化 agent 的 **219 个状态叶子逐项完全相同**，最大绝对差 0。
- 包含模型、优化器、agent RNG、动态 alpha/history、prior、EMA、更新计数。
- 最终 replay sampler RNG 完全相同；归一化 fingerprint 相同。
- Online 起点和终点评估也相同。

该实际续训使用实现提交 `8ef3be8`；后续 `271245b` 仅新增矩阵、汇总与说明文件，
未修改 checkpoint 或训练实现。单元测试额外注入了非零 alpha/history 状态验证恢复。
V1 只验收 offline 阶段边界续训；不承诺恢复任意 online 中途模拟器/replay 状态。

## 数据审计

复用已有 HDF5，经过原有 D4RL 单步加载器处理后，统计有效离线 transition：

| 任务 | 有效单步数 | H=1 窗口 | H=2 窗口 | H=5 窗口 | H=10 窗口 |
| --- | ---: | ---: | ---: | ---: | ---: |
| door-cloned-v1 | 995,642 | 995,642 | 991,284 | 978,210 | 956,420 |
| pen-cloned-v1 | 496,264 | 496,264 | 492,509 | 481,244 | 462,469 |

H=10 的跨边界排除比例分别约 3.9385% 与 6.8082%，分母为未过滤的
`N-H+1` 个候选起点；日志还保存相对于全部单步起点的不可用比例。
这不是丢失原始数据，而是训练时不使用不足 H 或跨 episode 的窗口。

数据 SHA-256：

```text
/home/lrh/d4rl_datasets/door-cloned-v1.hdf5
1f6a60ecc4d2a2e5c02f41dd3412be61bb265a33dcf120ceee9e9682d4fe4d48
/home/lrh/d4rl_datasets/pen-cloned-v1.hdf5
6d1a982cee1950501bedebbddedbcbb377e6284d379986c83abac8a6c0be0c22
```

## 汇总产物与正式运行边界

汇总脚本已对 16 个短测生成 `matrix-summary/summary.md`、`summary.json`、
`per_seed.csv` 和两个任务的 online 曲线 PNG，没有排除项或警告。
图表与文档醒目标注 **SMOKE VALIDATION / not performance evidence**。
其作用是验收 endpoint、AUC 和配对差值的报告链路，不用于判断 N 是否优于 B0。

正式矩阵已经 dry-run 核对为 32 条独立链，配置仍是每条 1M offline + 1M online、
seeds=1/2、每次评估 50 episodes。需要单独明确启动，不能把本轮 smoke 目录续作正式结果。
先阅读 [两任务协议与命令](CHUNK_PROTOCOL.md) 和两份 YAML，正式输出使用新目录。
