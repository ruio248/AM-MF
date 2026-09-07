# 两个任务的小范围有效性测试

本目录只保存**运行脚本和实验说明**，不保存新的训练日志、checkpoint 或结果。本轮
小范围验证的两个任务固定为：

1. `humanoidmaze-large-navigate-singletask-task1-v0`（OGBench，主指标为 success）；
2. `relocate-cloned-v1`（D4RL Adroit，主指标为 normalized return）。

它们分别覆盖长时域导航与高维灵巧手操作；前者用于观察 C5 候选选择在稀疏成功率
任务中的作用，后者用于观察 online adaptation 是否能改善高维动作任务。

## 1. 论文口径与已报告结果

MeanFlowQL 论文的候选动作数记为 `K`。为避免和本项目历史记录混淆，本文档把它称为
`C`（`num_candidates`）：`C5` 表示生成五个动作、由 Q 选择一个；`C1` 表示只生成一个
动作。它与 MeanFlow 的积分/NFE 步数不是同一个量。

| 任务 | 论文主评估设置 | MeanFlowQL 原文结果 | 指标 |
| --- | --- | ---: | --- |
| `humanoidmaze-large-navigate-singletask-task1-v0` | offline、C5 | `53 ± 5` | success (%) |
| `relocate-cloned-v1` | offline C5 -> offline-to-online C5 | `1 ± 1 -> 19 ± 8` | D4RL normalized return |

来源：MeanFlowQL [Table 1（HumanoidMaze Large Task1）](https://arxiv.org/html/2511.13035v1#S5.T1)、[Table 2（Relocate）](https://arxiv.org/html/2511.13035v1#S5.T2) 与 [Appendix E.2](https://arxiv.org/html/2511.13035v1#S5.SS2)。论文在这两个任务上都设置：

```text
alpha          = 10000
num_candidates = 5
time_steps     = 50
```

论文**不报告 C1 的任务分数**，因此 C1 只能作为本项目的诊断项，不能写成论文复现值。
论文也不报告 `humanoidmaze-large-navigate-singletask-task1-v0` 单任务的 online 数字；
因此该任务的在线运行脚本用于方法筛选，而不是用于复写一个不存在的原文单任务 online
表格值。

训练预算需要按 benchmark 分开理解：论文 Appendix D 说明 state-based OGBench offline
训练为 1M gradient steps、D4RL offline 训练为 500K gradient steps；其 Table 2 又将
offline-to-online 结果表述为在 1M offline steps 后开始 online fine-tuning，并报告 1M
和 2M 的数值。因此 `run_relocate_cloned.sh` 将两种预算显式拆开：`offline` 默认 500K，
`online` 默认 1M offline + 1M online；任一数值都可通过 `OFFLINE_STEPS` 和
`ONLINE_STEPS` 显式覆盖。正式论文级复现前，应先固定采用其中一种口径，并对所有方法
保持完全一致。

## 2. 运行脚本

从仓库根目录运行。脚本只会启动训练；默认 `WANDB_ONLINE=False`，结果会写入本地
`exp/` 和 `wandb_offline/`，不会写进 Git 仓库。

```bash
# HumanoidMaze Large Task1：offline / offline-to-online
bash scripts/small_scale_validation/run_humanoidmaze_large_task1.sh offline 1
bash scripts/small_scale_validation/run_humanoidmaze_large_task1.sh online 1

# Relocate Cloned：offline / offline-to-online
bash scripts/small_scale_validation/run_relocate_cloned.sh offline 1
bash scripts/small_scale_validation/run_relocate_cloned.sh online 1
```

两个脚本默认执行论文 MeanFlowQL baseline：

```text
--agent=agents/meanflowql.py
--agent.alpha=10000
--agent.num_candidates=5
--agent.time_steps=50
```

Humanoid 的脚本固定 `--agent.discount=0.995`，与官方仓库给出的该任务命令一致；
Relocate 使用 MeanFlowQL 默认的 `--agent.discount=0.99`。脚本还固定
`--balanced_sampling=0`、observation normalization、50 episodes/100K-step evaluation，
并关闭 early stopping 以避免实际训练预算少于请求预算。

## 3. C1/C5 的使用边界

论文的主配置为 C5。若将脚本中的环境变量改为：

```bash
NUM_CANDIDATES=1 \
bash scripts/small_scale_validation/run_humanoidmaze_large_task1.sh offline 1
```

则训练和采样都会切换为 C1。这是一个**新的 C1 训练协议**，不再是原论文 C5 主结果。
如果目标是“同一 C5 checkpoint 上的 C1/C5 配对评估”，应使用固定的 episode seeds 和
相同的第一份 candidate noise；当前训练入口未提供独立的 `eval_num_candidates` 参数，
不能仅凭一次训练命令无歧义地完成该配对评估。

## 4. 更改策略时应传入的参数

不要修改脚本里的训练逻辑。用 `AGENT_PATH` 选择 agent 文件，用 `EXTRA_AGENT_FLAGS`
传入该 agent 已有的 `--agent.*` 配置项即可。下面都是参数名称，不表示已经得到实验结果。

| 策略 | `AGENT_PATH` | 需要关注/修改的传参名称 |
| --- | --- | --- |
| 原始 MeanFlowQL | `agents/meanflowql.py` | `--agent.alpha`、`--agent.num_candidates`、`--agent.time_steps`、`--agent.discount`、`--agent.consistency_alpha` |
| MeanFlowQL + consistency | `agents/meanflowql.py` | `--agent.consistency_alpha`（大于 `0` 开启）以及上述 baseline 参数 |
| 原生 Native MeanFlow | `agents/native_meanflow.py` | `--agent.meanflow_coef`、`--agent.q_coef`、`--agent.critic_coef`、`--agent.bound_loss_weight`、`--agent.num_candidates` |
| 原始 AM-MF target | `agents/am_meanflow.py` | `--agent.adjoint_eta`、`--agent.am_loss_coef`、`--agent.native_regularizer_coef`、`--agent.alpha_mode`、`--agent.alpha_value`、`--agent.alpha_floor` |
| AM 作用于 MeanFlowQL target | `agents/am_meanflow_target_changed.py` | `--agent.adjoint_eta`、`--agent.meanflowql_direct_q_coef`、`--agent.alpha`、`--agent.num_candidates`、`--agent.time_steps` |

例如，启动“AM 作用于 MeanFlowQL target”的 Humanoid offline 筛选：

```bash
AGENT_PATH=agents/am_meanflow_target_changed.py \
EXTRA_AGENT_FLAGS='--agent.adjoint_eta=0.1 --agent.meanflowql_direct_q_coef=0.0' \
bash scripts/small_scale_validation/run_humanoidmaze_large_task1.sh offline 1
```

其中 `meanflowql_direct_q_coef=0.0` 表示 Q 只通过 AM target 进入 actor；设为大于零时，
会形成“AM target + 原始 MeanFlowQL Direct-Q”的混合消融，必须在结果中明确标注。

脚本会识别四个已实现 agent：`meanflowql.py`、`native_meanflow.py`、
`am_meanflow.py`、`am_meanflow_target_changed.py`。只有 MeanFlowQL 与
changed-target AM-MF 接收 `--agent.alpha` 和 `--agent.time_steps`；Native MeanFlow
与原始 AM-MF 会自动跳过这两个不属于其配置的参数。

每次切换策略时，仍应固定以下外部变量：`env_name`、`seed`、`offline_steps`、
`online_steps`、`alpha`、`num_candidates`、`time_steps`、`discount`、normalization、
`balanced_sampling`、评估 episode 数和评估随机种子。否则得到的是策略与协议同时变化的
比较，不能用于判断 AM 或 consistency 本身是否有效。
