# Online 阶段的 Adjoint：是否需要永久冻结的 Prior？

> 状态：研究假设与实验建议；尚未修改算法，也尚未通过消融验证。
>
> 基础分支：`global-mean-field-control-chunk`。本文讨论训练目标与 teacher 的来源，不依赖特定 chunk size。

## 核心论点

当前 N 方法在前 500,000 次**离线 learner 更新**后冻结行为 prior，后续仍以它的速度场为基础生成受控 teacher 轨迹。进入 online 阶段后，actor、critic 和所访问的状态分布都会变化，永久固定的 teacher 底座可能无法反映当前策略已经学到的行为。

值得检验的替代设计是：**不让冻结 prior 长期主导 online teacher 轨迹；改用当前 actor 的停止梯度快照生成轨迹和控制目标，同时继续使用离线＋在线 replay 上的 MeanFlow/BC 约束。** 这不等于取消 BC，也不等于仅凭 best-of-5 选择就完成策略学习。该论点是因果假设，不是由现有 B0–N 差距直接证明的结论。

## 当前实现：四个角色需要分清

| 角色 | 当前实现 | Online 阶段是否更新 |
| --- | --- | --- |
| 执行策略 / student | 当前 `actor_bc_flow`；生成候选动作 | 是 |
| 冻结 prior | `pre_actor_params`，提供 teacher 基础速度 | 否；达到 `behavior_warmup_updates` 后停止更新 |
| EMA actor | `target_actor_params`，提供 Q 梯度回传所用的 endpoint 映射 | 是，按 EMA 更新 |
| critic | 当前 critic 与 target critic | 是 |

实现证据：[`agents/am_meanflow_note.py`](../agents/am_meanflow_note.py) 中，`adjoint()` 通过 EMA actor 的 endpoint 映射对 target Q 求 VJP；`teacher_field()` 将该信号加到冻结 prior 的速度上；`teacher_path()` 再做 RK2 积分。`_finish_update()` 在 warmup 之后不再更新 `pre_actor_params`。Door/Pen 配置的 `behavior_warmup_updates` 均为 500,000，属于 learner update 数，不是 50M 次环境交互。

当前 N 在 warmup 后主要最小化受控轨迹的区间 transport、Jacobian matching 与边界损失；默认 `note_adjoint` 变体的直接 actor Q loss 为零。因此 B0 与 N 不只是“有无 adjoint”的区别，现有差距不能单独归因于冻结 prior。

## 与已有算法的关系

- **MeanFlowQL：**同一个 MeanFlow actor 接受直接 `-Q(s,a)` 目标和 MeanFlow Identity（MFI）行为约束；online 收集的 transition 进入 replay，继续使用相同训练目标。动作执行与 Bellman target 可使用 value-guided best-of-K；best-of-K 是选动作规则，不是 BC 损失的替代品。参见[论文算法](https://arxiv.org/html/2511.13035)及[官方在线循环](https://github.com/HiccupRL/MeanFlowQL/blob/main/main_meanflowql.py)。
- **FQL：**单独的 BC flow 提供 one-step actor 的蒸馏目标；actor 同时优化 Q 和蒸馏损失。原论文的 offline-to-online 方案将新 transition 加入数据，继续训练各组件，而非默认永久冻结 BC flow。参见[论文算法与在线细节](https://arxiv.org/html/2502.02538)。
- **原始 lean Adjoint Matching：**由当前微调策略采样轨迹，但保留固定 base flow 作为控制参照；lean adjoint 沿所采样轨迹反向计算时使用 base drift 的状态 Jacobian。因而“使用当前策略轨迹”不等于“删掉所有固定参照”。如果彻底移除 base，则不能直接援引原始 AM 的理论结论。参见[原始论文算法](https://arxiv.org/html/2409.08861)。确定性 AM-MF 与其随机最优控制设定之间也需单独论证。

## 可检验的设计方向（尚未实现）

1. **执行不变。** Online rollout 仍由当前 actor 生成候选，并由 critic 选 best-of-5；不让冻结 prior 直接执行动作。
2. **保留持续学习的行为约束。** 在离线＋在线 replay 上继续优化 MFI/BC，同时保留 actor 的直接 Q 目标。这样“取消冻结 prior”不会被混同为“取消 BC”。
3. **使 teacher 跟随当前策略。** 每次构造训练 target 时取当前 actor 的停止梯度快照。由该快照给出轨迹基础速度，并尽量使用同一快照定义 endpoint 回传，避免“冻结 prior 速度＋另一套 EMA endpoint”描述不同动力学。轨迹与 adjoint 作为 target，不穿过 teacher 对快照参数反传。
4. **准确命名。** 上述基于当前快照的 endpoint-VJP／轨迹蒸馏是一个 AM-MF 设计变体；它本身不是原始论文的严格 lean adjoint ODE。如需严格 lean AM，应另行实现当前策略采样轨迹、固定 base 参照与反向伴随递推。

## 最小消融与判定标准

先在不引入 chunk 差异的设置下，固定数据、critic、更新预算、评估种子及 best-of-5 规则，比较：

1. B0：持续 `Q + MFI` 的基线。
2. `Q + MFI + adjoint`，teacher 由冻结 prior 主导。
3. 与第 2 组相同损失和控制强度，但 teacher 使用当前 actor 快照。
4. 为第 2、3 组各设 `control_eta=0` 对照，隔离额外 teacher/transport 目标与真实控制信号的贡献。

报告每个 seed 的 offline 终点、online 起点和终点，以及 online 回报曲线与 AUC；同时记录当前 actor 与冻结 prior 的 endpoint 差异、两种回传方向的余弦相似度、控制项相对基础速度的范数、动作越界率、MFI/BC loss 和 actor Q loss。

若第 3 组在其他条件一致时稳定优于第 2 组，才支持“冻结 prior 阻碍 online 适应”的论点。若两组都输给 B0，还需检验 adjoint 信号、控制强度及蒸馏目标本身，而不能把失败都归于 prior。
