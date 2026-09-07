# AM-MF Target 变化版本：融入 MeanFlowQL Reformulation

本文描述第二套实现：由于仓库原有 `agents/meanflowql.py` 已经把原生 MeanFlow
速度场改写成直接动作映射 `g(o,x_t,t)`，AM 也必须相应地融入这个 reformulated
target，而不是继续使用 Native MeanFlow 的 `u` target。

对应代码：

```text
agents/am_meanflow_target_changed.py
agent_name=am_meanflow_target_changed
am_target_variant=meanflowql_reformulated_adjoint
```

## 1. MeanFlowQL 改变了什么

数据路径仍采用 action-to-noise 时间：

```math
x_t=(1-t)a+t\epsilon,
\qquad
v=\epsilon-a,
\qquad 0\le t\le1.
```

原生 MeanFlow 网络预测平均速度 `u(o,x_t,b,t)`；MeanFlowQL 将它改写为：

```math
g(o,x_t,b,t)=x_t-u(o,x_t,b,t).
```

当前仓库固定 `b=0`，网络接口因此是：

```text
g_theta(observation, state, time)
```

在 `t=1` 时，`g_theta(o,epsilon,1)` 直接输出动作，不再执行
`epsilon-u_theta(...)`。这也是 `MeanFlowQL_Agent.sample_actions()` 的真实接口。

仓库原 MeanFlowQL target 是：

```math
\boxed{
g_{\mathrm{tgt}}
=x_t+(t-1)v-tD_t^v g_\theta(o,x_t,t)
}
```

其中：

```math
D_t^v g_\theta
=\nabla_xg_\theta\,v+\partial_tg_\theta.
```

因此，target 变化版必须改造 `g_tgt`，而不是回到 Native MeanFlow 的
`v-(t-r)D_tu`。

## 2. `g` 对应的动作端点

虽然 `g` 在 `t=1` 直接等于动作，但在中间时间不能简单把 `g(o,x_t,t)` 当作
完整 endpoint。由：

```math
u=x_t-g
```

以及 Native 一步动作映射：

```math
F=x_t-tu
```

可得 MeanFlowQL 的隐含动作端点：

```math
\boxed{
F_g(o,x_t,t)
=(1-t)x_t+t\,g(o,x_t,t)
}
```

边界正确性：

```math
F_g(o,x_0,0)=x_0,
\qquad
F_g(o,x_1,1)=g(o,x_1,1).
```

实现计算 adjoint 时使用 frozen current actor snapshot 构造 `F_g`。这样不改变
MeanFlowQL 的网络参数树，也不凭空增加一个与官方 checkpoint 不兼容的 target
actor。

## 3. MeanFlowQL endpoint adjoint

target critic 在隐含动作端点上定义价值：

```math
Q_{\bar\phi}(o,F_g(o,x_t,t)).
```

对当前 noisy state 的 adjoint 为：

```math
\boxed{
\lambda_t
=J_{F_g}(x_t)^\top
\nabla_aQ_{\bar\phi}(o,F_g(o,x_t,t))
}
```

代码使用 `jax.vjp` 计算，不显式构造 Jacobian。endpoint、adjoint 和最终
regression target 都执行 `stop_gradient`。

critic 输入默认先裁剪到 `[-1,1]`：

```text
clip_actions_for_critic=True
```

这能避免在 AM 早期让 target critic 对严重越界动作提供未经数据支持的梯度。

## 4. AM-guided MeanFlowQL 路径速度

`v=epsilon-a` 指向 action-to-noise，而 `lambda_t` 指向提高 endpoint Q 的反向
生成方向。AM 修正后的路径速度为：

```math
\boxed{
v_{\mathrm{AM}}
=v-\eta(t-b)\lambda_t
}
```

当前 `b=0`，所以：

```math
v_{\mathrm{AM}}=v-\eta t\lambda_t.
```

这里有两个重要因素：

- 负号处理 MeanFlowQL 的 action-to-noise 时间方向；
- `t-b` 保留原 AM 的区间长度缩放，靠近动作端点时修正自然消失。

反向 transport 的新增位移是：

```math
-(t-b)(v_{\mathrm{AM}}-v)
=\eta(t-b)^2\lambda_t,
```

它与 Q 上升方向一致。

## 5. 完整的变化 target

JVP 必须沿 AM-guided velocity 重新计算：

```math
D_t^{v_{\mathrm{AM}}}g_\theta
=\nabla_xg_\theta\,v_{\mathrm{AM}}
+\partial_tg_\theta.
```

将 `v_AM` 代入 MeanFlowQL reformulation：

```math
\boxed{
g_{\mathrm{tgt}}^{\mathrm{AM}}
=x_t+(t-b-1)v_{\mathrm{AM}}
-(t-b)D_t^{v_{\mathrm{AM}}}g_\theta
}
```

当前仓库 `b=0`：

```math
\boxed{
g_{\mathrm{tgt}}^{\mathrm{AM}}
=x_t+(t-1)v_{\mathrm{AM}}
-tD_t^{v_{\mathrm{AM}}}g_\theta
}
```

actor 的 MeanFlow identity loss 为：

```math
L_{\mathrm{AM\text{-}MFI}}
=\operatorname{AdaptiveL2}
\left(
g_\theta(o,x_t,t)
-\operatorname{sg}(g_{\mathrm{tgt}}^{\mathrm{AM}})
\right).
```

实现继续复用 MeanFlowQL 的 adaptive MFI weighting、target clip `[-5,5]` 和可选
consistency loss。

## 6. `eta=0` 是硬回归条件

当：

```text
adjoint_eta=0
```

有：

```math
v_{\mathrm{AM}}=v,
```

所以：

```math
g_{\mathrm{tgt}}^{\mathrm{AM}}
=x_t+(t-1)v-tD_t^v g_\theta,
```

即仓库现有的 MeanFlowQL target。测试不仅检查公式，还会以相同 seed、网络、
batch 和 RNG 分别调用：

```text
MeanFlowQL_Agent.meanflow_loss()
AMMeanFlowTargetChangedAgent.meanflow_loss()
```

并要求 `eta=0` 时 loss 数值一致、`correction_norm=0`、
`target_shift_norm=0`。

## 7. 预训练与 AM 阶段

`pretrain()` 阶段不使用 critic guidance，严格训练原始 MeanFlowQL target：

```text
use_am=False
```

正常 `update()` 阶段切换为：

```text
use_am=True
```

这样避免尚未训练好的 critic 在行为模型预训练阶段污染 target。预训练只更新
actor；正常阶段同时执行 critic TD update 和 AM-guided MFI update。

## 8. 与 MeanFlowQL Direct-Q 的关系

原 MeanFlowQL 另有 endpoint Direct-Q actor loss。当前 AM 标准版本已经通过
adjoint 把 Q 写入 MFI target，所以默认关闭重复的 Direct-Q：

```text
meanflowql_direct_q_coef=0.0
```

边界损失仍然保留：

```math
L_{\mathrm{actor-extra}}
=c_{\mathrm{bound}}L_{\mathrm{bound}}.
```

若显式设置：

```text
meanflowql_direct_q_coef>0
```

则总目标中同时存在 adjoint target 和直接 policy gradient，必须把实验标成
`AM-MeanFlowQL + Direct-Q hybrid`，不能当作标准 target 变化版。

## 9. `alpha` 的含义

这个 agent 继承的 `alpha` 是 MeanFlowQL 的 MFI/BC loss 权重：

```math
L=L_Q+L_{\mathrm{bound}}+\alpha L_{\mathrm{AM\text{-}MFI}}.
```

它不是原 AM/Note 版本中用于 reward/bootstrap mixture 的 AlphaFlow `alpha`。
MeanFlowQL 原有的 cosine 或 dynamic alpha 调度继续有效，但实验记录必须把这两个
同名参数的语义区分开。

## 10. Checkpoint 兼容性

第二版直接继承 `MeanFlowQL_Agent`，网络参数树保持：

```text
modules_actor_bc_flow
modules_critic
modules_target_critic
```

它没有 Native MeanFlow 的四输入 actor，也没有
`modules_target_actor_bc_flow`。因此可以显式地从已恢复的 MeanFlowQL agent
切换：

```python
from agents.am_meanflow_target_changed import AMMeanFlowTargetChangedAgent

am_meanflowql = AMMeanFlowTargetChangedAgent.from_meanflowql_agent(
    meanflowql_agent,
    changed_config,
)
```

转换保留网络参数、optimizer state、动态 alpha 状态和 loss history，只替换 agent
类型与 AM target 配置。

不能用 `from_native_agent()` 导入 Native MeanFlow checkpoint，因为两者 actor
输入和输出语义不同。

## 11. 测试

专项测试：

```bash
python -m pytest -q tests/test_am_meanflow_target_changed.py
```

测试覆盖：

- MeanFlowQL 隐含 endpoint map 的 `t=0/t=1/中间值`；
- AM interval scaling 和 action-to-noise 修正符号；
- 反向路径修正确实沿 Q 上升方向；
- reformulated target 的闭式数值；
- `eta=0` 的公式退化；
- `eta=0` 与原 `MeanFlowQL_Agent.meanflow_loss()` 数值一致；
- changed agent 确实继承 MeanFlowQL，而不是 Native agent；
- 两者相同 seed 初始化得到相同网络参数；
- pretrain 使用无 AM 的原 target，且不更新 critic；
- update 使用 AM reformulated target；
- 默认 Direct-Q loss 为零；
- 采样继续使用 MeanFlowQL 的 direct map；
- MeanFlowQL agent 原位转换与 checkpoint round-trip。

完整回归：

```bash
python -m pytest -q \
  tests/test_native_meanflow.py \
  tests/test_native_meanflow_agent.py \
  tests/test_am_meanflow.py \
  tests/test_am_meanflow_agent.py \
  tests/test_am_meanflow_target_changed.py
```

## 12. 最小 smoke run

```bash
MUJOCO_GL=egl python main_meanflowql.py \
  --run_group=am_meanflowql_reformulated_smoke \
  --env_name=cube-triple-play-singletask-task2-v0 \
  --agent=agents/am_meanflow_target_changed.py \
  --seed=0 \
  --offline_steps=20 \
  --online_steps=0 \
  --pretrain_factor=0.5 \
  --log_interval=5 \
  --eval_interval=0 \
  --save_interval=20 \
  --enable_early_stopping=False \
  --wandb_online=False \
  --agent.batch_size=32 \
  --agent.adjoint_eta=0.1 \
  --agent.meanflowql_direct_q_coef=0.0 \
  --agent.num_candidates=1 \
  --agent.action_mode=normal
```

## 13. 必须记录的指标

```text
meanflow/am_enabled
meanflow/adjoint_norm
meanflow/correction_norm
meanflow/conditional_velocity_norm
meanflow/guided_velocity_norm
meanflow/derivative_norm
meanflow/baseline_target_norm
meanflow/target_norm
meanflow/target_shift_norm
meanflow/remaining_time
meanflow/endpoint_q
meanflow/endpoint_out_of_bounds_fraction
meanflow/mean_flow_loss
actor/bound_loss
actor/q_loss
critic/critic_loss
alpha_weight
evaluation/success
```

尤其需要观察：

- `correction_norm / conditional_velocity_norm` 是否过大；
- `target_shift_norm` 是否随 eta 单调增大；
- critic 预测的 `endpoint_q` 上升是否对应真实环境 return；
- endpoint 越界率是否因为 AM guidance 增大。

## 14. 推荐对照

| Run | Agent | 配置 | 目的 |
| --- | --- | --- | --- |
| MQL | `agents/meanflowql.py` | 官方配置 | 原 MeanFlowQL |
| AM-MQL-0 | `agents/am_meanflow_target_changed.py` | `eta=0`, Direct-Q 依实验对齐 | target 等价回归 |
| AM-MQL | `agents/am_meanflow_target_changed.py` | `eta>0`, Direct-Q=0 | AM 融入 reformulated target |
| AM-MQL-H | `agents/am_meanflow_target_changed.py` | `eta>0`, Direct-Q>0 | hybrid ablation |
| AM-Original | `agents/am_meanflow.py` | Note/AlphaFlow 配置 | AM target 不变版 |

标准 AM-MQL 与原 MeanFlowQL 的比较会同时改变 Q 的注入方式；若要做严格 target
等价检查，应先比较 `AM-MQL-0` 与关闭/对齐 Direct-Q 后的 MeanFlowQL。
