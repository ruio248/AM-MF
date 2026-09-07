# AM-MF 原 Target 版本：Note Average

本文只描述保持 AM-MF target 不变的版本。对应 agent：

```text
agents/am_meanflow.py
agent_name=am_meanflow
am_target_variant=note_average
```

该类会拒绝其他 `am_target_variant`，避免把原 AM/Note target 与
`meanflowql_reformulated_adjoint` 版本混用。

## 1. 原 target 定义

令：

```math
s_\alpha=\alpha r+(1-\alpha)t.
```

冻结 behavior MeanFlow 给出参考速度：

```math
u_{\mathrm{pre}}
=u_{\mathrm{pre}}(o,x_{s_\alpha},s_\alpha,t).
```

在 EMA rollout 得到的 `x_t` 上计算：

```math
\lambda(t)
=J_{F_{\bar\theta}}(x_t)^\top
\nabla_aQ_{\bar\phi}
\left(o,F_{\bar\theta}(x_t,t)\right).
```

原 AM-MF reward-guided average velocity 是：

```math
\boxed{
u_{Q,\mathrm{note}}^*
=u_{\mathrm{pre}}
+\eta(t-s_\alpha)\lambda(t)
}
```

`(t-s_alpha)` 不能删除。它把 endpoint reward 的局部梯度转换成当前区间上的
平均速度修正，并使短区间上的修正自然减弱。

## 2. 完整 AlphaFlow target

EMA actor 给出 bootstrap：

```math
u_{\mathrm{boot}}
=u_{\bar\theta}(o,x_r,r,s_\alpha).
```

原 target 为：

```math
\boxed{
y_{\mathrm{original}}
=\alpha u_{Q,\mathrm{note}}^*
+(1-\alpha)u_{\mathrm{boot}}
}
```

loss：

```math
\boxed{
L_{\mathrm{original}}
=\frac{1}{\alpha}
\left\|
u_\theta(o,x_r,r,t)
-\operatorname{sg}(y_{\mathrm{original}})
\right\|_2^2
}
```

默认 `normalize_am_loss_by_alpha=True` 对应正文形式。设为 `False` 只改变 loss
尺度，不改变 target 公式，应单独标记为 unnormalized ablation。

## 3. 端点行为

### `alpha=1`

此时 `s_alpha=r`，target 完全由 reward-guided 分支决定：

```math
y=u_{\mathrm{pre}}(o,x_r,r,t)+\eta(t-r)\lambda(t).
```

这个实验用于单独检验原始 adjoint average-velocity correction。

### 正的小 alpha

由于：

```math
t-s_\alpha=\alpha(t-r),
```

原 target 中进入 `u_Q^*` 的 Q correction 会随 alpha 同时缩小。最终混合 target
又乘一次 alpha，因此 target 内 Q offset 的量级为：

```math
\alpha\eta(t-s_\alpha)\lambda
=\eta\alpha^2(t-r)\lambda.
```

这是原 target 的数学行为，不应在 strict 版本中偷偷归一化掉。

### `alpha=0`

零点使用独立 endpoint-MeanFlow JVP 分支。此时 correction interval 为零：

```math
u_{Q,\mathrm{note}}^*=u_{\mathrm{pre}}.
```

因此原 target 的显式 Q correction 在微分极限自然消失。JVP target 为：

```math
y_{\mathrm{JVP}}
=u_{\mathrm{pre}}
+(1-t)
\frac{d}{dt}u_{\bar\theta}(o,x_t,t,1).
```

运行严格零点：

```bash
--agent.alpha_mode=fixed \
--agent.alpha_value=0
```

## 4. 代码边界

原 target 的专属方法是：

```text
AMMeanFlowAgent._correct_reward_velocity()
```

它只执行：

```python
base_velocity + adjoint_eta * interval * adjoint
```

其余 rollout、VJP、bootstrap、loss、EMA、critic 和 checkpoint 逻辑由同一个
`AMMeanFlowAgent` 管理。

默认配置必须满足：

```text
agent_name=am_meanflow
am_target_variant=note_average
q_coef=0
native_regularizer_coef=0
```

`q_coef=0` 是因为 Q 已通过 adjoint target 作用于 actor。若再启用 Native
Direct-Q，需要标记为 hybrid，而不是原 target AM-MF。

## 5. 测试

```bash
python -m pytest -q \
  tests/test_am_meanflow.py \
  tests/test_am_meanflow_agent.py
```

原 target 专属检查包括：

- `u_pre + eta*(t-s)*lambda` 的闭式数值；
- correction 正号；
- alpha 正文 target 的 `0/1/中间值`；
- target stop-gradient；
- endpoint adjoint 与有限差分一致；
- 正 alpha update 后 actor 有限且 target actor EMA 精确；
- AM update 后 pre actor 逐元素不变；
- `alpha=0` 确实进入 JVP 分支；
- 原 target 在零区间的 `correction_norm=0`；
- checkpoint round-trip 保留 frozen pre、EMA 和阶段计数器。

## 6. 最小 smoke run

```bash
MUJOCO_GL=egl python main_meanflowql.py \
  --run_group=am_mf_original_target_smoke \
  --env_name=cube-triple-play-singletask-task2-v0 \
  --agent=agents/am_meanflow.py \
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
  --agent.alpha_mode=fixed \
  --agent.alpha_value=0.5 \
  --agent.adjoint_eta=0.1 \
  --agent.num_candidates=1
```

正式运行时无需传 `am_target_variant`，因为本 agent 已固定为
`note_average`。如果显式传入其他值，创建阶段会失败。

## 7. 必须记录的指标

```text
am/alpha
am/jvp_branch
am/adjoint_norm
am/correction_norm
am/reward_target_loss
am/bootstrap_loss
am/mixed_target_loss
am/endpoint_q
critic/critic_loss
grad/norm
evaluation/success
```

原 target 的关键诊断是：`correction_norm` 是否随 `(t-s_alpha)` 缩小，以及小
alpha 时 Q correction 是否按公式变弱。如果它变弱，这是方法本身的预期行为，
不是实现 bug。

## 8. 实验命名

建议名称：

```text
am_mf_original_target
am_mf_note_average
am_mf_original_target_alpha1
am_mf_original_target_zero_jvp
```

不要把变化 target 的结果写在这些名称下。
