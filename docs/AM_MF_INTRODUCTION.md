# AM-MF 的两套版本：原 Target 与 MeanFlowQL Target

本仓库现在提供两条明确分开的 AM-MF 路线：

| 版本 | Agent | 底层策略参数化 | Target contract |
| --- | --- | --- | --- |
| AM target 不变版 | `agents/am_meanflow.py` | Native interval-average actor | `note_average` |
| MeanFlowQL target 变化版 | `agents/am_meanflow_target_changed.py` | MeanFlowQL direct map `g(o,x_t,t)` | `meanflowql_reformulated_adjoint` |

第二版的“target 变化”指的是：MeanFlowQL 已把原生 MeanFlow 的平均速度 target
改写成 direct-map target，因此 AM 必须作用在 MeanFlowQL 的新 target 中。它不是
Native MeanFlow target 变化版，也不是删除原 AM 公式中的 `(t-s)`。

## 1. 版本 A：AM target 保持不变

对应：

```text
agents/am_meanflow.py
agent_name=am_meanflow
am_target_variant=note_average
```

该版本在 Native MeanFlow actor 参数化上保留原 AM/Note 结构：

```math
s_\alpha=\alpha r+(1-\alpha)t,
```

```math
u_Q^*
=u_{\mathrm{pre}}+\eta(t-s_\alpha)\lambda(t),
```

```math
y_{\mathrm{original}}
=\alpha u_Q^*
+(1-\alpha)u_{\bar\theta}(o,x_r,r,s_\alpha).
```

它包含 frozen pre actor、EMA target actor、target critic adjoint、AlphaFlow
bootstrap 和严格 `alpha=0` JVP 分支。完整说明见
`AM_MF_ORIGINAL_TARGET.md`。

## 2. 版本 B：AM 融入 MeanFlowQL target

对应：

```text
agents/am_meanflow_target_changed.py
agent_name=am_meanflow_target_changed
am_target_variant=meanflowql_reformulated_adjoint
```

MeanFlowQL 的网络直接预测：

```math
g_\theta(o,x_t,t),
```

官方 target 为：

```math
g_{\mathrm{tgt}}
=x_t+(t-1)v-tD_t^v g_\theta,
\qquad v=\epsilon-a.
```

`g` 在中间时间对应的隐含动作端点是：

```math
F_g(o,x_t,t)
=(1-t)x_t+t g(o,x_t,t).
```

target critic 通过该端点产生 adjoint：

```math
\lambda_t
=J_{F_g}(x_t)^\top
\nabla_aQ_{\bar\phi}(o,F_g(o,x_t,t)).
```

保留 AM 区间缩放后，action-to-noise 路径速度改为：

```math
v_{\mathrm{AM}}=v-\eta t\lambda_t.
```

最终把它代入 MeanFlowQL target，并沿 `v_AM` 重新计算 JVP：

```math
\boxed{
g_{\mathrm{tgt}}^{\mathrm{AM}}
=x_t+(t-1)v_{\mathrm{AM}}
-tD_t^{v_{\mathrm{AM}}}g_\theta
}
```

完整推导和运行说明见 `AM_MF_CHANGED_TARGET.md`。

## 3. 两个版本的结构差异

| 维度 | AM target 不变版 | MeanFlowQL target 变化版 |
| --- | --- | --- |
| 父类 | `NativeMeanFlowAgent` | `MeanFlowQL_Agent` |
| actor 输入 | `(o,x_t,r,t)` | `(o,x_t,t)` |
| actor 输出 | interval-average velocity `u` | direct map `g` |
| 一步动作 | `epsilon-u(o,epsilon,0,1)` | `g(o,epsilon,1)` |
| AM Q 注入 | Note reward velocity target | MeanFlowQL path velocity与 `g_tgt` |
| JVP | Native/endpoint JVP | MeanFlowQL reformulated JVP |
| target actor | EMA target actor | 不新增；使用 frozen current snapshot 构造 endpoint |
| pre actor | 冻结 | 不需要 |
| checkpoint 来源 | Native MeanFlow | MeanFlowQL |
| alpha 语义 | AlphaFlow mixture | MeanFlowQL MFI/BC coefficient |

两个版本不共享 actor checkpoint；即使 action dimension 相同，输入签名和输出语义
也不相同。

## 4. 为什么 MeanFlowQL 版不能继承 Native AM agent

若让第二版继续继承 `AMMeanFlowAgent`，会出现三个根本错误：

- 网络仍接收 `(r,t)`，并不是 MeanFlowQL 的 single-time direct map；
- target 仍然是 `v-(t-r)D_tu`，而不是 MeanFlowQL 的 `g_tgt`；
- 一步动作仍通过 `epsilon-u` 生成，而不是直接调用 `g(epsilon,1)`。

当前第二版直接继承 `MeanFlowQL_Agent`，并保留其网络参数树、采样接口、adaptive
MFI loss、critic 和动态 BC 系数。

## 5. 阶段边界

MeanFlowQL target 变化版采用两阶段语义：

```text
pretrain(): 原 MeanFlowQL target，use_am=False
update():   AM-guided MeanFlowQL target，use_am=True
```

这样 critic guidance 不会在 behavior pretrain 时提前进入。若
`pretrain_factor=0`，则从第一步 update 开始使用 AM target；这应被标记为
no-pretrain ablation。

## 6. Q 注入规则

标准 changed-target 版本只让 Q 通过 stopped adjoint 进入 MFI target：

```text
meanflowql_direct_q_coef=0
```

若再设置非零 Direct-Q，实验属于 hybrid：

```text
meanflowql_direct_q_coef>0
```

边界损失仍沿用 MeanFlowQL 的 `bound_loss_weight`，因为 direct map 在训练初期可能
输出动作范围外的值。

## 7. Checkpoint 导入

原 target 版从 Native agent 转换：

```python
original_am = AMMeanFlowAgent.from_native_agent(native, original_config)
```

MeanFlowQL target 变化版从 MeanFlowQL agent 转换：

```python
changed_am = AMMeanFlowTargetChangedAgent.from_meanflowql_agent(
    meanflowql,
    changed_config,
)
```

不要交叉使用这两个转换接口。

## 8. 测试入口

```bash
python -m pytest -q \
  tests/test_native_meanflow.py \
  tests/test_native_meanflow_agent.py \
  tests/test_am_meanflow.py \
  tests/test_am_meanflow_agent.py \
  tests/test_am_meanflow_target_changed.py
```

测试职责：

| 文件 | 职责 |
| --- | --- |
| `test_am_meanflow.py` | 原 AM/Note 数学工具、AlphaFlow 和 endpoint JVP |
| `test_am_meanflow_agent.py` | 原 target agent、freeze、EMA、checkpoint |
| `test_am_meanflow_target_changed.py` | MeanFlowQL endpoint、AM-guided `g_tgt`、eta=0 等价、更新和迁移 |

## 9. 最小实验矩阵

| Run | Agent | 目的 |
| --- | --- | --- |
| Native | `agents/native_meanflow.py` | Native MeanFlow 基线 |
| AM-Original | `agents/am_meanflow.py` | AM target 保持不变 |
| MeanFlowQL | `agents/meanflowql.py` | 官方 reformulated target 基线 |
| AM-MQL-0 | `agents/am_meanflow_target_changed.py`, `eta=0` | MeanFlowQL target 等价控制 |
| AM-MQL | `agents/am_meanflow_target_changed.py`, `eta>0` | AM 融入 MeanFlowQL target |

所有对照应记录各自 checkpoint schema。AM-Original 与 AM-MQL 是两种不同 actor
参数化上的算法比较，不应声称只改变一个标量超参数。

## 10. 文件清单

```text
agents/am_meanflow.py
agents/am_meanflow_target_changed.py
utils/am_meanflow.py
tests/test_am_meanflow.py
tests/test_am_meanflow_agent.py
tests/test_am_meanflow_target_changed.py
docs/AM_MF_ORIGINAL_TARGET.md
docs/AM_MF_CHANGED_TARGET.md
```
