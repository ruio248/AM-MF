# consistency_eval 一致性判别模块

## 1. 目的

`consistency_eval` 使用固定的 observation-noise probe，检查 MeanFlow 模型在强化
学习前后是否仍保持可用的一致性映射。它独立于环境 success/return，也不会把轨迹
是否笔直直接等同于 MeanFlow consistency。

模块支持当前仓库中的四类 agent：

| 参数化 | 支持的 agent |
| --- | --- |
| Native MeanFlow 区间平均速度 | `native_meanflow`、`am_meanflow` |
| MeanFlowQL reformulated direct map | `meanflowql`、`meanflowql_beta`、`am_meanflow_target_changed` |

实现文件：

```text
utils/consistency_eval.py   # 可复用数学指标和 agent adapter
consistency_eval.py         # checkpoint 命令行评估入口
tests/test_consistency_eval.py
```

## 2. 为什么必须分开两种参数化

Native MeanFlow 输出完整区间平均速度：

\[
u_\theta(o,x_t,r,t), \qquad 0\le r < t\le 1.
\]

MeanFlowQL 输出的是单时间 direct map：

\[
g_\theta(o,x_t,t).
\]

它隐含的动作端点为：

\[
F_g(o,x_t,t)=(1-t)x_t+t g_\theta(o,x_t,t).
\]

MeanFlowQL 不提供任意 \((r,t)\) 区间的原生平均速度，因此不能伪造一个
`split_consistency_mse`。模块只对 Native MeanFlow 报告 split identity；对
MeanFlowQL 则检查其真实存在的 endpoint-map consistency。

## 3. 三层核心判别

### 3.1 Native MeanFlow interval split identity

当前 Native MeanFlow 使用 endpoint-state convention：网络在 \(x_t\) 上预测从
\(r\) 到 \(t\) 的平均速度。直接区间位移是：

\[
\Delta_{r\leftarrow t}=(t-r)u_\theta(o,x_t,r,t).
\]

在中间时间 \(s\) 分段后的位移是：

\[
\Delta_{r\leftarrow s\leftarrow t}
=(t-s)u_\theta(o,x_t,s,t)
+(s-r)u_\theta(o,x_s,r,s).
\]

一致性要求：

\[
\boxed{\Delta_{r\leftarrow t}\approx\Delta_{r\leftarrow s\leftarrow t}}.
\]

对应指标：

```text
split_consistency_mse
split_consistency_nmse
split_consistency_mse_r0_s5_t10
split_consistency_mse_r0_s2_t5
split_consistency_mse_r2_s5_t10
...
```

这一项只出现在 `native_meanflow` 和 target 不变版 `am_meanflow` 中。

### 3.2 Endpoint-map consistency

先用固定噪声执行一条 K-step 参考轨迹：

\[
x_1\rightarrow x_{(K-1)/K}\rightarrow\cdots\rightarrow x_0.
\]

然后从每个 \(x_t\) 计算网络预测的单步动作端点 \(F_\theta(x_t,t)\)，并与同一条
参考轨迹的最终 \(x_0\) 比较：

\[
\mathcal E_F(t)=\left\|F_\theta(o,x_t,t)-x_0\right\|^2.
\]

Native MeanFlow 使用：

\[
F_u(o,x_t,t)=x_t-t u_\theta(o,x_t,0,t).
\]

MeanFlowQL 使用：

\[
F_g(o,x_t,t)=(1-t)x_t+t g_\theta(o,x_t,t).
\]

对应指标：

```text
endpoint_map_mse
endpoint_map_nmse
endpoint_map_mse_t1
...
endpoint_map_mse_t10
```

其中 NMSE 用整条参考轨迹的端点位移能量归一化，便于不同任务和动作尺度之间
比较。

### 3.3 Endpoint Jacobian/JVP consistency

AM-MF 的 single-step adjoint 依赖 endpoint map 的 Jacobian：

\[
\lambda_t
=J_{F_\theta}(x_t)^\top\nabla_a Q(o,F_\theta(x_t)).
\]

因此只检查函数值一致并不充分。模块在固定随机单位方向 \(d\) 上比较：

\[
J_{F_\theta}(x_t)d
\]

和从同一状态继续执行剩余分段 rollout 的经验映射 \(\Phi_K\) 的 JVP：

\[
J_{\Phi_K}(x_t)d.
\]

对应指标：

```text
endpoint_jvp_mse
endpoint_jvp_nmse
endpoint_jvp_mse_t1
...
endpoint_jvp_mse_t10
jacobian_probe_pairs
```

这个指标对 AM-MF 尤其重要，因为 adjoint 的信用信号会经过该 Jacobian。

## 4. 公共辅助指标

两类模型都报告：

```text
k1_k2_mse
k1_k4_mse
k1_k10_mse
k1_action_std
k10_action_std
k1_clip_fraction
k10_clip_fraction
trajectory_turning_residual
trajectory_turning_active_fraction
trajectory_path_length_ratio
```

- `k1_kN_mse`：同一 observation 和同一初始噪声下，一步端点和 N 步端点之差；
- `action_std`：检测动作坍缩；
- `clip_fraction`：检测动作大量落在 critic 或环境支持域外；
- `turning_residual`：轨迹转向程度；
- `path_length_ratio`：轨迹长度与端点直线距离之比。

`turning_residual` 和 `path_length_ratio` 只是轨迹几何辅助量。弯曲轨迹可能仍然
满足 endpoint consistency，笔直轨迹也可能映射到错误端点，因此不能用它们替代
前面三层结构指标。

## 5. 从 checkpoint 运行

训练目录必须包含：

```text
flags.json
params_<STEP>.pkl
```

在仓库根目录执行：

```bash
MUJOCO_GL=egl python consistency_eval.py \
  --run_dir=/absolute/path/to/run \
  --restore_epoch=1000000 \
  --validation_states=1024 \
  --noises_per_state=4 \
  --eval_seed=20260824 \
  --eval_nfes=1,2,4,10 \
  --trajectory_steps=10 \
  --inference_batch_size=256 \
  --jacobian_probe_pairs=128
```

脚本会从 `flags.json` 自动恢复：

- `agent_name` 和完整 agent config；
- 环境名称；
- observation normalization 设置；
- `offline_steps`、`online_steps` 和 `pretrain_factor`；
- 训练 seed。

默认输出：

```text
<run_dir>/consistency_eval_step1000000_seed20260824.json
```

也可以显式指定：

```bash
--output_path=/absolute/path/to/result.json
```

## 6. 使用共享 probe 做公平对照

比较 Native MeanFlow、MeanFlowQL 和两种 AM-MF 时，必须使用同一批 observation 和
noise。可以准备：

```python
import numpy as np

np.savez(
    "shared_consistency_probe.npz",
    observations=normalized_observations,
    noises=fixed_noises,
)
```

然后每个 checkpoint 都传入：

```bash
--probe_path=/absolute/path/to/shared_consistency_probe.npz
```

共享 probe 内的 observations 必须采用对应训练运行的输入口径。如果不同运行使用
完全相同的训练集 normalization stats，可以共享规范化后的 observations；如果
normalization 不同，应先统一协议，不能在 evaluator 中静默混用。

## 7. 二值判别阈值

模块默认只报告连续指标，不默认宣布“通过”或“失败”。不同动作维数、任务、训练
阶段和 probe 分布没有一个理论上通用的数值阈值。

正式实验应先在同任务的 behavior-only/reference checkpoint 上校准阈值，然后显式
传入：

```bash
--max_endpoint_map_nmse=0.05 \
--max_endpoint_jvp_nmse=0.10 \
--max_k1_k_reference_mse=0.01
```

Native MeanFlow 还可以加：

```bash
--max_split_consistency_nmse=0.05
```

输出中的 `judgement` 会包含每个阈值的 value、maximum 和 passed。如果不传阈值，
则输出：

```json
{
  "status": "not_requested",
  "passed": null,
  "checks": {}
}
```

MeanFlowQL 没有 Native interval split target。如果给 MeanFlowQL 传入
`--max_split_consistency_nmse`，脚本会明确报错，而不会生成一个没有数学定义的
split 判别结果。

## 8. 建议实验协议

每个 task、seed、checkpoint 和 candidate 配置固定使用：

```text
1024 validation states × 4 noises = 4096 state-noise pairs
trajectory_steps = 10
NFE = 1, 2, 4, 10
inference_batch_size = 256
jacobian_probe_pairs = 128
eval_seed = 20260824
```

最少同时报告：

```text
success / return
k1_k10_mse
endpoint_map_mse / nmse
endpoint_jvp_mse / nmse
split_consistency_mse / nmse（仅 Native）
action_std
clip_fraction
```

建议 checkpoint 对照：

```text
Native MeanFlow behavior-only boundary
AM-MF target 不变版 final
MeanFlowQL baseline final
AM-MF target 变化版 final
```

这样可以分别判断性能、endpoint function consistency、endpoint Jacobian
consistency、Native interval split identity，以及性能变化是否伴随动作越界、坍缩
或 K1/K10 分叉。

## 9. 单元测试

```bash
python -m pytest -q tests/test_consistency_eval.py
```

测试包括：

- 常数 Native velocity 的 split、endpoint、JVP 零误差；
- 精确 MeanFlowQL endpoint map 的 K1/KN、endpoint、JVP 零误差；
- 非一致 direct map 产生非零误差；
- 当前时间方向和 rollout 状态顺序；
- 非法 shape、NFE 和 split triplet 拒绝；
- 显式阈值的 pass/fail 判别；
- Native AM-MF 与 MeanFlowQL target-changed adapter 分流。
