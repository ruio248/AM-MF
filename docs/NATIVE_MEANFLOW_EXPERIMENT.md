# 原生 MeanFlow 实验流程

本文档描述仓库中的原生 MeanFlow 基线：它学习区间平均速度
`u(o, x_t, r, t)`，并可选地使用 critic 对一步动作端点做 Direct-Q
优化。它与 `agents/meanflowql.py` 中的官方 MeanFlowQL endpoint
reformulation 相互独立，不共享 checkpoint 参数树。

## 1. 实现边界

新增代码位于：

| 文件 | 作用 |
| --- | --- |
| `utils/native_meanflow.py` | `(r,t)` 采样、JVP target、一步端点映射和反向 transport |
| `agents/native_meanflow.py` | Native MeanFlow actor、critic、Direct-Q、EMA target 和 K-step 采样 |
| `tests/test_native_meanflow.py` | 数学恒等式、边界条件和梯度方向测试 |
| `tests/test_native_meanflow_agent.py` | 创建、更新、EMA、候选动作和 K1/K4 agent smoke test |
| `agents/__init__.py` | 注册 `native_meanflow` agent |

原始 `agents/meanflowql.py` 保持不变。原生实现使用单独的
`NativeMeanFlowAgent`，因此不能直接加载 MeanFlowQL actor checkpoint；
critic 参数也只有在结构完全一致且显式转换时才可迁移。

## 2. 数学定义

给定数据动作 `a` 和高斯噪声 `epsilon`，使用线性概率路径：

```math
x_t = (1-t)a + t\epsilon, \qquad v = \epsilon-a.
```

网络显式预测从当前时间 `t` 到目标时间 `r` 的区间平均速度：

```math
u_\theta(o,x_t,r,t), \qquad 0 \le r \le t \le 1.
```

训练 target 使用 MeanFlow JVP identity：

```math
u_{\mathrm{target}}
= v-(t-r)\frac{d}{dt}u_\theta(o,x_t,r,t),
```

其中总导数沿数据路径计算，并固定 `r`：

```math
\frac{d u}{dt}
= \nabla_x u\,v + \partial_t u.
```

target 在进入 MSE 前执行 `stop_gradient`。当 `r=t` 时，target 自动退化为
瞬时条件速度 `v`，这也是 `flow_ratio` 所控制的边界样本。

从 `t=1` 的噪声一步生成 `t=0` 动作时：

```math
\hat a = \epsilon-u_\theta(o,\epsilon,0,1).
```

K-step 诊断把 `[0,1]` 等分，在每个反向区间使用：

```math
x_r=x_t-(t-r)u_\theta(o,x_t,r,t).
```

当前标准环境评估入口调用 K1 `sample_actions()`；K4/K10 通过
`sample_actions_nfe(..., num_steps=K)` 提供给诊断脚本和测试。不要把增加
NFE 的结果与 K1 结果混成同一组计算预算。

## 3. 与原 MeanFlowQL 的区别

| 项目 | 原 MeanFlowQL | Native MeanFlow |
| --- | --- | --- |
| 网络输入 | `(o, x_t, t)` | `(o, x_t, r, t)` |
| 输出语义 | reformulated endpoint `g(o,x_t,t)` | interval-average velocity `u(o,x_t,r,t)` |
| 训练 target | `x_t + (t-1)v - t dg/dt` | `v - (t-r) du/dt` |
| 一步动作 | 网络在 `t=1` 直接输出 endpoint | `epsilon - u(o,epsilon,0,1)` |
| 多步 transport | 非原生接口 | 显式区间反向 transport |
| target actor | 无独立 target actor | EMA `target_actor_bc_flow` |

因此，Native MeanFlow 是机制诊断与独立实验基线，不能被描述成对
`meanflowql.py` 增加一个普通 consistency MSE。

## 4. 环境准备

完整 OGBench/MuJoCo 实验建议在 Linux + NVIDIA GPU 上运行：

```bash
conda env create -f environment.yml
conda activate flowrl
```

无显示器服务器使用 EGL：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

确认关键依赖和设备：

```bash
python -c "import jax, flax, ogbench; print(jax.devices())"
```

下载仓库列出的全部数据集：

```bash
python download_all_datasets.py
```

只准备本文示例所需的 cube-triple 数据集：

```bash
python -c "import ogbench; ogbench.download_datasets(['cube-triple-play-v0'], dataset_dir='./dataset')"
```

代码中的 OGBench 数据路径固定为仓库根目录下的 `./dataset`，所以训练命令
应从仓库根目录执行。

## 5. 测试流程

先运行纯数学测试：

```bash
python -m pytest -q tests/test_native_meanflow.py
```

再运行 agent 级测试：

```bash
python -m pytest -q tests/test_native_meanflow_agent.py
```

最后运行全部 Native MeanFlow 测试：

```bash
python -m pytest -q \
  tests/test_native_meanflow.py \
  tests/test_native_meanflow_agent.py
```

测试覆盖：

- `0 <= r <= t <= 1` 和 `r=t` 边界样本；
- affine field 的 JVP 闭式解；
- MeanFlow target 的 stop-gradient；
- 一步 endpoint map 与 reverse transport 等价性；
- 常速度场上的 K1/K4 一致性；
- Direct-Q 梯度方向；
- agent 注册、创建和一次真实 update；
- pretrain 时 critic 参数隔离；
- actor/critic target 的精确 EMA；
- best-of-N 和 K-step 输出 shape、有限值及非法参数检查。

## 6. 最小 smoke experiment

下面的命令只用于确认数据加载、网络初始化、JIT、日志和 checkpoint 链路。
它不是可报告的正式结果：

```bash
MUJOCO_GL=egl python main_meanflowql.py \
  --run_group=native_meanflow_smoke \
  --env_name=cube-triple-play-singletask-task2-v0 \
  --agent=agents/native_meanflow.py \
  --seed=0 \
  --offline_steps=20 \
  --online_steps=0 \
  --log_interval=5 \
  --eval_interval=0 \
  --save_interval=20 \
  --enable_early_stopping=False \
  --wandb_online=False \
  --agent.batch_size=32 \
  --agent.num_candidates=1
```

成功标准：

- 训练进入第一个 update 且没有 NaN/Inf；
- `meanflow/mean_flow_loss`、`actor/q_loss` 和 `critic/critic_loss` 可记录；
- `exp/<project>/<run_group>/<run>/flags.json` 被创建；
- 第 20 步生成 `params_20.pkl`。

## 7. Native MeanFlow SFT

SFT 只训练原生 MeanFlow identity。将 `q_coef`、`critic_coef` 和
`bound_loss_weight` 都设为 0，可以复用同一训练入口且不更新 critic：

```bash
MUJOCO_GL=egl python main_meanflowql.py \
  --run_group=native_meanflow_sft \
  --env_name=cube-triple-play-singletask-task2-v0 \
  --agent=agents/native_meanflow.py \
  --seed=0 \
  --offline_steps=1000000 \
  --online_steps=0 \
  --log_interval=5000 \
  --eval_interval=100000 \
  --save_interval=100000 \
  --enable_early_stopping=False \
  --wandb_online=True \
  --agent.meanflow_coef=1.0 \
  --agent.q_coef=0.0 \
  --agent.critic_coef=0.0 \
  --agent.bound_loss_weight=0.0 \
  --agent.num_candidates=1
```

这组实验用于回答“完整 `(r,t)` MeanFlow 是否能在动作数据上稳定学习”，不应
被标成 RL 结果。重点记录：

- `meanflow/mean_flow_loss`；
- `meanflow/identity_residual`；
- `meanflow/instantaneous_fraction`；
- `meanflow/mean_interval`；
- K1 与 K4 的动作差异和环境成功率。

## 8. Native MeanFlow + Direct-Q

正式 Direct-Q 实验同时训练 critic、MeanFlow identity 和一步 endpoint 的
Q 目标：

```bash
MUJOCO_GL=egl python main_meanflowql.py \
  --run_group=native_meanflow_direct_q \
  --env_name=cube-triple-play-singletask-task2-v0 \
  --agent=agents/native_meanflow.py \
  --seed=0 \
  --offline_steps=1000000 \
  --online_steps=0 \
  --pretrain_factor=0.1 \
  --log_interval=5000 \
  --eval_interval=100000 \
  --save_interval=100000 \
  --early_stopping_metric=evaluation/success \
  --early_stopping_patience=10 \
  --wandb_online=True \
  --agent.meanflow_coef=10.0 \
  --agent.q_coef=1.0 \
  --agent.critic_coef=1.0 \
  --agent.num_candidates=5 \
  --agent.q_agg=mean
```

`pretrain_factor=0.1` 会先执行 `0.1 * offline_steps` 次 behavior-only
`pretrain()`，随后执行完整的 `offline_steps` 次 Direct-Q update。它增加了总
update 数，做公平对照时必须让其他方法使用相同预算，或者单独报告预训练成本。

## 9. Checkpoint 恢复

继续训练某个 checkpoint：

```bash
MUJOCO_GL=egl python main_meanflowql.py \
  --run_group=native_meanflow_resume \
  --env_name=cube-triple-play-singletask-task2-v0 \
  --agent=agents/native_meanflow.py \
  --seed=0 \
  --restore_path='/absolute/path/to/run_directory' \
  --restore_epoch=100000 \
  --offline_steps=100000 \
  --online_steps=0 \
  --eval_interval=10000 \
  --save_interval=100000 \
  --wandb_online=False
```

恢复时必须使用 `agents/native_meanflow.py` 和兼容的网络尺寸。原 MeanFlowQL
checkpoint 不能作为 Native MeanFlow actor 直接恢复。

## 10. 实验矩阵与记录

最小可解释对照建议为：

| 组别 | Agent | `meanflow_coef` | `q_coef` | `critic_coef` | 目的 |
| --- | --- | ---: | ---: | ---: | --- |
| A | Native MeanFlow | 1 | 0 | 0 | 验证原生 identity/SFT |
| B | Native MeanFlow | 10 | 1 | 1 | 验证原生 MeanFlow + Direct-Q |
| C | Original MeanFlowQL | 原论文配置 | 原实现 | 原实现 | 同任务主基线 |

正式比较至少固定：

- environment 和 dataset 版本；
- seed 集合；
- offline/online update 数；
- batch size、网络宽度和深度；
- critic 结构、discount、tau 和 Q aggregation；
- `num_candidates` 与评估 episode 数；
- checkpoint 选择规则和 early stopping 规则；
- K1 或 K-step 的 NFE 预算。

每个 run 建议记录：

```text
commit:
environment:
seed:
mode: native_sft | native_direct_q | original_meanflowql
dataset:
offline_steps:
pretrain_steps:
num_candidates:
evaluation_nfe:
best_success:
final_success:
checkpoint:
wandb_url:
notes:
```

## 11. 常见问题

### JAX 看不到 GPU

先运行 `python -c "import jax; print(jax.devices())"`。若只显示 CPU，检查当前
Conda 环境是否安装了与服务器 CUDA 驱动兼容的 `jaxlib`/CUDA plugin。

### MuJoCo EGL 初始化失败

确认 `MUJOCO_GL=egl`、NVIDIA 驱动库和 MuJoCo 库在运行进程的
`LD_LIBRARY_PATH` 中。不要只在另一个 shell 中设置变量。

### OGBench 找不到数据

本仓库从 `./dataset` 读取数据。确认在仓库根目录启动命令，并确认对应基础
dataset 已下载。

### 第一次 update 很慢

JAX 会为网络、JVP 和 optimizer 做首次编译。先用 `offline_steps=20` 的 smoke
命令区分正常 JIT 时间和真正的进程卡死。

### K1 与 K4 不一致

一般网络下二者不要求完全相等；常速度场才应严格一致。正式实验中把差异作为
一致性诊断，并分别报告 NFE，不能把 K4 的额外计算隐藏在 K1 基线中。
