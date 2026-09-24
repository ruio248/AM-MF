# QAM's time schedule and a proposed N control ablation

Status: source-checked implementation note, **not** an AM-MF result or a claim of equivalence to QAM. This note changes no training code or existing experiment configuration.

## What the official QAM implementation does

Source: [ColinQiyangLi/qam, commit `2726d767c9a0a7a46d49693f0391f73dc2cf58ac`](https://github.com/ColinQiyangLi/qam/tree/2726d767c9a0a7a46d49693f0391f73dc2cf58ac), especially [`agents/qam.py` lines 45–93](https://github.com/ColinQiyangLi/qam/blob/2726d767c9a0a7a46d49693f0391f73dc2cf58ac/agents/qam.py#L45-L93), [lines 116–131](https://github.com/ColinQiyangLi/qam/blob/2726d767c9a0a7a46d49693f0391f73dc2cf58ac/agents/qam.py#L116-L131), and [default configuration](https://github.com/ColinQiyangLi/qam/blob/2726d767c9a0a7a46d49693f0391f73dc2cf58ac/agents/qam.py#L366-L404). See also the [QAM paper, Section 4](https://arxiv.org/html/2601.14234).

QAM uses generation time `s=0` at Gaussian noise and `s=1` at the action. With `K=flow_steps`, `h=1/K`, and grid time `s_i=i/K`, its implementation uses

$$
\sigma_i = \sqrt{\frac{2(1-s_i+h)}{s_i+h}}.
$$

The `+h` terms regularize the endpoint singularities. The default is `K=10`. The terminal lean-adjoint initialization is `g_K = -tau * grad_a Q(s_env, a_K)`, where `tau` is the fixed `inv_temp` configuration value (default `0.3`); the adjoint is then propagated backward through the behavior drift using VJPs. QAM does **not** configure a separate scalar named `eta(t)`.

The QAM actor matching residual is

$$
\frac{2(f_\theta-f_\beta)}{\sigma_i}+\sigma_i g_i.
$$

Setting this residual to zero gives the **algebraic target**

$$
f_\theta-f_\beta=-c_i g_i,\qquad
c_i=\frac{\sigma_i^2}{2}=\frac{1-s_i+h}{s_i+h}.
$$

For the default `K=10`, `c_i` is 11 at `s=0`, 1 at `s=0.5`, and 0.2 at `s=0.9`. This is a time-dependent coefficient implied by the loss, **not** a literal multiplier applied by the action sampler and **not** the measured correction norm. That norm also depends on the propagated `g_i`, the learned networks, and how closely the matching residual is optimized. `inv_temp` scales the terminal Q gradient; it is not a generation-time schedule.

## What N currently does

In this repository, generation time is reversed: `t=1` is noise and `t=0` is the action ([`utils/note_flow.py`](../utils/note_flow.py)). The [N teacher field](../agents/am_meanflow_note.py) uses

$$
\lambda_t=\nabla_{x_t}Q_{\rm tar}\bigl(o,\operatorname{clip}(F_{\rm EMA}^{0\leftarrow t}(o,x_t))\bigr),
\qquad
v_{\rm guided}(o,x_t,t)=v_{\rm pre}(o,x_t,t)-\eta\lambda_t.
$$

The current `control_eta=0.1` is constant **within a generated trajectory**. `lambda_t` still varies with time and action-space state. The guided field is re-evaluated during RK2 integration, after which interval-average velocities from the controlled path supervise the student. The [chunk protocol](CHUNK_PROTOCOL.md) fixes `teacher_steps=8` and keeps the same coefficient for all action chunk sizes.

QAM's `g_i` and N's `lambda_t` are different objects: QAM solves a reverse lean-adjoint recursion on a memoryless SDE trajectory; N pulls target-critic gradients through an EMA endpoint map and then integrates a deterministic controlled teacher. QAM's schedule and its optimality argument therefore do not transfer automatically to N. A constant N coefficient is a design choice worth testing, not a demonstrated code bug.

## Proposed schedule implementation for N

Keep `control_eta` as the overall amplitude and add an explicit `control_time_schedule` setting. The default must be `constant` so existing checkpoints and results retain their meaning. Calculate the effective coefficient **inside** `teacher_field` at each actual RK2 evaluation time, including midpoint evaluations:

$$
v_{\rm guided}(o,x_t,t)=v_{\rm pre}(o,x_t,t)
-\eta_{\rm eff}(t)\lambda_t,
\qquad
\eta_{\rm eff}(t)=\eta_0 w(t).
$$

Three modes would isolate the shape question:

| Mode | Weight in N's reverse-time convention | Interpretation |
| --- | --- | --- |
| `constant` | `w(t)=1` | Existing N behavior; exact regression check. |
| `linear_remaining` | `w(t)=2t` | Bounded schedule, zero at the action endpoint; mean weight 1 over the RK2 midpoint grid. This is an ablation, not QAM's derivation. |
| `qam_shape_bounded` | `w(t)=min((t+h)/(1-t+h), cap) / Z`, `h=1/teacher_steps` | QAM coefficient after the time reversal `s=1-t`, capped and normalized for a controlled ablation. |

For `qam_shape_bounded`, define `Z` as the arithmetic mean of the capped numerator over the `teacher_steps` RK2 midpoint times `t_j=1-(j+1/2)/teacher_steps`. A cap such as `3` is a proposed *experimental setting*, not a value from QAM. It avoids the raw factor `teacher_steps+1` at the noise endpoint (`9` when N uses eight teacher steps). Normalization keeps the midpoint-grid mean of `eta_eff` equal to the existing `eta_0`; it does not equate the mean of `||eta_eff(t) lambda_t||`, which must be measured. Never identify N's `eta_0=0.1` numerically with QAM's `inv_temp`.

Implementation touchpoints, if this ablation is approved for code:

1. Put a pure, JAX-compatible time-weight helper in `utils/note_flow.py`; add the new settings and validation in `agents/am_meanflow_note.py`. Apply the weight only in `teacher_field`, so both ordinary teacher trajectories and the Jacobian teacher use the same field.
2. Add schedule keys to the N YAMLs under `configs/chunk/`, pass them through the existing `make_config` path in `agents/chunked.py`, and include them in effective configuration/checkpoint validation. B0 configuration remains unchanged. Any CLI override in `main_chunked.py` must also be recorded in the manifest.
3. Log `eta_eff(t)`, `||lambda_t||`, and `||eta_eff(t) lambda_t||` by generation-time bin, as well as the fraction at the cap. Update the current fixed-`t=0.5` control metric to use `eta_eff(0.5)`.
4. Test constant-mode parity with the current implementation, finite weights at every RK2 stage, the intended time direction, zero-control parity, and rejection of a checkpoint whose saved schedule differs. Then compare matched N runs at fixed task, H, seed, critic, and environment-step budget; report offline endpoint and online curves/AUC alongside the realized control norms.

First compare `constant` against `linear_remaining`; add the bounded QAM-shaped mode once the implementation and time-binned diagnostics are verified. Any improvement would be evidence about N's control schedule, not a reproduction of QAM's stochastic-control theorem.
