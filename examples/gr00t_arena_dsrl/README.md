# GR00T Arena DSRL (latent-noise steering)

DSRL ([Diffusion Steering via Reinforcement Learning](https://arxiv.org/abs/2506.15799),
DSRL keeps the **whole VLA frozen** and trains only a small SAC policy over the flow-matching **initial noise `x0`**:

```
obs ──frozen backbone──▶ pooled VL features ┐
obs ──processor────────▶ raw state  ────────┴─▶ noise actor (tanh Gaussian, ~0.5M params)
                                                   │  steering noise x0  (the SAC action)
                                                   ▼
                              frozen flow head, deterministic Euler ODE
                                                   │
                                                   ▼
                                              env action chunk
```

- **Actor**: `verl_vla.models.dsrl.DSRLNoiseActor` — MLP over the frozen pooled
  backbone features + raw state, outputs one tanh-bounded noise vector
  (`max_action_dim`, GR00T: 128) broadcast over the action horizon.
- **Critic**: the existing SAC critic ensemble, scoring the *steering noise*
  (defaults auto-switch to `action_dim=max_action_dim`, `action_horizon=1`).
- **Replay**: `full_action` stores the steering noise; `action` stays the
  decoded env chunk. Trainer / replay pool / env are unchanged.
- **Generic**: the same `adapter.dsrl.*` keys work for pi0/pi05
  (`model/adapter/pi0.yaml`); for pi0 also set `critic.input_dim`
  to `prefix_embed_dim + state_dim + max_action_dim` and `flow_sde_enable=False`.

## Launch

Same Docker / paths as `examples/gr00t_arena_sac` (see its README):

```bash
# GR1 fridge task
ARENA_TASK=gr1 INNER_SCRIPT=examples/gr00t_arena_dsrl/run_gr00t_arena_dsrl.sh \
  OUTPUT_ROOT=/eval/outputs/arena_gr00t_gr1_dsrl \
  examples/gr00t_arena_sac/run_docker.sh

# Arena LIBERO (Franka)
ARENA_TASK=libero TASK_SUITE=libero_spatial TASK_ID=3 \
  INNER_SCRIPT=examples/gr00t_arena_dsrl/run_gr00t_arena_dsrl.sh \
  OUTPUT_ROOT=/eval/outputs/arena_gr00t_libero_dsrl \
  examples/gr00t_arena_sac/run_docker.sh
```

## Key knobs (env vars)

| Var | Default | Meaning |
| --- | --- | --- |
| `NOISE_ACTOR_LR` | `3e-4` | Noise-actor lr (`actor.optim.lr`; the VLA is frozen) |
| `CRITIC_LR` | `3e-4` | SAC critic lr |
| `CRITIC_TAU` | `0.005` | Target critic Polyak coefficient |
| `AUTO_ENTROPY` | `True` | SAC entropy auto-tuning |
| `TARGET_ENTROPY` | `-64.0` | Target entropy over the 128-dim steering noise (≈ −dim/2) |
| `BACKUP_ENTROPY` | `False` | Keep the −α·logπ term out of the critic TD target (RLinf parity) |
| `CRITIC_WARMUP_STEPS` | `100` | Critic-only steps before actor updates |
| `EMA_DECAY` | `null` | Actor EMA over the tiny noise actor (null = off) |
| `CRITIC_POOL_PROJ_DIM` | `0` | Critic pooled-feature projection (SAC baseline 256) |
| `CRITIC_LAYERNORM` | `True` | LayerNorm in critic heads (SAC baseline True) |
| `ACTOR_POSITIVE_SAMPLE_RATIO` | `0.8` | Positive replay ratio for actor batches |
| `EVAL_EPISODES` | `GPUs×NUM_ENV` | Trajectories averaged per eval SR |
| `EPISODIC_REPLAY` | `True` | Episodic replay collection |

The SAC launcher's `FREEZE_ACTION_IO` / `FLOW_SDE_*` knobs are intentionally
absent: DSRL freezes the whole VLA and owns the exploration noise
(`flow_sde_enable=true` raises at model init).

Adapter-level knobs live under `cluster.actor_rollout_ref.model.adapter.dsrl.*`
(`hidden_dims`, `feature_latent_dim`, `state_latent_dim`, `noise_per_step`,
`noise_bound`) — see `src/verl_vla/workflows/config/model/adapter/gr00t.yaml`.

## Caveats

- Mutually exclusive with `flow_sde_enable` (DSRL owns the exploration noise).
- TD3+BC and offline RLPD prefill are incompatible: demos are env actions,
  while the DSRL SAC action space is the steering noise.
- Eval (`eval=True`) uses the deterministic steering noise `tanh(mean)`.
- Checkpoints: the frozen policy is exported unchanged; the noise actor is
  saved alongside as `dsrl_noise_actor.pt` (critic as `critic.pt`).
