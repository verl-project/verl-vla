# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers for IsaacLabArenaEnv."""

import logging

logger = logging.getLogger(__name__)


def arena_task_success_reward(env, success_func, success_params):
    """RL reward = Arena composite-task success (+1.0 the step the whole task is solved).

    Reuses the sequential-task success function so the *stateful* subtask state
    machine (``_current_subtask_idx`` / ``_subtask_success_state`` /
    ``env.extras['subtask_success_state']``) is advanced exactly once per step --
    it now runs inside the RewardManager instead of the (removed) TerminationManager.
    ``IsaacLabArenaEnv.env_step`` derives ``done`` / ``success_once`` from this
    reward, so this term is what makes a success visible to training (the raw
    Arena task defines no reward term at all).
    """
    success = success_func(env, **success_params)
    return success.float()


def arena_subtask_graded_reward(env, success_func, success_params):
    """Graded RL reward for a SEQUENTIAL task = fraction of subtasks completed.

    Returns 0 / 0.5 / 1.0 (for a 2-subtask task) — the latched progress, so e.g. +0.5 once
    pick-and-place is done and +1.0 once the door is also closed. ``success_func`` (the
    sequential-task success fn) is still called first so it advances the subtask state machine
    and writes ``env.extras['subtask_success_state']`` (a per-env list of per-subtask latched
    bools); we read that to compute the graded progress. This gives the long-horizon task an
    early (PnP) learning signal instead of a single composite +1 only after BOTH subtasks.

    NB: with this reward, ``reward > 0`` no longer means full success, so the
    Arena env wrapper must derive episode success from the composite threshold.
    """
    import torch

    composite = success_func(env, **success_params)  # advances state machine + writes extras
    state = getattr(env, "extras", {}).get("subtask_success_state", None)
    if not state:
        return composite.float()
    progress = torch.tensor(
        [(sum(1 for x in s if x) / max(len(s), 1)) for s in state],
        device=composite.device,
        dtype=torch.float32,
    )
    return progress


def build_env_cfg_without_recorder(env_builder):
    """Build the Arena env cfg and disable IsaacLab's demo/metric recorder.

    BaseEnv owns rollout/video recording for verl-vla. IsaacLab-Arena's internal
    recorder is aimed at demo datasets and task metrics; in RL mode we also move
    the success termination into a reward and disable terminations, which makes
    Arena's success-rate recorder assert during reset. Disable the internal
    recorder manager entirely before env construction.

    Returns the built ``env_cfg`` (ready to be patched further / handed to
    ``env_builder.make_registered``).
    """
    _, env_cfg = env_builder.build_registered()
    env_cfg.recorders = None
    return env_cfg


def apply_rl_reward_and_disable_autoreset(env_cfg, subtask_reward: bool = False) -> None:
    """Turn the Arena composite-success TERMINATION into a sparse RL REWARD and
    disable IsaacLab auto-reset (verl owns episode resets + horizon).

    Mirrors the LIBERO RL setup (franka_libero_rl_env_cfg + isaac_env.py):
      * success DoneTerm -> RewTerm(weight = 1 / step_dt), so the Arena wrapper
        can derive episode success from reward. The raw Arena task otherwise has
        no reward term, so a success would be invisible to training.
      * every termination term -> None, so reset_buf stays False (no auto-reset
        mid-rollout, which would corrupt fixed-length trajectories).

    RL-only: patches the env cfg built inside verl; the shared Arena task /
    eval / mimic configs are untouched. Gate off with
    ``env.train.rl_success_reward=False``.

    Args:
        env_cfg: the Arena env cfg to patch in place.
        subtask_reward: if True, use the graded subtask reward (0/0.5/1.0 = fraction
            of subtasks done) for earlier long-horizon credit, vs the single
            composite +1. Gate: ``env.train.subtask_reward``.
    """
    import dataclasses

    from isaaclab.managers import RewardTermCfg
    from isaaclab.utils import configclass

    term_cfg = getattr(env_cfg, "terminations", None)
    succ_term = getattr(term_cfg, "success", None) if term_cfg is not None else None
    if succ_term is None:
        logger.warning("[arena_env] terminations.success not found; skipping RL success-reward patch")
        return

    # step_dt = sim.dt * decimation (Arena default 1/200 * 4 = 0.02s -> 50 Hz).
    # RewardManager scales every term by step_dt, so weight = 1/step_dt emits
    # exactly +1.0 per step the task is solved (matches LIBERO weight=20 @ 0.05s).
    sim_dt = float(getattr(getattr(env_cfg, "sim", None), "dt", 1.0 / 200.0))
    decimation = int(getattr(env_cfg, "decimation", 4))
    step_dt = sim_dt * decimation
    weight = 1.0 / step_dt

    @configclass
    class _ArenaRLRewardsCfg:
        task_success: RewardTermCfg = None

    # Sequential-task option: graded subtask reward vs the single composite +1.
    reward_func = arena_subtask_graded_reward if subtask_reward else arena_task_success_reward

    rewards = _ArenaRLRewardsCfg()
    rewards.task_success = RewardTermCfg(
        func=reward_func,
        weight=weight,
        params={"success_func": succ_term.func, "success_params": succ_term.params},
    )
    env_cfg.rewards = rewards

    # Disable auto-reset: null every termination term (composite success now
    # lives in the reward above; subtask terms like object_dropped and any
    # time_out are dropped so verl controls when envs reset).
    disabled = [f.name for f in dataclasses.fields(term_cfg)]
    for name in disabled:
        setattr(term_cfg, name, None)

    logger.info(
        "[arena_env] RL patch: success->RewTerm weight=%.3f (step_dt=%.4fs); "
        "terminations disabled (%s) -> no auto-reset",
        weight,
        step_dt,
        ", ".join(disabled) or "none",
    )


_LIGHTWHEEL_SSL_PATCHED = False


def disable_lightwheel_ssl_verify() -> None:
    """Skip TLS cert verification for lightwheel asset-registry calls only.

    Arena loads the kitchen/object USDs from the lightwheel registry
    (``LW_API_ENDPOINT``, default the dev host). Its SDK calls ``requests`` with no
    ``verify=`` option, so an expired/invalid server cert makes every env's scene load
    die with ``SSLCertVerificationError: certificate has expired`` before the local
    asset cache is even consulted. Patch ``requests.Session.request`` to pass
    ``verify=False`` for lightwheel hosts (other hosts keep normal verification).
    Idempotent; assets themselves are integrity-checked by the cache, not the TLS cert.
    """
    global _LIGHTWHEEL_SSL_PATCHED
    if _LIGHTWHEEL_SSL_PATCHED:
        return
    try:
        import requests
        import urllib3

        _orig_request = requests.Session.request

        def _request(self, method, url, *args, **kwargs):
            if "lightwheel" in str(url):
                kwargs.setdefault("verify", False)
            return _orig_request(self, method, url, *args, **kwargs)

        requests.Session.request = _request
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        _LIGHTWHEEL_SSL_PATCHED = True
        logger.warning("Disabled TLS verification for lightwheel asset-registry requests (expired cert workaround)")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Could not patch lightwheel SSL verification: {exc}")
