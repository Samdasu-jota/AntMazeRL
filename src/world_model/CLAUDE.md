# src/world_model/ — guardrails (Phase 4)

read when: editing the rollout collector, RolloutDataset, or the (future 4.2) world model.
Phase 4.1 = collect policy rollouts → dataset. 4.2 Transformer World Model = pending.

## STORE RAW OBS — normalization is for the policy, not the dataset
- `scripts/collect_rollouts.py` builds the env with NO `FixedNormalizeObs` (evaluate_comparison
  `build_env_raw` pattern) so `env.step/reset` return RAW 109-D obs. It normalizes a COPY only to
  feed `model.predict`: `clip((raw-mean)/(std+1e-6), -10, 10)`. The dataset stores raw.
- Do NOT recover raw by inverting normalization — `FixedNormalizeObs` clips to ±10 AFTER
  normalizing (wrappers.py:35-37); saturated dims are unrecoverable. Inversion is lossy.
- `vecnormalize.pkl` (next to the ckpt) is REWARD-norm only (`norm_obs=False` always) — irrelevant
  to obs. Obs stats are `data/obs_norm_stats.npz` (BC/PPO policy-input stats; load via
  `train_ppo.load_norm_stats`). RolloutDataset fits its OWN stats — see below.

## obs / keys (verified)
- obs is 109-D here (AntMazeWaypoint-v0, `include_up_z=False`): obs[0:2]=absolute torso xy,
  obs[2:107]=Ant proprio, obs[107:109]=goal-relative vec. up_z is NOT in obs (term/reward only).
- success key = `info["reached_goal"]` (NOT is_success). `terminated`=reached OR fell(z<0.2 or
  flip); `truncated`=TimeLimit 1000. dataset `dones`=terminated only (truncation→0).
- seed0=20000 (det subset uses canonical eval seeds → comparable to logged 88%).

## NON-MARKOV waypoint switch (the load-bearing caveat)
- WaypointFollower flips `env.unwrapped.goal_pos` to the next sub-goal when within wp_reach=0.8
  (waypoint_follower.py:60-66), so obs[107:109] JUMPS on the switch step. Active `_wp_idx` is
  hidden state NOT in obs → `(obs,a)→obs'` is NOT Markov across a switch (≤2/episode).
- The dataset stores `wp_idx`, `wp_switch`, and `subgoal_xy` so 4.2 can drop/mask those targets or
  recompute the goal vector analytically (`subgoal_xy = obs[0:2]+obs[107:109]`, an integrity check
  asserted at collect time, <1e-3). obs[0:2] (abs xy) stays continuous; only obs[107:109] jumps.

## RolloutDataset (rollout_dataset.py)
- Split is per-EPISODE (not per-transition — adjacent transitions near-identical → leakage). Same
  `(val_frac, split_seed)` ⇒ train/val agree on the partition.
- Norm stats are fit on the TRAIN split only (every instance computes the same train subset), saved
  to `data/world_model_norm.npz` (state/action/delta mean+std, +1e-6 guard, NO ±10 clip). Distinct
  from obs_norm_stats.npz.
- x=[norm(state)|norm(action)] (117,); y default=norm_delta(next-state) (delta beats absolute);
  `predict="absolute"` and `include_reward=True` supported. `__getitem__` returns `wp_switch` so
  4.2 can mask non-Markov targets.

## INVARIANTS
- EXPERIMENTS.md is append-only. The collector appends its OWN Phase-4.1 block (mode "a"); do NOT
  reuse `train_ppo.append_experiment_log` (PPO-shaped: model.logger/approx_kl/PASS-FAIL are nonsense
  here). Never reformat prior blocks.
- Run as a module from repo root: `python -m scripts.collect_rollouts` (imports src.envs to register
  AntMaze*). `--smoke` = 2 ep + integrity + policy round-trip, no EXPERIMENTS write, temp out.
- Don't touch src/envs, train_ppo, configs, vecnormalize.pkl, obs_norm_stats.npz.
