# src/training/CLAUDE.md
read when: editing train_ppo.py or touching PPO warm-start / normalization / eval gates.

## Normalization (fragile — double-norm silently kills warm-start)
- Two layers of obs normalization must NEVER both run. Order: env -> Stage-1 wrapper -> `FixedNormalizeObs` -> VecMonitor -> VecNormalize.
- `FixedNormalizeObs` applies BC's FIXED stats `(obs-mean)/(std+1e-6)` (wrappers.py:35-37); stats are loaded by `load_norm_stats` from `obs_norm_stats.npz` (train_ppo.py:31-34) and passed into __init__. It is the ONLY obs normalizer.
- VecNormalize is conditional on config `norm_reward` and MUST stay `norm_obs=False` (train_ppo.py:413-419). Turning norm_obs on = double-normalize = warm-start destroyed. It only scales reward (raw ~-800 -> O(1)).
- Stage-1 wrapper goes INSIDE FixedNormalizeObs (make_env: train_ppo.py:80-84; apply_env_wrapper warning :47) — wrapper rewrites raw obs/goal, THEN normalize.
- VecMonitor is inside VecNormalize so `ep_rew_mean` stays raw (train_ppo.py:409).
- Any eval/deploy must re-wrap with the SAME stats or the bug returns (final_eval/evaluate_warmstart both do: :141,:166).
- BCPolicy normalizes internally in forward() too (behavior_cloning.py:66) — that's why the fixed-stats env wrapper is required for transferred weights to see matching scale.

## target_kl=0.05 is load-bearing (never relax)
- Passed only in the scratch/warm-start branch (train_ppo.py:461, `config["target_kl"]`).
- Korean comment at config line: "절대 풀지 말 것: 풀면 더 큰 step이 박혀 악화" = never relax; relaxing bakes a bigger step and makes divergence worse. It guards warm-start from blowing up early.

## Flip-fix lever = override_log_std (NOT an up-threshold)
- Applied ONLY on the `init_from` (curriculum-continue) branch, AFTER `PPO.load`, by overwriting `model.policy.log_std` (train_ppo.py:441-445).
- Production value -1.5 (σ≈0.22) in p2/p3 configs; shrinks exploration noise so posture-termination doesn't starve learning. Default checkpoint σ≈0.53 caused ~93% flip. `log_std_init` in those configs is ignored when init_from set.
- Do NOT confuse with `up_thresh` (flip_thresh): that's only the eval/measurement definition of "flipped", not the fix.

## Warm-start / init_from (two mutually exclusive paths)
- `init_from` SET => curriculum continuation: `PPO.load(init_from)`, BC transfer SKIPPED (train_ppo.py:433-435).
- `init_from` NONE => fresh PPO; if `bc_policy_path` exists, `transfer_bc_weights` copies BC weights (train_ppo.py:468-469); else scratch random init.
- Weight map (transfer_bc_weights :88-128): BC `net` Linear[0,1,2] (the 3 hidden) -> PPO `mlp_extractor.policy_net`; BC Linear[3] (output) -> PPO `action_net` (Gaussian mean head). BC Tanh dropped. `log_std` NOT copied (set via log_std_init). Shape mismatch -> skip + warn, never crash. SB3 2.8.0 paths.
- BC arch is fixed 109->256->256->256->8 (behavior_cloning.py:53-62); PPO net_arch must match [256,256,256] (train_ppo.py:448) or transfer silently skips layers.

## Two eval gates (different purpose — don't swap them)
- `evaluate_warmstart` (train_ppo.py:131): PRE-train reference, deterministic, `warmstart_eval_episodes`=20, seed0=10000. ~0% is OK (BC ceiling ~5%) — NOT a gate. The real pre-train check is `verify_warmstart_match` MAE<0.15 (:198,:472-474).
- `final_eval` (train_ppo.py:156): AUTHORITATIVE post-train gate. 100 episodes, seed=20000 (reset seed=20000+ep), deterministic. Returns (success, mean_ep_len, flip_rate, flip_term_rate, mean_min_up_z). PASS = success ≥0.80 AND mean_ep_len <1000 (train_ppo.py:528).

## EXPERIMENTS.md logging (append-only)
- `append_experiment_log` (train_ppo.py:311) opens `EXPERIMENTS.md` in mode "a" (:356) — APPEND ONLY. Never reformat/rewrite prior blocks; it is the experiment ledger. Runs only on full (non-smoke) runs (:538).

## CLI flags (argparse :363-370; smoke logic :375-386)
- `--smoke`: in-memory only — total_timesteps=200000, wandb off, run_name="200k_smoke_stabv2"; saves `ppo_smoke.zip`, NOT ppo_final, and SKIPS EXPERIMENTS.md + final_eval.
- `--timesteps N`: override total_timesteps. `--run-name S`: override run_name. `--no-wandb`: force wandb off. Explicit flags apply after smoke defaults (smoke does not re-override them except its own three).

## Compute
- CPU MuJoCo, no GPU assumed: device = cuda if available else cpu (train_ppo.py:388), `MUJOCO_GL=glfw` (:13). n_envs=4, vec_env="dummy" (DummyVecEnv) in config — Mac-safe, not Subproc.
