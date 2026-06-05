# scripts/ — AI guardrail (command + invariant reference)

Read when running/editing anything in scripts/. Verified against source 2026-06-03.

## Hard invariants (all scripts)
- Run as modules from PROJECT ROOT: `python -m scripts.<name>`. They `import src.envs` to register `AntMaze*-v0` (gym.make fails without it). Never run as a file path.
- Every script sets `MUJOCO_GL=glfw` via setdefault. Black screen → export `MUJOCO_GL=egl` (or `osmesa`) before running.
- Render gotcha: pass BOTH `camera_id=-1` AND `default_camera_config=CAM` or you get the useless ground-level track cam. Top-down/follow/angled CAM dicts are hardcoded per script.
- PPO eval wiring: `FixedNormalizeObs(env, obs_mean, obs_std)` then `model.predict(obs, deterministic=True)`. BC is the exception: feed RAW obs (BCPolicy normalizes internally) — see evaluate_comparison build_env_raw.
- Default norm stats: `data/obs_norm_stats.npz`. Eval seed base is `seed0=20000` almost everywhere (per-ep seed = seed0+ep).
- `AntMaze-v0` obs is 109-D; obs[-2:] = goal-relative vec (GOAL_POS − ant_xy), NOT absolute pos.

## Authoritative eval
- THE success number = `final_eval()` in src/training/train_ppo.py:156 — 100 episodes, seed=20000, deterministic=True, runs automatically at end of training. PASS gate = success ≥0.80 AND mean_ep_len <1000 (train_ppo.py:528).
- `eval_flip.py` is the POSTURE check (up_z flip diagnostic), NOT the success authority.
- `evaluate_comparison.py` reproduces the cross-policy table (random/BC/RL + ablations), same 100ep/seed0=20000/deterministic.
- NO lint / pytest / unit-test framework exists. `test_env.py` + `check_install.py` + `check_maze_env.py` are the de-facto smoke tests. Do not invent a test runner.

## Script map
- check_install.py — MuJoCo+Gym sanity via plain Ant-v5 (no src.envs). Run first on a new machine.
- check_maze_env.py — minimal AntMaze-v0 reset/step dim check + 1 frame PNG. Quick env-registered check.
- test_env.py — full AntMaze smoke: dim asserts, 3 random eps, reward = speed−energy−collision assert, path PNG, 2 cam mp4s.
- preview_expert.py — eyeball scripted expert gait (stage A/B/C); tune before collecting.
- collect_demos.py — 200-ep scripted-expert demos → data/expert_demos.npz (BC training data).
- collect_rollouts.py — Phase 4.1: runs p3-upright PPO in AntMazeWaypoint-v0 with NO FixedNormalizeObs (RAW obs) → (s,a,s',r,done)+wp_idx/subgoal/action_mode npz for the world model; appends a Phase-4.1 block to EXPERIMENTS.md. `--smoke`=2ep+verify.
- train_bc.py — trains BC from bc_config.yaml; reads `configs/bc_config.yaml` (path hardcoded, no flags).
- eval_flip.py — up_z posture diagnostic; up_z = 1−2*(qx²+qy²) (qpos[4],qpos[5]). Detects inverted-crawl that z<0.2 termination misses.
- eval_waypoint.py — Stage1 waypoint-following success (hardcoded WAYPOINTS or A*).
- eval_omnidir.py — 8-octant×distance success for the omnidir walker; prints Phase-2 gate (≥80% overall & ≥60% worst octant).
- evaluate_comparison.py — Phase 3.3 master comparison → outputs/evaluation_results.json (read for the canonical numbers).
- plot_comparison.py — re-plots that JSON → comparison.png + factor_impact.png (no re-eval).
- verify_placement.py — eval-only ablation: is maze failure geometry or turn-limit (hardcoded MODEL, no flags).
- make_flip_video.py — before/after inverted-vs-upright evidence video. `--cam angled` (default, torso-tracking side angle) + min-up_z PNG, or `--cam topdown` (vertical overhead, shows the A*-path navigation; PNG skipped). Config-driven env build, so any checkpoint/maze via --config.
- make_maze_result_video.py / make_maze_short_video.py — preset top-down result videos (hardcoded MODEL+paths, no flags); scan one maze success + one maze fail.
- make_videos.py — Stage-0 plane-walk success clip + Stage-1 maze stuck/blocked clip (two different envs, not a success/fail pair; no flags).

## Command reference
Setup / smoke:
```
python -m scripts.check_install
python -m scripts.check_maze_env
python -m scripts.test_env
```
Data + BC:
```
python -m scripts.preview_expert C        # A|B|C positional, default A
python -m scripts.collect_demos
python -m scripts.train_bc
```
World model (Phase 4.1):
```
python -m scripts.collect_rollouts [--episodes 1000] [--det-frac 0.5] [--out data/world_model_rollouts.npz] [--smoke]
# stores RAW obs (env built without FixedNormalizeObs); normalizes only to feed model.predict.
```
Train (module, NOT in scripts/):
```
python -m src.training.train_ppo --config configs/ppo_config.yaml [--timesteps N] [--run-name NAME] [--no-wandb] [--smoke]
# --smoke = 200k steps, wandb off, does NOT write ppo_final / EXPERIMENTS.md
```
Eval:
```
python -m scripts.eval_flip --config CFG.yaml --checkpoint CKPT.zip [--n 100] [--flip-thresh 0.0] [--stochastic] [--norm-stats PATH]
python -m scripts.eval_waypoint --checkpoint CKPT.zip [--episodes 100] [--video PATH] [--use-astar] [--norm-stats PATH]
python -m scripts.eval_omnidir --checkpoint CKPT.zip [--reps 6] [--norm-stats PATH]
python -m scripts.evaluate_comparison [--n 100] [--out outputs/evaluation_results.json]
python -m scripts.verify_placement        # no flags
```
Compare / plot:
```
python -m scripts.plot_comparison [--json outputs/evaluation_results.json] [--out outputs/images/comparison.png]
```
Video:
```
python -m scripts.make_flip_video --config CFG.yaml --checkpoint CKPT.zip --out OUT.mp4 \
    [--cam angled|topdown] [--want-inverted] [--n-scan 40] [--seed0 20000] [--max-frames 700] [--norm-stats PATH]
python -m scripts.make_videos                 # no flags, fixed checkpoints
python -m scripts.make_maze_result_video      # no flags
python -m scripts.make_maze_short_video       # no flags
```

## Pre-flip-fix regime gotcha
- evaluate_comparison.py forces `up_thresh=-1.1` (PRE override, line 43) on pre-flip-fix policies to turn posture-termination OFF — required to reproduce the logged 58/81/87%. Post-flip-fix policies keep env default up_thresh=0.0. Editing this silently breaks reproducibility.
