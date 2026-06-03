# configs/ — PPO/BC run config guardrail
read when: editing or creating a YAML in configs/ (curriculum run specs for train_ppo.py).

Configs are flat YAML, loaded by `yaml.safe_load` then read key-by-key with `config.get/[]`
in src/training/train_ppo.py:main (def at :360; config loaded :372-373). NO schema validation — a typo'd/missing required key
crashes at access time. There is no config inheritance: every run is a full standalone copy.

## How a run is composed (the 5 keys that define behavior)
- `env_id` picks the registered env (src/envs/__init__.py): `AntMaze-v0` (maze, obstacle on),
  `AntMazeOpen-v0` (empty plane, reward_mode=direct), `AntMazeWaypoint-v0` (obstacle on, goal_bonus=0
  so the wrapper owns bonuses). gym.make(env_id, **env_kwargs).
- `env_kwargs` overrides that env's registration defaults per-run (e.g. goal_y, pillar_half_len,
  start_xy, up_thresh, up_bonus_coef, time_penalty, stall_penalty, energy_coef). NOT validated.
- `env_wrapper` (+ `env_wrapper_kwargs`) inserts a Stage-1 wrapper INSIDE FixedNormalizeObs
  (train_ppo.py:45-71,82). Values: `random_goal`, `waypoint_follow`, `random_sequence`. Omit → none.
- `init_from` = curriculum continuation: PPO.load(ckpt) and **BC warm-start is SKIPPED** (433).
  If absent → new PPO; BC warm-start applied only if bc_policy_path exists, else scratch (446-484).
- `override_log_std` only takes effect WITH init_from; forces policy.log_std (441-445). With
  init_from, `log_std_init` is ignored (hence the inline "init_from이 있으면 무시" comments).

## Non-obvious invariants (do not "fix")
- `target_kl: 0.05` — load-bearing anti-divergence; comment says never loosen it.
- `ent_coef: 0.005` — lets σ widen so KL pressure isn't dumped on μ (warm-start stability).
- `norm_reward: true` but obs normalization stays OFF: VecNormalize is `norm_obs=False` always
  (train_ppo.py:413-419). obs is normalized solely by FixedNormalizeObs; enabling norm_obs would
  double-normalize and destroy warm-start. Don't add a norm_obs key expecting it to work.
- `bc_policy_path` pointing at a missing file (e.g. `models/NO_BC_stage0.pt`) is INTENTIONAL —
  it's how an init_from/scratch run disables BC warm-start. Don't "fix" the path.
- `vec_env: "dummy"` vs `"subproc"`; dummy is the Mac-safe default.
- log_std_init -1.0 ≈ σ0.37, -0.5 ≈ σ0.61, override -1.5 ≈ σ0.22 (low noise = anti-flip).
- `--smoke` overrides total_timesteps=200000, use_wandb=False, run_name in-memory only; does NOT
  edit the YAML and does NOT write ppo_final/EXPERIMENTS.md (train_ppo.py:376-380,506-511).
- CLI flags override config: --timesteps, --run-name, --no-wandb (381-386).

## Naming convention: ppo_<stage>_<variant>.yaml (eval_ for eval-only)
- `stage0_*` → AntMazeOpen-v0, plane locomotion (goal_y distance curriculum). scratch tops ~81%.
- `stage1_*` → command-following / maze (random_sequence or waypoint_follow). Maze stalled ~58-65%.
- `p2_*` → AntMazeOpen-v0 upright fix (up_thresh + up_bonus_coef + override_log_std). v2 ≈98%/flip2%.
- `p3_*` → AntMazeWaypoint-v0 maze fine-tune of the p2 upright plane walker.
  `ppo_p3_maze_upright.yaml` (pillar 1.0, shrunk) ≈88%; `ppo_p3_maze_fullpillar.yaml` (pillar 3.0).
- `eval_*` = eval-only. **NOT consumed by train_ppo** (that would re-train). They are passed to
  `python -m scripts.eval_flip --config <yaml> --checkpoint <ckpt>` to rebuild the env; the model
  comes from a separate --checkpoint, not from init_from in the YAML.

## Worked example: ppo_p3_maze_upright.yaml
- `env_id: AntMazeWaypoint-v0` + `env_kwargs: {energy_coef 0.02, time_penalty 0.0, stall_penalty 0.0,
  pillar_half_len 1.0, start_xy [0,0], up_thresh 0.0, up_bonus_coef 0.1}` — honest spawn at origin,
  shrunk pillar, posture-termination on (up_thresh 0.0) + upright bonus, no time/stall bleed.
- `env_wrapper: waypoint_follow` + `env_wrapper_kwargs: {goal_radius 1.0, final_bonus 100.0,
  subgoal_bonus 10.0, waypoints [[-1,1.4],[-1,4.6],[0,6]]}` — A* path; wrapper owns all bonuses.
- `init_from: models/checkpoints_p2_upbonus_plane_v2/ppo_final.zip` — the upright PLANE walker
  (NOT a maze checkpoint; a maze ckpt would seed an inverted-crawl mean). BC warm-start skipped.
- `override_log_std: -1.5` (σ0.22) — keeps exploration low so posture-termination doesn't starve
  learning. Run = fine-tune the upright plane walker on the shrunk-pillar maze → ~88% success.
