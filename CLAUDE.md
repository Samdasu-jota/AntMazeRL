# CLAUDE.md — AntMazeRL (AI context index)

Custom MuJoCo Ant maze RL (BC + PPO + flip-fix), ~3,500 LoC. This file is the always-loaded
index: read the linked file ON DEMAND instead of re-reading source. Per-folder `CLAUDE.md`
auto-load only when you edit that folder. Keep every context file ≤100 lines (`python -m
scripts.check_context_md`).

## CROSS-MODULE INVARIANTS (bite on any edit)
- `EXPERIMENTS.md` is machine-appended by `train_ppo.py` (append_experiment_log, train_ppo.py:311, mode "a") — APPEND ONLY, never reformat. It is the experiment ledger.
- No GPU: CPU MuJoCo, n_envs=4, fps ~3300; 1M steps ≈ 5 min, 2M ≈ 10 min. Don't estimate in hours.
- Touch env / reward / termination? Re-run `python -m scripts.eval_flip` to confirm posture (up_z) didn't regress — height-only metrics are blind to inverted-crawl.
- Run code as modules from repo root (`python -m scripts.<name>`, `python -m src.training.train_ppo`); scripts `import src.envs` to register `AntMaze*-v0`. A bare file path fails.
- Warm-start chain is fragile (FixedNormalizeObs vs `VecNormalize norm_obs=False`, `target_kl=0.05`, `override_log_std`) — read `src/training/CLAUDE.md` before editing training.
- Maze fine-tune seeds from the upright PLANE walker, not a maze checkpoint (maze ckpts are belly-up even deterministically) — see `configs/CLAUDE.md`.

## COMMANDS (full reference: scripts/CLAUDE.md)
- Setup / smoke: `python -m scripts.check_install` · `python -m scripts.check_maze_env` · `python -m scripts.test_env`
- Train: `python -m src.training.train_ppo --config configs/<name>.yaml [--smoke|--no-wandb|--timesteps N|--run-name NAME]`
- Authoritative eval: `final_eval` runs inside training (100 ep, seed0=20000, deterministic; PASS = success ≥0.80 AND ep_len <1000). Posture: `python -m scripts.eval_flip --config <cfg> --checkpoint <ckpt>`
- Compare / plot: `python -m scripts.evaluate_comparison` · `python -m scripts.plot_comparison`
- No lint / unit-test framework — `test_env.py` + `check_install.py` are the smoke tests.

## STAGE / PHASE NAMING KEY (configs/ prefixes; det 100-ep success)
- `stage0_*` — AntMazeOpen-v0, open-plane walking. scratch 81% (3m) → 87% (6m).
- `stage1_*` — maze nav debugging (omnidir / cmdfollow / maze_short + A*). 8 → 58–65%.
- `p2_*` — upright fix on plane → 98% (flip 34%→2%).
- `p3_*` — upright fix on maze → 88% (shrunk pillar). `ppo_p3_maze_fullpillar` = full maze.
- Phase A — full maze (pillar 3.0) fine-tune → 73%.
- Pattern: `ppo_<stage>_<variant>.yaml` (train). `eval_*.yaml` = eval-only (rebuild env for `scripts.eval_flip --config ... --checkpoint ...`).

## NAVIGATION INDEX
Per-folder guardrails (auto-load when you edit that folder):
- `src/envs/CLAUDE.md` — AntMazeEnv internals: obs 109/110, up_z/flip, reward modes, registration gotchas.
- `src/training/CLAUDE.md` — PPO warm-start, normalization, target_kl, override_log_std, eval gates, EXPERIMENTS appender.
- `configs/CLAUDE.md` — config schema, stage/phase naming, how a run composes, worked example.
- `scripts/CLAUDE.md` — full train/eval/video/plot command reference + script map.

Read on demand (cross-cutting):
- `docs/context/01-architecture.md` — pipeline (incl. future World-Model/Compression), 3 envs, A*+RL split, Stage→Phase table.
- `docs/context/02-discoveries.md` — durable lessons (flip-fix, geometry, BC ceiling) → links into EXPERIMENTS.md.

Other docs:
- `EXPERIMENTS.md` — Korean raw tuning journal (append-only). `outputs/README.md` — video↔experiment map. `README.md` — human/portfolio.

## MAINTENANCE
- Context files are guardrails for the AI, not READMEs: non-obvious invariants only, never restate what code shows.
- ≤100 lines/file (hard cap); at the cap, split by sub-topic and add it to this index.
- Mistake → one-line rule: when the AI gets something wrong, add ONE line to the relevant file (or here if edit-universal).
- New durable discovery: it's born in local auto-memory; promote the proven kernel into `docs/context/02-discoveries.md`.
- `python -m scripts.check_context_md` enforces the ≤100-line cap and this index's sync (run it before committing context edits).
