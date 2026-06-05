# 01 — Architecture (cross-cutting)

read when: orienting on the pipeline, the 3 envs, the A*+RL split, or the stage→phase arc.

## Pipeline (README.md:30-36)
1 Environment (custom maze + reward) · 2 BC (clone scripted expert) · 3 PPO ·
5 Eval/ablation — DONE (only step with ✅).
- **Step 4 World Model — 4.1 data DONE, 4.2 transformer pending.** `scripts.collect_rollouts` runs the p3-upright policy → `data/world_model_rollouts.npz` of RAW (s,a,s',r,done)+wp_idx. Caveat: under waypoint_follow obs[107:109] jumps at sub-goal switches (≤2/ep, non-Markov) — flagged via `wp_switch`. See `src/world_model/CLAUDE.md`. Step 6 Compression — FUTURE, not built.
- Maze env is hand-built, NOT gymnasium-robotics AntMaze (README.md:41).
- obs=109-d, action=8-d. obs[-2:] = goal-relative vector (scripted_expert.py:21). obs[15:17] = body lin vel.

## The 3 envs (src/envs/__init__.py)
- **AntMaze-v0** — full maze, defaults; `max_episode_steps=1000` (timeout=truncated).
- **AntMazeOpen-v0** — `obstacle=False`, open plane, `reward_mode="direct"`, goal_y=3. Stage 0: "can it walk to a goal at all?" with no maze.
- **AntMazeWaypoint-v0** — `obstacle=True`, direct reward, **goal_bonus=0**. The 0 is load-bearing: env must NOT emit +50 per sub-goal — the WaypointFollower wrapper owns all bonuses (__init__.py:27, waypoint_follower.py:29).

## A* + RL split (classic, ANYmal/Barkour-style)
- Planner = classic A* gives sub-goal waypoints; RL policy walks between them. Separation is the whole point.
- `WaypointFollower` (waypoint_follower.py:22) **never edits AntMazeEnv** — it overwrites `env.unwrapped.goal_pos` to the current sub-goal each step, so the env's own `_get_obs`/`_step_direct` recompute relative-to-sub-goal for free.
- Termination/success judged on the **final** goal only, wrapper overrides (l.70-82); sub-goal arrival never terminates.
- On sub-goal advance it resets `_prev_goal_dist=None` to kill a progress-reward spike (l.64). `fell = z<0.2 OR _flipped()` (l.74).
- A* (planning/astar.py): occupancy grid → inflate by ant radius → 8-dir A* → RDP `simplify` to corners. `plan(pillar_half_len=...)` swaps only the central wall length; shrinking pillar (1.0 vs full 3.0) yields a gentler detour (astar.py:136-152).
- Other wrappers (same file): `RandomGoalDirection`, `RandomWaypointSequence` (angle curriculum via `max_turn_angle`).

## Stage → Phase arc (verified vs EXPERIMENTS.md / outputs/README.md; deterministic 100-ep, seed0=20000)
| Step | Env / setup | Det success |
|---|---|---|
| Exp #1-7 | full maze, reward tuning (divergence→stable) | 0–5% |
| Stage 0 | open plane, scratch, curriculum 3→4→6m | 81→73→**87%** |
| Stage 1 | maze: omnidir→waypoint→cmdfollow/turn30 | 8→18% (turn bottleneck) |
| Stage 1 | maze rebalance (spawn/radius) | 43% (r1.5) |
| Stage 1 | short pillar (1.0) + A* replan | 58% (r1.0), 65% (r1.5) |
| **Phase 2** | upright on plane (posture-term + upright bonus) | 87→**98%**, flip 34→2% |
| **Phase 3** | upright on short maze (same env as 58%) | 58→**88%**, flip-step 60→0.1% |
| **Phase A** | full maze (pillar 3.0): zero-shot → fine-tune | 39 → **73%** (r1.5 76%) |

- The same-env posture-only delta is **58%→88% (+30pp)** (identical short maze + A*). The README "8%→88%" headline additionally folds in the geometry fix (full→short pillar + A*, 8→58), per the README's own ablation table. Full-maze honest jumped 21%→73% (README.md:6-25).
- Flip-fix recipe: seed from the **upright plane** walker (NOT a maze ckpt — maze ckpts are inverted even deterministically) + `override_log_std=-1.5` (σ≈0.22) so low-noise rollouts survive posture-termination. Threshold tuning alone fails (starves exploration).

## Promoted invariants
- **Stage 0 proved locomotion was the real bottleneck, not the maze/pillar.** Scratch (81%) beat BC-warm (66%) on open plane — the U-path BC heading bias + frozen action std (~0.36) hurt (stage0-open-plane-locomotion-verified).
- **Scripted expert caps BC quality.** `scripted_expert.py` (open-loop sinusoidal torque trot) reaches goal ~40% of eps with ~13% topple; it is NOT "never falls". `collect_demonstrations` is topple-aware: records only upright (z≥`TOPPLE_Z`=0.4) transitions, ends episode after `TOPPLE_PATIENCE` low steps — clean demos but BC bounded by ~40% reach.
- BC self-normalizes obs inside the model (behavior_cloning.py:64-67) — feed raw obs, do NOT double-normalize. Pure BC scored **0% in maze** (upright but can't path-find).

## Detail lives in
- EXPERIMENTS.md — full chronological tuning log (Korean; per-experiment blocks 1:1 with outputs/videos/00..07).
- docs/context/02-discoveries.md — curated durable discoveries (flip-fix, geometry, BC ceiling).
- outputs/README.md — video↔experiment map.
- Per-folder guardrails: src/envs/CLAUDE.md, src/training/CLAUDE.md, configs/CLAUDE.md, scripts/CLAUDE.md (auto-load when you edit that folder).
