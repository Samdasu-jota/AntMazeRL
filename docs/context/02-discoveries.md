# 02 — Durable Discoveries (AntMazeRL)

Read when: about to re-derive a "physical limit", trust a maze success number, or design a flip/geometry fix.

> Memory = where a finding is born (local ~/.claude/.../memory, dated, mine).
> docs/context = where a finding lives once proven (git, shared, durable).
> CLAUDE.md = the few findings that bite on every edit.

## Flip / inverted-crawl diagnostic — `fell 0/40` is a blind metric
- The Stage-1 maze "walker" was inverted ~60% of steps (flip_rate 99%, mean_min_up_z −0.97); it
  crawled on its back into the goal. The plane walker flips far less (~12% of steps, 34% rate).
- Old termination/diagnostics only checked height `z < 0.2`; an inverted ant keeps `z > 0.2`, so
  `fell 0/40` never saw it. Posture is measured by `up_z = 1 − 2*(qx²+qy²)` from unwrapped
  `data.qpos[3:7]` (+1 upright, 0 on side, −1 fully inverted) — `compute_up_z` (src/envs/ant_maze_env.py:37-41),
  read-only sweep in `scripts/eval_flip.py`. So the earlier "mid-motion turn weakness / geometry
  bottleneck" was diagnosed while the agent was on its back — turn weakness is largely a *consequence*
  of the flip (can't steer inverted), not a physical wall.
- See EXPERIMENTS.md ("검증 — up_z 뒤집힘 진단" → Phase 2 → "Phase 3 미로 재테스트").

## Flip-fix recipe — and the non-obvious lever is exploration noise, not the threshold
- Recipe: seed from the **upright plane walker** (NOT a maze checkpoint — its mean gait is belly-up,
  unusable as seed) + orientation termination `up_z < up_thresh` + upright bonus
  `up_bonus_coef·max(0,up_z)` + time/stall penalties at 0 (else suicide-collapse).
  Levers are `AntMazeEnv` constructor params (src/envs/ant_maze_env.py:108, stored :127-128):
  up_thresh (used in `_flipped`, :179) and up_bonus_coef (applied as up_bonus, :218).
- Gotcha: turning on flip-termination at the checkpoint's σ≈0.53 **collapsed** the 87% walker to 30%
  — stochastic exploration flips ~93% of episodes, so termination ends them instantly and starves
  learning. Threshold tuning can't fix it (flips reach −1.0). Fix = `override_log_std: -1.5` (σ≈0.22)
  right after init_from, keeping the policy near its clean gait. Result: plane 87%→98%, maze 58%→88%
  (flip steps 59.7%→~0.1%), all at ~5-10 min/run (fps ~3300, CPU-bound — "2-3h" was a bad estimate).
- See EXPERIMENTS.md (Phase 2 자세종료+직립보너스; Phase 3 미로 재테스트, +30pp).

## Full-maze-upright resolves the shrunk-pillar "band-aid" doubt
- The 88% maze number was on a shrunk pillar (`pillar_half_len=1.0`). On the **honest full maze**
  (`pillar_half_len=3.0`, full 6m pillar, spawn (0,0), goal_radius 1.0) the same upright recipe gives
  zero-shot **39%** but fine-tuned-in-place **73%** (76% at r=1.5).
- The D1(39%) ≪ D2(73%) gap proves the full maze is **not a geometry/turn wall** — it's
  under-training / distribution shift. Zero-shot loss was NOT spawn-jam: moving spawn off the pillar
  base (0,0)→(1,0) gave only +1pp; loosening radius +8pp. Real cause = flipping on unseen full-maze
  geometry (6m pillar-hug straight + one sharp ~90° turn at (1,6)). So the shrunk pillar was a
  curriculum **stepping-stone, not a result-inflating band-aid**. Residual ~27% are flips at the sharp
  turn (mean wp_idx 1.63 = straight is cleared) — a turn-curriculum / σ / longer-training problem.
- See EXPERIMENTS.md ("Phase A 풀맵(pillar_half_len=3.0) 직립 재테스트", 21%→73%).

## Maze difficulty is mostly geometry + A* re-planning, not a turn limit
- Frozen-model eval sweep: removing the pillar ~2× success; loosening goal_radius 1.0→2.0 ~3×. Two
  fixable geometry flaws: spawn (0,0) sits jammed at the pillar base (wall_mid y∈[0,6]) blocking the
  first leg (~53% stuck at wp0); and goal (0,6) is glued to the pillar tip (tight approach).
- Fix levers: `pillar_half_len` param (default 3.0 = full 6m, byte-identical → regression-safe;
  src/envs/ant_maze_env.py:58, mirrored in astar.py:137). Shrinking to 1.0 (y∈[2,4]) frees both
  spawn and goal (spawn-jam 40%→10%). **A* re-planning is the key lever**: for a short pillar, a
  gentle path `[(-1,1.4),(-1,4.6),(0,6)]` instead of the conservative sharp-90° `[(2.5,0),(2.5,6.3),(0,6)]`
  lifted completion (wp2) 15%→44% — proof the planner/RL split actually pays off. Honest maze
  (spawn (0,0), r=1.0): 21% (full) → 58% (shrunk), 65% at r=1.5. Residual turn weakness was real
  here but later mostly dissolved by the flip-fix above.
- See EXPERIMENTS.md ("검증 — goal/기둥 배치 가설"; "stage1_maze_short_v1", 8%→58~65%).
