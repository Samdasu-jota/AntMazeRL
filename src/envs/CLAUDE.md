# src/envs/ — guardrails

read when: editing AntMazeEnv, its wrappers, or env registration.

## obs layout (verified by runtime: AntMaze-v0=109, include_up_z=110)
- base AntEnv obs = 107 because env passes `exclude_current_positions_from_observation=False` (ant_maze_env.py:144) → x,y kept.
- `_get_obs` appends goal-relative vec `goal_pos - qpos[:2]` (2 dims) → 109. (ant_maze_env.py:163-171)
- if `include_up_z=True`: appends up_z (1 dim) → 110. up_z is LAST element, after the goal vec. (ant_maze_env.py:170)
- obs_dim is computed from the parent's actual `observation_space` + extra, NOT hardcoded — version-safe. (ant_maze_env.py:150-151)

## up_z / posture (flip) — single source of truth
- up_z = world-z of body's up-axis = `1 - 2*(qx²+qy²)` from quat qpos[3:7]. +1 upright, 0 on-side, -1 fully inverted. (ant_maze_env.py:37-41)
- `UP_THRESH_DEFAULT=0.0` is the flip-termination boundary, NOT arbitrary: plane-walker min-up_z histogram is bimodal with [0.0,0.5) EMPTY, so 0.0 never kills normal gait (≥0.5) but catches deep flips (≤-0.5). (ant_maze_env.py:28-34)
- `_up_z()`/`_flipped()` (ant_maze_env.py:173-179) are the ONE flip oracle; wrappers call `u._flipped()` — do not reimplement.

## termination (BOTH conditions, every step path)
- `fell_over = z < 0.2 OR self._flipped()` → terminated if `reached_goal or fell_over`. (ant_maze_env.py:222-223, 267-268)
- z<0.2 alone was blind to inverted-crawl (z stays >0.2 while flipped); the `_flipped()` OR-term is the fix. The WaypointFollower wrappers duplicate this same `z<0.2 or u._flipped()` check (waypoint_follower.py:74 and :198); wrappers.py has NO flip/termination logic (only FixedNormalizeObs).

## reward_mode
- `"direct"`: potential-based progress to straight-line goal dist + stall/time penalties + reduced goal_bonus. `_step_direct`. Used by Open + Waypoint envs. (ant_maze_env.py:237-286)
- `"waypoint"`: legacy maze branch (hardcoded WAYPOINTS path-distance reward, goal_bonus 200). Experiment #7 = 0% success; kept but NOT used by any registered env or the wrapper. (ant_maze_env.py:193-235, waypoint_follower.py:13)

## INVARIANTS (do not break)
- AntMazeEnv custom kwargs MUST be EXPLICIT params, never via `**kwargs` — parent AntEnv raises TypeError on unknown kwargs. (ant_maze_env.py:103-109)
- `self.include_up_z` MUST be set BEFORE `super().__init__` — parent calls overridden `_get_obs` during init. (ant_maze_env.py:129,142,168)
- `reset_model` must NOT re-append the goal vec: parent's `reset_model` already returns via overridden `_get_obs`. Double-append was a fixed bug. (ant_maze_env.py:288-294)
- env value-DUPLICATES WAYPOINTS/WP_REACH locally (ant_maze_env.py:46-51) to avoid importing scripted_expert (cycle: scripted_expert imports GOAL_POS from this file, ant_maze_env.py:44-45). Keep the two copies in sync by hand.
- `close()` deletes the temp maze XML — leak guard; don't drop it. (ant_maze_env.py:305-311)

## registered envs (src/envs/__init__.py)
- `AntMaze-v0`: bare class defaults (obstacle=True, reward_mode="waypoint"). max_episode_steps=1000 → TimeLimit owns `truncated`. (__init__.py:5-9)
- `AntMazeOpen-v0`: Stage 0 — `obstacle=False` (no center wall = open plane) + `reward_mode="direct"`, goal_y=3.0, goal_bonus=50. (__init__.py:14-22)
- `AntMazeWaypoint-v0`: Stage 1 — obstacle=True + direct reward; `goal_bonus=0.0` ON PURPOSE so the WaypointFollower wrapper owns all bonuses (env would otherwise emit +bonus per subgoal). (__init__.py:27-34, waypoint_follower.py:29-31)

## make_maze_xml (ant_maze_env.py:57-97)
- `obstacle=False` removes only `wall_mid` (center pillar) → open plane; 4 boundary walls + goal marker stay.
- `pillar_half_len` (default 3.0 → y∈[0,6]); 1.0 = 1/3-size pillar that separates spawn(0,0) & goal(0,6). `:g` format makes 3.0→"3" so default geom string is byte-identical to original (regression-safe — kept verbatim for experiment #1–#7 reproducibility).

## render (promoted-memory; no render code in src/envs/, lives in scripts/test_env.py)
- For rgb_array, `default_camera_config` is SILENTLY ignored unless you also pass `camera_id=-1` — gymnasium defaults to the fixed ground-level `track` cam (camera_id=0). `lookat` must be an np.ndarray. Free cam (id=-1) needs manual per-frame lookat to follow.
