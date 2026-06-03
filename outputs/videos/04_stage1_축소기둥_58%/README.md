# 04_stage1_축소기둥_58% — 기둥 1/3 축소 + A* 경로
> EXPERIMENTS.md: **stage1_maze_short_v1**

중앙 기둥을 1/3로 줄이고(pillar_half_len 1.0, y∈[2,4]) A* 부드러운 경로로 honest 성공률을 3배 가까이 끌어올림(21%→58%). **단, 이때도 개미는 60% 스텝을 뒤집힌 채 기고 있었다** — 이건 다음에 발견·수정(`06_phase3_미로직립_88%/`).

| 파일 | 무엇을 보여주나 | 성공률 |
|---|---|---|
| `축소기둥미로_실패.mp4` | 축소기둥 단계 실패 케이스 | — |
| `축소기둥미로_성공_58-65%.mp4` | 축소기둥+A*로 완주 | honest 58% (r1.0) / 65% (r1.5) |

- 모델: `models/checkpoints_stage1_maze_short/ppo_final.zip` — **06번의 'before'(뒤집힘 58%)가 바로 이 워커**.
> 재생성: `scripts/make_maze_short_video.py`.
> 다음 → `05_phase2_평지직립_98%/`.
