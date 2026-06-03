# 02_stage1_미로정체_턴병목 — 미로 정체·턴 약점 진단기
> EXPERIMENTS.md: **stage1_omnidir_v1 ~ stage1_maze_waypoint_v1 ~ stage1_cmdfollow_turn30_v1** (미로 8~18% 정체)

평지 보행은 되는데 미로는 8~18%에서 정체하던 시기. 당시엔 "mid-motion +y 턴 약점/기하 병목"으로 진단했다(나중에 큰 부분이 **전복의 결과**로 밝혀짐 → `06_phase3_미로직립_88%/`).

| 파일 | 무엇을 보여주나 |
|---|---|
| `평지워커_위쪽턴_막힘.mp4` | 첫 다리(+x)는 통과하나 **위쪽(+y) 턴에서 막힘** — 대표 병목 데모(웨이포인트 1에서 정체) |
| `초기_stage1평가.mp4` | 초기 Stage 1 평가 기록(체크포인트 불명확, 참고용) |

> 재생성: `scripts/make_videos.py`의 `make_maze_stuck()`.
> 다음 → `03_stage1_rebalance_43%/`.
