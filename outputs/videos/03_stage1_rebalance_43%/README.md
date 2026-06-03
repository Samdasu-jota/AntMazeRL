# 03_stage1_rebalance_43% — 스폰/반경 기하 수정
> EXPERIMENTS.md: **stage1_maze_rebalance_v1**

사용자 영상 관찰("스폰이 기둥 밑동에 끼임 + goal이 벽끝에 붙어 접근 공간 없음")을 받아 **스폰 (0,0)→(0,−1) + goal_radius 1.0→1.5**로 기하만 수정. frozen 평가 20%→40%(학습 0)로 배치 가설 검증.

| 파일 | 무엇을 보여주나 | 성공률 |
|---|---|---|
| `리밸런스미로_실패.mp4` | rebalance 단계 실패 케이스 | — |
| `리밸런스미로_성공_r1.5.mp4` | 스폰(0,−1)+반경 1.5로 완주 | 결정적 43% (r1.5) |

- 모델: `models/checkpoints_stage1_maze_rebalance/ppo_final.zip`.
> 재생성: `scripts/make_maze_result_video.py`.
> 다음 → `04_stage1_축소기둥_58%/`.
