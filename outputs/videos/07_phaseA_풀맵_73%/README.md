# 07_phaseA_풀맵_73% — 풀맵 정직 재테스트 (임시방편 검증)
> EXPERIMENTS.md: **Phase A 풀맵(pillar_half_len=3.0) 직립 재테스트**

"축소기둥(04번) 88%는 임시방편(band-aid) 아니냐"는 의심을 **진짜 풀맵(pillar_half_len 3.0, 기둥 6m)**에서 검증. A* 경로 `[[1,0],[1,6],[0,6]]` = x=1.0 6m 직선 + (1,6) sharp ~90° 좌회전 → goal.

| 파일 | 무엇을 보여주나 | 성공률 | flip |
|---|---|---|---|
| `풀맵_zeroshot_39%.mp4` | 축소맵 88% 정책을 풀맵서 **그냥** 평가(학습 0) | 39% | 59% |
| `풀맵_after_직립_73%.mp4` | 직립 레시피를 **풀맵에 직접 fine-tune** | **73%** (r1.5 76%) | 26% |

- **결론: 축소기둥은 결과 부풀린 band-aid가 아니라 학습용 디딤돌** — 풀맵 직접학습으로 70% 게이트 정직 통과(전복수정 前 honest 풀맵 21% → 73%). zero-shot 39% 손실은 스폰끼임(+1pp) 아닌 **풀맵 기하서의 전복**(분포shift/미학습).
- 모델: `models/checkpoints_p3_maze_fullpillar/ppo_final.zip`.
> 재생성: `scripts/eval_waypoint.py --checkpoint … --use-astar --video`.
