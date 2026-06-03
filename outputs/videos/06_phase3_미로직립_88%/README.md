# 06_phase3_미로직립_88% — ★ 전복 수정 before ↔ after (핵심)
> EXPERIMENTS.md: **Phase 3 미로 재테스트 (58→88%)**

이 프로젝트의 **가장 중요한 발견·수정.** 종료조건이 키(z<0.2)만 봐서, 개미가 **배를 위로 한 채(뒤집힌 채) 기어다녀도** 에피소드가 안 끝났다(`fell 0/40`은 눈먼 지표). 자세(up_z) 종료 + 직립 보너스로 고침. before/after는 **완전히 같은 미로 설정**(축소기둥+A*+honest spawn+r1.0)이라 **자세만 바뀐 직접 비교**.

| 파일 | 무엇을 보여주나 | 성공률 | 전복 스텝 | mean min up_z |
|---|---|---|---|---|
| `미로_before_뒤집힘_58%.mp4` | 뒤집힌 채 기어 골 도달 (문제) | 58% | **59.7%** | −0.97 |
| `미로_after_직립_88%.mp4` | **같은 미로**를 **직립** 주행 (해결) | **88%** | **0.1%** | +0.80 |
| `*_장면.png` | 각 영상 up_z 최저(가장 기운) 순간 정지컷 — before는 배가 위로, after는 다리가 바닥 |

- before: `models/checkpoints_stage1_maze_short/ppo_final.zip`(=04번 워커) · after: `models/checkpoints_p3_maze_upright/ppo_final.zip`
- **핵심 메시지: 턴 약점이라 믿었던 정체가 사실 '전복'의 결과였다.** (축소기둥 없는 풀맵 후속은 `07_phaseA_풀맵_73%/`.)
> 재생성: `scripts/make_flip_video.py`.
