# 01_stage0_평지보행_87% — 기본 보행 검증
> EXPERIMENTS.md: **stage0_openplane_scratch_6m_v1**

미로(기둥) 없이 **빈 평지**에서 6m 목표까지 직진. "병목은 미로가 아니라 보행/탐색이었다"를 확인한 Stage 0 결과.

| 파일 | 무엇을 보여주나 | 성공률 |
|---|---|---|
| `평지_직진_87%워커.mp4` | 빈 평지 6m 직진(커리큘럼 3m→4m→6m) | 87% (결정적 100ep) |

- 모델: `models/checkpoints_stage0_scratch_6m/ppo_final.zip` — **이후 모든 미로/자세수정 실험의 출발점(seed)**.
> 재생성: `scripts/make_videos.py`의 `make_open_plane_walk()`.
> 다음 → `02_stage1_미로정체_턴병목/`.
