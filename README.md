# AntMazeRL 🐜

MuJoCo Ant 로봇이 미로를 **빠르게, 에너지 효율적으로, 벽에 부딪히지 않고**
통과하도록 학습시키는 강화학습 프로젝트.

## 📊 결과 (Results)

영상에서 개미가 **뒤집힌 채 기어다니는 것**을 발견·수정한 게 게임체인저였습니다.
미로 성공률 **8% → 88%**(축소맵, 같은 환경에서 자세만 교정), 진짜 풀맵에서도 **73%**(이전엔 '난망'이라던 70% 게이트 통과).

![Per-factor performance comparison](outputs/images/comparison.png)

각 개입(factor)이 성능을 얼마나 올렸나 — **환경을 고정하고 한 변수만** 바꿔 측정 (결정적 100ep, seed0=20000):

| Factor (intervention) | Controlled env | Δ Success |
|---|---|---|
| Training (random → RL) | short maze | **+88pp** |
| Imitation (BC) vs scratch | plane 3m | **+15pp** (scratch wins) |
| Geometry: full → short pillar +A* | maze | **+37pp** |
| ★ **Flip-fix (upright posture)** | short maze (**same env**) | **+30pp** (58→88) |
| Map difficulty: short → full | upright policy | **−49pp** (harder map) |
| Full-map fine-tune | full maze | **+34pp** (39→73) |

- 전체 실험 일지: **[EXPERIMENTS.md](EXPERIMENTS.md)** · 실험순 영상: **[outputs/videos/](outputs/videos/)** · 평가 재현: `python -m scripts.evaluate_comparison && python -m scripts.plot_comparison`
- 핵심 before↔after (같은 미로, 자세만): [before 뒤집힘 58%](outputs/videos/06_phase3_미로직립_88%/미로_before_뒤집힘_58%.mp4) ↔ [after 직립 88%](outputs/videos/06_phase3_미로직립_88%/미로_after_직립_88%.mp4)

## 파이프라인
1. **환경** — 커스텀 미로 + 보상 함수 (속도 + 에너지 + 충돌 회피)
2. **모방 학습 (BC)** — 스크립트 전문가를 따라하며 초기 정책 확보
3. **강화 학습 (PPO)** — 시행착오로 정책 개선
4. **World Model** — 미래 상태를 예측하는 트랜스포머 모델
5. **평가 & 문서화** ✅ — random vs BC vs RL + 팩트별 ablation (`scripts/evaluate_comparison.py` → `outputs/evaluation_results.json`, `scripts/plot_comparison.py` → `outputs/images/comparison.png`)
6. **압축** — 양자화로 추론 속도 개선 (엣지 배포 대비)

## 기술 스택
Python 3.12 · MuJoCo · Gymnasium · PyTorch · Stable-Baselines3 · W&B

> 미로 환경은 직접 만든다 (gymnasium-robotics의 AntMaze를 쓰지 않음) — 환경 설계 능력을 보여주기 위함.

## 구조
```
src/        # 핵심 코드 (환경, 모델, 학습 로직)
configs/    # YAML 설정 (하이퍼파라미터 등)
data/       # 데이터셋 (시연 데이터, rollout 등)
models/     # 학습된 모델 체크포인트 (저장소엔 주요 ppo_final.zip만 — 나머지는 configs로 재학습)
notebooks/  # 분석용 주피터 노트북
scripts/    # 실행 스크립트 (학습, 평가, 테스트)
outputs/    # 결과물 (그래프·영상·JSON; 원시 로그 outputs/logs/는 제외)
```

> 저장소엔 **최종 체크포인트만** 포함(용량). 전체 중간 체크포인트·W&B 로그·전문가 시연 데이터는 제외 — 모두 `configs/`로 재학습/재수집 가능.

## 설치 & 실행
```bash
python3.12 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_install.py   # 환경 정상 작동 확인 (obs에 x,y 포함 + 첫 프레임 저장)
```

## 트러블슈팅
- **렌더링이 검은 화면이면**: `scripts/check_install.py`의 `MUJOCO_GL` 값을 `"glfw"` → `"egl"` 또는 `"osmesa"`로 변경.
- **`pillow`/`numpy` 빌드 실패**: 파이썬이 너무 최신(예: 3.14)이라 미리 빌드된 휠이 없을 수 있음 → **Python 3.12**로 venv 재생성 (`rm -rf venv && python3.12 -m venv venv`).
- **`Ant-v5` 없음**: gymnasium이 오래된 버전 → `pip install -U "gymnasium[mujoco]"`.
