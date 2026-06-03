# AntMazeRL 실험 일지

> **새 세션은 이 파일을 먼저 읽으세요.** 아래는 시간순 튜닝 기록입니다.
> 형식: 설정 / 결과 / 교훈 / 다음. 숫자 곡선은 W&B(project: AntMazeRL) 참조.
>
> **현재 상태 (2026-06-02, Stage 0 완료):** **병목은 미로가 아니라 보행/탐색이었음(확정).** 기둥 제거(빈 평지, `AntMazeOpen-v0`) + wander-trap 제거(stall+시간 페널티 + 도달보너스 200→50)로 성공률 ~1.5~5% → **scratch 81%(3m) → 커리큘럼 6m 87%**(결정적100ep, ep_len~557 ≪1000). 개미는 **원래 목표 거리(6m)를 빈 평지에서 87% 완주.** 의외: warm-start(BC, 66%)가 scratch(81%)보다 **낮음** — U-path 전문가 헤딩 바이어스 + 좁은 std(0.36 vs 0.55).
> 다음 액션 = **Stage 1**: 기둥 복귀(`obstacle=True`) + 검증된 워커(`models/checkpoints_stage0_scratch_6m/ppo_final.zip`) 위에 A*/웨이포인트 서브목표 얹기(업계 표준 분리: 길찾기=고전 A*, 보행=RL).

---

## 실험 #1 — warm-start PPO 첫 2M (발산)
- **설정:** warm-start O, log_std_init -1.0, obs 정규화(클리핑 **없음**), target_kl **없음**, energy 0.1, 2M
- **결과:** 성공률 0%, **approx_kl 755,619**, ep_rew -855→-1040(악화), std 0.089, clip_frac 0.95 → 완전 발산
- **교훈:** ① warm-start 정책 + 백지 critic + 무제한 업데이트 = KL 폭발  ② FixedNormalizeObs에 클리핑이 없어 분포 밖 관측이 폭발(|obs|→97)
- **다음:** target_kl + obs 클리핑 추가

## 실험 #2 — 진단 & 안정화 수정 (단축 격리실험)
- **설정:** 통제 실험(각 40k): 순정 PPO / warm-start / +target_kl / +lr↓
- **결과(approx_kl):** 순정 PPO **0.029✅** · warm-start **12.4❌** · warm+target_kl=0.05 **0.035✅** · warm+lr5e-5 **0.043✅**
  - obs 측정: std<0.05인 차원 **32/109**(최소 0.0025), 탐색 중 정규화 |obs| **97** → 클리핑 필수
  - NaN 추적: 보상/관측 NaN **0회** (NaN 가설 기각), 최대 |보상| 4.62 (보상폭발 아님)
- **교훈:** 범인은 **warm-start 발산**(환경/보상은 무죄). 수정 = **target_kl 0.05 + obs 클리핑[-10,10]** → approx_kl 0.04로 안정(160k 유지). warm-start MAE는 클리핑해도 0.064 그대로
- **다음:** 보상 가설 검증

## 실험 #3 — 보상 리밸런스 단축 테스트 (각 150k)
- **설정:** 안정화 적용 + energy 계수 비교
- **결과(성공률 50k→100k→150k):** ① warm+energy0.1: 2→2→0%  ② warm+energy0.01: 2→0→0%  ③ 순정+energy0.01: 0→0→0%
- **교훈:** ① energy 단독 범인 **아님**  ② **150k는 너무 짧음** — AntMaze는 sample-hungry hard-exploration 과제라 단축 A/B로는 못 가림  ③ (미검증 가설) 보상에 **목표 도달 보너스가 없음** → 약한 인센티브일 수 있음
- **다음:** 2M 풀런 제대로

## 실험 #4 — 2M 풀런 (안정화 적용)
- **설정:** warm-start O, log_std -1.0, obs클리핑, **target_kl 0.05**, energy 0.1, run_name `2M_warmstart_energy0.1_v1`, 2M
- **결과:** 성공률 0%, approx_kl(최종) 8.61, std 0.366, clip_frac 0.40, ep_rew -757→-1030(악화)
  - **궤적:** 학습직전 **5%** → 60k **10%**(critic 미숙 ev0.55) → ~147k부터 KL **8→14→25→62→…→240** 발산 → 영구 0%
  - std 2M 내내 0.367 **고정**(발산이 행동 평균 μ에서만), 매 iter "step 0" early-stop, 끝 critic ev=**0.986**인데도 KL 발산
- **교훈:** ① **"발산 해결"은 미완** — #2의 target_kl+클리핑은 발산을 *지연*시켰을 뿐(40k OK였지만 147k 재발). 짧은 통제실험이 못 잡은 늦은 발산
        ② 진짜 주범 = **log_std -1.0(σ0.367) 고정 + ent_coef=0** → 학습 압력이 평균 μ에만 쏠려 한 step이 KL 폭발. 보조 = 보상 정규화 없음(원시 -800) + lr 3e-4 고정 (#2에서 lr5e-5가 통한 것과 일치)
        ③ value망 미초기화는 **주범 아님** — 끝 ev=0.99인데도 발산하는 게 결정적 반례
        ④ target_kl=0.05는 이미 박힌 step을 못 되돌림 → **풀면 악화**(직관과 반대)
        ⑤ 별개 보상 결함: 유클리드 진행 보상이 미로 우회((2.5,0)行)를 처벌 + 도달 보너스 없음 + 에너지 항 지배
- **다음:** 실험 #5 안정화 v2(변수 하나씩) — **ent_coef 0.005 + lr 1e-4 선형감쇠 + VecNormalize(norm_reward만)**, target_kl 0.05 유지. 200k 스모크로 "warm-start 파괴 중단"(KL<0.2 · ep_rew 비악화 · 성공률 5% 유지) 확인 후 2M. 통과 후 실험 #6 보상 셰이핑(도달 보너스 +200 + 웨이포인트 진행 + 에너지 계수↓)

## 실험 #5 — 안정화 v2 (정책측 불안정 직격)
- **설정:** warm-start O, log_std -1.0, **ent_coef 0.005**, **lr 1e-4 + 선형감쇠**, **VecNormalize(norm_reward만, norm_obs=False)**, target_kl 0.05 유지, energy 0.1, 2M, run_name `2M_stabv2_entcoef_rewardnorm`
- **결과:** 성공률 ~0~5%(끝 5%), **approx_kl 0.0012**, **clip_frac 0.00**, std 0.357, ev **0.991**, value_loss **0.005**, **ep_rew -945→-641(개선)**, 모델 `models/checkpoints/ppo_final.zip` + `vecnormalize.pkl`
  - 200k 스모크 게이트 통과(approx_kl max 0.053, 폭발 0회) → 2M. 2M 전체 `KL>0.5` 스파이크 **3회**(2.2/0.52/1.17) 전부 고립·자가회복(#4의 8→240 연쇄 발산과 질적으로 다름)
  - early-stop이 #4의 `step 0`(첫 에폭 폭발) → `step 1~9`로 분산 = 거의 풀 업데이트를 KL 통제하에 수행
- **교훈:** ① **안정화 성공** — 2M 내내 approx_kl 0.001대, 발산 0, 정책 파괴 없음. 효과 레버는 **보상 정규화(value_loss 200→0.005) + lr↓**이고 **ent_coef 0.005는 거의 무효**(std 0.368→0.357로 안 넓어짐). KL∝‖Δμ‖²/(2σ²)에서 σ(분모) 대신 **Δμ(분자)를 줄여** 해결 — #4 진단의 메커니즘은 맞고, 고친 방향이 분자 쪽
        ② **그런데 성공률은 BC 수준(~5%)에서 정체** — 보상은 오르는데(-945→-641) 성공은 안 오름 = **보상 오설계(reward misspecification)**. 정책이 "에너지 아끼고 살짝 전진"으로 보상만 올리고 미로는 안 풂
        ③ **변수 분리 완료: 안정성은 더 이상 병목이 아님 → 진짜 병목 = 보상 설계**
- **다음:** 실험 #6 보상 셰이핑(안정화 셋업 유지, 보상만 변경) — ① **도달 보너스 +200**(현재 도달이 return에 안 보임) ② **웨이포인트 진행 보상**으로 유클리드 대체(우회 처벌 제거) ③ **에너지 계수 0.1→0.02**

## 실험 #6 — 보상: 도달 보너스 +200 단독
- **설정:** 안정화 셋업(#5) 유지 + `step()`에 `reached_goal` 시 **+200 보너스**만 추가, energy 0.1·웨이포인트 미적용, 2M, run_name `2M_reward_goalbonus_v1`
- **결과:** **거의 무효.** 성공률 시간구간 평균 **Q1 1.80% → Q2 1.00% → Q3 0.90% → Q4 2.30%**(상승 추세 없음, 전체 ~1.5%). 안정성 완벽 유지: approx_kl 0.0021, clip_frac 0.00, std 0.361, ev 0.982, ep_rew -792→-645(개선). 200k 스모크 통과(KL max 0.091, ev -4.6→0.96 회복) 후 2M
- **교훈:** ① **도달 보너스 단독은 효과 미미** — 개미가 미로 U턴 경로를 거의 못 찾아(성공 ~1.5%) **희소 +200 보너스가 발화를 안 함** → 학습을 못 끎. ② 보너스로 critic이 초반 흔들렸다(ev 음수) 회복하지만 정책 성공률은 안 오름. ③ 안정화 셋업은 보너스 스파이크에도 견고(KL 0.002). ④ **확정: 병목 = 탐색/경로**(인센티브 문제 아님). dense 유클리드 진행 보상이 우회를 처벌하는 게 핵심
- **다음:** 실험 #7 — **웨이포인트 진행 보상**으로 유클리드 진행 대체(전문가 경로 따라 남은 거리). 도달 보너스 유지, 에너지·안정화 그대로(변수 하나씩)

## 실험 #7 — 보상: 웨이포인트 진행 보상 (경로/탐색 직격) ← 지금
- **설정:** 안정화 셋업 + 도달 보너스 유지 + speed_reward를 **웨이포인트 경로((2.5,0)→(2.5,6.3)→(0,6)) 남은거리 진행**으로 교체(우회 처벌 제거), energy 0.1 유지, run_name `2M_reward_waypoint_v1`
- **결과:** _(200k 스모크 → 통과 시 2M, 자동/수동 기록)_
- **교훈:** _(채우기)_
- **다음:** _(채우기)_

## 실험 — 2M_reward_waypoint_v1 (2026-06-02 17:45)
- 설정: warm-start=True, log_std_init=-1.0, target_kl=0.05, total_timesteps=2,000,000
- 결과: 최근성공률 0.0%, approx_kl 0.0004, std 0.350, clip_frac 0.00
- 모델: models/checkpoints/ppo_final.zip
- 교훈: (직접 채우기)
- 다음: (직접 채우기)

## 실험 — stage0_openplane_warm_3m_v1 (2026-06-02 18:59)
- env: AntMazeOpen-v0 {'goal_y': 3.0}, init_from=None
- 설정: warm-start=True, log_std_init=-1.0, gamma=0.99, target_kl=0.05, total_timesteps=1,000,000
- 보상: goal_bonus=(reg기본), stall=(reg기본), time=(reg기본)
- 결과(학습말미 rolling20): 성공률 55.0%, approx_kl 0.0039, std 0.363, clip_frac 0.01
- 결정적평가(100ep): 성공률 66.0%, 평균 ep_len 457 → Stage-0 FAIL ❌
- 모델: models/checkpoints_stage0_warm/ppo_final.zip
- 교훈: **'걷기' 검증 성공** — 미로 ~1.5~5% → **66%**(결정적100ep), 성공 ep_len **457**(≪1000 = 배회 아니라 완주). **병목은 미로가 아니라 보행/탐색이었음(가설 확정).** 단 warm이 scratch(81%)보다 **낮음** — BC가 U-path 전문가((2.5,0)行) 기반이라 평지 직진과 **헤딩 바이어스가 어긋남** + std 0.363으로 좁아 탈출 못 함.
- 다음: warm 살리려면 log_std_init↑(-0.3쯤)로 탐색 넓혀 BC 바이어스 탈출, 또는 'heading' 전문가로 BC 재수집. 아니면 scratch 트랙으로 커리큘럼 진행.

## 실험 — stage0_openplane_scratch_3m_v1 (2026-06-02 19:00)
- env: AntMazeOpen-v0 {'goal_y': 3.0}, init_from=None
- 설정: warm-start=False, log_std_init=-0.5, gamma=0.99, target_kl=0.05, total_timesteps=1,000,000
- 보상: goal_bonus=(reg기본), stall=(reg기본), time=(reg기본)
- 결과(학습말미 rolling20): 성공률 55.0%, approx_kl 0.0015, std 0.565, clip_frac 0.00
- 결정적평가(100ep): 성공률 81.0%, 평균 ep_len 384 → Stage-0 PASS ✅
- 모델: models/checkpoints_stage0_scratch/ppo_final.zip
- 교훈: **Stage 0 PASS (81%, ep_len 384).** BC 없이도 PPO가 빈 평지에서 목표 보행 학습 — 핵심 = wander-trap 제거(stall+시간 페널티) + **넓은 탐색**(log_std -0.5 → std 0.565). warm(66%, std 0.363)보다 높음 = **U-path BC 바이어스가 평지에선 오히려 해**. 곡선은 1M에서 아직 상승 중(lr→0이라 정체).
- 다음: 커리큘럼 4m→6m(init_from으로 이어학습) 또는 2M로 늘려 더 끌어올리기. Stage 1(미로 복귀 + A* 길찾기)로 갈 준비됨 — 보행은 검증됨.

## 실험 — stage0_openplane_scratch_4m_v1 (2026-06-02 19:15)
- env: AntMazeOpen-v0 {'goal_y': 4.0}, init_from=models/checkpoints_stage0_scratch/ppo_final.zip
- 설정: warm-start=False, log_std_init=-0.5, gamma=0.99, target_kl=0.05, total_timesteps=1,000,000
- 보상: goal_bonus=(reg기본), stall=(reg기본), time=(reg기본)
- 결과(학습말미 rolling20): 성공률 90.0%, approx_kl 0.0009, std 0.549, clip_frac 0.00
- 결정적평가(100ep): 성공률 73.0%, 평균 ep_len 533 → Stage-0 FAIL ❌
- 모델: models/checkpoints_stage0_scratch_4m/ppo_final.zip
- 교훈: 커리큘럼 이어학습(init_from) 동작 확인 — 3m 워커가 4m로 일반화(학습 전 63%→). 결정적 **73%**, rolling20 90%(결정적이 더 보수적·안정 지표). 3m(81%)→4m(73%) **거리 늘수록 소폭 하락** = 먼 목표 = 스텝↑ = 표류/전복 기회↑ + 1M 미수렴 가능. ep_len 533(3m 384보다 김, 당연).
- 다음: 6m 이어학습으로 커리큘럼 마무리. 80% 확실히 넘기려면 거리별 2M 또는 stall/시간 페널티 재튜닝.

## 실험 — stage0_openplane_scratch_6m_v1 (2026-06-02 19:21)
- env: AntMazeOpen-v0 {'goal_y': 6.0}, init_from=models/checkpoints_stage0_scratch_4m/ppo_final.zip
- 설정: warm-start=False, log_std_init=-0.5, gamma=0.99, target_kl=0.05, total_timesteps=1,000,000
- 보상: goal_bonus=(reg기본), stall=(reg기본), time=(reg기본)
- 결과(학습말미 rolling20): 성공률 80.0%, approx_kl 0.0117, std 0.532, clip_frac 0.04
- 결정적평가(100ep): 성공률 87.0%, 평균 ep_len 557 → Stage-0 PASS ✅
- 모델: models/checkpoints_stage0_scratch_6m/ppo_final.zip
- 교훈: **Stage 0 커리큘럼 완주 — 6m(원래 목표 거리) 87% PASS.** 3m(81%)→4m(73%)→6m(87%): 6m이 4m보다 높음 = 커리큘럼 누적학습(3+4+6=3M 누적)으로 최종 정책이 가장 강함, 4m 딥은 중간 미수렴. **결론: 개미는 원래 목표 거리(6m)를 빈 평지에서 87% 완주 → 미로의 유일한 미해결은 기둥(=A* 문제).** ep_len 557(전구간 ≪1000 = 완주, 배회 아님).
- 다음: **Stage 1** — 기둥 복귀 + 이 워커 위에 A*/웨이포인트 서브목표(개미 obs의 목표-상대벡터를 '현재 서브목표' 기준으로). 워커가 6m까지 견고하니 서브목표 간(2.5~6.3m) 커버. (옵션: warm 트랙 살리기, 거리별 2M로 90%+.)

## 실험 — stage1_omnidir_v1 (2026-06-02 20:17)
- env: AntMazeOpen-v0 {'goal_y': 6.0}, init_from=models/checkpoints_stage0_scratch_6m/ppo_final.zip
- 설정: warm-start=False, log_std_init=-0.5, gamma=0.99, target_kl=0.05, total_timesteps=2,000,000
- 보상: goal_bonus=(reg기본), stall=(reg기본), time=(reg기본)
- 결과(학습말미 rolling20): 성공률 45.0%, approx_kl 0.0008, std 0.519, clip_frac 0.00
- 결정적평가(100ep): 성공률 51.0%, 평균 ep_len 701 → Stage-0 FAIL ❌
- 모델: models/checkpoints_stage1_omnidir/ppo_final.zip
- 교훈: (직접 채우기)
- 다음: (직접 채우기)

## 실험 — stage1_maze_waypoint_v1 (2026-06-02 20:26)
- env: AntMazeWaypoint-v0 {}, init_from=models/checkpoints_stage1_omnidir/ppo_final.zip
- 설정: warm-start=False, log_std_init=-0.5, gamma=0.99, target_kl=0.05, total_timesteps=1,000,000
- 보상: goal_bonus=(reg기본), stall=(reg기본), time=(reg기본)
- 결과(학습말미 rolling20): 성공률 15.0%, approx_kl 0.0013, std 0.512, clip_frac 0.00
- 결정적평가(100ep): 성공률 8.0%, 평균 ep_len 982 → Stage-0 FAIL ❌
- 모델: models/checkpoints_stage1_maze/ppo_final.zip
- 교훈: (직접 채우기)
- 다음: (직접 채우기)

## 실험 — stage1_cmdfollow_rebalance_v1 (2026-06-02 21:59)
- env: AntMazeOpen-v0 {'energy_coef': 0.02, 'time_penalty': 0.0, 'stall_penalty': 0.0, 'goal_bonus': 100.0}, init_from=models/checkpoints_stage1_omnidir/ppo_final.zip
- 설정: warm-start=False, log_std_init=-0.5, gamma=0.99, target_kl=0.05, total_timesteps=2,000,000
- 보상: goal_bonus=100.0, stall=0.0, time=0.0
- 결과(학습말미 rolling20): 성공률 5.0%, approx_kl 0.0011, std 0.529, clip_frac 0.00
- 결정적평가(100ep): 성공률 4.0%, 평균 ep_len 980 → Stage-0 FAIL ❌
- 모델: models/checkpoints_stage1_cmdfollow_rebalance/ppo_final.zip
- **frozen 미로 평가(eval_waypoint 100ep): 성공률 13.0%, ep_len 950, mean wp_idx 0.61** (vs 직전 미로 fine-tune 8%)
- **미로 진단(40ep wp_idx 분포 + 종료원인):** wp_idx 0(첫 +x 다리 전 막힘) 40% · wp_idx 1(+y 턴서 막힘) 55% · 완주 5%. **종료: timeout 35/40, fell 0/40, reached 5/40.**
- 교훈: **① C0가 붕괴를 확정적으로 고침 — goals/ep 0.05 → ~1.1 (2M 내내 0.7~1.65 안정, 재붕괴 0). per-step 페널티(time/stall) 제거 + energy floor가 self-termination 유인을 없앰 → fall 0/40 (붕괴 시그니처였던 '빨리 넘어져 페널티 탈출'이 완전히 사라짐).** 진단(페널티 누적 > 도달보너스)이 정확했고 수정이 작동. ② **그러나 붕괴 수정 ≠ 미로 완주.** 미로 8→13%로 소폭만 개선, 50% 게이트 미달. 잔여 병목 = **mid-motion +y 턴(wp_idx 1서 55% 막힘)** = 정확히 예측된 leg-2 턴. ③ **부작용: 페널티 0 → wander-forever.** 40%는 첫 목표도 못 잠그고 1000스텝 timeout(예전엔 넘어졌을 것). 후속 단계서 작은 time_penalty 복원으로 교정 필요. ④ 시퀀스 완주(3목표) 4~15%서 정체 — 임의 방향 mid-motion 턴은 여전히 hard.
- 다음: **각도 커리큘럼(±30→60→90) 발동** (사용자 사전승인: "leg-2에서만 막히면"; 지배 병목 55%가 +y 턴). `RandomWaypointSequence`에 `max_turn_angle` 추가 → resample 목표를 현재 진행방향(qvel) ±각도로 바이어스 → 부드러운 턴부터 자주 성공 → 점차 확대. init_from=이 C0 모델. **턴30도 못 배우면(시퀀스 완주 안 오르면) 물리적 한계로 보고 omni+A* 정리 후 Phase 4.**

## 실험 — stage1_cmdfollow_turn30_v1 (2026-06-02 22:24)
- env: AntMazeOpen-v0 {'energy_coef': 0.02, 'time_penalty': 0.0, 'stall_penalty': 0.0, 'goal_bonus': 100.0}, init_from=models/checkpoints_stage1_cmdfollow_rebalance/ppo_final.zip
- 설정: warm-start=False, log_std_init=-0.5, gamma=0.99, target_kl=0.05, total_timesteps=2,000,000
- 보상: goal_bonus=100.0, stall=0.0, time=0.0
- 결과(학습말미 rolling20): 성공률 15.0%, approx_kl 0.0009, std 0.539, clip_frac 0.00
- 결정적평가(100ep): 성공률 13.0%, 평균 ep_len 938 → Stage-0 FAIL ❌
- 모델: models/checkpoints_stage1_cmdfollow_turn30/ppo_final.zip
- **frozen 미로 평가(100ep): 성공률 18.0%, ep_len 937, mean wp_idx 0.63** (8%→13%(C0)→**18%** 점증)
- **미로 진단(40ep):** wp_idx 0(첫 다리 전 막힘) 52%↑ · wp_idx 1(+y 턴) 35%↓ · 완주 12%↑. 종료: reached 8 / timeout 32 / fell 0.
- 교훈: **① ±30° 각도 바이어스가 미로를 점증 개선(13→18%) — 일부 개미가 +y 턴을 통과(wp_idx 2: 5%→12%, wp_idx 1 막힘 55%→35%).** ② **그러나 ±30° 시퀀스 완주조차 결정적 13%** = 부드러운 mid-motion 턴도 턴당 ~43%만 성공. 커리큘럼 전제(쉬운 설정서 '자주' 성공→부트스트랩)가 안 채워짐 — 즉 **mid-motion 조향이 이 워커의 본질적 약점**(걷기·도달은 견고한데 달리며 방향전환은 ~43%). ③ trade-off: ±30° 학습이 mid-motion 턴(leg-2)은 도왔지만 **출발 reorientation(leg-1)은 오히려 악화**(wp_idx0 40%→52%) — 평지 커리큘럼이 미로 첫 다리와 어긋남. ④ 추세(8→13→18%, 단계당 ~5pp)로는 70% 게이트 도달 난망. 물리적 한계에 근접.
- 다음: **의사결정 지점.** 후보 — (A) 각도 ±60→90 계속(점증, 70% 난망) (B) **turn30서 미로 직접 fine-tune(리밸런스 보상)** — 실제 90° 턴을 직격, 예전 미로 fine-tune 8%는 붕괴 보상 탓이라 리밸런스로 재시도 가치 (C) 물리적 한계로 보고 omni+A*+진단을 Stage 1 결과물로 정리→Phase 4. 사용자와 상의해 결정.

## 검증 — goal/기둥 배치 가설 (frozen turn30, eval-only, 2026-06-02)
> 사용자 가설(영상 관찰): "goal(0,6)이 기둥 끝에 붙어 접근 공간 없음 = 배치 문제, 꺾기 한계 아닐 수도." → scripts/verify_placement.py로 검증(60ep/조건).

- **E1 기둥 격리 (동일 waypoints):** 미로(기둥) **16.7%** → 빈 평지(기둥X) **38.3%** = **기둥 제거로 2배↑.** (wp0 막힘: 미로 53% vs 평지 27% — 기둥이 출발 leg를 2배 막음. 개미가 (0,0)=기둥 밑동에 스폰돼 끼임.)
- **E2 goal_radius sweep (미로):** r=1.0 **16.7%** → r=1.5 **43.3%** → r=2.0 **50.0%** = **반경 완화로 3배↑.** 벽 끝(0,6) tight 접근이 큰 병목 — 완화하면 개미가 목표 근방엔 50% 도달.
- **E3 goal 위치 이동 (미로):** (0,6.8) 벽 위(센터라인 유지) **10%**(악화), (-1.5,6) 측면 **25%**(개선). 센터라인 근처는 나쁘고, 측면으로 빼면 좋아짐.
- 교훈: **사용자 가설 상당부 적중 — "물리적 턴 한계"가 아니라 배치/기하 문제가 큼.** ① 기둥 제거 2배, ② 반경 완화 3배. 단 빈 평지서도 r=1.0 38%(62% 미완) = 턴 난이도도 일부 실재. 두 기하 결함: **(가) (0,0) 스폰이 기둥 밑동에 끼임**(wp0 막힘 53%), **(나) goal이 벽 끝(0,6)+tight 1m 반경**(마지막 접근 어려움). 둘 다 **고칠 수 있음** → 미로 재오픈.
- 다음: 물리 한계 결론 철회. **리밸런스 미로 fine-tune(turn30 init) + 합리적 goal_radius(1.5) + 스폰/경로 점검**으로 70% 게이트 재도전. 사용자와 방향 확정.

## 실험 — stage1_maze_rebalance_v1 (2026-06-02 23:40)
- env: AntMazeWaypoint-v0 {'energy_coef': 0.02, 'time_penalty': 0.0, 'stall_penalty': 0.0, 'start_xy': [0.0, -1.0]}, init_from=models/checkpoints_stage1_cmdfollow_turn30/ppo_final.zip
- 설정: warm-start=False, log_std_init=-0.5, gamma=0.99, target_kl=0.05, total_timesteps=2,000,000
- 보상: goal_bonus=(reg기본), stall=0.0, time=0.0
- 결과(학습말미 rolling20): 성공률 50.0%, approx_kl 0.0007, std 0.542, clip_frac 0.00
- 결정적평가(100ep): 성공률 43.0%, 평균 ep_len 808 → Stage-0 FAIL ❌
- 모델: models/checkpoints_stage1_maze_rebalance/ppo_final.zip
- **종합 평가(100ep, spawn(0,-1)):** r=1.5 det **43%**/stoch 51% · r=1.0 det **21%**/stoch 26%. wp_idx(r=1.5 det): wp0 40% · wp1 45% · wp2 15%. fell 0.
- 교훈: **① 배치 가설 검증됨 — 기하 수정(스폰 (0,0)→(0,-1) + r=1.5)만으로 frozen 20%→40%(학습 0).** 사용자 영상 관찰이 옳았음: 미로 난이도의 큰 부분이 "물리 한계"가 아니라 배치(기둥 밑동 스폰 + 벽끝 goal + tight 반경). ② **그러나 2M fine-tune은 기하수정 위에 거의 못 보탬(40%→43% @ r=1.5).** 70% 게이트 미달. ③ **strict r=1.0은 21%** — 정직한 1m 반경에선 거의 개선 없음(턴/leg-1 난이도 실재). ④ **det↔stoch 격차 작음(~5-8pp)** — 정책 sharpening으로 70% 도달 불가. ⑤ 잔여 누수 = **wp0 40%(첫 +x 다리 reorientation 실패)** — 스폰 수정해도 지속. ⇒ 진실: 미로 난이도 = 기하(고침, ~20pp) + 턴/leg-1(잔존, 실재). 8%→**43%(r=1.5)/21%(r=1.0)**.
- 다음: **diminishing returns 도달.** 70% 게이트는 PPO+커리큘럼+기하수정으로 미달. 후보: (A) Stage 1 정리(붕괴수정+가설검증+8→43%+영상+진단)→Phase 4 ← 추천 (B) A* 웨이포인트 평가+영상(포트폴리오 마감) (C) SAC+HER 에스컬레이션. 사용자 결정.

## 실험 — stage1_maze_short_v1 (2026-06-03 00:25)
- env: AntMazeWaypoint-v0 {'energy_coef': 0.02, 'time_penalty': 0.0, 'stall_penalty': 0.0, 'pillar_half_len': 1.0, 'start_xy': [0.0, 0.0]}, init_from=models/checkpoints_stage1_maze_rebalance/ppo_final.zip
- 설정: warm-start=False, log_std_init=-0.5, gamma=0.99, target_kl=0.05, total_timesteps=2,000,000
- 보상: goal_bonus=(reg기본), stall=0.0, time=0.0
- 결과(학습말미 rolling20): 성공률 45.0%, approx_kl 0.0016, std 0.552, clip_frac 0.00
- 결정적평가(100ep): 성공률 58.0%, 평균 ep_len 710 → Stage-0 FAIL ❌
- 모델: models/checkpoints_stage1_maze_short/ppo_final.zip
- **종합 평가(100ep, 축소기둥+A*+honest spawn(0,0)):** r=1.0 det **58%**/stoch 54% · r=1.5 det **65%**/stoch 61%. wp(r=1.0 det): wp0 10% · wp1 46% · wp2 **44%**. fell 0.
- 교훈: **① 사용자 아이디어(기둥 1/3 축소) + A* 재계획이 정직한 성공률을 3배 가까이 끌어올림: full-maze honest r=1.0 21% → 축소기둥 58%(r=1.0)/65%(r=1.5).** ② 축소 기둥(y∈[2,4])이 스폰-끼임 제거(wp0 40%→10%) + goal 위 2m 열림. ③ **A* 재계획이 핵심 레버** — 짧은 기둥엔 conservative (2.5,...)의 sharp 90° 코너 대신 gentle 경로([(-1,1.4),(-1,4.6),(0,6)]) → wp2 완주 15%→44%. 분리 아키텍처(고전 플래너가 맵에 맞춰 경로 재계산)의 실효 입증. ④ det↔stoch 격차 작음(~4pp) = 안정적 정책. ⑤ 70% 게이트엔 살짝 못 미침(턴 약점 잔존) but **honest 58%는 강한 결과**. ⑥ **풀맵(pillar_half_len 기본 3.0) 무변경 보존** — 회귀 PASS, 사용자 추후 풀맵 재테스트 가능.
- 다음: 영상(축소기둥 완주) 생성. 70% 더 원하면 (A) r=1.5(65%, goal 깨끗해 정직) 채택 또는 (B) 추가 fine-tune/커리큘럼. 아니면 이 결과(8%→58~65%)로 Stage 1 마감 → Phase 4. 풀맵 70%는 future work(유연한 모델 업그레이드 후, 사용자 예정).

## 검증 — up_z 뒤집힘 진단 (eval-only, 2026-06-03)
> 리서치 + 사용자 영상 관찰: "개미가 뒤집힌 채 기어다닌다. 종료조건이 z(키)만 봐서(z<0.2) 못 잡는다 — 뒤집혀도 z>0.2라 안 끝남." 검증: `scripts/eval_flip.py`로 체크포인트별 매 스텝 **up_z = 1 - 2*(qx²+qy²)** 직접 측정(env 무수정, unwrapped qpos만 읽음 → 87% 워커 재현성 무손상). up_z: **+1 똑바로 · 0 옆으로 누움 · -1 완전 전복(배가 위)**. flip_thresh=0.0, 결정적 100ep.

- **평지 워커(stage0_scratch_6m):** 성공 **87.0%**/ep_len 557 (EXPERIMENTS 재현 일치) · **flip_rate 34%** · mean min up_z **0.231** · 뒤집힘 스텝비율 **11.6%** · 종료 {reached 87, timeout 13, **fell 0**}. min up_z 분포 **이봉형(bimodal)**: **33ep이 [-1.0,-0.5)**(거의 완전 전복), **58ep이 ≥0.7**(깨끗한 직립) — 가운데 [-0.5,0.5)는 단 4ep(빈 골). **성공 ep 중에도 최저 up_z -1.0 존재 = 뒤집힌 채 골 도달.**
- **미로 워커(stage1_maze_short):** 성공 **58.0%**/ep_len 710 (재현 일치) · **flip_rate 99%** · mean min up_z **-0.966** · **뒤집힘 스텝비율 59.7%** · 종료 {reached 58, timeout 42, **fell 0**}. min up_z 분포: **97/100ep이 [-1.0,-0.5)**, 직립(>0.7) 단 1ep = **사실상 항상 뒤집힌 채 기어다님(전 스텝의 ~60%가 배가 위/옆).**
- 교훈: **① 가설 결정적 확정.** `fell 0/40`은 z만 본 **눈먼 지표** — 두 워커 모두 fell 0인데 평지 1/3, 미로 ~전부가 (거의)완전 전복. 자세를 안 봐서 못 잡았을 뿐. ② **미로 워커는 '걷는' 게 아니라 '뒤집혀 기어가는' 정책**(60% 스텝 inverted). ⇒ **Stage 1의 'mid-motion 턴 약점/기하 병목' 진단은 개미가 등으로 기는 상태에서 내려진 것** — 턴 약점이 전복의 *결과*일 개연성 큼(뒤집힌 채 조향은 불가). ③ **왜 미로가 평지보다 훨씬 심한가:** Stage 1이 omnidir→turn30→rebalance→short로 턴 과제에 누적 fine-tune되며 방향전환 압력을 받자, 자세 종료/페널티가 없는 상태에서 정책이 **'구르기=싸게 방향 바꾸기'(전복 basin)**로 더 깊이 표류(평지 12%→미로 60% 스텝). ④ **UP_THRESH=0.0 데이터로 정당화:** 평지 min up_z 히스토그램의 [0.0,0.5) 구간이 거의 빔 → 0.0은 깨끗한 직립(≥0.5)은 안 건드리고 깊은 전복(≤-0.5)만 잡는 안전 경계(정상 보행 흔들림으론 안 걸림).
- 다음: **Phase 2 — 자세 종료 + 직립 보너스 묶음**(termination에 `up_z<0.0` 추가 + 보상에 `r_up=up_bonus_coef·max(0,up_z)`; time/stall 페널티 0으로 suicide 방지), **87% 워커 init_from 이어학습**(평지). 검증 신호: **flip_rate→~0 · ep_len↓ · 성공률 ≥87% 유지**. 통과 시 미로 재테스트로 58~65% 정체가 풀리는지 확인 — 사용자 가설("전복 수정이 geodesic보다 근본적")의 직접 시험.

## 실험 — Phase 2 자세종료+직립보너스 (평지, 87%워커 이어학습) (2026-06-03)
> 구현: env에 up_z=1-2*(qx²+qy²) 자세종료(`up_thresh`) + 직립보너스(`+up_bonus_coef·max(0,up_z)`),
> wrapper 2곳 종료도 `u._flipped()`로 통일, train_ppo final_eval/콜백에 flip 지표 추가.
> ⚠️ 속도 정정: "재학습 2-3h"는 잘못된 추정 — 실측 **fps ~3300, 1M ≈ 5분**(CPU, MuJoCo는 원래 CPU-bound).

- **v1 실패 (up_thresh=0.0, σ유지):** 87% 워커가 **30%로 붕괴, 회복 안 됨**(440k까지 추세 없음).
  ep_len ~120, flip종료 ~60/윈도(에피소드 ~75%가 즉시 전복종료). **원인 진단(eval_flip --stochastic):**
  체크포인트 σ0.53 탐색이 **에피소드 93%를 (거의)완전 전복**시킴 → 자세종료(0.0)가 78%를 즉시 끝내
  **학습을 굶김**(연구가 경고한 "too-strict UP_THRESH → 탐색 죽음"). std는 0.51 고정 — on-policy가
  저노이즈 롤아웃을 못 봐서 "노이즈 낮추면 낫다"를 못 배움. **임계값으론 못 고침**: 전복이 깊어
  (-1.0) up_thresh -0.5도 74% 종료(평지 stochastic 측정).
- **핵심 측정 — flip율 vs σ(전복종료 OFF, 87%워커):** σ0.53→flip 93%/성공53% · σ0.37→50%/87%
  · **σ0.22(log_std -1.5)→33%/100%** · σ0.14→37%/100%. ⇒ **레버는 임계값이 아니라 탐색 노이즈.**
- **v2 고침 (override_log_std=-1.5, σ0.22):** init_from 직후 log_std를 강제로 낮춰 탐색을 깨끗한
  보행 근처로 제한 → 대부분 에피소드 생존(전복 33%) → 신호 충분 → 자세종료+직립보너스가 잔여
  전복 제거. **결정적100ep: 성공 98%(87%→**상승**) · flip_rate 34%→2% · mean_min_up_z 0.23→0.91
  · 뒤집힘 스텝 11.6%→0.0% · ep_len 557→100(빠르고 직진).** min up_z 히스토그램 이봉형→단봉형
  (82/100ep이 [0.9,1.0], 전복 0). 학습 ~5분. 모델: `models/checkpoints_p2_upbonus_plane_v2/ppo_final.zip`.
- 영상: before `outputs/videos/06_phase3_미로직립_88%/미로_before_뒤집힘_58%.mp4`(미로 배깔고 골 도달) ↔ after `outputs/videos/05_phase2_평지직립_98%/평지_after_직립_98%.mp4`(평지 직립 보행).
- 교훈: **① 자세종료+직립보너스가 평지 전복을 사실상 제거**(34%→2%) **하면서 성공률은 오히려 상승**
  (87%→98%) **+ 2배 이상 빨라짐**(ep_len 557→100). 연구 예측("더 빨리 끝남, 코너링 개선") 적중.
  ② **숨은 함정 = 탐색 노이즈.** 자세종료를 켜면 σ0.53 탐색이 매 에피소드 전복→즉시 종료→학습
  굶음(87%→30%). 임계값 튜닝으론 못 고침(전복이 깊어 어떤 threshold도 ~74% 종료). **해법은
  override_log_std로 탐색 자체를 줄여** 정책이 깨끗한 보행에 머물게 하는 것 — 그러면 자세종료가
  잔여 소수 전복만 잡고 직립보너스가 마무리. ③ **suicide 재발 0**(time/stall 페널티 0 + 직립보너스
  +부호 + 전복종료 보상 0). flip_term 2%는 전복 자체가 거의 없어서지 self-termination 유인 아님.
  ④ 속도 추정 오류 교훈: 추정 말고 실측(fps) — 1M은 2-3h가 아니라 5분, 반복 비용 거의 0.
- 다음: **Phase 3 미로 재테스트** — 이 직립 워커를 `AntMazeWaypoint-v0`(축소기둥+A*)에 init_from
  올려 자세종료+직립보너스로 fine-tune, **58~65% 정체가 풀리는지** 확인. 미로 워커는 60% 스텝
  전복이었으므로 가장 큰 효과 기대 지점(사용자 가설 "전복 수정이 geodesic보다 근본적"의 직접 시험).
  before/after 미로 영상으로 마무리. (옵션: up_z를 obs에 추가하는 P3는 obs 110차원 scratch 필요라 보류.)

## 실험 — Phase 3 미로 재테스트: 직립 워커로 미로 (2026-06-03)
- env: AntMazeWaypoint-v0 {energy 0.02, time 0, stall 0, pillar_half_len 1.0, start_xy(0,0), **up_thresh 0.0, up_bonus_coef 0.1**}, A* waypoints [[-1,1.4],[-1,4.6],[0,6]], goal_radius **1.0(honest)**, **init_from=직립 평지워커 p2_v2**(미로 체크포인트 아님), **override_log_std=-1.5(σ0.22)**, 2M
- 결과(학습말미 rolling20): 성공률 85%, approx_kl 0.0027, std 0.224, clip_frac 0.00
- 결정적평가(100ep): **성공률 88.0%, 평균 ep_len 118 → PASS ✅**
- 자세(100ep, 독립검증 eval_flip): **flip_rate 11.0% · mean_min_up_z 0.798 · 뒤집힘 스텝 0.1%** · 종료 {reached 88, term_other(전복) 11, timeout 1}. min up_z 히스토그램: 58ep [0.9,1.0] · 29ep [0.7,0.9] · 11ep [-0.5,0)(잔여 턴 전복) → **87%가 ≥0.7 직립.**
- 영상: before `outputs/videos/06_phase3_미로직립_88%/미로_before_뒤집힘_58%.mp4`(같은 미로, 배깔고 김) ↔ after `outputs/videos/06_phase3_미로직립_88%/미로_after_직립_88%.mp4`(같은 미로, 직립 주행). 모델: `models/checkpoints_p3_maze_upright/ppo_final.zip`.
- 교훈: **★ 사용자 가설 결정적 입증 — 전복 수정이 'turn/기하 한계'보다 근본적이었다.** 같은 honest 설정(축소기둥+A*+spawn(0,0)+r=1.0)에서 **58%→88%(+30pp)**, **전복 스텝 59.7%→0.1%**, **ep_len 710→118(6배 빠름).** 그동안 "mid-motion 턴 약점(~43%)·기하 병목"으로 진단했던 정체(58~65%)는 **개미가 60% 스텝을 배깔고 기던 상태에서 측정된 것** — 턴 약점의 큰 부분이 전복의 *결과*였음(뒤집힌 채 조향 불가). 직립으로 세우니 같은 미로가 **이전에 '난망'이라던 70% 게이트를 넘어 88%.** ② **핵심 레시피: 직립 평지워커(미로워커 아님!) seed + override_log_std로 저노이즈 + 자세종료 + 직립보너스.** 미로 체크포인트는 결정적으로도 전복(mean_min_up_z -0.97)이라 seed로 못 씀 — 깨끗한 평지워커에서 출발해야 미로 턴을 직립으로 학습. ③ 학습 ~10분, suicide 재발 0(time/stall 0 + 직립보너스 +). ④ 잔여 11% 전복 = 가장 어려운 턴에서 기울어 종료 — 추가 튜닝(σ 미세조정/더 긴 학습) 여지지만 88%는 강한 결과.
- 다음: **Stage 1 사실상 해결(8%→58~65%→88%).** 선택지: (A) **풀맵(pillar_half_len 3.0)** 직립 재테스트 — 축소기둥 hack 없이 얼마나 가나(직립이면 풀맵도 오를 가능성). (B) up_z를 obs에 추가(P3a, 110차원 scratch) — 자세종료+보너스만으로 이미 풀려 한계효용 낮음, 선택. (C) Phase 4(world-model/압축/A* 통합)로 진행 — Stage 1 결과물 확정. 사용자 결정. **포트폴리오 서사: "영상에서 전복 발견 → up_z 진단(fell 0은 눈먼 지표) → 자세종료+직립보너스 → 평지 87→98%, 미로 58→88%. 전복이 진짜 병목이었음을 데이터로 입증."**

## 실험 — Phase A 풀맵(pillar_half_len=3.0) 직립 재테스트 (2026-06-03)
> **가설 시험:** 미로 88%는 **축소기둥(pillar_half_len=1.0)** 산물 = 임시방편(band-aid) 아닌가? **진짜 풀맵(3.0, 기둥 6m)**에서 직립 워커가 얼마나 가나. 88% 런과 **단 한 변수(기둥 크기 1.0→3.0)**만 바꿔 정직 A/B. 사용자 결정 = (A) 둘 다(zero-shot eval → fine-tune) + 최대정직 기하(spawn(0,0), r=1.0; r=1.5 보조).
- **풀맵 기하(코드 확인):** A* 웨이포인트 `[[1,0],[1,6],[0,6]]` = x=1.0으로 **6m 직선 주행(긴 기둥 옆)** + **(1,6)서 sharp ~90° 좌회전** → goal. (축소맵의 부드러운 [[-1,1.4],[-1,4.6],[0,6]]보다 어려움.) [1,0]·[1,6] 충돌안전(팽창벽 |x|<0.9<1.0). spawn(0,0)은 풀기둥 밑동.

- **Step 0 하니스 재검(축소 config 재현, 신뢰성 게이트):** 같은 p3 체크포인트를 원래 축소맵 config로 eval_flip → **성공 88.0% · flip 11.0% · min_up_z 0.798 · ep_len 118 → EXPERIMENTS 정확 재현 ✅.** 평가 하니스 신뢰 확보 → 풀맵 숫자 신뢰 가능.

- **D1 zero-shot 풀맵(학습 0, 축소맵 88% 정책을 풀맵서 그냥 평가):**
  - eval_flip(canonical, (0,0) r1.0): **성공 39.0%** · flip_rate 59.0% · mean_min_up_z 0.281 · 뒤집힘스텝 2.6% · 종료 {reached 39, term_other(전복) 59, timeout 2}. min_up_z 이봉형(59ep [-0.5,0) 전복종료 · 36ep ≥0.7 깨끗직립).
  - eval_waypoint(교차검증, --use-astar): **성공 39.0%** · ep_len 296 · 평균 wp_idx 0.48 → **두 경로 정확 일치(±5pp 내) = 교차검증 통과.**
  - 보조 r=1.5: 47.0%(flip 52%). **스폰끼임 분리 A/B** spawn(1,0)[기둥서 비킴]: **40.0%**(flip 58%).
  - **D1 귀인:** 88%(축소) → **39%(풀맵)** 급락. **원인은 스폰끼임 아님** — spawn을 기둥서 비켜도 39→40%(**단 +1pp**), flip_rate 58% 유지. r 완화도 +8pp뿐. **진짜 원인 = 풀맵의 더 어려운 기하(6m 기둥-옆 직선 + sharp 90° 턴)에서 전복(flip 59%).** = 정책이 학습 안 한 기하 → 분포shift/미학습.

- **D2 fine-tune 풀맵(직립 레시피를 풀맵에 직접 학습, 2M ≈ 10분):**
  - env: AntMazeWaypoint-v0 {energy 0.02, time 0, stall 0, **pillar_half_len 3.0**, start_xy(0,0), up_thresh 0.0, up_bonus 0.1}, A* `[[1,0],[1,6],[0,6]]`, goal_radius 1.0(honest), **init_from=직립 평지워커 p2_v2**(미로/축소 체크포인트 아님), **override_log_std=-1.5(σ0.22)**. (88% 레시피와 기둥·웨이포인트만 다름.)
  - 학습: 시작 로그 `log_std 강제설정 -1.5 (σ≈0.223)` 확인 ✅, 200k 스모크 게이트 통과(붕괴 0, ep_len 300~580, flip종료 ~10~28/윈도 = v1 붕괴와 질적으로 다름). rolling20 말미 55~90%, min_up_z 0.4~0.77, flip종료 2~13.
  - **결정적평가(100ep): 성공 73.0% · 평균 ep_len 324** · 자세 flip_rate **26.0%**(D1 59%→26%로 반감) · mean_min_up_z **0.615**(D1 0.281→) · 뒤집힘스텝 2.0% · 종료 {reached 73, term_other 26, timeout 1}.
  - eval_waypoint 교차검증: **73.0%** · ep_len 324 · **평균 wp_idx 1.63**(D1 0.48→ 직선 통과·턴 도달) → 두 경로 정확 일치.
  - 보조 r=1.5: **76.0%**(flip 24%).
- 영상: zeroshot `outputs/videos/07_phaseA_풀맵_73%/풀맵_zeroshot_39%.mp4`(풀맵 39%, 직선/턴서 전복) ↔ after `outputs/videos/07_phaseA_풀맵_73%/풀맵_after_직립_73%.mp4`(풀맵 73%, 직립 주행). 모델: `models/checkpoints_p3_maze_fullpillar/ppo_final.zip`.
- **교훈: ★ 임시방편 의심 해소 — 직립 레시피가 진짜 풀맵도 정복.** 축소맵 88%는 쉬운 커리큘럼 단계였고, **진짜 풀맵(기둥 3.0, 무축소)에서도 직립 워커가 73%(r=1.0)/76%(r=1.5)** = **이전에 '난망'이라던 70% 게이트를 정직하게 통과**(전복수정 前 honest 풀맵은 21%였음 → **21%→73%**). ① **축소기둥은 결과를 부풀린 band-aid가 아니라 학습용 디딤돌**이었음 — 풀맵에서 직접 학습하면 디딤돌 없이도 게이트 통과. ② **zero-shot은 일반화 안 됨(39%)** — 그러나 그 39% 손실의 원인은 스폰끼임(+1pp)도 반경(+8pp)도 아닌 **풀맵 기하서의 전복(flip 59%)**, 즉 정책이 안 본 6m 직선+sharp턴 = 분포shift/미학습. **D1(39%)≪D2(73%) 격차가 "기하 한계 아니라 미학습"임을 증명** — 풀맵 기하를 학습하면 전복이 반감(59→26%)되고 성공이 거의 2배. ③ 잔여 27% 실패 = wp_idx 1.63(직선은 통과)·종료 term_other 26 → **sharp 90° 턴/최종접근서 기울어 전복**(스폰·직선 아님). = 튜닝거리(σ 미세조정/더 긴 학습/턴 커리큘럼)지 물리 한계 아님. ④ suicide 재발 0(time/stall 0 + 직립보너스 +). 학습 ~10분, 반복 비용 거의 0.
- **다음: Stage 1 완전 마감 — 축소 88% + 풀맵 73%(정직) 둘 다 확보.** band-aid 의심 해소됐으니 **Phase 4(world-model/압축/A* 통합)** 진행 추천. 풀맵 73%→80%+ 더 원하면 (선택) sharp턴 커리큘럼 또는 σ -1.2 미세조정 또는 더 긴 학습. **포트폴리오 서사 보강: "축소맵 88%가 임시방편 아니냐 의심 → 진짜 풀맵서 zero-shot 39%(전복이 원인, 스폰 아님을 A/B로 입증) → 풀맵 직접학습 73%로 70% 게이트 정직 통과 → 축소기둥은 band-aid가 아니라 디딤돌이었음을 데이터로 확정."**

## 실험 — Phase 3.3 정식 평가·비교 (random vs BC vs RL + 팩트별 ablation) (2026-06-03)
> README 5단계("평가·문서화 — random vs BC vs RL 비교") 구현. 다른 AI 제안(정식 비교+도표+JSON)을 검토→**업그레이드**: 좁은 3자 대신 **'한 개입(팩트)이 성능을 몇 pp 올렸나'를 맵별로** ablation. 산출: `scripts/evaluate_comparison.py`(한 하니스 9개 평가→`outputs/evaluation_results.json`), `scripts/plot_comparison.py`(→`outputs/images/comparison.png`·`factor_impact.png`). 결정적 100ep, seed0=20000.
- **배선 정직성:** PPO=`FixedNormalizeObs`(고정통계) · BC=**raw obs**(내부 자체정규화, 이중정규화 금지) · random=`action_space.sample()`. **전복수정 前 정책은 자세종료 OFF(up_thresh=-1.1)로 원래 측정 regime 복원** — 안 하면 현재 env 기본 up_thresh=0.0이 뒤집힌 워커를 즉시 종료해 일지값(58/81/87)을 못 냄.
- **하니스 검증(전 항목 일지 재현 ✅):** RL전복前 58% · RL전복後 88% · 풀맵 zero-shot 39% · 풀맵 fine-tune 73% · 평지 scratch3m 81% · BC-warm3m 66% · scratch6m 87% — 전부 ±0pp 재현. random 0% · **BC 0%**(미로). 
- **★ BC 발견:** BC는 미로서 **직립이지만(mean_min_up_z 0.959) 길찾기 실패로 timeout 0%** — "모방만으론 미로 못 풂"을 시각적으로 입증. 또 평지서 **BC-warm(66%) < scratch(81%)** = U-path 바이어스로 **BC가 오히려 해**(원래 계획 'BC→RL warm-start'를 뒤집은 발견).
- **팩트별 임팩트(Δ 성공률, 통제=env 고정·한 변수):**
  | 팩트 | env | Δ |
  |---|---|---|
  | 학습 자체 (random→RL) | 축소미로 | **+88** |
  | 모방(BC) vs scratch | 평지3m | **+15**(scratch 우위) |
  | 기하: 풀맵→축소기둥+A* | 미로 honest | **+37**(21→58) |
  | ★ **전복수정(flip-fix)** | **축소미로 동일** | **+30**(58→88, flip 99→11) |
  | 맵 난이도: 축소→풀맵 | 직립워커 동일 | **−49**(88→39, 더 어려움) |
  | 풀맵 직접학습(일반화) | 풀맵 honest | **+34**(39→73) |
- 도표: `outputs/images/comparison.png`(4패널: 팩트토네이도·baseline→RL·전복수정·맵난이도) + `factor_impact.png`(히어로). 영상: `outputs/videos/06_phase3_미로직립_88%/`(전복 before↔after) · `07_phaseA_풀맵_73%/`(풀맵).
- 교훈: **① 정직한 비교의 핵심은 'env 고정'** — 단일 막대 0→5→58→88→73은 3개 다른 맵을 한 축에 섞어 88→73이 '퇴행'처럼 보이는 오류. 맵별로 분리하니 각 팩트 기여가 명확. ② 다른 AI의 "BC 5%"는 오독(그건 원래 풀맵 warm-start 정체값)이었고, **순수 BC는 처음 측정(미로 0%)**. ③ **전복수정(+30, env 완전 동일)이 단일 최대 게임체인저**, 학습 자체(+88)·기하(+37)·풀맵 일반화(+34)가 뒤따름. ④ 모든 숫자가 한 하니스서 일지 재현 → 신뢰.
- 다음: **Phase 3 전체 마감(3.1 PPO·3.2 보상/커리큘럼·3.3 평가/비교 + 보너스 전복 발견/수정).** → **Phase 4(World Model)**.
