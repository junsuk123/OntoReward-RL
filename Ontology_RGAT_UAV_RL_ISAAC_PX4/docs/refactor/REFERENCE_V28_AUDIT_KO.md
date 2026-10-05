# 2026-10-04: 2D v2.8 대조, 운용 점검, 시스템 리팩터링

## 판정

기존 시스템은 **축 확장 이외에도 알고리즘적으로 동등하지 않았다**.
v2.5 기반 Python 경량 환경과 Shin/Isaac 실험이 별도로 동작했고,
루트 실행 명령은 후자를 선택했다. v2.8 핵심을 경량 경로에 이식하고 실행·체크포인트
경계를 분리했다. 이후 별도 causal spatial 경로에서 실제 `(ax,ay,az)` 정책의
PPO 학습→저장→재로딩→실제 Isaac/PX4 평가까지 실행했다. **연결·실행의 검증과
학습 착륙 성능의 검증은 다르며, 2D 성능 재현/강건 착륙 판정은 아직 불가**하다.

### 최신 상태 — 승인 연장 구간 진행 중

- 기본 계약 v5는 그대로이며 v6–v10은 명시적 opt-in 후보다.
- 사용자 선택에 따라 Isaac 외란은 유지하고 v8 로컬에도 같은 시드 draw를 적용했다.
- 이전 승인 구간의 v8 actual PPO→저장→local reload→새 실제 seed 13000 평가를 완료했다.
  세 모델 모두 SAFE_ABORT이며 비영 R-GAT 출력은 확인했지만 acceptance=false다.
  전체 15회(학습 6/validation 6/test 3)의 종료·disarm은 확인됐다.
- 앞선 v8 seed 823 실제 3-arm×2회는 vector SAFE_ABORT/UNSAFE_CONTACT,
  flat·R-GAT SAFE_ABORT/SAFE_ABORT였다. 이전 unsafe 기록은 그대로 보존한다.
- 종료/arming gate, 같은 시각 ABG 처리, 광학 프레임 변환, 한정된 소유 스택 복구를
  수정했다. 에피소드별 cold PX4 초기화와 v8 로컬 커리큘럼을 완료했다.
- 10월 5일 사용자 추가 승인으로 04:13 KST까지 정합·재검증을 진행한다.
  v9는 축별 reference 9×12 문맥, 공동 3D 하강 evidence, 촬영 시각/own EKF 정합 및
  로컬 영상 지연을 추가했다. v9의 부분 결과는 보존하고 해당 학습을 종료했다.
  별도 v10은 trust/latch/제동거리/terminal corridor, 외란 중 자체 위치 backup,
  실제 optical measurement 기반 view reward를 추가했다. 두 seed × 세 arm의 로컬 학습을
  완료했다. actual PPO 학습·저장 6개와 local reload 12회도 완료했으며 actual 고정 시험은
  진행 중이다. 최초 실행의 pre-policy 중단과 미완료 graph829 재실행 비용을 별도로 보존했다.
- 최신 완료 전체 회귀는 **1,358 passed/2 skipped**이며 명령 sim-time pacing,
  owned-child 회수 및 초기 identity 준비 대기도 포함한다.
  수정 후 두 optical-loss 진단은 terminal hold/cleanup, nominal 비학습 진단은
  25.148s 착륙/8.177s cleanup을 확인했다. 앞선 cleanup 180s×2 실패 및 terminal hold
  미충족은 그대로 실패 기록으로 남는다.
  테스트 통과는 착륙 성공의 증거가 아니다.
- 원본 2D와 추가 대조한 episode 배치·rollout advantage·12→3초 학습 loss curriculum을
  명시적 옵션으로 반영했다. 물리·시각 분포·안전 감독기 차이는 여전히 남는다.
- 최신 v10 고정 로컬 40회/arm(2개 학습 seed)의 착륙은 vector/flat/graph **1/0/0**,
  unsafe는 **2/3/1**이다. 두 graph의 비영 residual은 확인했지만 제안 착륙/강건성은
  미통과다. checkpoint 선택에 이 test 결과를 사용하지 않았다.

현재 정합 범위를 다음처럼 구분한다. 동일 구조/서명만으로 동일 알고리즘이나 성능을
판정하지 않는다.

| 항목 | 현재 공간 경로 | 남은 차이/검증 |
|---|---|---|
| 제어·정보 경계 | ENU `(ax,ay,az)`, causal actor/critic, truth는 평가 전용 | 실제 PX4/로컬 reduced plant의 동역학 차이 |
| R-GAT 구조 | v9: x/y 각각 9×12, 동일 26 typed edges/shared encoder, 축별 4-group readout, raw bypass | 평면 slice의 전체 feature 수식 수치 일치 검사 통과; 공간 policy 성능 검증 중 |
| 하강 residual gate | v9: confidence×position/speed/attitude/rate evidence 곱, 두 축 공동 위험 | hard supervisor는 그대로; 두 평면/전체 3D 위험 결합은 명시적 축 확장 |
| 공통 hard supervisor | v10: uncertainty trust/latch, 응답지연·외력 범위 제동거리, bounded terminal corridor | 실제 nadir blind zone와 persistent force에 맞춘 명시적 수정; 원본과 bitwise 동일하지 않음 |
| 보상·학습 | 공통 fixed potential, readiness, raw-action PPO, episodic/rollout 정규화 옵션 | near-field 하강 목표·차원별 비용 정규화·GAE·학습/검증 예산 차이; 주요 scalar 보상 계수는 동일 |
| 외란 | v8에서 동일 seed 외력·토크·인계 충격을 로컬/Isaac에 적용 | gain/servo/카메라의 exact sim-to-sim 대응은 아님 |
| 궤적·시각 사건 | v9: 동일 시드 analytic CV–CA–CV, 실제 capture timestamp/own EKF 정합, 로컬 75ms delivery delay | synthetic optical과 실제 PnP/noise/dropout 분포 차이는 남음 |
| 운용 | armed/OFFBOARD gate, cold reset, bounded cleanup, 미검증 snapshot 배포 차단 | 검증한 횟수 밖의 장시간/다중 조건 안정성을 보장하지 않음 |
| 성능·통계 | frozen 정책 test와 unsafe를 함께 보고 | proposed learned landing, 다중 학습 seed, capacity-matched 성능 소거 실험 미충족 |

아래는 **시간순 감사 기록**이다. 초기 섹션의 “Isaac 통합 미완료/실행 금지”나
당시 테스트 수는 해당 시점의 상태이며, 현재 spatial 경로 전체를 설명하지 않는다.
기존 legacy privileged critic 경로는 여전히 분리돼 있고 reference로 재명명하지 않는다.
최신 실행 근거는 문서 끝의 추가 승인 구간과 `progress.json`에 기록한다.

## 최초 감사와 기준 출처

알고리즘 대조 기준: [upstream immutable commit](https://github.com/junsuk123/ugv_landing_2d_workspace/tree/83f10d6c5d90fc9350d67af1f729f83fc49009f4).
13:18 UTC 원격 재조회/isolated clone fetch에서 최신 HEAD
[`1bc69db080bf177d187a450ffc71072768d5dc1e`](https://github.com/junsuk123/ugv_landing_2d_workspace/tree/1bc69db080bf177d187a450ffc71072768d5dc1e)를 확인했다.
추가 두 커밋 `a233a88`/`1bc69db`의 8개 변경 파일은 온톨로지 explorer와 문서다.
`git diff --exit-code 83f10d6 origin/main -- src/algorithms src/simulations
src/orchestration/+landing2d/+config`는 exit 0이며 학습 기준을 중간에 교체하지 않는다.
새 MATLAB 시각화 UI를 Isaac 정책 알고리즘 누락으로 분류하지 않는다.
별도로 공간 `core.observation`과 원본 `contextGraph.m`을 다시 대조했다. 공간 그래프의
MotionTracking/VisibilityRecovery 노드는 현재 signed position/velocity/acceleration,
예측 bearing/margin을 재배치한 것으로, 원본의 correction/recoveryNeed/방향성 support
및 통일된 risk/signed/valid/confidence/uncertainty/trend/urgency 열을 그대로 확장한
것은 아니다. DescentEligibility와 공통 supervisor는 존재하지만 **9×12 모양·typed
edge·grouped readout 일치만으로 문맥 특징 수식 동등성을 주장할 수 없다**.
이는 센서/물리 차이와 별도로 남은 representation-prior 차이다. 현재 동결된 실험의
graph 해석을 사후에 바꾸지 않으며, 이를 수정하려면 새 feature registry/checkpoint
서명 및 재학습·paired 검증이 필요하다. 기존 2D oracle의 수치 fixture 통과와 구분한다.
최초 방정식 fixture는 `7efcc10`에 고정해 출처를 보존했다.
`contextSchema.m`, `contextGraph.m`, `encoderForward.m`, `agentInit.m`,
`relationPolicyResidual.m`, `pretrainCausalEncoder.m`, `computeReward.m`,
`updatePadTrack.m`, `normalizePacket.m`, `sampleEvents.m`, `selectionScoreV2.m`,
`primaryConfig.m` 및 실제 환경 `reset/step`을 대조했다.
문서의 실행 지시를 자동 수행하지 않고 사용자가 요청한 검토·수정 범위에만 적용했다.

## 사용자 지적 후 재검토: 관계형 기여 판정 정정

원격을 다시 fetch하여 `da105b7`(결과 문서 갱신), `83f10d6`(소스 계층 정리)을 확인했다.
최신 `paper_architecture_audit.csv`는 actor/critic readout norm
`0.0001665372 / 0.0006056019`, relation head norm `0.1516933 / 0.4792230`,
활성 여부 1을 기록한다. 최신 README의 test 평균 절대 행동 residual은
`[1.27e-5, 3.86e-5]`, 선택 scale은 `0.001`이다.
**현재 최종 R-GAT의 관계형 기여가 0이라는 앞선 판정은 틀렸으며 정정한다.**
`7efcc10`에도 이미 activation/guard 코드가 있었지만 최초 이식에서 누락했었다.
오래된 결과 설명과 구현 커버리지를 충분히 대조하지 못한 점도 함께 정정한다.

이번 추가 이식은 `ensureRelationalPath`, `guardRelationalCandidate`,
`relationalPathActive`, `scaleRelationalReadout`, `evaluateV2`를 대조했다.
현재 source는 `src/{algorithms,simulations,orchestration}`으로 나뉜다.

- raw actor/value/log-std 고정 후 실제 관계 전용 PPO; 인위적 epsilon 삽입 없음.
- validation만으로 내림차순 scale을 검사한다. 성공률 비하락,
  unsafe/abort/timeout 비증가, return/score 허용 하락 0.25,
  실제 평균 residual norm `[1e-6, 0.005]`, 양 head/readout 활성 조건을 적용한다.
- 모두 탈락하면 anchor를 유지한다. 승인 시 별도 `checkpoint_relational.pt`를 만들고
  `checkpoint_best.pt`는 보존한다. test는 repair·scale 선택에 쓰지 않는다.
- 새 profile은 `two_axis_reference_v28_active.yaml` / `two-axis-reference-v28-active-port2`.
  repair의 추가 25×2048 decisions를 별도 보고한다. MATLAB의 25×6 완결 에피소드와
  예산 단위가 다른 점은 여전히 남는다.

최신 test 100회 성공률은 vector/flat/graph **74/75/72%**로 유지됐다.
R-GAT 평균 return은 **17.873 → 17.930**. 작은 비영 경로는 입증됐지만,
우월성·실용적으로 큰 관계 효과·다중 seed 유의성은 별도 검증 대상이다.
Python의 짧은 warmup smoke가 `Wg=0`인 것은 그 smoke의 상태이며,
최신 upstream 최종 checkpoint 상태로 일반화하면 안 된다.
위 최신 성능 수치는 upstream 소스·공개 결과표를 교차 확인한 것이며,
이번에 MATLAB 최종 checkpoint의 100개 test를 독립 재실행한 결과는 아니다.

## Claude 기록과 실제 산출물

프로젝트의 최근 세 세션(`91447234…`, `dac5a184…`, `391972fc…`)에서 코드 변경과
최종 보고를 읽고 실제 JSON/CSV와 교차 확인했다. 사용자 대화 전체를 저장소로 복사하지 않았다.
수정 전 dirty/untracked 파일은 `/tmp/ugv-pre-refactor-NKFJbY/worktree-changes.tar.gz`에
보존했다. 기존 코드·실험 결과·체크포인트를 삭제하거나 과거 결과를 재작성하지 않았다.

- v2.5 수정: 접촉 허가 불가능, 액추에이터 권한 부족, 월드 원점 기준 envelope,
  회복 불가능한 abort latch, terminal 보상 순서 등의 수정은 코드에 존재했다.
- 그 뒤 sink-rate readiness, 할인 시간상수 350초, terminal table 변경 및
  curriculum 비활성화까지 진행됐다. 옛 `FINAL_REPORT.md`/`progress.json`의 일부
  “학습 미실행”, “pilot running”, -12/-40 같은 수치는 최신 상태가 아니었다.
- 10월 4일 `two_axis_gate_release`는 실제로 800 iterations × 3 arms를 완료했다.
  마지막 400개 학습 에피소드의 착륙은 모두 0%; timeout은 96.5/99/97.5%였다.
  **마지막 학습 창과 선택된 checkpoint의 성능을 혼동하면 안 된다.**

| 선택 checkpoint / validation 40회 | 착륙 | unsafe | timeout |
|---|---:|---:|---:|
| vector, iteration 300 | 17.5% | 72.5% | 2.5% |
| semantic-flat, iteration 500 | 50.0% | 45.0% | 5.0% |
| R-GAT, iteration 200 | 0.0% | 0.0% | 100.0% |

출처: `results/two_axis_gate_release/runs/*/summary.json`. 이는 선택에 반복 사용한
validation이며 새로운 test 결과가 아니다. 성공률을 우선하는 기존 사전식 선택이
안전하지 않은 후보를 선택하는 문제도 드러난다. 새 reference는 upstream의
`1000 success − 2500 unsafe − 10 timeout − 100 abort + mean return`을 쓴다.

Isaac pilot의 presentation 요약에는 학습 150회씩에서 안전 착륙 1회/3회가 기록돼 있다.
`evaluation/per_episode.csv`는 PN 11행, 각 학습 arm 10행으로 불완전하다.
이를 완료된 3-arm 평가나 graph 우월성 근거로 사용하지 않는다.

## 축 차이와 알고리즘 차이

현재 Isaac `PlanarFlightController`의 세 명령은 `[a_fwd,a_z,tilt]`다.
측방 이동과 heading은 제약한다. 이는 x/y/z 세 공간축으로의 확장이 아니다.
독립 tilt action 추가는 **정책의 자유도·물리 해석 변경**이므로 별도의 교란이다.
실제 3D 렌더링/접촉/PX4 inner-loop 차이와 이 변경을 분리해야 한다.

사용자는 후속 통합을 **실제 공간 3축 `(ax,ay,az)`**으로 선택했다.
`SpatialAccelerationController`는 고정 world ENU 좌표를 사용한다.
roll/pitch는 합력에서 유도하며, yaw는 own-state handover heading으로 고정한다.
속도 handover는 자신의 측정 속도로 시작하고, xy 합성 tilt·총 추력·축별 속도 제한을
동시에 적용한다. 새로운 `SpatialCommand` / `spatial_velocity_action`은
legacy tilt 명령과 호환되지 않으며 gateway는 이를 SITL에서만 허용한다.
ENU 속도·가속도는 yaw로 재회전하지 않고 NED로 한 번 변환한다.
이는 **제어 경계 구현**이지 공간 관측·graph·actor/critic·supervisor 통합 완료가 아니다.
`config/control/spatial_acceleration_v1.yaml`은 runnable experiments 밖에 두고,
live trainer가 실행하지 못하도록 차단했다.

| 항목 | 수정 전 경량 / 기존 Isaac | 새 reference |
|---|---|---|
| graph topology | 11 nodes·query readout / 별도 9-node graph | 9×12, 5 types, 17 semantic+9 self edges |
| raw 정보 경로 | graph 병목 / 별도 recurrent 구조 | semantic-flat와 동일 raw MLP + 독립 관계 residual |
| readout | PolicyNode/ValueNode | perception/tracking/vehicle/safety 4 groups |
| 하강 residual | 없음 | eligibility에 따른 추가 하강만 gate, 상승 보정 보존 |
| 초기화·적응 | arm 크기에 따라 RNG 소비 차이 | 동일 raw 초기값, zero-context, component RNG, staged adaptation |
| 사전학습 | 없음 | train-only same-time masked node reconstruction, 교사/보상/미래값 제외 |
| estimator | 10Hz Kalman / learned visual recurrent | 100Hz causal ABG, own-velocity prior, 중복 시각 무갱신 |
| 예측·정규화 | 현재 bearing·선형 scale | 0.5초 예측, smooth signed normalization, 별도 registry hash |
| 사건 분포 | 매회 short+sustained dropout | 50% clean/25% short/25% sustained, 독립 25% pitch event |
| 잡음 pairing | 보이지 않으면 RNG 미소비 | 매 센서 tick 동일 draw 수 |
| readiness | 수직속도 추가된 곱 gate | closing/sink-rate target, pitch-rate 포함 Gaussian risk |
| potential | 없음 | 공통 고정 potential, γΦ(next)−Φ(now), terminal Φ=0 |
| curriculum | disabled 또는 정체 가능한 승급 | train-only 낮은 고도→nominal, 80% budget까지 floor, replay 분리 |
| 선택·평가 | nominal 표기만 확인 / 불완전 실험 | 실제 nominal episode 후 선택, unsafe 가중, ineligible test 미평가 |
| 최종 비활성 관계 경로 | 최초 Python 이식에서 복구 절차 누락 | raw 고정 PPO + validation outcome/residual guard, 실패 시 anchor 유지 |
| 정직한 결과 보고 | seed 동일 여부 hardcoded true | 기록된 seed·arm 행렬·서명·eligibility를 비교 |

모델 파라미터 수는 vector **7,445**, flat **15,317**, graph **17,001**로 기준과 일치한다.
이 수치 일치가 학습 성능 일치를 뜻하지는 않는다. 사전학습의 추가 environment steps,
staged adaptation, graph 선택 margin도 비교 요인으로 기록한다.
원시 의미 경로를 보존하므로 `Wg=0`이면 flat-equivalent다. 추가한
`models_v28.relational_contribution`은 attention이 아니라 실제 actor/critic 출력 기여를 측정한다.

## 운용 안정성

읽기 전용 점검 시각: 2026-10-04 15:20 KST. 비행/학습 프로세스 및
UDP 8888, 14650–14657 listener 없음. **상태는 stopped이지 healthy 검증 완료가 아니다.**
16:13 KST 재점검에서도 관련 프로세스·포트가 없고 stopped임을 확인했다.
`logs/runs`에는 294개 run 디렉터리, 최신은 `run-20261003-2013`이었다.
디렉터리 수 자체를 장애 횟수로 해석하지 않는다.

실제 발견·수정:

1. `collect_episode_resilient`: 동료가 재시작했으면 반환 False를 받아 재시도 횟수가
   증가하지 않았다. 반복적인 공유 스택 장애가 무한 재시도로 이어질 수 있어,
   **버린 모든 시도를 예산에서 차감**하게 고쳤다. 부분 궤적은 여전히 PPO에 넣지 않는다.
2. 루트 실행이 자동 full 학습·프로세스 takeover를 수행했다. `reference`,
   `reference-smoke`, `status`, `isaac-legacy`를 분리했고 bare/help/status는 비행 상태를 바꾸지 않는다.
3. 접촉 판정은 보간 시각인데 반환 상태·reward dt는 physics-step 끝이었다.
   상태도 보간하고 접촉 이후 관측을 estimator에 넣지 않게 고쳤다.
4. PX4 호환 helper가 자기 pitch/가속도 scale을 상대 controller scale로 간주했다.
   실제 controller envelope를 필수 입력으로 받고 실현 불가능한 요청을 거절한다.
   반환값은 gateway wire packet이 아니라 **legacy controller 입력**임을 명확히 했다.
   PX4 velocity+acceleration feed-forward와 surrogate thrust/pitch는 같은 plant가 아니다.
5. output/config를 전파하고 기존 학습 디렉터리 덮어쓰기를 차단했다.
   다른 config·normalizer·graph checkpoint를 암묵적으로 재사용하지 않는다.
6. v1에서 비활성화된 curriculum의 unsafe 시작값 -32는 timeout -30보다 낮지만
   safe-abort -40보다 높다. 과거 검증은 timeout만 비교해 이 역전을 놓쳤다.
   이제 safe-abort보다도 낮아야 통과한다. 활성화한 테스트용 ramp도 -45로 수정했다.
7. 루트의 비어 있는 중복 `config/observation`, `python/ontology_rgat/contracts`
   디렉터리를 정리했다. 소스나 데이터 파일은 삭제하지 않았다.

10월 2일 20시간 무산출 run의 정확한 원인이 이 결함이라고 단정하지 않는다.
당시 기록의 keypoint 준비 지연 및 스택 재구축 반복과, 이번에 입증한 재시도 결함을 구분한다.
새 비행의 heartbeat·episode commit·재시작 원인/횟수/경과시간 측정이 필요하다.

## 아직 같은 시스템이라고 부를 수 없는 부분

- **Isaac integration 미완료**: `ppo/recurrent.py` critic은 여전히 `true_relative_state`를
  받는다. 기존 auxiliary estimator target, Shin/active-perception reward, keypoint 인지,
  3-channel controller, deck schedule는 reference와 다르다. 기존 ID의 재현성을
  보존하며 별도 경로로 남겼다. 새 정책이 Isaac에서 실행된다는 주장은 금지한다.
  새 spatial 명령 경계도 기존 position-backed gateway를 공유하며 그 위치 한계는
  deck 높이를 참조한다. 이를 causal 3D supervisor 구현으로 간주하지 않는다.
- **Python supervisor 차이**: v2.5 terminal-descent corridor/flare commit 및 abort
  감독기를 유지했다. upstream은 confidence 기반 inhibit와 상대 추종 recovery를 쓴다.
  이 감독기 차이는 축 확장과 무관하며 성공률에 영향을 줄 수 있는 남은 명시적 차이다.
  감독기를 조용히 완화하여 성공률을 맞추지 않는다.
- **분포/학습 절차 차이**: NumPy와 MATLAB RNG, 학습 seed 집합, 현재 고정-decision PPO와
  MATLAB episode-batch PPO, curriculum 승급 window/초기 고도 끝점·abort ramp가 다르다.
  비교할 때 manifest를 교환하고 동일 evaluation 시나리오를 고정해야 한다.
  PPO epoch 수·학습률·초기 log-std/하한·파라미터 초기화 분포도 동일화 전이다.
- **수치 검증 범위**: MATLAB 생성 fixture는 관측 정규화·graph feature·reward 방정식이다.
  encoder/backprop/전체 trajectory의 MATLAB bitwise parity는 아직 검증하지 않았다.
- **성과 미검증**: 이식한 버전의 긴 PPO 학습, multi-seed held-out 평가,
  Isaac 재비행, 통신 fault-injection soak test는 이번 작업에서 하지 않았다.

## 검증과 후속 완료 기준

리팩터링 직후 회귀: `python -m pytest -q tests/` → **1,134 passed, 2 skipped,
15 warnings**, 119.26초. 별도 집중 검사 48개 통과.
경고는 기존 matplotlib/pyparsing 사용 및 Axes3D import 문제이며,
실제 Isaac 렌더링·비행은 이 suite에서 검증하지 않는다.
`git diff --check`, 두 launcher의 `bash -n`도 통과했다.

새 active profile로 `./run.sh reference-smoke --steps 32 --ppo-minibatch` 실행:
세 arm 모두 finite reward/gradient 확인. 관계 활성화 테스트는
raw/log-std 불변, outcome 악화·zero/NaN residual 거절, test seed 차단,
실제 작은 PPO repair 및 synthetic 0.2초 task의 저장·재로딩을 포함한다.
이 synthetic task는 착륙 성능 평가가 아니다.
3축 테스트는 실제 gateway 수신·출력 메서드를 fake IO로 실행해 lateral 보존,
ENU/NED 변환, legacy 전환 초기화, hardware 거절을 확인한다. 실통신 시험은 아니다.

`tools/reference_v28_fixture.m`을 실제 MATLAB로 실행해 생성한 값을
`tests/fixtures/reference_v28_matlab.json`에 보관했다. 생성 코드, 기준 SHA, 원시 packet과
기대값을 함께 남겨 Python 구현의 자체 계산으로 만든 oracle이 되지 않게 했다.
`tests/test_reference_v28_alignment.py`는 모델 크기, 동일 초기 raw 경로, gradient,
하강 gate, paired sensor noise, 100Hz 추정, 접촉 시각, curriculum floor,
checkpoint signature, 재시도 상한, adapter scale, fairness, masked pretrain을 검증한다.

추가 실행 증거: `results/refactor_v28_smoke_20261004/`에는 3 arms × 2 iterations ×
32 decisions의 bounded trainer 결과가 있다(seed 101, 별도 train-only graph pretraining 포함).
세 `checkpoint_final.pt`를 서명 검증 후 다시 불러와 추론했다. 이 예산은 nominal에
도달하지 않아 selected checkpoint가 없는 것이 정상이며, 성능 비교 자료가 아니다.
R-GAT actor/critic readout norm과 출력 residual이 모두 0이고 semantic-flat과 action이
일치했다. 이는 warmup 계약 검증이며 R-GAT 성능 증거가 아니다.

### 추가 요청 재검증 (2026-10-04 16:24–16:29 KST)

구현 소스를 변경하지 않고 다시 검증했다. 판정은 **오프라인 구현 검증 통과,
3D Isaac 통합 및 학습 성능 검증 미완료**다.

- 전체 suite 재실행: **1,134 passed, 2 skipped, 15 warnings**, 141.16초.
  XML: `/tmp/ugv-validation-20m3MT/pytest.xml`. 두 skip은 현재 teacher가
  `privileged_relative_state_velocity_pd_v4`인 기존 분기에서 발생했다.
- 새 reference profile, 3 arms × 64 decisions + PPO minibatch smoke 통과.
  reward·gradient finite; 성능 비교용 실행이 아니다.
- 실제 MATLAB로 방정식 fixture를 다시 생성했다. 새 JSON과 저장 fixture는
  JSON 값 기준 동일하다(텍스트에는 정렬/공백 및 `-0`/`0` 표현 차이만 존재).
  해당 contextGraph/computeReward/normalizePacket 소스는 최초 SHA와 `83f10d6`
  사이에 이동 외 변경이 없음을 git diff로 확인했다.
- 3축 제어: 100개 무작위 envelope·yaw·dt 설정 × 100 actions = **10,000회**.
  축별 속도/가속도, 합성 tilt·추력, 속도 적분, protocol 검증 전부 통과.
  유도 roll/pitch에서 합력 재구성 최대 오차 `3.55e-15 m/s²`.
- 가짜 evaluator 없이 nominal 환경에서 seed 804의 random-initialized graph에
  관계 전용 PPO **64 decisions**를 적용하고 validation seed `[2000,2001]`로
  guard를 실행했다. scale `0.75` 승인, 평균 절대 raw residual
  `[0.00461098, 0.00154513]`, raw actor/value/log-std는 bitwise 보존됐다.
  두 episode는 anchor와 candidate 모두 **SAFE_ABORT**다. 이는 경로 활성화 및
  guard 연결 확인이며, 착륙 성공·성능 향상·upstream 재현 증거가 아니다.
- `bash -n`과 `git diff --check` 통과. 전후 runtime audit에서 비행 프로세스와
  gateway UDP listener 없음; 마지막 점검 16:29 KST, stopped.

미검증 범위는 실제 DDS/heartbeat/비행/접촉/화면, 장시간 fault soak,
공간 causal actor/critic·supervisor 통합, multi-seed 성능이다.
기존 Isaac critic의 `true_relative_state` 경로도 그대로 남아 있으므로,
위 검증 통과를 전체 시스템 causal 계약 충족이나 비행 준비 완료로 해석하면 안 된다.

완료 기준은 다음과 같다. 이 목록은 성능 달성의 약속이 아니라 미완료 작업이다.

1. 감독기/episode scheduler/시나리오 manifest 차이를 고정한 별도 parity 프로파일 마련.
2. Isaac에서 실제 관측만으로 **3D packet**을 만드는 backend 구현 및 privileged critic 차단.
   y축 인과 추정, 두 bearing/FOV 축, roll/pitch 동역학, xy 하강 gate, 3-action
   Gaussian·graph/raw 공통 관측과 서명, simulator truth 없는 supervisor를 함께 구현.
   기존 신경망에 2채널 padding만 붙여 이식 완료로 처리하지 않는다.
3. bounded Isaac smoke: 100Hz/10Hz timestamp, frame stale/dropout, 접촉·재시도 종료,
   nominal 체크포인트 load, proposed pair side-view를 실제 화면으로 확인.
4. 승인된 동일 학습 예산으로 다중 seed 학습. validation으로 선택한 checkpoint만
   한 번 고정한 test manifest에 평가. PN 결과는 물리 feasibility에만 사용.
5. full/no-relations/uniform-attention/relation-label-shuffle 및 parameter-matched
   MLP를 비교하고, 관계 residual이 실제 출력에 기여하는지 함께 보고.
6. upstream과 같은 성공률이 아니라 신뢰구간·unsafe/abort/timeout·제어 지연까지 비교.
   최신 upstream은 selected graph가 비영 활성화됐지만 single-seed test 100회 기준
   vector/flat/graph 74/75/72%로 R-GAT 우월성을 확인하지 못했다.

장시간 PPO, 실차 명령, push는 수행하지 않았다. 기존 side-view 설정/선택 로직은 유지했다.

### 전체 파이프라인 실행 검증 (2026-10-04 16:31–16:49 KST)

**최종 판정: 전체 시스템 통과 아님.** 기본 2D reference의 학습→선택→재로딩→평가→집계와
새 공간 제어의 실제 UDP/DDS 경계는 실행됐다. 자동 튜닝의 재현성, 기존 통합 검사,
설치된 gateway 동기화에는 문제가 있으며, causal 3D Isaac backend와 비행 검증은 미완료다.
이 절의 실행 중 production 소스·설치 gateway·기존 checkpoint는 변경하지 않았다.
검증 산출물은 `/tmp/ugv-whole-pipeline-5UdeLI/`에 있으며 임시 파일이므로 영구 보관은 아니다.
앞 절의 1,134-pass suite는 직전 재검증 결과이며, 이번 절에서는 CLI와 실통신을 추가 실행했다.

#### 1. 기본 reference: 실행 경로 통과, 착륙 성능 미달

루트 launcher에서 다음 실행을 수행했다. 모든 명령에 `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`을 사용했다.

```bash
./run.sh reference --stage train --output /tmp/ugv-whole-pipeline-5UdeLI/nominal_bounded \
  --seeds 809 --iterations 8 --decisions 512 --evaluation-every 8 \
  --evaluation-episodes 2 --workers 3 --threads 1 --progress-every 1
./run.sh reference --stage evaluate --output /tmp/ugv-whole-pipeline-5UdeLI/nominal_bounded \
  --seeds 809 --threads 1
./run.sh reference --stage aggregate --output /tmp/ugv-whole-pipeline-5UdeLI/nominal_bounded \
  --seeds 809 --threads 1
```

세 arm 모두 nominal checkpoint를 선택하고 실제 파일을 다시 불러와 validation 40회,
test 40회씩 완료했다. `aggregate.json`의 `fairness.passes=true`지만,
선행학습·관계 전용 PPO·validation scale 선택 등 명시된 차이 때문에
`single_factor_claim_allowed=false`다. 동일 representation-only 비교가 아니다.

| arm | test 착륙 | test SAFE_ABORT | test unsafe |
| --- | ---: | ---: | ---: |
| vector | 0/40 | 26/40 | 14/40 |
| semantic-flat | 0/40 | 38/40 | 2/40 |
| ontology-RGAT | 0/40 | 38/40 | 2/40 |

이 짧은 단일 training seed 실행은 성능 재현용 예산이 아니다. 그러나 실제 관측된
unsafe 결과가 있으므로 이 checkpoint를 비행 준비 완료로 취급해서도 안 된다.
각 arm 기본 학습은 4,096 decisions. graph에는 train-only pretraining 960,
자동 관계 활성화 PPO 51,200이 추가되어 총 56,256 decisions를 썼다.
선택 파일은 `checkpoint_relational.pt`, 승인 scale은 `0.02`이며 test에서 평균 절대
raw 관계 residual은 `[0.0002438495732, 0.0036881891915]`다.
관계 기여가 0이라는 이전 판단은 적용되지 않는다. 비영 기여가 우월성의 증거는 아니다.

#### 2. `--stage all`: 실행 완료와 acceptance 판정이 다름

별도 `synthetic_e2e.yaml`은 active profile을 상속하되 mission/reward duration 0.2초,
T1/T2/T3 각각 0.04–0.06초, curriculum off, pretraining 2×4 decisions/2 epochs,
activation 2×16 decisions로 제한했다. `primary=false`,
`publication_claim_allowed=false`이며 착륙 성능 시험으로 사용하지 않았다.

`--stage all --seeds 807 808 --iterations 2 --tune-iterations 1 --decisions 8
--evaluation-every 2 --evaluation-episodes 2 --workers 3 --threads 1`로
18 tuning jobs→6 training runs→6 checkpoint reload/evaluations→aggregate를 완료했다.
모든 checkpoint가 eligible이고 arm/seed matrix도 완전했으나,
vector entropy coefficient가 `0.02`, 다른 두 arm은 `0.005`여서
`fairness.identical_hyperparameters=false`, `fairness.passes=false`다.
이 상태에서도 CLI exit code는 0이었다. CI에서 exit code만 확인하면 acceptance 실패를 놓친다.

추가로 같은 18 tuning trials를 `--workers 1`로 재실행했다.
trial별 score·overrides·probe 결과는 전부 동일하고 모든 score가 정확히 2.0인데,
vector winner가 trial 1에서 trial 0으로 바뀌었다.
원인은 `python/run_two_axis_pipeline.py`의 `stage_tune`: `as_completed` 순서로
결과를 쌓고 `max(subset, key=score)`로 동점을 처리하여 첫 완료 작업이 선택된다.
안정적인 tie-break와 per-arm tuning/동일 hyperparameter 비교의 구분이 필요하다.
증거: `synthetic_all/tuning.json`, `synthetic_all/aggregate.json`,
`tuning_single_worker/tuning.json`.

#### 3. 실제 UDP/DDS 소스 경계: 제한된 통과

설치본을 덮어쓰지 않고 repository source를 `PYTHONPATH` 선두로 올려 ROS domain 187,
localhost UDP 24772/24773에서 실행했다. PX4/Isaac 대신 모의 odometry/deck/EKF telemetry를
발행하고 실제 `PX4Bridge`, UDP server, ROS executor, ROS subscriber를 사용했다.
기존 loopback fixture에 누락된 PX4 `xy_global/ref_lat/ref_lon/ref_alt`를 scratch harness에서
공급했다. 기존 gateway의 pad-track 필터 때문에 첫 packet이 단순 truth subtraction과
같다고 가정하지 않았다. 따라서 pad 추정 정확도/수렴 검증은 포함하지 않는다.

- hello/state, pad-frame 표기, legacy goto/attitude 출력 수신 통과.
- spatial controller→bridge→UDP→ROS TrajectorySetpoint 전달 통과.
  lateral 보존, velocity/acceleration ENU↔NED, held-yaw 변환을 수치 검사했다.
- 잘못된 spatial frame 거절, legacy 전환 시 spatial feedforward 제거 통과.
- telemetry 중단 후 `estimator_valid=false` 확인.
- heartbeat 40 samples, 평균 50.055 Hz, 최대 간격 0.02155초.
  짧은 loopback 결과일 뿐, Isaac 부하/장시간 fault soak의 지연 보장은 아니다.

증거: `ros_source_probe.py`, `ros_source_probe.log`. 이 실행은 학습된 3D 정책,
PX4 동역학, 실제 접촉, 화면, 설치된 runtime의 정상 동작을 증명하지 않는다.

#### 4. 실제로 실패한 기존 통합 검사와 배포 preflight

- `scripts/sync_gateway.sh --check` exit 1. ASCII runtime mirror의 `protocol.py`,
  `ros2_gateway.py`가 repository와 다르다. 일반 launcher에는 자동 동기화 경로가 있지만,
  이번 검증에서는 배포본을 변경하지 않았다. 소스 loopback 통과를 설치본 통과로 해석하지 않는다.
  증거: `gateway_sync_check.log`.
- `ONTOLOGY_RGAT_FAKE_PORT=24770 timeout 45s bash scripts/check_learner_protocol.sh`
  exit 1. 첫 reset에서 20 simulated seconds 뒤 `EntryResetError`.
  fake의 `goto`는 측정 위치를 target으로 두지만 truth에는 `[+1,-0.7,0]` offset을 유지한다.
  `PX4Bridge.entry_state`는 truth를 사용하므로 offset `1.22065556 m`가
  entry tolerance `0.90 m`보다 커서 영원히 handover되지 않는다.
  이는 fake fixture/진입 계약 불일치이며 실제 PX4 비행 고장으로 해석하지 않는다.
  full episode 검사에는 도달하지 못했다. 증거: `learner_protocol.log`.

새로운 3D profile은 여전히 `execution_status: control_boundary_only`다.
reference 2D actor/critic을 Isaac spatial backend에 연결한 것이 아니며,
기존 Isaac critic의 privileged truth 경로도 남아 있다.
proposed pair 측면 viewport 설정 `[0,-12,4.5]`와 pair 선택 코드가 존재함은 확인했으나
Isaac 화면을 열어 확인하지 않았으므로 가시성 완료 판정은 하지 않는다.

검사 시점 JSON 산출물 54개 strict JSON 파싱, launcher/script `bash -n`,
`git diff --check` 통과. 실행 후 소유한 probe가 모두 종료됐고 flight process와
기본/검사용 UDP listener가 남지 않았음을 확인했다(`runtime_after.json`).
우선순위는 튜닝 tie-break/acceptance gate와 fake fixture 수정, gateway 배포 동기화,
그 후 causal spatial backend 구현 및 bounded Isaac 비행·화면 검증이다.

### 2026-10-04 후속 구현·실제 Isaac 검증 (단기 기능 검증 완료, 성능 미완료)

이 절은 위의 **검증 당시 상태**를 대체하는 후속 기록이다. 기존 기록은 실패 재현
근거로 보존한다. 증거를 `results/validation_spatial_20261004_3pQvUf`에 보존했다.
scratch 원본은 `/tmp/ugv-pipeline-implementation-2N6RDn`이다.

- Tuning 동점은 trial 번호로 결정하고, 세 arm에 공통 pooled winner를 적용한다.
  실제 root CLI `reference --stage all`의 18 tuning→6 train→6 reload/eval→aggregate를
  다시 실행해 fairness=true를 확인했다 (`reference_fixed_cli/`). fairness=false는 exit 2다.
- Fake goto는 센서 바이어스를 보존하면서 물리 entry target을 만족시킨다.
  armed/landed 및 종료 확인 계약도 수정했고 UDP full episode 검사가 통과했다.
- ASCII runtime gateway는 원본 백업 후 동기화했다. `sync_gateway.sh --check`가 통과한다.
- 공간 정책은 별도 `spatial-causal-rgat/2`: 35-field packet, 9×12 graph, ENU ax/ay/az.
  actor/critic/graph/추정기/감독기는 allowlist 측정값만 받고 truth는 평가기에만 전달된다.
  카메라→ENU 방향 변환은 PnP+own IMU이며 완전한 simulator deck yaw를 사용하지 않는다.
- Isaac physics clock을 임무·추정·gateway 적분에 사용하고 PX4 message timestamp와 분리했다.
  실제 camera는 20Hz, 추정 propagation substep은 최대 10ms다. 물리 250Hz나
  정책 목표 10Hz를 정확한 측정 update/decision 간격으로 보고하지 않는다.
- 조인트를 비활성화해 시각 하위 트리를 숨기거나, 제거된 rigid body 간 joint를
  생성하던 문제를 수정했다. instance proxy를 포함해 실제 Ranger mesh 9개를 확인했다.
  차량·패드·기체 측면 카메라와 환경광을 실제 창 캡처로 확인했다.
  환경광은 perception에도 영향을 주므로 deployment 해시에 포함된다.
- 접촉 평가는 physics 직전 free-flight 상태를 latch한다. 충돌 반응으로 발생한
  각속도를 접근 각속도로 오인하지 않는다. 새 클라이언트가 seed/seq를 재사용해도
  고유 episode ID와 episode 시작 이후 timestamp가 일치해야 현재 접촉으로 인정한다.
  stale contact로 0-step SUCCESS가 나온 `feasibility-isaac-coast`는 무효로 표시했고
  성공 근거에서 제외했다. 회귀 테스트에 동일 seed/seq·다른 episode ID를 포함했다.
- 실제 camera/PX4를 사용하는 **격리된 인과적 진단 제어기**로 이동 패드에 2회 안전 착륙했다:
  `feasibility-isaac-fresh/`, `feasibility-isaac-repeat/`. 진단 행동은 PPO 학습·사전학습·
  보상 타깃으로 절대 사용하지 않는다. 이는 물리적 가능성과 반복 reset 검증이지 PPO 성과가 아니다.
- 단기 로컬 학습 3-arm 저장/reload, 실제 3-arm 짧은 비행과 land/disarm 종료는 확인했다.
  더 긴 단기 학습 정책에는 unsafe/abort가 있었으므로 전체 성능 acceptance는 아직 실패다.
  로컬 nominal preflight가 unsafe인 체크포인트의 Isaac 배포는 거절한다.

현재 공간 프로파일은 2D 성능 동등성 실험이 아니다. 패드 운동(로컬 CV–CA–CV vs
Isaac legacy segmented cruise), 20Hz 실제 ArUco와 100Hz 기준 센서, 90°/50° FOV,
observer gain·uncertainty 모델, 초기조건·물리 plant·domain randomization, 학습용
curriculum 세부와 0.5초 예측 정보가 아직 reference와 동일하지 않다. 이 차이들은
단순히 2축→3축으로 늘어서 생긴 것이라고 주장하지 않는다. 장시간 학습·성능 동등성
검증은 완료하지 않았으며 단기 TIMEOUT을 학습 착륙 성공으로 집계하지 않는다.

최종 단기 실행 `spatial-actual-final/`은 **실제 Isaac 수집 → PPO 갱신 → validation →
checkpoint 저장 → local reload/eval → Isaac reload/eval**의 3-arm 전체 경로를 완료했다.
arm별 2×8 PPO decisions, 임무 시간 0.8초이며 graph는 실제 상태 수집 사전학습 32스텝,
raw 고정 관계 전용 PPO 8스텝과 실제 validation 가드를 추가로 수행했다.
guard scale 1.0이 선택됐고 실제 test 평균 절대 residual은
`[0.0001133424, 0.0011913697, 0.0020323189]`였다. 추가 비용은 summary에 별도 기록했다.
실제 test는 모두 TASK_TIMEOUT, unsafe 관측 0이었다. 이 짧은 설정에서 착륙 성능을
주장할 수 없으므로 `isaac_acceptance.json: passes=false`, CLI exit 2를 그대로 유지한다.

격리된 진단 착륙은 26.036초/27.424초, 수평 오차 0.04597m/0.08875m,
접촉 직전 수직 속도 −0.23723m/s/−0.24530m/s였다. 동일 seed의 독립 클라이언트
재실행에서도 첫 스텝은 RUNNING이었고, 새 접촉 시점에서만 SUCCESS가 나왔다.
착륙 기준이나 원시 실패 기록을 이 결과에 맞춰 완화·삭제하지 않았다.

최신 full pytest는 **1163 passed, 2 skipped, 15 warnings, 350.78초**다.
`sync_gateway.sh --check`, fake UDP full-episode, strict JSON, `bash -n`,
`git diff --check`가 통과했다. 마지막 읽기 전용 상태는 PX4 `armed=false`,
`landed=true`, `estimator_valid=true`; 실제 runtime 설정 해시가 배포 manifest와 일치했다.
GUI는 제안 모델을 마지막에 실행한 물리 pair 0의 밝은 측면 뷰로 남겼다.
캡처는 `isaac-proposed-side-view.png`다. 장시간 PPO 성능 검증 승인은 별도로 요청했다.

### 승인된 추가 학습 및 장시간 경로 점검 (2026-10-04)

사용자가 최대 2시간 추가 PPO/Isaac 검증을 승인했다. 예산은 UTC 09:07:01–11:07:01,
산출물은 `results/spatial_approved_20261004_2PY7qZ`다. 아래에는 중간 발견과 최종
결과를 순서대로 보존한다. learned-performance acceptance는 충족하지 못했다.

추가로 드러난 축 확장 이외의 차이/결함:

- 공간 선택식의 unsafe 가중치가 -200이었다. reference의 -2500을 공용 함수로
  재사용하고 timeout도 -10으로 맞췄다. 기존 잘못된 선택 결과는 round1에 보존했다.
- 단순 budget floor만 있던 공간 curriculum에 reference scheduler의 성과 승급,
  easy/bridge replay, 속도 tolerance와 unsafe penalty ramp를 적용했다(schema 3).
  nominal completion만 eligible로 세며 replay 성과는 승급에 넣지 않는다.
- raw MLP의 기본 PyTorch 초기값은 random bias와 큰 출력층이었다. 원본 `mlpInit.m`의
  zero bias, Gaussian fan-in, output gain 0.1을 공간 경로에 명시적으로 적용했다.
  MATLAB/PyTorch 난수열의 bitwise 동일성을 주장하지 않는다.
- `reference-v28-scratch` 옵션은 공통 epoch 8/LR 5e-4/entropy .0025/log-std -1.1이다.
  후속 round7/8부터 원본의 critic-only 2-iteration value warmup도 적용했다.
  round6 이전에는 warmup이 없었다. 최저 log-std/RNG/episode batching 등까지
  모든 MATLAB 학습 세부가 동일한 것은 아니다.
- 쉬운 3D 단계에서도 각도/각속도 때문에 접촉 경험이 지나치게 희소했다.
  **별도 실험 요인**인 학습 전용 angular tolerance 2/4배→1배 ramp를 추가했다.
  검증/Isaac 기준은 5도·10도/초 그대로다. 완화 단계의 SUCCESS를 nominal 성과로
  세지 않는다. `angular_probe`의 checkpoint는 의도적으로 eligible=false다.
- 긴 Isaac 비행 후 두 setup 결함을 확인했다. (1) rate를 0으로 고정하면 착륙 후
  기울어진 자세가 복원되지 않아 preflight가 거절됐다. 공간 disarmed 상태에서만
  물리적 leveling torque를 적용하고 PX4가 armed이면 토크 보조를 금지했다.
  (2) armed AUTO.LAND를 prearm으로 오인해 호버 보조가 다시 켜져 종료를 막았다.
  armed/controlled 상태를 분리해 AUTO.LAND에는 setup force가 들어가지 않게 했다.
  pose teleport, PX4 preflight 검사 해제, 비행 중 교사 제어는 사용하지 않았다.
- 로컬 시각 모델과 packet bearing이 실제 roll/pitch/yaw를 무시했다. schema 4는
  공통 nadir mount의 optical-frame projection을 사용한다. 렌더 geometry와 수치 대조
  테스트를 추가했고 schema 3은 명시적 역사 경로로 보존했다. 체크포인트 재라벨링은 거절한다.
- 실제 카메라 mount는 접촉 전 사각 구간을 만든다. schema 4 readiness는 0.45m 아래에서
  기존 `.8 × touchdown_z_speed` 목표를 유지해 blind hover를 보상하지 않는다.
  이는 **공간 카메라용 보상 변경**이며 원본 2D 식의 완전 일치로 주장하지 않는다.
  진단 제어기 행동을 목표·label·replay로 사용하지 않는다.

PPO 결과가 나쁘더라도 strict nominal 기준과 acceptance gate를 낮추지 않는다.
진단 실패 원본/중단된 trial/완화된 curriculum 성과를 보존하고 최종 비교에서 분리한다.

장시간 운용 중 `round5-flat-isaac-validation`에서 OFFBOARD-loss failsafe가 발생했다.
이 실행은 불완전 인프라 실패로 보존하며 RL 성과에 합치지 않는다. cleanup은
armed failsafe/일시적 invalid estimator에도 bounded 대기를 계속하고, fresh valid
`landed=true, armed=false`를 모두 받아야 성공한다. 정상 RL step의 failsafe 거절은 유지한다.
실제 후속 상태 조회에서 해당 비행의 착륙·disarm을 확인했다.

PX4의 `Tools/msg/templates/ucdr`는 wire timestamp에 DDS session offset을 더하고
수신 시 뺀다. slow rendered SITL에서 wall clock이 physics보다 빨리 진행하면
offset 갱신 사이에 heartbeat가 만료될 수 있다. 공간 SITL gateway는 최신 PX4 wire
timestamp에 실제 physics 경과만 더하도록 수정했다. legacy/hardware는 변경하지
않고 COM_OF_LOSS_T, estimator/arming 검사를 완화하지 않았다. 이 변경의 긴 실제
비행 검증은 아래 round9/10에서 수행했다. 또 running world의 source SHA256을 broadcast하여 소스
수정 후 낡은 세계를 adopt하면 arming 전에 거절한다.

중간 회귀 검증은 1179 passed/2 skipped/15 warnings(862.50초), 이후 clock/cleanup/
source guard 중심 110 tests pass다. full suite는 추가 clock 수정 이전 시점이며,
최종 재검증 결과로 별도 갱신한다.

#### v5: 시각 상실 중 추정/수평 복구 결함

v4 진단 seed 14002에서는 표식 상실 후 가속도 추정을 계속 적분하여 실제 패드가
거의 횡이동하지 않는데 추정 pad vy가 커졌고, 수직 전용 abort 중에도 기체가 이를
추적해 횡방향 24m까지 이탈했다. 원본 2D의 1.5초 acceleration decay와 own vx
기반 abort braking이 공간 코드에 누락된 것으로 확인했다. 이는 축 수 증가 자체의
필연적 차이가 아니라 이식 누락이다.

별도 `spatial-causal-rgat/5`에서는 추정 가속도를 매 100Hz substep에서 감쇠하고,
abort 상태의 x/y 명령을 자체 EKF 속도 기반 제동으로 바꿨다. 정상 비행의 정책 명령,
truth 격리, 물리 착륙 한계, 시각 불확실성 게이트는 완화하지 않았다. v3/v4는
명시적 역사 경로로 유지하며 엄격한 loader가 v4를 v5로 조용히 받아들이지 않는다.

v5 실제 진단 재실행은 시각 재획득 후 295 decision에서 SUCCESS, fresh disarm 확인까지
완료했다(`v5-causal-feasibility-14002/`). 진단 제어기의 성공이며 PPO 교사/성과가 아니다.
v4 seed 14003의 UNAUTHORIZED_CONTACT 및 seed 14002 SAFE_ABORT도 보존했다.

`round9-v5`는 round8의 완료된 eligible **PPO** 가중치를 명시적으로 이전한 실험이다.
원본/대상 서명과 SHA256, optimizer reset, `eligibility_transferred=false`를 기록한다.
세 arm 모두 새 v5 nominal 에피소드 완료 후에만 선택 가능하다. 세 arm 공통 200×2048,
reference-v28-scratch preset, GAE lambda .99, curriculum 없음이다. GAE의 credit 기간을
늘린 별도 실험이며 원본 .95와 동일하다고 주장하지 않는다. test를 초기화/선택에 쓰지 않는다.

남은 전이 차이도 명시한다. `LocalBackend.advance`는 acceleration feed-forward +
velocity-error plant이고, 실제 gateway `_publish_velocity_setpoint`는 별도 적분된
position target도 PX4에 전달한다. 실제 cascade의 위치/속도/자세 루프 및 내부 기준값과
로컬 fixture는 동일 동역학이 아니다. 이 차이와 실제 PnP의 분기/가림/잡음, nominal
0.1초 대비 실제 decision 간격 차이를 단순 축 확장이나 unit-test 통과로 해소됐다고
주장하지 않는다. 또 2D의 명시적 0.5초 예측 특징 전체를 공간 packet에 이식한 상태는
아니다. 현재 v5 결과에서 2D 성능 동등성·sim-to-sim 전이 성공을 주장할 근거가 없다.

로컬/PX4 소스 대조의 구체적 차이는 다음과 같다. 최초 확인은 stock source 기본값
대조였다. 후속 실제 비행 ULog `10_48_28.ulg`의 initial parameters에서도 아래 PX4
값을 확인했고 해당 파라미터의 비행 중 변경은 없었다. 이는 기록된 비행의 값이며
live parameter service를 조회했다는 뜻은 아니다.

| 제어 항목 | 공간 local fixture | PX4 source (`mc_pos_control_params.c`, `PositionControl.cpp`) |
|---|---|---|
| 위치 오차 피드백 | 없음 | XY P=.95, Z P=1.0; gateway가 position target도 제공 |
| 속도 P | 모든 축 1.8 | XY=1.8, Z=4.0 |
| 속도 I / D | 없음 | XY I=.4/D=.2, Z I=2.0/D=0 |
| 자세 목표 | 요청 net acceleration에서 유도 | 위치/속도 PID까지 합한 acceleration에서 유도 |

따라서 단순 추가 학습만으로 전이를 보장할 수 없다. 다음 단계는 실행 파라미터·
causal 자체 상태·명령 추적으로 plant/관측 정합을 확인하고, 바뀐 계약은 별도 버전에서
재학습·재검증하는 것이다. 평가에서 본 test seed를 curriculum이나 선택에 넣으면 안 된다.

카메라 UI 검토에서는 고도 상승 시 UAV가 잘리는 문제를 확인했다. 관전자 전용 side
offset 8m, pair span 2.5m로 조정하고 차량 하단/고도 15m UAV까지 framing 테스트를
추가했다. 이 변경 전후 scientific YAML SHA256은 동일하다. 실제 화면에서 드론·패드·
차량이 함께 보이는 것을 `isaac-release-side-view.png`로 확인했다.

최신 release 회귀: **1190 passed, 2 skipped, 15 warnings, 491.48초**
(`pytest-v5-release.xml`). 두 skip은 legacy privileged teacher 전용 테스트이며,
새 causal spatial 경로의 필수 테스트를 건너뛴 것은 아니다. 별도 reference 3-arm
32-decision + PPO minibatch smoke, gateway source sync, compileall, strict JSON,
bash syntax 및 whitespace 검사도 통과했다.

`v5-vector-isaac-validation`의 실제 70초 설정 validation seeds 2000/2001은 각각
34.836초/22.004초 SAFE_ABORT였다. 모두 착륙·disarm 확인 후 종료했고 두 번째 reset도
완료했다. 이 둘은 운용 검증 근거이지 PPO 착륙 성공이 아니다. 비정상 인프라 종료 없이
`status.json: complete=true, exit_code=2`와 고정 checkpoint 사본을 남겼다.

#### 고정된 최종 3-arm 평가 (round9-v5)

세 arm 모두 v4 round8의 400×2048 PPO 이후 v5에서 추가 200×2048 PPO를 완료했다.
v5에서 완료한 nominal 학습 에피소드는 vector 423 / flat 405 / graph 424다.
학습 중 SUCCESS는 9 / 1 / 0이며 아래 validation/test 성과와 혼합하지 않는다.
graph에는 별도 관계 전용 PPO 2560 steps가 추가됐다. 선택 scale은 .1이고,
raw actor/value/log-std 보존 및 validation guard를 적용했다.

| 모델 | 로컬 test (9000,9001) | 실제 Isaac test (12000, 70초) | 실제 평균 return |
|---|---|---|---:|
| Vector | 2 TIMEOUT | TASK_TIMEOUT, 70.016초 | -9.0652 |
| Semantic-flat | 2 TIMEOUT | TASK_TIMEOUT, 70.032초 | -9.8356 |
| Ontology R-GAT | 2 TIMEOUT | TASK_TIMEOUT, 70.108초 | -9.6161 |

이 실제 matrix에서는 infra 오류/unsafe outcome이 관측되지 않았고 매 terminal의
착륙·disarm 확인이 완료됐다. 실제 decision 평균은 .10419/.10563/.10558초,
최댓값 .120/.116/.120초였다. R-GAT 실제 평균 절대 residual은
`[.0022849517, .0000146846, .0002692274]`로 비영이다. 착륙 우월성의 증거는 아니다.

`round9-v5/isaac_acceptance.json`은 `complete_matrix=true`,
`no_unsafe_outcomes=true`, **`landing_observed=false, passes=false`**다. CLI exit 2를
유지한다. 제안 모델을 마지막에 실행했고 종료 후 PX4 armed=false/landed=true/
estimator_valid=true 및 측면 뷰를 확인했다(`isaac-proposed-final-side-view.png`).

남은 승인 시간의 `round10-actual-graph-probe`는 제안 모델만의 **실제 PPO 업데이트
기능 점검**이다. 완료된 round9 비교를 수정하거나 이 추가 학습을 섞지 않는다.
원본 체크포인트를 보존하고 새 output에 학습/validation trace를 구분해 기록한다.

이 점검은 완료됐다. 실제 학습 701 decisions 및 12 PPO minibatches에서
`policy_updated=1`; 학습 에피소드 1회는 70.008초 TASK_TIMEOUT이었다. validation
2000/2001은 각각 70.036/70.060초 TASK_TIMEOUT, unsafe 0, landing 0이었다. 저장된
정책을 다시 읽은 local validation도 두 번 모두 TASK_TIMEOUT이었다.
원본 SHA256 `bf69a4c7…`는 유지됐고 새 정책 SHA256은 `7952452c…`다.
실제 tensor 대조에서 actor/critic 관계 encoder 및 residual이 바뀌었고,
raw actor/value와 log-std는 bitwise 동일했다. 관계 경로의 실제 갱신 증거이지
착륙 성능의 증거는 아니다. 이후 `round11-actual-graph-probe`는 이 점검 정책을
초기값으로 한 동일한 bounded 점검이며, frozen 3-arm 비교에는 합치지 않는다.

실제 로그에는 transient `Disarming denied! Not landed`, preflight drift/velocity
경고 및 round10의 DDS timesync reset 1회가 남아 있다. 이를 숨기거나 arming 거절로
바꿔 보고하지 않는다. 후속 reset과 70초 에피소드는 완료했고 fresh disarm도
확인했지만, 무경고 운용/장기 안정성까지 입증된 것은 아니다. 화면의 disarmed
hover는 setup support 상태이며 학습 정책의 착륙 성공 장면으로 해석하지 않는다.

#### 마지막 반복에서 발견한 arming 진입 게이트 결함

`round11-actual-graph-probe`는 validation 2000 진입 시 arming이 거절됐는데도
disarmed setup 보조 상태를 정상 entry로 받아들였다. 원인은 공용 `wait_at_entry`의
성공 조건에 `armed`가 빠져 있어, pose/speed/view가 맞으면 arming grace 이전에
반환하는 것이었다. 이는 운용 안정성 결함이며 축 수 차이가 아니다. 해당 실행을
SIGTERM으로 중단하고 `complete=false`로 보존했으며 학습/성능 비교에서 제외했다.

공용 entry gate에 armed 조건을 넣고, spatial reset/매 정책 step에서 armed 및
OFFBOARD(nav_state=14)를 확인하도록 수정했다. 같은 seed·현재 episode의 신선한
pre-impact contact만 terminal 평가용 예외를 허용한다. 낡은 contact로 비행 검사를
통과할 수 없다. 실패는 TIMEOUT/착륙 보상으로 변환하지 않고 인프라 오류로 중단한다.
`flight_armed`/`flight_nav_state`를 audit info에 추가했으며 actor/critic packet은
변경하지 않았다. 실제 disarmed 스택의 상태를 새 guard가 거절하는 것도 확인했다.

관련 114 tests가 통과했다. 전체 release 회귀는 아래 최종 기록을 따른다.
수정 후 정상 70초 비행의 재실행까지 완료했다고 주장하지 않는다. 현재 인스턴스의
preflight compass/accelerometer 경고 원인 및 반복 reset 안정성은 추가 점검 대상이다.

이전 결과도 ULog `vehicle_status`로 재확인했다. round9 vector/flat/graph 로그의
armed+OFFBOARD 시간은 각각 70.860/70.824/70.560초, round10 validation 두 비행은
71.028/71.436초다. round11 실패 validation에는 새 armed 비행 로그가 없었다.
이는 이전 실제 비행의 보조 확인이며 per-decision flight flag가 없는 과거 trace를
새 형식으로 꾸미지 않는다. 전체 acceptance는 계속 false이며 parity/안정 착륙을
주장하지 않는다. 최종 tensor/ULog/상태 근거는 `final_postflight_audit.json`에 모았다.

최종 gate 수정 포함 전체 회귀는 **1198 passed, 2 skipped, 15 warnings, 329.13초**다
(`pytest-final-flight-gate.xml`). 마지막 fresh 상태는 armed=false, landed=true,
estimator_valid=true였다. `isaac-final-disarmed-side-view.png`에서 차량·패드·UAV가
함께 보이는 것을 확인했다. 코드 snapshot `v5-final-code.tar.gz`의 SHA256은
`60cbcfb201b4c4af57be5af69607b7554b08f364bc80fc2476f50b980dc236a0`이다.
학습은 승인된 예산 안에 중지했으며 자동으로 추가 학습을 예약하지 않았다.
후속 작업에는 반복 reset/센서 health 복구, 최종 gate 적용 후 실제 재비행,
local–PX4 plant/관측 정합 및 새 계약의 재학습·평가가 필요하다.

### 추가 승인 구간 — 2026-10-04 11:13:29~15:13:29 UTC (진행 중)

산출물 루트는 `results/spatial_extension_20261004_TxwRik`다. 이전 frozen v5 결과는
덮어쓰지 않는다. `lifecycle-guard-v5`에서 마지막 armed/OFFBOARD gate를 적용한
validation 2000/2001의 정상 70초 비행·종료 확인을 완료했다. 둘 다 TASK_TIMEOUT,
unsafe 0이며 비행 중 `flight_armed`/`flight_nav_state`를 기록했다. 위의 “positive
rerun pending”은 이전 승인 종료 시점 기록이며 이 재검증으로 해소됐다.

| 후보 | 명시적인 차이 | 현재 증거 |
|---|---|---|
| v6 | velocity PID를 제거한 net ENU acceleration-only; PX4 HTE; reference 2차 자세/1차 추력 로컬 모델; 0.5초 예측 시야; log-std 하한 -2.5 | 집중 136 tests, local causal 진단 SUCCESS; 실제 seed 14002 70초 TIMEOUT |
| v7 | 절대 pad ABG(.20,.02,.0001), 1.5초 감쇠, 2/3/2 초기 std, q=1.5/noise=.02, 가속도 std·이전 명령 packet | 100/20 Hz 평면 불변 부분공간 수치 대조 포함 140 tests; local 진단 SUCCESS; 실제 진단/종료 오류 보존 |
| v8 | v7 + Isaac 공용 `sample_domain_randomization(seed)` 외력·토크·handover 속도/각속도 충격 및 자세 gain spread의 reduced-order 근사 | 집중 153 tests, 세 arm 공통 외란 학습 진행 중; 실제 재검증 진행 중 |

v6의 PX4 acceleration 내부 tilt가 g를 기준으로 정규화하는 성질을 로컬 PX4 소스로
확인하고, 수평 입력에 `g/(g+az)`를 적용했다. z에는 net acceleration을 그대로 보내며
중력을 이중으로 더하지 않는다. 모든 실제 후보에서 position/velocity setpoint는 NaN,
acceleration만 활성이다. 기존 v1 속도+가속도 경로와 wire ID를 구분한다.

사용자는 Isaac의 상속된 Table-II 외란을 끄는 대신 **로컬에도 동일하게 추가**하도록
선택했다. v8은 같은 시드의 body 외력 ±.75 N, body torque ±.004 Nm, 초기 속도
±1 m/s, 초기 각속도 ±10 deg/s를 적용한다. training-only difficulty에서만 0~1로
줄이며 nominal 1에서는 원래 draw 전체를 적용한다. 실제 물리 기준 Iris USD의 mass
1.5 kg, inertia `[.029125,.029125,.055225] kg m²`를 확인했다. 표본 일치는 검증했지만
PX4 servo/카메라 appearance/센서 지연까지 reduced model과 동일하다고 주장하지 않는다.
과거 v6/v7의 로컬에는 이 외란이 없었다는 차이를 숨기지 않는다.

진단에서 새 운용 결함 두 가지를 확인했다. (1) 패드 밖 지면 착륙의 첫 disarm 요청은
NAV_LAND로 변환되는데, 착지 후 일반 disarm을 다시 보내지 않아 종료 확인이 지연됐다.
fresh landed 후 일반 disarm 및 armed=false 확인을 추가했다. (2) 첫 arm→OFFBOARD
사이 setup 지지가 해제돼 entry 이전에 낙하할 수 있었다. reset별 “한 번이라도 policy
handover 했는가” latch로 pre-entry와 post-policy AUTO.LAND를 구분했다. 후자에서는
지지를 재활성화하지 않는다. 관련 entry/lifecycle 21 tests 통과 후 실제 재실행 중이다.

반복 reset에서는 High Accelerometer/Gyro Bias, heading estimate 경고와 arming 거절도
재현됐다. 검사 기준을 해제하지 않았다. 명시적인 `--reset-recoveries`는 소유한 스택의
정책 전 EntryResetError만 제한적으로 복구하며, 같은 seed·0 transitions와 재기동 비용을
별도 기록한다. 무제한 자동 재시도나 실패 episode 삭제가 아니다. 실제 반복 복구 성공은
완료 산출물이 있을 때만 추가한다. 현재 전체 learned-landing acceptance는 여전히 false다.

### 추가 재검증: 시각 좌표계와 rollout 단위

`pytest-v8.xml`: **1,224 passed / 2 skipped / 15 warnings, 556.25 s**.
동일 Isaac clock에 새로운 DDS 영상이 먼저 도착할 때 ABG가 0에 가까운 시간으로
innovation을 나눠 발산하던 결함을 수정했다. 같은 시각은 sample ID를 소비하지 않고
다음 증가한 clock에서 처리한다. 발산이 있었던 `v8-owned-lifecycle-entryfix`는
성능에서 제외했으며, `v8-owned-lifecycle-clockfix` 2회는 SAFE_ABORT/SAFE_ABORT로
종료·disarm이 확인됐다. 로컬의 단조 증가 시간 경로 수식은 변하지 않았다.

PnP의 잘못된 roll/pitch가 횡위치로 결합되는 문제에는 광학 상대벡터의 프레임 변환을
수정했다: `R_own_IMU × R_PnPᵀ × p_PnP`. 자신의 측정 자세와 카메라만 사용하고
pad truth는 사용하지 않는다. 독립적인 PnP/IMU 자세 차이 12° 검사, 위치 innovation,
기하·착륙·terminal coast 기준은 그대로다. 광학/제어 집중 테스트 33개 통과.
외란이 켜진 실제 `v8-imu-ray-feasibility` seed 14002/14003은 각각 **SUCCESS /
UNAUTHORIZED_CONTACT**였다. 후자는 기계적 접촉 한계는 만족했지만 coast 시간이
초과됐으므로 실패로 유지한다. 이는 분리된 causal 진단(+적분 gain .15)이며 PPO의
teacher, reward, checkpoint 또는 학습 성능으로 사용하지 않는다.

upstream `ppoTrain.m`과 추가 대조에서 완결 에피소드 단위 수집 및 전체 rollout에서
한 번 수행하는 advantage 정규화가 기존 Python 공간 반복과 달랐음을 확인했다.
기존 decision-budget 실행은 보존하고 `--episodes-per-iteration 6` 및
`--ppo-preset reference-v28-episodic` 옵션을 추가했다. 각 arm은 동일 에피소드 예산을
사용하되 길이에 따라 실제 decision 수는 달라지므로 누적 실측 steps를 별도 기록한다.
에피소드별 수집 상한, terminal/no-truncation, 기존 수집과 수치 일치, step 회계 및
optimizer 분기를 포함한 집중 79 tests가 통과했다. 새 `v8-episodic`은 seed 824,
1,000 iterations × 6 complete episodes, GAE .99, 세 arm 공통 외란·안전·보상이다.
기존 v8 seed 823과 seed/배치/정규화/예산이 달라 단일요인 ablation으로 주장하지 않는다.
KL early-stop, 20 Hz 카메라, reduced plant 등의 잔여 차이는 여전히 명시적으로 남는다.
소스 snapshot SHA256: `176b6e8cf4a128222e71c35d1a3036cf8289b5c7028b4a253532f2fc15430e1c`.

후속 전체 회귀 `pytest-optical-episodic.xml`: **1,240 passed / 2 skipped / 15 warnings,
511.10 s**. 원본 `primaryConfig.m`/`trainingEpisodeConfig.m`의 train-only visual-loss
12→3초 커리큘럼도 이전 공간 이식에서 빠져 있었다. 별도
`--curriculum-loss-timeout-start 12` 옵션을 추가했으며 nominal difficulty=1에서
cfg 객체·관측·reward/outcome이 기존과 동일한지 포함한 집중 54 tests가 통과했다.
`v8-reference-curriculum` seed 825에 세 arm 공통으로 적용하며, 앞선 실행을 덮어쓰지
않는다. snapshot SHA256: `d959cc27e6f02274742717364b9e5b8a4c420d5234204d525313611383b32e4a`.

legacy entry의 .9 m/.6 m/s 허용 범위는 spatial의 2~2.5 m 초기 분포에 비해 컸다.
direct 후보는 시드 draw를 유지한 채 인계 이전 수렴 기준을 .2 m/.15 m/s/1초로
강화했다. nominal 착륙 기준 완화가 아니며, 외란 충격은 여전히 그 이후 적용된다.
추가 실제 `v8-converged-entry` 진단 14002/14003은 SUCCESS / SAFE_ABORT,
두 번 모두 종료 확인. 반복 실험의 초기 상태·HTE 적응 차이 때문에 앞선 결과와
완전한 단일요인 ablation으로 간주하지 않는다. 또 legacy full-board FOV conditioning,
실제 PX4 HTE/servo, 실제 segmented trajectory와 로컬 reduced 모델의 차이는 남는다.

`v7-round1` 실제 3-arm × seed 12000은 모두 SAFE_ABORT, unsafe=0, 관계 residual
비영이지만 착륙 0이다. `v8-round1` 로컬 test 9000/9001은 세 arm 모두 TASK_TIMEOUT,
unsafe=0이다. 실제 v8 3-arm×2회 평가 중 vector는 SAFE_ABORT / UNSAFE_CONTACT를
기록했다. 완료 전 다른 arm 결과를 추정하지 않으며 learned acceptance는 false다.

actual acceptance v2는 같은 계약·paired seeds·중복 없는 완결 행렬·outcome/rate 일치,
unsafe 없음, **제안 R-GAT의 실제 착륙과 비영 관계 출력**을 요구한다. baseline만
착륙한 결과로 전체 통과하지 않는다. 작은 표본의 integration pass도 성능 동등성이나
통계적 우월성은 아니다. `tools/summarize_spatial_run.py`는 보존된 trace로 시간 가중
명령 RMS/포화/자세/추정오차, accepted-track loss/reacquisition, strict 접촉 기하,
return 평균/중앙값, episode-level Wilson 95% 구간을 보고한다. geometric FOV와
optical availability를 혼동하지 않으며, 재사용한 개발 test를 새 confirmatory test로
표시하지 않는다. 아직 여러 독립 학습 seed에 걸친 최종 성능 추론은 하지 않는다.

v8 실제 6회 완료 결과: vector SAFE_ABORT/UNSAFE_CONTACT, flat 및 R-GAT는
SAFE_ABORT/SAFE_ABORT. graph 평균 절대 residual은
`[.0038152011,.0021063322,.0001688403]`이며, 착륙 0으로 acceptance=false다.
graph 첫 seed 12000의 arming 거절은 소유 스택 복구 1회(96.664 s) 후 같은 시드로
재개됐다. 재시작 전에는 transition 0이며 실패 자체를 RL episode로 숨기지 않았다.
앞선 에피소드의 PX4 EKF/HTE 상태가 다음 에피소드에 남고 중간 복구가 발생하므로,
이 행렬은 **cold PX4 초기 상태가 엄밀히 paired된 최종 비교는 아니다**.

후속 `--fresh-stack-per-episode`는 소유 스택에만 허용하며 매회 confirmed stop 이후
예정된 새 프로세스로 시작한다. 이는 실패 재시도가 아니고 기존 결과를 버리지 않는다.
로그·시드·재시작 비용을 `episode_isolation.jsonl`에 보존하며 adopted stack이나 미확인
종료 상태에서는 거절한다. 인계 오류 복구 예산은 별도다. lifecycle 집중 60 tests와
read-only runtime identity 확인 이후 strict reset/measurement 검증 62 tests가 통과했다.
`v8-actual-ppo`는 seed 823의 동일 세 checkpoint에서 출발해 실제 Isaac에서만 arm당
4 complete episodes로 PPO fine-tuning, nominal actual validation 2회를 수행한다.
새 optimizer이며 exact resume가 아니다. extra relation repair는 0, teacher/BC는 없고,
실제 수집 step 수·초기 checkpoint SHA·raw/relational 단계는 산출물로 구분한다.

`pytest-isolated-release.xml`: **1,247 passed / 2 skipped / 15 warnings, 496.48 s**.
이 수집 이후의 read-only identity constructor 수정 및 실제 holdout seed 옵션은
별도 집중 테스트를 통과했으며 최종 전체 회귀에서 다시 포함한다. 실제 PPO의 첫
두 예정된 cold restart는 각각 87.154/85.139초에 완료됐고, 이전 세 학습 에피소드의
SAFE_ABORT와 confirmed stop을 모두 보존했다. PPO 정책 갱신은 3번째 반복에서
시작됐다(처음 두 반복은 preset의 critic-only warmup).

완료된 `v8-episodic`의 nominal local test는 세 arm 모두 SAFE_ABORT/SAFE_ABORT,
unsafe=0, landing=0이다. 완결 episode/advantage 정합만으로 착륙 문제가 해결됐다고
주장하지 않는다. `v8-episodic-descriptive.json`에 별도로 기록했다.
실제 frozen 후속 비교에는 명시적 `--isaac-seed-start`로 새 holdout을 지정할 수 있고,
checkpoint bytes SHA256·paired episode seeds·cold 상태 분리 옵션을 결과 행에 남긴다.

`v8-long-credit`는 같은 seed 823의 v8-round1 checkpoint를 출발점으로 한 별도
200×6 nominal complete-episode PPO fine-tuning이다. GAE λ=1로 지연된 보상의
credit-assignment 가설을 점검하며, 보상/외란/안전/센서 계약은 바꾸지 않는다.
새 optimizer, 세 arm 공통 설정이며 validation만으로 checkpoint를 선택한다.
실행 상한은 4,500초로 승인 종료 전이며, 미완료 progress를 배포하지 않는다.

현재 source까지 포함한 후속 전체 회귀 `pytest-final-source.xml`은
**1,248 passed/2 skipped/15 warnings, 515.22 s**다. 그 수집 뒤의 read-only status
프로세스 인식 수정은 집중 12 tests 통과 후 전체 재검증 중이다.
v8 selected checkpoint의 batch=1/CPU 1 thread/500회 추론 probe에서 파라미터 수는
vector/flat/graph 9,127/15,367/17,055, actor median은 .368/.385/.677 ms였다.
동시 학습·Isaac·pytest 부하 중 한 관측에서 잰 기술 통계이며 실시간 보장이나
시뮬레이션 전체 latency가 아니다(`v8-inference-probe.json`).

### Actual PPO 중단과 direct-acceleration 종료 결함

`v8-actual-ppo`는 vector validation 2001의 cleanup 실패로 중단됐다. 3개 iteration은
progress에 보존됐지만 4번째 update 후 validation 완료 전에 실패하여 summary/eligible
checkpoint는 없다. 기다리던 local 평가도 취소했고 partial 정책을 배포하지 않았다.
소유 프로세스 종료와 PX4 disarm 확인은 다른 사실이다: 전자는 완료, 후자는 실패다.
`v8-actual-ppo/status.json` 및 원본 ULog SHA256 `7557ef7e…`를 보존했다.

ULog에는 마지막 OFFBOARD mode timestamp 19.072 s, AUTO.LAND 전환 38.556 s가
남고, 그 사이 마지막 상승 acceleration이 계속 실행됐다. EKF local-origin 기준
최대 고도는 533.62 m까지 증가해 일반 LAND의 180초 wall budget 안에 내려오지 못했다.
해당 경로의 DDS clock rebase 경고도 보존한다. 모든 상승을 정책의 정상 제어 성능으로
보고하지 않는다. 수정 전 source archive SHA256은
`7760834745c4f05c112f4348d1ffc0951712664f6ea8502305f85566a614e831`다.

SITL direct 경로는 NAV_LAND 전에 자신의 EKF 위치로 임시 braking hold를 원자적으로
발행한다. OFFBOARD 재진입은 금지하고 PX4의 모드 변경/해제 후 hold를 지운다.
첫 실제 2초 종료 probe에서는 AUTO.LAND 후 임시 position setpoint가 남아 외부 명령이
자동 하강을 덮어쓰는 추가 결함을 확인했다. gateway의 mode-change callback과
explicit disable 양쪽에서 임시 hold를 해제하도록 추가 수정했다. 후속 실제
`v8-cleanup-release-probe`는 seed 2001의 2초 진단 종료 뒤 **21.651초에 착지·disarm을
확인**했다. 과제 판정은 TASK_TIMEOUT이며 PPO/nominal 성공이 아니다. 첫 probe의
180초 cleanup 실패와 재시도 기록을 그대로 보존한다.

`flight_terminal.jsonl`은 cleanup 전에 evaluator terminal 상태/시각/물리 상태/명령을
먼저 기록하고 cleanup 확인을 별도 append한다. 이제 cleanup 오류가 terminal 판정의
유일한 기록까지 지우지 않는다. 이 로그는 policy input이나 RL 성공 transition이 아니다.

### 종료 수정 후 전체 재검증

`pytest-atomic-cleanup-release.xml`: **1,263 passed / 2 skipped / 15 warnings,
350.02 s**. 자동 hold 해제, 실제 seed/공통 warmup 옵션, 미검증 update snapshot의
선택 차단을 포함한다. 진단 `checkpoint_last_update.pt`는 항상 eligible=false이고
optimizer 없이 보존하며, validation 실패 후 이를 정상 선택 결과로 승격하지 않는다.

`v8-actual-release`는 동일한 v8-round1 세 arm checkpoint로부터 새 optimizer,
warmup 0, arm당 실제 완결 학습 2회와 validation 2회를 수행한다. frozen reload 후
local test와 새 실제 seed 13000 평가까지 한 명령으로 연결했다. 모든 에피소드는
소유한 새 PX4/Isaac 프로세스에서 시작하며 외란·명목 기준을 유지한다. source archive
SHA256 `f272e1d0e6eaea41eaa1fa653a383bf7209a1f4bfcb8becb2e3219c541e3a5df`.
최초 두 학습 비행은 SAFE_ABORT이며 cleanup은 각각 38.135/23.906초에 확인됐다.
진행 중인 후속 결과나 전체 반복 안정성을 이 두 사례에서 추정하지 않는다.

`v8-long-credit`는 세 arm 모두 200×6 nominal 학습 에피소드를 완료했다.
GAE λ=1 후 local test 9000/9001에서 vector와 R-GAT는 TIMEOUT/TIMEOUT,
flat은 TIMEOUT/SAFE_ABORT다. unsafe=0이지만 착륙도 0이다. R-GAT는 validation-only
scale .05를 선택했고 test 평균 절대 residual은 `[.00295286,.00119613,.00101630]`다.
`v8-long-credit-descriptive.json`은 재사용 개발 test의 기술 통계이며 새로운
confirmatory 결과나 2D 성능 재현으로 보고하지 않는다.

고정 정책의 별도 local 시드 9100–9119 확대 평가도 완료했다. 원본 v8-round1은
vector 20 TIMEOUT, flat/graph 각각 16 TIMEOUT+4 SAFE_ABORT, 착륙/unsafe=0이다.
v8-long-credit는 vector 19 TIMEOUT+1 envelope 위반, flat 10 TIMEOUT+7 SAFE_ABORT+
1 SUCCESS+2 UNSAFE_CONTACT, graph 16 TIMEOUT+3 SAFE_ABORT+1 UNAUTHORIZED_CONTACT다.
flat seed 9113의 7.736초 착륙은 명목 접촉 기하·속도·각도·rate와 1.5초 coast를
만족한 실제 **로컬 PPO 평가 성공**이지만, graph 성공이나 Isaac 성공은 아니다.
모든 arm의 위험 비율을 포함하므로 장기 보상 후보를 안전한 배포 후보로 승격하지 않는다.
이 평가는 정책 bytes를 먼저 고정했고 추가 학습/선택에 사용하지 않았다.
`frozen-holdout-comparison.json`은 평균·중앙값 및 episode-level Wilson 구간을 담는다.
0/20의 95% 구간은 [0, .1611], 1/20은 [.00888, .23613]으로 단일 학습 seed만으로
우월성/강건성을 주장하지 않는다. 고정 평가 CLI 옵션을 포함한 최종 전체 회귀
`pytest-holdout-release.xml`: **1,272 passed / 2 skipped / 15 warnings, 360.31 s**.

14:19 UTC 원격 HEAD 재조회도 `1bc69db`로 동일했다. 12→3초 커리큘럼의 완결
에피소드가 초기 시간 추정보다 길어 로컬 실행의 6,000초 wrapper만 종료하고,
기존 승인 15:13:29 UTC보다 이른 **14:55 UTC 종료 감시**로 교체했다.
먼저 PID/start-ticks를 검사하는 guardian을 실행한 뒤 wrapper PID 3982910만
종료했으며 학습 PID 3982911과 1,000×6 계약은 중단/수정하지 않았다.
따라서 기존 wrapper의 exit 137을 학습 완료나 학습 실패로 해석하지 않는다.
완료 여부는 최종 summary로만 판단한다(`local_budget_reallocation.json`).

`v8-actual-release`의 세 arm은 실제 PPO/validation/저장을 모두 완료했다.
명목 완결 학습 2회씩, 실측 PPO steps는 vector/flat/graph 283/819/809이며
minibatch update 수는 16/26/31이다. 모든 arm에서 actor/critic 실제 가중치 변경과
초기 checkpoint SHA 일치를 검사했다. graph의 양 encoder/readout head도 변경됐다.
전체 실행의 tensor 차이는 `v8-actual-release-checkpoint-audit.json`에 보존했으며
이는 단계별 raw 고정의 실측 증명이나 착륙 성능 증명으로 과장하지 않는다.
actual validation은 세 arm 모두 SAFE_ABORT/SAFE_ABORT다. source와 달리 미검증
last-update snapshot만 남은 실행이 아니라, 검증된 eligible 파일을 실제로 저장했다.

추가 capacity 검사에서 semantic-flat의 hidden width를 52로 하면 actor/critic 합계
17,063개로 R-GAT 17,055개와 8개 차이다. 기본 flat width 48은 15,367개다.
`v8-capacity-audit.json`은 구조/개수만 확인한 결과이며 이 별도 MLP를 학습·평가하지
않았다. 따라서 capacity-matched 성능 소거 실험은 아직 남은 검증 항목이다.

### Actual release 최종 결과와 화면

`v8-actual-release`는 exit 2로 정상 판정을 마쳤다. `isaac_acceptance.json`은
complete_matrix/common_contract/paired_episode_seeds/outcome_rates_consistent/
no_unsafe_outcomes/proposed_relation_active가 true지만, proposed landing이 없어
**passes=false**다. 새 seed 13000의 vector/flat/graph는 모두 SAFE_ABORT였고
graph 평균 절대 residual은 `[.0123589486,.0002065754,.0005350276]`이다.
raw/residual이 실제 갱신됐다는 사실과 작은 upstream residual의 정확한 재현은 다르다.

전체 실제 15회(학습 6+validation 6+frozen test 3)의 terminal과 confirmed cleanup을
모두 기록했고, cleanup 실패 0, 최대 38.135초였다. 예정된 cold isolation은 13회,
누적 재시작 비용 1,125.880초다. 첫 부팅 두 번은 이 숫자에 포함하지 않는다.
새 run에서 정책 전 복구 재시도는 필요하지 않았다. `v8-actual-release-isaac-descriptive.json`
및 local 대응 보고서에 실제 trace 기반 기술 통계를 저장했다. 작은 횟수의 반복
성공은 장시간 운용 자격·2D 성능 동등성·강건 착륙 검증을 대신하지 않는다.

14:34:54 UTC 종료 후 소유 actual 스택/비행 UDP listener가 모두 없음을 확인했다.
그 다음 별도 viewer를 열어 측면 pair 추적 및 창 확대를 적용했다.
14:37 UTC read-only 상태는 armed=false, landed=true, estimator_valid=true다.
이 landed 상태는 **비무장 setup support**이며 학습 착륙 증거가 아니다.
`final-disarmed-side-view.png`, `final-disarmed-view-state.json`을 보존했고 창 제목에도
DISARMED/no policy running을 표시했다. 남은 로컬 학습은 이 화면에 비행 명령을 보내지 않는다.

### 이전 승인 구간의 로컬 최종 결과

`v8-reference-curriculum`은 세 arm 모두 **6,000 완결 에피소드/명목 804회**로
완료했다. vector/flat/graph PPO steps는 3,860,372/3,458,015/3,471,868이며 graph는
별도 pretraining 128/activation 5,120 steps를 썼다. guardian의 process_exited 기록은
14:51:25 UTC다. 고정 local seed 9100–9119 추가 평가에서 vector는 18 TIMEOUT+
2 SAFE_ABORT, flat/graph는 각각 16 TIMEOUT+4 SAFE_ABORT였다. 착륙/unsafe는 모두 0이다.
이는 학습 진행 중이 아니라 **완료했으나 제안 성능 미달**인 후보로 보존한다.

별도 causal feasibility 20회는 10 SUCCESS/6 UNSAFE_CONTACT/4 TIMEOUT이었다.
이 진단기는 PPO teacher/BC/보상 입력/초기 가중치가 아니며, 도달 가능한 사례와
위험 사례가 함께 존재한다는 검사일 뿐이다. 학습 성능이나 강건 controller 검증으로
전환하지 않는다. 이전 승인 마지막 tested-source archive는
`release-tested-source-20261004T1453.tar.gz`, SHA256
`d8901efbb882bdb0f84128e57d5ea8df8544db52d238d3e299ff43a8099b96d9`다.

## 10월 5일 추가 4시간: v9 문맥·관측·궤적 정합

사용자가 기존 종료 시각 이후 4시간을 추가 승인했다. 종료 한도는
**2026-10-04 19:13:29 UTC / 10월 5일 04:13:29 KST**이며, 실제 하드웨어/push는
범위 밖이다. 산출물은 `results/spatial_alignment_20261005_gzhmiY`다.
기존 v8 viewer는 소유 helper PID 4085913에만 SIGTERM을 보내 종료했다.
다른 Claude 세션/기존 결과는 중단·삭제하지 않았다.

v9는 기존 v3–v8/default v5를 재해석하지 않는 별도 서명이다.

- x/y 각각 upstream 9×12 의미 채널을 구성한다. 단순 원시 상태 재배치 대신
  confidence/uncertainty/trend/urgency/correction/recoveryNeed 수식을 사용한다.
  typed encoder를 두 평면에서 공유하고 각 4-group readout을 연결한다.
  flat/R-GAT raw input은 동일한 216개 feature이며 초기 raw 출력도 일치한다.
  vector는 같은 47-field causal packet을 받는다. 입력/파라미터 증가를 숨기지 않는다.
- 하강 residual의 evidence는 confidence×position/speed/attitude/rate risk의 곱이며
  x/y 공동 3D 위험을 반영한다. ENU thrust-axis tilt/미분을 사용해 yaw가 있어도
  body roll/pitch를 world 축으로 오인하지 않는다. hard supervisor/terminal 기준은 유지한다.
- 카메라 픽셀과 `rendering_time`을 **같은 acquisition callback**의 snapshot에서 읽는다.
  Isaac의 `get_rgba()`는 annotator의 더 새 이미지를 반환할 수 있으므로 그 이미지와
  예전 timestamp를 조합하지 않는다. gateway는 이미 수신한 own EKF history만 보관하고
  촬영 시각의 자체 위치/자세를 보간한다. deck truth는 이 경계를 넘지 않는다.
  own EKF 시각은 최신 simulation-clock callback과 대응하므로 하드웨어 시각 교정과
  동등하다고 주장하지 않는다. 미래/역순/반복 프레임은 track age를 새로 고치지 않는다.
- 로컬에는 75ms delivery queue와 20Hz capture를 적용했다. 예비 실제 진단의 검출
  frame age 중앙값 96ms, 10–90% 60–140ms에 근거한 근사이며 PnP 오차 분포의 완전한
  재현은 아니다. 첫 reset은 이미 관측된 pre-policy track으로 시작한다.
- 기존 actual `segmented_cruise_slow`는 세 속도 사이의 가감속이며 로컬 CV–CA–CV와
  다르므로, v9 actual에 `spatial_reference_cv_ca_cv`를 별도로 등록했다.
  로컬과 같은 seeded draw 및 analytic position/velocity를 사용한다. phase/switch time은
  policy 입력이 아니다. 학습 초기 draft `v9-round1`은 명목 완료가 모두 0일 때 중단했고
  가중치 이전 없이 새 output `v9-aligned`에서 다시 시작했다. 옛 archive/실패 기록은 보존했다.

평면 단면에서 reference graph의 **모든 feature 채널** 수치 일치, x/y 부호 보존,
공동 하강 gate, 유실 회복/불확실성, 세 arm의 실제 PPO update/save/reload,
지연 frame의 own-state 짝, 픽셀/timestamp 짝 및 7시드의 local/Isaac 궤적 일치를 검사했다.
궤적·capture fail-closed까지 포함한 전체 회귀는 **1,296 passed/2 skipped/15 warnings
(419.43s)**다(`pytest-v9-aligned.xml`). 이후 추가한 독립 terminal-hold 감사/acceptance
검사는 집중 76 passed(4.44s)이며, 기존 학습의 수치 계약/보상/선택 점수는 바꾸지 않는다.

예비 실제 optical probe 14004는 UNAUTHORIZED_CONTACT, atomic-frame probe 14005와
shared-scenario probe 14006은 SAFE_ABORT였다. 어느 것도 learned-policy 성공이 아니다.
14004의 96ms 수치는 atomic-pixel 수정 전 예비 측정으로도 명시한다.
`v9-quick-smoke --stage all`의 로컬 학습/재로딩은 완료했으나 실제 단계는 이미 소유 중인
진단 flight lock에 의해 거부됐다(exit 1). 다른 실행을 인계/중단하지 않았고 actual pass로
집계하지 않는다.

현재 새 PPO는 독립 seed 826/827 × 세 arm, 각 700×6 완결 episode 예산이며
GAE .99, reference episodic preset, train-only angular 3→1/loss 12→3 curriculum이다.
외력/토크/인계 충격은 유지한다. source archive `v9-aligned-source.tar.gz` SHA256:
`29435673fef756b6fbbcf93d7c885d7f24845f8de68e2a361b74911d7884410e`.
현재 결과는 **진행 중**이며 nominal/frozen test/실제 착륙이 확보되기 전까지 승격하지 않는다.

### SAFE_ABORT의 의미와 독립 운영 감사

2D `SafetySupervisor.abort_complete`는 정지/자세 조건을 계산하지만, 실제 환경의
종료 분기는 `abort_expired`에 의해 SAFE_ABORT를 기록한다. 공간 환경도 복구 시간 만료를
같은 이름으로 기록하므로, 이것만으로 안정 호버를 확인했다고 쓰면 안 된다.
기존 결과·reward·checkpoint 선택은 보존하고 actual acceptance를 별도 `/3`로 강화했다.
모든 episode의 confirmed stop, 그리고 SAFE_ABORT의 **종료 직전** 비접촉/높이≥.5m/
자체 수평 속도≤.2m/s/수직 속도≤.1m/s/합성 tilt≤5°를 확인해야 한다.
자체 속도를 pad 상대 속도와 혼동하지 않는다. 이 검사는 단일 snapshot이며 지속 안정성
증명은 아니다. 외란을 끄거나 touchdown 기준을 낮춘 변경도 아니다.

별도 `historical-terminal-hold-audit.json`에서 이전 v8 frozen test 3회는 이 snapshot
조건을 만족했다. 새 v9 causal probe 14005/14006은 각각 수평 .313/.358m/s가 남아
조건을 만족하지 않았다. 두 진단 비행은 후속 LAND/disarm cleanup은 확인됐지만
**안정된 abort hold가 검증됐다고 보고하지 않는다**. 과거 SAFE_ABORT label을 소급
SUCCESS나 unsafe로 바꾸지 않는다. `tools/audit_spatial_terminal_holds.py`로 재검사한다.

### 축 증가만으로 설명할 수 없는 잔여 차이

다음은 v9에서 해소됐다고 주장하지 않는 알고리즘/운영 차이다. 단위 테스트 통과와
기능 실행의 완결성을 2D 성능 재현으로 바꾸지 않는다.

| 항목 | 2D 기준 | 공간 v9의 잔여 차이 |
|---|---|---|
| 감독기 trust/latch | 위치/속도 std 한계, trustworthy 재관측 때 abort 해제 | 현재 공간 gate는 주로 age/geometry, uncertainty는 terminal coast에서 사용; 같은 latch 이식 아님 |
| 제동/복구 수직 명령 | 추력·자세 응답지연 정지거리, abort 시 수직 override | 공간 sink envelope 및 최소 상향 명령; actor의 더 큰 상향 명령을 유지할 수 있음 |
| terminal corridor | shrinking footprint + bounded commitment | 실제 nadir camera mount blind interval을 위한 별도 coast gate; 접촉 mesh 높이도 다름 |
| view reward | optical measurement bearing/validity | 현재 공간 reward는 추정 위치로 계산한 bearing/검출 여부; 측정 분포까지 같은 보상은 아님 |
| 초기/센서 분포 | pad 속도와 맞춘 centered start, reference dropout mixture | parked entry 뒤 pad 출발·인계 충격, 실제 PnP/자체 EKF; 로컬 ideal own-state/Gaussian 광학 근사 |
| plant/탐색 | 기준 thrust/attitude 식과 해당 sampler | PX4 HTE/servo/PhysX vs reduced plant; log-std 하한·KL early-stop·확장 난이도 차이 |
| 비교 설계 | upstream의 특정 seed/예산/활성 scale | 독립 두 seed 후보; capacity-matched flat 및 큰 표본 강건성 실험 미완료 |

공간 v9의 nominal 성능을 보고 나서 이 차이를 없앴다고 간주하지 않는다. 학습 중인
서명의 감독기/보상을 몰래 수정하지 않으며, 추가 수치 변경에는 별도 계약·재학습이 필요하다.

### 별도 v10: 감독기 이식 누락 수정

위 코드 대조로 발견한 감독기 및 view reward 차이를 별도 v10 서명에서 수정했다.
진행 중인 v9는 그대로 보존하며 그 checkpoint를 v10으로 이름만 바꾸거나 이전하지 않는다.

- 위치 std≤.75m, 속도 std≤1m/s, age≤.5s를 모두 요구하며, abort latch는
  trustworthy 재관측 전까지 해제하지 않는다.
- 추력 time constant .05s와 attitude response 2/10s, net az 한계에서 외력 norm의
  보수적 상계를 뺀 제동력을 사용해 정지거리를 계산한다. 낮은 고도에서는 shrinking
  nadir footprint·자세/속도 조건을 만족한 terminal corridor로 대체하며 1.5s commitment는
  uncertainty/range/speed/attitude 조건을 벗어나면 허용하지 않는다.
- 외력 때문에 속도 PD만으로는 정지 후 drift가 남으므로, latched own EKF 위치에 대한
  bounded PD backup(위치 gain 1.5, 속도 2.4)을 **모든 축**에 적용한다. 지속 하강 시
  수직 최대 제동을 유지한다. 이는 reference의 무외란 velocity backup과 의도적으로
  다르며, teacher/BC/nominal landing controller가 아니다. abort 종료 label은 유지하고
  별도 terminal snapshot 감사가 이를 검증한다.
- command tilt 20°/hard tilt 21°로 제한한다. 기존 touchdown 속도·tilt·rate 및 3s loss,
  8s recovery 기준과 외란은 완화하지 않는다. view reward는 추정 bearing이 아니라
  capture-aligned optical measurement bearing을 사용한다.

극값 ±.75N 힘과 ±.004Nm 토크 하에서 reduced-plant backup 정지, 잘못된 재관측으로
latch가 풀리지 않음, 정책 상향 명령 override, 도달 가능한 contact corridor, 동일 시드
3-arm PPO/save/reload/signature 거절 등을 포함해 집중 **91 passed(3.17s)**다.
이는 actual hold/landing 증거가 아니며 actual 진단 seed 14007을 별도로 실행한다.

`v10-evaluation-plan.json`을 먼저 저장하고 seed 828/829 × 세 arm을 각 500×6 완결
episode, graph 추가 pretraining128/activation5120 steps로 시작했다. v9와 예산이 달라
동일 예산 후보 우열로 해석하지 않는다. 고정 local9400–9419/actual16000–16001은
v9 시험에 사용하지 않은 새 범위이며 checkpoint 선택에는 쓰지 않는다.
소스 `v10-aligned-source.tar.gz` SHA256:
`467d16a53d9a0e4e588eb46298f52933aaf13a6d1da5504320d2783293e1cc3f`.
local timeout 90분은 소유 학습만 정리하며, 전체 작업 한도 19:13:29 UTC를 늘리지 않는다.

v10 전체 회귀는 **1,318 passed/2 skipped/15 warnings(535.65s)**,
`pytest-v10.xml`이다. 이후 진단 전용 optical blackout 옵션의 별도 검사 1 passed를
추가했으며 numerical PPO/environment 변경은 없다. 실제 causal 진단 seed14007은
27.680s에서 SUCCESS: 접촉 상대 위치 `[.002860,.000914,.069792]m`, 속도
`[.010919,-.008042,-.180137]m/s`다. 이어 41.399s에 cleanup 확인 후 소유 스택만 종료했다.
이는 **learned policy가 아니며** replay/teacher/보상 데이터로 사용하지 않는다.
seed14008의 2초 이후 optical blackout은 별도 고장 주입 진단이다.

고장 진단 seed14008은 blackout 이전 1.116s에 tilt hard envelope를 넘었고,
SAFETY_ENVELOPE_VIOLATION 및 22.309s cleanup 확인을 그대로 보존한다. 유실 후 hold
검증으로 집계하지 않는다. 같은 feasible seed14007을 사용한 다음 paired probe는
정책 transition 0개인 최종 인계 시점에 armed/OFFBOARD 조건을 만족하지 못했다.
이는 episode가 아니라 infrastructure 실패이며 cleanup은 확인됐다.

이 과정에서 두 운영 결함을 보강했다. (1) 공간 entry settle streak 전체에 OFFBOARD를
요구하고, 최종 pre-policy refusal은 명시적 EntryResetError로 기록해 승인된 동일-seed
owned retry만 소비한다. 정책 중 RuntimeError는 여전히 재시도하지 않는다.
(2) train과 test가 별도 live-stack context를 열어도 `reset_recovery.jsonl`의 소비량을
이어 받아 run-wide retry budget을 새로 충전하지 않는다. 손상된 journal은 fail-closed다.
해당 gateway/entry/lifecycle 집중 검사는 126 passed(10.51s)이며 최신 전체 재검증은
`pytest-v10-deployment.xml`로 별도 실행한다. 앞선 `pytest-v10-release`는 bridge 수정 전
import 상태였으므로 소유 pytest만 중단했고 partial log를 통과 결과로 쓰지 않는다.

완료된 6개 v10 summary와 최신 전체 회귀를 기다리는 task-local launcher를 두었다.
실제 PPO2+validation2+frozen test2씩/arm/학습 seed를 수행하고 19:04 UTC에 종료를
요청해 원래 승인 한도 내 cleanup·화면 확인 시간을 남긴다. 기존 진단 flight lock을
우회하지 않는다. 이전 v9 학습은 numerical/episode 예산 변경 없이 소유 worker nice를
10으로 낮춰 v10/실제 실행에 CPU를 우선 배분했다. 두 후보의 wall time을 속도 우열로
비교하지 않는다. 새 `tools/show_spatial_view.py`는 최종 확인 후 사용할 **비무장**
side-view 전용 도구이며 정책/reset/arm 명령을 보내지 않는다.

후속 `v10-blackout-final-entry-probe`는 12.964s에 SAFE_ABORT였지만 terminal hold가
아니었으며 cleanup 두 번 모두 180s 제한에 실패했다. finally가 소유 스택을 종료했으나
이를 착륙·안전 정지 성공으로 세지 않는다. PX4 ULog에는 OFFBOARD 위치 유지 setpoint는
있고 NAV_LAND 수신 기록은 없다. UDP gateway ACK는 PX4 command ACK가 아니므로
fresh armed/airborne/OFFBOARD 상태에서만 LAND를 1초 간격으로 재전송하도록 보강했다.
AUTO.LAND 진입 후 재전송하지 않으며 기존 deadline 및 공중 force-disarm 금지를 유지한다.
명령 유실/모드 전환/제한 시간 검사를 포함한 집중 회귀는 **93 passed(1.29s)**다.
같은 seed/blackout actual 재검증은 `v10-land-retry-probe`에 별도 보존한다.

보강 이전의 전체 회귀는 **1,326 passed/2 skipped/15 warnings(1,077.18s)**로 완료했다.
이 결과를 후속 LAND retry의 전체 회귀라고 부르지 않는다. 문제 확인 후 idle actual
launcher만 중단했고, 이전 v9의 소유 local parent도 SIGTERM으로 정리했다. 부분 결과와
완료된 vector seed826은 보존하지만 v9 6행 matrix 완료로 보고하지 않는다. 현재 v10
local 수치·예산은 바꾸지 않았다. 상세 운영 증거는 `cleanup-investigation.json`이다.

동일 seed14007/광학 blackout 재시험은 SAFE_ABORT, cleanup **21.175s 확인**으로
완료했다. terminal own vertical speed .11094m/s는 .1m/s hold 기준을 넘으므로 안정
정지 통과가 아니다. `v10-land-retry-hold-audit.json`에 별도 기록했다. 최대 상향 제동의
반복 전환이 진동에 미치는 영향은 `continuous-abort-diagnostic`으로 분리해 진단한다.
해당 결과는 명시적 supervisor override이며 v10 계약/학습 데이터/teacher로 사용하지 않는다.

### 실제 시간/물리 시간의 명령 발행률 불일치

continuous-PD 진단도 hold에 실패했으므로 그 수식은 production에 채택하지 않았다.
수동 ROS subscriber 및 PX4 ULog로 명령 경로를 분리해 확인했다. 이전 wall 50Hz
발행은 느린 실제 실행의 마지막 15 wall seconds(2.532 simulated seconds)에서
trajectory 750개, 즉 **296.21개/simulated second**였다. 동시에 OFFBOARD mode도
발행된다. 127개 정책 명령의 ROS 발행 대응은 모두 찾았으며 median 발행 지연은
15.45ms wall /3.19ms sim이었다. 따라서 정책→ROS 송신 지연만으로 설명되지 않는
PX4 내부 입력 지연과 lockstep 수신 경로의 과도한 유량을 구분해 조사했다.

공간 SITL의 연속 setpoint는 받은 physics clock에서 최대 50Hz로만 발행하도록
수정했다. wall timer의 deadman/maintenance, hardware/legacy cadence는 유지하고,
clock 정지 시 가짜 heartbeat나 catch-up burst를 만들지 않는다. prestream은 실제
발행 수만 센다. 0.05–1.0 real-time factor/정지/clock reset/legacy 분기 검사를 포함한
집중 회귀는 **103 passed(8.95s)**이며 gateway 실행 mirror도 재빌드·일치 확인했다.
그 이전 LAND-retry 전체 회귀는 **1,328 passed/2 skipped/15 warnings(1,213.52s)**다.
cadence를 포함한 새 전체 회귀는 `pytest-v10-sim-cadence.xml`로 별도 수행한다.

동일 seed14007·2초 뒤 optical blackout·외란 유지·원래 v10 supervisor에서 실제
발행률은 관측 구간 **49.74Hz(sim)**였다. 복구 진동이 가라앉았고 terminal hold 감사는
수평 속도 .007873m/s, |수직 속도| .000675m/s, tilt .03868rad로 모든 기준 통과다.
cleanup도 87.241s에 확인됐으며 이전 실패를 이 결과로 덮어쓰지 않는다.
한 terminal snapshot의 통과는 장시간 강건성이나 learned landing 증거가 아니다.
기록: `v10-sim-cadence-probe`, `v10-sim-cadence-hold-audit.json`, 두 passive transport
JSONL; 후속 다른 seed14008 진단을 별도로 실행한다. 학습 중 v10 local 수치/서명은
바꾸지 않았다. 변화는 실제 명령 전달 주기의 simulator-time 정합이다.

seed14008의 독립 재시험도 SAFE_ABORT 시 hold 기준 전부 통과: 수평 .04554m/s,
|수직| .01198m/s, tilt .07042rad, clearance 1.119m; cleanup 57.208s 확인이다.
각 trace의 마지막 연속 compliant 구간은 seed14007 **6.704s**, seed14008 **3.704s**다.
이는 그 두 짧은 외란/유실 진단의 관측치이며 장시간/다중 조건 보장은 아니다.

동일 500×6 학습 예산을 모두 끝내기 위해 task-local 90분 guard를 17:43 UTC까지
재배치했다. 전체 사용자 승인 종료 19:13:29 UTC는 그대로다. 수치/seed/episode 예산은
불변이며 `v10-local-guard-reschedule.json`에 정확한 소유 monitor/learner identity를
기록했다. pending 원래 timeout은 learner 종료 또는 새 cutoff에 재개하므로 외부
monitor exit124와 완결 summary의 존재를 구분한다. 다른 사용자 프로세스는 변경하지 않았다.

중간 eligible graph828을 고정한 **validation-only** trace에서 약 1m plateau가 관측됐다.
동일 bytes SHA `63e50129c4dc550a2767176d09ffc7cf9042621f53b2d9881f644471a4f775f6`에
commitment reset 높이만 1.0→1.1로 바꾼 분리 진단은 두 validation seed 모두 여전히
TASK_TIMEOUT이므로 production에 채택하지 않았다. activation/contact/외란 기준은
이 진단에서도 유지했고, 고정 test 선택이나 새 deployable checkpoint를 만들지 않았다.

cadence를 포함한 최신 전체 회귀는 **1,336 passed/2 skipped/15 warnings(1,466.60s)**로
완료했다(`pytest-v10-sim-cadence.xml`). 두 skip은 기존 legacy teacher 계약과 불일치하는
역사적 검사이며 actual 검증을 숨긴 결과가 아니다. 수정 후 외란을 유지한 nominal
비학습 진단 seed14007은 **25.148s SUCCESS**, 접촉 상대 위치
`[.001413,.002603,.069764]m`, 상대 속도 `[.011480,-.004178,-.249196]m/s`였고,
8.177s에 cleanup을 확인했다. 원시 기록은 `v10-paced-landing-probe`다.
이 결과는 PPO 착륙이나 학습 데이터로 사용하지 않는다.

`transport-timebase-audit.json`은 동일 seed14007의 전체 policy 구간을 재계산한다.
baseline은 **464.84**, paced는 **49.66** trajectory messages/simulated second다.
위의 296.21/49.74 값은 서로 다른 짧은 관측 구간이며 모순되는 동일 구간 집계가 아니다.
ULog own-velocity 정합으로 추정한 setpoint 지연은 약 .35s에서 해상도 수준으로 줄었다.
이는 시계 정합을 통한 추정이며 보정된 하드웨어 latency 측정이나 음의 실제 지연 주장이
아니다. 두 부팅의 초기 상태도 bitwise 동일하지 않다. 측정된 hold와 발행률을 증거로
사용하고, 이 진단으로 learned policy의 성능을 대체하지 않는다.

### v10 로컬 학습 및 동결 holdout 완료 — actual 검증 진행

17:41:50 UTC에 6개 완료 summary를 확인했다. 각 실행은 500 iterations/3,000 episodes,
nominal 404 episodes, actor update 498 iterations(처음 2회 value warmup)이다.
graph는 각 128 masked-pretraining + 5,120 관계 전용 train-only steps를 추가한다.
validation만으로 선택한 관계 scale은 seed828 **.01**, seed829 **.005**다.
원래 timeout monitor는 learner 종료를 확인한 뒤 재개했다. outer exit124는 기록하지만,
6개 완결 summary 및 후속 checkpoint reload의 완료와 구분한다.

고정 test 9400–9419 × 두 독립 학습 seed의 결과:

| 모델 | 에피소드 | SUCCESS | TASK_TIMEOUT | SAFE_ABORT | UNSAFE_CONTACT |
|---|---:|---:|---:|---:|---:|
| Vector | 40 | 1 | 24 | 13 | 2 |
| Semantic-flat | 40 | 0 | 30 | 7 | 3 |
| R-GAT | 40 | 0 | 31 | 8 | 1 |

28개 SAFE_ABORT terminal hold snapshot은 모두 기준을 만족했으나 장시간 안정성 증명이
아니다. 모든 시험 checkpoint는 선택된 원본과 bytes SHA256가 일치한다. 두 graph의 평균
절대 residual은 각각 `[3.5311e-4,1.8161e-4,2.8474e-5]`,
`[1.2138e-5,9.5654e-5,5.7770e-5]`로 0이 아니다. 출력 사용과 성능 우월성은 별개다.
원시 결과/서명/예산/hash 감사는 `v10-frozen-local-matrix.json`에 있으며 unsafe도 보존했다.
이 결과로 default 변경이나 실제 배포 승인을 하지 않는다.

17:41:52 UTC부터 같은 계약의 actual 2 PPO + 2 validation + 2 frozen test / 실행,
총 36회 계획을 `v10-actual-release`에서 실행한다. test는 미사용 16000/16001이며
각 에피소드에 cold owned stack을 쓴다. source archive SHA256:
`a68dc010bfdc82ab9d31db4d5287fc138c5e2e410bebdea256aa5c37fffae5ae`.
19:04 UTC에 종료를 요청하도록 제한했으며 전체 승인 19:13:29 UTC를 넘기지 않는다.
완료·착륙·/3 acceptance는 아직 주장하지 않는다.

### 남은 차이의 재분류 — 축 증가로 설명하지 않는 항목

동결된 upstream `83f10d6`의 `defaultRlConfig.m`, `primaryConfig.m`,
`defaultPlanarVisibilityConfig.m`, `computeReward.m`, `updatePadTrack.m`을 다시 대조했다.
현재 run의 설정을 바꾸지는 않았다. 아래 차이를 동일 알고리즘/동일 예산으로 숨기지 않는다.

| 항목 | 원본 2D 설정 | 이번 v10 실행 |
|---|---|---|
| scratch PPO budget | 2,500 iterations × 6 episodes | 500 × 6 / 학습 seed |
| GAE lambda | .95 | 명시적 .99 override |
| validation | 기본 100 episodes, 25 iterations마다 | 2 episodes, 50 iterations마다 |
| masked pretraining | Python reference profile 12×80 steps / 8 epochs | spatial 4×32 steps / 4 epochs |
| nominal 궤적 | 초기 속도 .5–2.5, 가속 .3–1.5, 구간 0.5–4s, 높이 4–8m | .3–.8, .1–.4, T1 .5–2s/T2 1–2s, 높이 2–2.5m |
| optical | 100Hz, 50° FOV, 지정 clean/짧은/긴 유실 분포 | 20Hz 카메라, 90°×약73.7°, 실제 PnP 및 local 75ms delay 근사 |
| terminal 근접 readiness | 높이에 따라 줄어드는 desiredVz | ≤.45m에서 −.8×touchdown speed 유지: offset camera 사각 구간의 명시적 수정 |
| abort | 원본 reference backup | persistent force용 자체 EKF position anchor; 모든 축 override |

보상의 goal/view/control/readiness/potential scalar 계수 **2/1/.25/8/2**와 terminal
**25/−12/−15/−40**, reference/discount time **70s**는 같다. 제어 비용은 두 축의 평균에서
세 축의 평균으로 확장했고, readiness의 XY norm·roll/pitch/rate 결합은 축 확장이다.
ABG 가속 innovation의 `.0001`은 원본 `.00005 × 2`와 같으므로 gain 누락으로 세지 않는다.
다만 100Hz와 20Hz의 실제 정보 유입률·영상 잡음 분포는 같지 않다.

따라서 현재 결과가 낮은 원인을 한 가지 결함이나 단순 학습량 부족으로 단정하지 않는다.
원본과 동일한 검증 표본 수·학습 예산, matched-capacity 재학습, 센서/plant 분포별 소거 실험은
여전히 별도 필요하다. 이번 고정 holdout으로 해당 요인을 튜닝하지 않았다.

### 실행 중 추가 진단의 해석 제한

`vertical-observation-timing-audit.json`은 이미 끝난 비학습 trace의 own EKF 수직 속도와
평지 pad의 truth 수직 상대 속도를 offline 대조한 것이다. seed14007의 paced nominal
RMSE는 .01293m/s, paced blackout은 .02189m/s였지만 seed14008의 transient 구간은
.19382m/s였다. 단일 지연 fitting은 일관되지 않았고 특히 seed14008은 탐색 경계 근처의
음수 shift가 선택됐다. 이를 센서 지연이나 미래정보라고 해석하지 않으며, 시각 정합·EKF
예측·전달 지연을 이 0.1초 trace만으로 분리할 수 없다. 보정 수식이나 actor 입력을
추가하지 않았다. 앞서 통과한 짧은 terminal hold도 전 구간 물리 안정성 보장은 아니다.

actual release 첫 vector828은 실제 228 transitions/10 PPO minibatches로 actor와 critic이
모두 갱신됐다(`v10-actual-checkpoints-1of6.json`, 아직 부분 감사). validation은
TIMEOUT/SAFE_ABORT였고 cleanup은 둘 다 확인했다. 후자의 own vertical speed
.13392m/s는 .1m/s 기준을 넘으므로 안정 정지 통과가 아니다. 후속 checkpoint·다른 모델·
frozen test의 결과와 섞지 않는다.

actual 실행 중 20초 bounded passive subscriber도 제어 포트나 arm/reset에 접근하지
않았다. `v10-actual-passive-validation.jsonl`에서 wall19.54s 동안 physics7.964s가
진행했고 302 trajectory messages를 수신했다. reader 시작/종료 및 timer 양자화 때문에
정확한 50Hz 측정으로 부르지 않지만, 과거 수백 개/sim-second flood와는 구분된다.
observer의 exit124/KeyboardInterrupt는 지정한 20초 제한 종료이며 flight 실패가 아니다.

18:03:32 UTC부터 같은 승인 한도 내 task-local resource monitor를 추가했다.
실제 main thread가 E-core 14에 배치된 것을 확인해, 초기화가 끝난 **소유 Isaac main
thread만** P-core CPU `[0,2,4,6,8,10]`으로 제한한다. 다른 기존 thread/사용자 process,
수치/시드/외란/episode budget은 변경하지 않는다. 정확한 parent/start-tick 검증과
변경 전후 affinity는 `v10-actual-main-affinity.jsonl`에 남긴다. monitor는 19:04 UTC
또는 소유 learner 종료에 끝나며 남은 thread의 원래 affinity를 복구한다. 이 실행의
wall time을 정책 속도 우월성이나 통제된 성능 비교의 근거로 사용하지 않는다.

### 하강 정체의 추가 validation-only 진단

최종 local graph828(`4c37dd2b590986dcc427f949d06fef4d1a38b1b9e7e360ca2d8cf5b195ee29bb`)
기존 validation 2000/2001을 고정했다. 다음 결과는 test 선택/재학습/실제 배포에 사용하지
않으며 실행 중인 actual v10은 변경하지 않았다.

- aligned corridor의 entry/clear 높이만 1.25m로 겹치고 incoming 하강 속도를 제한한
  `v10-corridor-overlap-diagnostic`도 TIMEOUT/TIMEOUT, mean return −9.7513이었다.
  구간 밖 1m 제동, contact·외란·hard attitude 기준은 유지했다. 채택하지 않았다.
- `v10-gate-attribution-validation`은 조건별 진단값만 기록했으며 원래 validation
  metrics와 **정확히 동일**함을 확인했다. t≥10s, 추정 높이 .2–1.5m의 각 600개
  상태에서 각속도 조건 실패는 **385/600, 357/600(64.2%,59.5%)**, 높이 실패는
  299/600,279/600이다. tilt 실패는 47/600,44/600이며 위치/상대속도/추적 신뢰
  검사 실패는 해당 구간에서 없었다. 각속도 median은 12.15/11.62°/s로 기준 10°/s를
  넘었다. 조건들은 중첩되므로 이를 독립적 인과 효과로 해석하지 않는다.
- τ=.25s(기존 thrust/attitude 응답 시간에서 유도)의 causal 명령 필터를 **감독기 앞에만**
  둔 `v10-command-filter-diagnostic`은 오히려 SAFE_ABORT/SAFE_ABORT(13.2/14.3s),
  mean return −15.2179로 악화됐다. truth/목표/teacher를 쓰지 않고 backup은 즉시
  override했지만 previous-action 의미가 달라졌으므로 원래 계약과도 동등하지 않다.
  production에 채택하지 않았다. 데이터/가중치는 원본 그대로 보존했다.

**원본 hard-supervisor와의 잔여 차이를 더 정확히 명시한다.** 원본
`updateDecisionContext.m`은 최근 accepted track과 `lastConfidence >= .25`를 사용한다.
공간 v10은 `.75m position std / 1m/s velocity std / .5s age`를 사용하며 accepted
confidence 메모리의 동일 threshold 조건을 구현한 것은 아니다. 원본 `safetySupervisor.m`
의 지면 기준 stoppingHeight와 달리 공간에는 1m guard·정렬된 terminal corridor가 있다.
이는 이미 기재한 보수적 공간 감독기 차이이지 단순 축 증가도, 완전한 수식 이식도 아니다.
동등성 실험에는 별도 version/signature에서 이 차이를 명시적으로 정합·검증해야 하며,
이번 동결 실험 중 즉석에서 제거하거나 10°/s contact 기준을 완화하지 않았다.

### 종료된 소유 프로세스의 zombie 대기 보강

실제 release의 cold isolation은 반복적으로 약 83–85 wall seconds였다. 정지 대기 중
소유 agent PID79410이 `Zs`였고, 부모 learner38914 아래에 회수되지 않은 것을 확인했다.
기존 `_kill_group`는 `killpg(pid,0)`만 검사하여 종료된 group leader도 살아 있는 것처럼
각 SIGINT/SIGTERM 대기 10초를 모두 소비할 수 있었다.

`ExternalStack.stop`가 자신이 생성한 Popen을 전달하고, group 검사 전에 `poll()`로
그 **소유 자식만 회수**하도록 수정했다. leader 회수 뒤에도 전체 group 생존 확인을
계속하므로 살아 있는 PX4 자식을 버리지 않는다. 기존 10+10초 escalation 제한을
유지하고, broad/current group 및 PID 불일치 owner는 신호 전 거부한다. SIGKILL 후
owned wait도 최대 1초로 제한했다. 실제 zombie/정상 SIGINT fixture, surviving-child
모형, ownership/시간 상한을 포함한 집중 회귀는 **33 passed(1.90s)**다.

현재 실행 중인 actual learner의 import를 hot-reload하지 않았다. 따라서 그 36회 계획은
기존 source archive의 lifecycle을 그대로 사용하고, 새 소스의 전체 회귀는
`pytest-v10-owned-reaping.xml`로 분리한다. PPO/물리/시드/안전 기준 변경은 없다.
최종 owned disarmed Isaac 기동·종료 때 이 lifecycle 변경을 별도로 확인할 예정이다.

이 변경을 포함한 전체 회귀는 **1,345 passed / 2 skipped / 15 warnings, 349.33s**로
완료됐다(`pytest-v10-owned-reaping.xml`). 두 skip은 과거 teacher 계약 전용 검사이며
Isaac 비행을 통과 처리하기 위한 skip이 아니다. 실제 disarmed 기동·종료 확인과는
별개 증거이며, 실행 중인 actual learner는 계속 기존 import를 사용한다.

### 실제 cold-start 식별 정보 준비 경합과 명시적 복구

원래 `v10-actual-release`는 21회 비행/21회 confirmed cleanup 후 graph829의 두 번째
학습 episode를 시작하기 전 `running Isaac profile is stale or unavailable`로 exit1했다.
완료된 5개 summary와 미완료 graph829의 1회 PPO update는 모두 보존했다. 그때의
raw identity 응답을 저장하지 않았으므로 최초 원인을 확정할 수는 없다. 다만 UDP 준비
직후 clock 구독 정보가 아직 없을 수 있는 단발 조회 경합이 코드에 존재했다. 실행 archive의
19개 `isaac_sim/*.py`와 현재 파일은 byte-for-byte 같고 source hash는
`eb07ad9fcde432670278f7ce1d59562345bdb9029252ba1888c8790016325e3f`로 같다.

`wait_runtime_identity`는 reset/arm 없이 최대 30초 동안 state만 조회한다. 이미 보고된
profile/source가 다르면 **즉시 거절**하며, 값이 없거나 clock이 아직 stale일 때만 기다린다.
단일 request에도 남은 deadline을 적용하고, 늦은 일치 응답으로 deadline을 늘리지 않는다.
기존 policy-time clock/estimator 기준은 그대로다. 집중 회귀 **79 passed, 2.61s**;
전체 회귀는 `pytest-v10-identity-readiness.xml`로 재실행한다.

`v10-disarmed-reaping-probe`는 initial/latest 모두 armed=false, landed=true,
estimator_valid=true였다. arm/reset/policy 없이 1분 표시 후 agent/Isaac/gateway 세 owned
leader를 모두 회수하고 handle을 닫았다. stop은 **1.75284s**였다. 이 증거는 새 lifecycle의
실제 disarmed 기동·종료 확인이지 learned landing이나 공중 안전 종료 검증이 아니다.

`v10-actual-recovered`는 원래 5개 완료 run을 hash 검사 후 그대로 복사하며,
graph829만 원래 local checkpoint에서 2회 PPO+2회 validation을 다시 수행한다.
미완료 optimizer state를 저장하지 않았으므로 **정확한 resume가 아니다**. 미완료 1회는
삭제/성공 처리하지 않고 추가 비용으로 기록한다. 기존 policy-time unsafe 1회도 유지한다.
계획된 frozen test는 원래 그대로 16000/16001이며 test를 보고 선택하지 않았다.
성공적으로 끝나면 원래 21회+복구 16회=37회 실비행(기존 계획 대비 추가 학습 비용 1회)다.
복구는 기존 19:04 UTC stop/19:13:29 UTC 승인 한도 안에만 실행한다.
소스 archive SHA256은 `081ccd3b1752f314ef9b0f6937e4f574a4adf50384732c697dfd8918a1b3b2b6`이다.
별도 `recovery-manifest.json`에 복사 hash·중단 사유·단계·원본 경로를 기록한다.

준비 대기를 포함한 전체 회귀는 **1,358 passed / 2 skipped / 15 warnings, 442.07s**로
완료됐다(`pytest-v10-identity-readiness.xml`). 이후 production 수치/동작 변경은 없다.

복구 학습/validation 4회가 모두 끝나면서 6개 actual training summary가 완성됐다.
`v10-actual-checkpoints-complete.json`은 동일한 최초 source hash/계약, 실제 nominal
2회 eligibility 재획득, actor·critic 수치 갱신을 6개 모두 검사했다. 완료 5개는 그대로이며
graph829의 새 PPO 데이터는 810 transitions다. 두 실제 validation은 SAFE_ABORT이며
cleanup은 확인됐지만 own vz `+.22369/−.18047m/s`로 terminal hold는 미통과다.
후속 고정 local reload 9000/9001, 총 12회는 TIMEOUT11/SAFE_ABORT1, 착륙0/unsafe0이다.
graph mean absolute relation output은 seed828 `[.005044,.009291,.000845]`,
seed829 `[.004929,.016668,.003075]`로 비영이다. 이 값은 감독기 이전 actor residual이며
실제 applied command에 동일한 크기로 전달되거나 우월성을 만든다는 주장이 아니다.
원래 계획한 actual frozen test 16000/16001은 별도 단계에서 시작했다.
