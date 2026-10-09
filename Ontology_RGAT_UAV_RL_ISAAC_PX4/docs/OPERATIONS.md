# 운영: 실행 프로파일, 산출물, 상태 확인

## 현재 진입점 (2026-10-05)

공간 기본값은 v5, 현재 별도 검증 후보는 v10이다. 실제 v8 PPO 배관은 15회 종료 확인까지
완료했으나 proposed learned landing이 없어 acceptance=false다. v10은 문맥/촬영 시각/
감독기 정합 후 6개 로컬 학습·120회 고정 시험을 완료하고 실제 PPO·재로딩 평가 중이다.
최신 판정은 [감사 보고서](refactor/REFERENCE_V28_AUDIT_KO.md)를
따른다. 두 v10 광학 유실 진단의 terminal hold/cleanup 통과를 PPO 착륙으로 세지 않는다.

앞선 실제 반복에서 arming 거절 상태의 setup hover가 entry geometry를 만족하는
결함을 수정했다. 공용 entry는 armed를, spatial reset/정책 step은 armed+OFFBOARD를
필수로 확인한다. 이를 충족하지 못한 실행은 인프라 실패로 제외하며 TIMEOUT이나
학습 착륙 성공으로 세지 않는다. 새 gate 적용 후 70초 비행 두 번과 종료 확인은
완료됐지만 모두 TIMEOUT이며, 광범위한 반복 reset 안정성은 아직 미검증이다.
Preflight compass/accelerometer 경고가 있는 인스턴스에서 검사
기준을 해제하지 말고, 소유권을 확인한 후 해당 스택의 복구·재검증을 수행한다.

루트에서 `./run.sh`는 도움말, `./run.sh status`는 읽기 전용 점검,
`./run.sh reference-smoke --ppo-minibatch`는 경량 수치 검증이다.
`./run.sh reference --stage ...`만 새 v2.8 프로파일을 사용한다.
이 프로파일은 아직 Isaac backend를 실행하지 않는다.

현재 reference는 `two_axis_reference_v28_active.yaml`이다. 비활성 nominal graph가
선택되면 관계 전용 PPO 25×2048 train decisions와 validation scale 검사 비용이
추가되며 summary에 분리 기록된다. 원본 best checkpoint는 보존한다.
`config/control/spatial_acceleration_v1.yaml`은 제어 경계 시험용이고 live 학습 실행용이 아니다.
새 공간 실행은 아래처럼 별도로 요청한다. 최신 schema/profile hash가 다른
체크포인트는 거절한다. 출력 디렉터리는 실험마다 새로 만들며 기존 파일을 덮어쓰지 않는다.

```bash
# 저장소 루트, 단기 배관 검사 (착륙 성능을 보장하지 않음)
./run.sh spatial --stage train --output /tmp/new-spatial-run --seeds 811 \
  --iterations 8 --decisions 512 --horizon 30
./run.sh spatial --stage evaluate --output /tmp/new-spatial-run --seeds 811 --horizon 30
# nominal 로컬 안전 preflight 후 실제 비행; 기존 스택은 명시적 --adopt-stack 필요
./run.sh spatial --stage isaac --output /tmp/new-spatial-run --seeds 811 --horizon 30
```

학습 기본값은 `local-spatial`이며 `--training-backend isaac`를 명시하면
사전학습 상태 수집·PPO rollout·validation·관계형 가드까지 실제 Isaac/PX4에서 실행한다.
이때 curriculum은 끄고 모든 episode를 해당 공간 프로파일의 difficulty 1에서 실행한다.
평가 backend는 `isaac-px4-spatial`이다. 초기화 전에 실행 중인 Isaac이 방송하는
설정 해시와 현재 프로파일을 비교하므로 YAML만 수정하고 스택을 재사용할 수 없다.
`isaac_acceptance.json`은 완전한 3-arm 행렬·unsafe=0·학습 착륙 관측을 요구하며,
불충족하면 exit 2다. 진단 제어기의 착륙이나 종료 후 AUTO.LAND는 학습 착륙이 아니다.
최신 실제 비행 증거와 남은 실패는 refactor audit에 별도로 기록한다.

`--contract-version 6/7/8/9/10`은 실험 후보이며 기본값은 여전히 5다. v6는 독립적인
가속도 OFFBOARD 계약, v7은 기준 ABG/불확실성/이전 명령 packet, v8은 공용 시드의
외력·토크·초기 속도/각속도 외란을 로컬에도 적용한다. v8의 mass/inertia는 Iris USD의
1.5 kg 및 `[.029125,.029125,.055225] kg m²`를 쓴다. 전체 PX4/렌더러 동등성을 뜻하지
않으며 기존 2D/무외란 결과와 동일 실험으로 합치지 않는다.

v9는 별도 `spatial-isaac-system-v9.yaml`을 사용한다. capture-time 필드가 없는
gateway를 허용하지 않으므로, 소유한 이전 stack이 종료됐는지 확인하고
`scripts/sync_gateway.sh`로 runtime mirror를 갱신한 후 새로 시작한다.
47-field packet과 두 개의 9×12 reference context, 동일 시드 CV–CA–CV 및 로컬
75ms 영상 지연이 계약에 포함된다. 외란/명목 contact 기준은 유지한다.
`--stage train`은 학습만, `--stage evaluate`는 local reload 평가만 실행한다.
`--stage all`에는 실제 Isaac도 포함되므로 다른 진단 flight와 동시에 시작하지 않는다.
flight lock 거절을 우회하거나 상대 실행을 인계하지 않는다.

Cold start에서 UDP listen은 Isaac clock 식별 정보 준비를 뜻하지 않는다. spatial backend는
reset/arm 전 read-only state를 최대 30초 기다리며, 이미 보고된 profile/source hash 불일치는
즉시 거절한다. 기다린 횟수·시간은 `runtime_identity_ready` event에 기록한다. 정보 미도착을
RL 실패나 성공으로 세지 말고 원본 로그/미완료 checkpoint를 보존한다. 종료 시 stack은
자신의 Popen 자식만 회수하며 leader 종료 후에도 살아 있는 process group을 확인한다.
미완료 checkpoint는 nominal eligibility가 없고 optimizer snapshot이 아니므로 임의로
`summary.json`을 만들거나 성공한 run처럼 재사용하지 않는다.

v10은 동일 v9 Isaac YAML을 사용하지만 다른 checkpoint 서명이다. 공통 supervisor의
trust/latch/제동/corridor 및 외란 중 own-position backup을 바꾸므로 v9 가중치를
자동 이전하지 않는다. 기존 touchdown 기준/외란을 유지하고 새 학습·재검증이 필요하다.
actual acceptance `/3`는 모든 episode의 stop 확인과 SAFE_ABORT의 별도 terminal
hold snapshot을 요구한다. timeout 이름, setup의 landed 신호, cleanup AUTO.LAND를
성공으로 해석하지 않는다. 단일 hold snapshot도 지속 안정성 증명은 아니다.

`tools/check_spatial_landing.py --optical-blackout-after-s 2`는 명시적 진단 전용
광학 입력 유실 시험이다. physics/own EKF/외란을 변경하지 않고 PPO teacher나
학습 데이터로 사용하지 않는다. 이런 고장 주입 결과를 nominal PPO 성능과 합치지 않는다.

소유한 스택에서만 `--reset-recoveries N`(0~5, 기본 0)을 명시할 수 있다. 정책 시작
전 entry 실패만 run-wide 예산을 소비해 같은 seed로 재시도한다. 각 실패/재기동 시간은
`reset_recovery.jsonl`에 기록하며 RL outcome으로 세지 않는다. `--adopt-stack`으로
빌린 프로세스는 재기동하지 않는다. 정책 중 오류는 이 복구로 숨기지 않는다.
지면 착지 후 자동 disarm이 꺼져 있어도, fresh landed 확인 뒤 일반 disarm을 보내고
실제 armed=false까지 확인한다. 공중 강제 disarm으로 종료 기준을 충족시키지 않는다.
재시도 소비량은 `reset_recovery.jsonl`에서 이어 받으므로 같은 output의 train/test
스택을 다시 열어도 한도가 복원되지 않는다. 공간 진입 settle 전 구간에 OFFBOARD를
요구한다. 최종 인계 guard의 실패도 정책 전 실패로만 기록하고, 정책 시작 후에는 이
경로로 재시도하지 않는다.

direct spatial 종료에서는 own-EKF 위치 제동을 먼저 시작하고 LAND를 요청한다.
UDP ACK는 PX4의 LAND 수신 확인이 아니다. fresh armed/airborne/OFFBOARD 상태일 때만
1초 간격으로 재요청하며, 모드 전환 후 멈추고 원래 cleanup deadline을 늘리지 않는다.
공간 SITL의 연속 setpoint는 entry/cleanup까지 **물리 시계 기준 최대 50Hz**다.
느린 렌더링에서 wall 50Hz를 그대로 보내면 PX4 lockstep 경로에 과도한 명령이 쌓인다.
wall deadman/maintenance 및 hardware/legacy 동작은 별개이며, 멈춘 물리 시계에 가짜
heartbeat를 생성하거나 밀린 ticks를 몰아서 보내지 않는다.

모든 실제 실행을 마친 뒤 화면만 열려면 프로젝트 디렉터리에서 아래 명령을 실행한다.
소유 stack/flight lock을 사용하고 타 실험을 인계하지 않으며, 비무장 telemetry만 조회한다.
setup의 landed=true는 착륙 실적이 아니다.

```bash
PYTHONPATH=python python tools/show_spatial_view.py --contract-version 10 \
  --minutes 60 --output /path/to/new-view
```

```bash
# 매우 짧은 실제 PPO 배관 검사: 학습 성능/착륙률 측정용이 아님
./run.sh spatial --stage all --training-backend isaac --output /tmp/new-isaac-smoke \
  --seeds 816 --iterations 2 --decisions 8 --horizon 0.8 \
  --activation-iterations 1 --activation-decisions 8 --isaac-episodes 1
```

0.8초 임무 시간으로 착륙 성공을 기대해서는 안 된다. 전체 실행이 정상이어도
학습 착륙 미관측이면 acceptance는 false/exit 2를 유지한다. 장시간 학습은 별도 승인 후 실행한다.

**아래는 기존 Isaac 운용 안내**다. 아래의 `./run.sh --...`는 이제
`./run.sh isaac-legacy --...`로 읽어야 하며, 인자 없는 실행이 full을 시작한다는
설명은 폐기됐다. 기존 스택 takeover도 기본 비활성화되었다. 의도적으로 이전
비행을 종료할 때만 `--takeover`를 명시한다.
[현재 안정성·미완료 항목](refactor/REFERENCE_V28_AUDIT_KO.md)을 확인한 뒤 실행한다.

[문서 안내](README.md) · [제안 알고리즘](ONTOLOGY_RGAT_STATE.md) ·
[평면 엔벨로프](PLANAR_ENVELOPE.md) · [아키텍처](ARCHITECTURE.md)

## 1. 실행 프로파일

```bash
./run.sh                 # 기본: 평면 3-arm 비교, full 예산
./run.sh --mode quick    # 전 구간 배관 검증 (수십 분)
```

인자 없는 `./run.sh`는 `config/experiments/planar_three_arm_comparison.yaml`을 그
파일이 선언한 예산으로, 기계가 재는 만큼의 페어에서 실행한다. 각 pair는 별도 PX4
instance, ROS namespace, gateway/learner UDP port, controller/reset state를 쓰고,
각 worker가 별도 model, rollout buffer, optimizer를 소유하며 GPU update만 lock으로
직렬화한다.

| 항목 | 값 |
|---|---:|
| 설정 | `config/experiments/planar_three_arm_comparison.yaml` |
| system 설정 | `config/shin2026-planar-system.yaml` |
| arm | 3 (학습 2 + 비학습 PN 유도 대조군 1) |
| 액션 | 평면 3채널 `[a_fwd, a_z, tilt]`, 세 arm 공통 |
| 덱 | `segmented_cruise_{slow,medium,fast}` 셋 |
| training / 학습 arm | `training.episodes_full` (`--mode full`) |
| evaluation / arm | 덱당 400 episodes (총 1200) |
| 결과 | `results/planar_ontology_graph_state/<mode>/` |

**학습 페어는 학습 arm에만 나뉜다**(4 ÷ 2 = 각 2 replica). 대조군은 학습 중 페어를
갖지 않고 평가에서만 참여하므로 `pair_count % len(pipelines) == 0` 불변식이 유지된다.

대체된 설계와 예비 프로파일도 그대로 남아 있다.

```bash
./run.sh --config config/experiments/three_arm_burst_comparison.yaml  # 은퇴한 보상항 방법
./run.sh --config config/experiments/two_pipeline_comparison.yaml     # 6-덱 2-arm
./run.sh --pipelines shin_se_fixed shin_se_onto_gat_state             # 소거 실험
./run.sh --seminar-fast   # publication_claim_allowed: false — 논문 결과로 보고 금지
```

> **full 전에 `--mode quick`을 먼저 돌릴 것.** quick은 수십 분에 keypoint → PN 유도
> 시연 → 행동복제 → PPO → checkpoint 선택 → 3-arm 평가 → 리포트 전 구간을 실행한다.
> 평면 설계에는 오프라인 보상 설계 단계가 없다 — 그래프 부호기는 PPO가 학습한다.
> **2026-09-23 개편 이후 quick 패스는 아직 수행되지 않았다.** 2026-09-22의 quick 패스는 full이었다면 하루치
> 연산 뒤에야 드러났을 결함 세 개를 잡았다 — 비학습 arm의 평가 크래시, 은퇴한 덱의
> 부활, 완주한 run의 manifest를 죽이는 NaN.

## 2. 실행 순서: 수집 단계 → 학습 단계

실행은 두 단계이고 순서는 고정이다. `--stage`가 이번 명령이 어느 쪽인지를 말한다.

```bash
./run.sh                   # 수집 → 학습 (한 프로세스, 기본값)
./run.sh --stage collect   # 데이터셋만 모으고 종료
./run.sh --stage train     # 모은 데이터로 학습·평가 (수집 비행 없음)
```

**1단계 — 수집.** 비행해서 재사용 가능한 데이터셋을 만드는 일만 한다.
공통 keypoint encoder 준비와 실기 카메라 측량 검증 → 교사 시연 비행과 행동 복제 →
FOV-risk rollout 수집. PPO는 한 에피소드도 돌지 않으므로 **모든 물리 페어가 수집에
쓰인다.** 산출물은 `results/<실험>/<mode>/`와 누적 datastore
(`results/datastore/collected.sqlite3`)에 남는다.

**기본 덱은 `straight_escape_burst_track` 하나다**(`COLLECTION_BASE_SCENARIO`).
20 m 직선 두 개를 6 m 반경 등속 반원으로 이은 닫힌 오벌을 등속으로 돌다가, 직선에서
급가속해 카메라 프레임을 벗어난다. 시험 대상 사건이 모든 에피소드에 포함되고, 덱은
32×15 m를 절대 벗어나지 않는다(300 에피소드 측정). 기하는 `pad.benchmark_track_straight_m`
/ `pad.benchmark_track_radius_m`로 조정한다. 프로파일이 덱을 선언하면 그쪽이 이긴다.

| 덱을 정하는 키 | 적용 대상 |
|---|---|
| `training.scenarios` | PPO + 모든 수집의 기본 |
| `behavior_cloning.scenarios` | 교사 시연 |
| `fov_risk_design.scenarios` | FOV-risk rollout |
| `rgat_design.scenarios` | semantic R-GAT rollout |
| `adaptive_reward_design.scenarios` | adaptive reward rollout |

per-design 키가 `training.scenarios`를 덮고, 아무것도 없으면 기본 덱이 쓰인다.

> **덱을 바꾸면 FOV 누적은 은퇴한다.** 덱은 궤적 분포 그 자체이므로
> `fov_risk_data_fingerprint`에 들어간다(2026-09-22 추가). 다른 덱에서 모은
> episode는 DB에 남아 감사 가능하되 새 지문에는 보이지 않는다. 바꾸기 전에
> 현재 누적량을 확인할 것.

**2단계 — 학습.** R-GAT validation-best 학습·동결 → **모든 arm의 PPO를 동시에 시작**
(완성된 같은 보상 설계를 상대로, 동일 예산·동일 seed) → held-out checkpoint 선택 →
paired/crossover 평가 → 표·그림 저장.

`--stage train`은 수집을 위해 비행하지 않는다. 필요한 데이터셋이 없으면 몇 시간짜리
비행을 조용히 시작하는 대신 `--stage collect`를 지목하는 오류로 멈춘다. 두 명령에는
**같은 `--config`와 `--system-config`를 준다.**

중단 후 같은 명령을 실행하면 호환되는 checkpoint와 완료된 평가 행을 재사용한다.
수집도 같다: datastore에 이미 있는 seed는 다시 날지 않는다.

> 왜 나눴는지는 [실험 설계 5.11](TWO_PIPELINE_COMPARISON.md#511-수집-단계와-학습-단계의-분리-2026-09-22)에
> 있다. 요약하면 예전 순서는 수집을 사실상 1페어로 묶었고, 페어 이중 점유를 허용했고,
> 제안 arm의 동결 readout을 비교 arm이 수백 에피소드 학습한 뒤에야 만들었다.

## 3. 짧은 개발 실행과 정적 검증

```bash
./run.sh --mode quick --train-episodes 1 --eval-episodes 1 \
  --rgat-data-episodes 2 --rgat-epochs 1 --headless --no-dashboard --no-rviz
```

```bash
python python/run_two_pipeline.py --help   # primary-only 옵션
./scripts/check_workspace.sh               # simulator 없는 전체 정적·단위 검사
```

실제 stack smoke test 전에는 UDP port 충돌을 피하도록 이전 실행을 종료한다.
`run.sh`는 flight lock으로 두 실행이 같은 stack을 동시에 채택하는 것을 막는다.

### 진행 상황을 보는 법

러너의 stdout은 파일로 redirect하면 **블록 버퍼링**된다. 로그가 수 분간 비어 있어도
스택은 정상 비행 중일 수 있다. 진행을 보려면:

```bash
PYTHONUNBUFFERED=1 setsid nohup ./run.sh </dev/null >> run.log 2>&1 &
```

이미 돌고 있는 run은 산출물을 직접 본다. teacher의 per-step trace는 매 스텝
flush되므로 실시간 신호다.

```
results/<experiment>/<mode>/training/teacher_trace_<fingerprint>_pair<N>.jsonl
results/<experiment>/<mode>/training/teacher_attempts_<fingerprint>.csv
results/<experiment>/<mode>/models/*/*_training.csv
results/<experiment>/<mode>/evaluation/per_episode.csv
/tmp/ontology_rgat_stack/isaac.log
```

### 게이트웨이를 고쳤다면

`ros2_ws/src/.../protocol.py` 등을 수정했으면 설치 미러를 갱신해야 한다. 하지
않으면 게이트웨이가 기동 중 종료되며 그 사실을 로그에 명시한다.

```bash
./scripts/sync_gateway.sh
```

## 4. 산출물

| 경로 | 내용 |
|---|---|
| `manifest.json` | resolved config, 두 spec, seed/budget, 독립 pair와 artifact provenance |
| `rgat/fov_risk_rollouts.npz` | episode ID를 포함한 미래 FOV 비가용 dataset |
| `rgat/fov_risk_model.pt` | 동결 validation-best R-GAT과 checksum |
| `rgat/fov_risk_training_history.csv` | epoch별 train/validation huber·contract |
| `models/<pipeline>/` | 독립 PPO latest/best/selected checkpoint |
| `evaluation/per_episode.csv` | paired/crossover 원자료 |
| `evaluation/paired_summary.csv` | 시나리오별 계층 bootstrap 요약 |
| `tables/primary_comparison.*` | 두 pipeline 주 비교 |

Dashboard는 두 agent를 나란히 표시하며 공통 `active_perception`과 proposed-only
`ontology_fov_reward`를 별도 series로 그린다.

## 5. 실행 중 상태 확인

브라우저 없이 진행 상황을 보려면 대시보드 API를 그대로 읽는 읽기 전용 도구를 쓴다.
이미 비행 중인 run에도 붙을 수 있고 run에 아무것도 쓰지 않는다.

```bash
tools/watch_run.py              # 스냅샷 1회 (127.0.0.1:8770)
tools/watch_run.py --follow     # 중단할 때까지 갱신
```

stage/phase, 두 arm의 PPO episode와 최근 50회 성공률, pair별 상태, FOV 데이터 수집·
readout 학습·동결 모델 검증 진행을 한 화면에 출력한다.

## 6. 부록: 장애 대응

실험 설계와 무관한 환경 문제만 다룬다.

### 제어가 느리고 덱이 도망가는 것처럼 보일 때

수평 오차가 스텝당 1.5–2.4 m씩 뛰거나, 기체가 0.1초에 2.75 m 상승한 것처럼 보이거나,
teacher가 패드에 닿았다가 곧바로 멀어진다면 **제어 대역폭을 먼저 재라.** 물리가
아니라 사라진 시간이다.

```bash
python3 - <<'EOF'
import json, glob
from collections import defaultdict
eps=defaultdict(list)
for f in glob.glob("results/*/*/training/teacher_trace_*.jsonl"):
    for line in open(f):
        r=json.loads(line); eps[(f, r['seed'])].append(r)
for k,v in sorted(eps.items())[-6:]:
    v.sort(key=lambda r: r['step'])
    px4=(v[-1]['px4_time_us']-v[0]['px4_time_us'])/1e6
    wall=v[-1]['wall']-v[0]['wall']; n=len(v)-1
    print(f"{k[1]} steps={len(v):4d} px4/step={px4/n:.3f} wall/step={wall/n:.3f} "
          f"sim:wall={px4/wall:.2f} eff={n/px4:.2f} Hz")
EOF
```

`px4/step`이 0.104이면 정상(약 9.6 Hz)이고, 0.7–0.95이면 열화(약 1.2 Hz)다. 후자면
Isaac이 실시간에 가깝게 자유 주행 중이라는 뜻이며, **teacher 게인이나 하강 임계를
만지기 전에 이것부터 해결하라** — 그 상태의 측정값은 대역폭이 통제되지 않은 값이다.

**`isaac.max_sim_speed_ratio`를 켜지 말 것.** 구현되어 있지만 Isaac의
`world.current_time`을 제어하는데 학습기가 읽는 것은 PX4의 `px4_time_us`이고, 두
시계는 8.6배 어긋난다. 켜면 Isaac이 잠든 만큼 PX4가 CPU를 더 얻어 제어율이 오히려
떨어진다(1.34 → 1.1 Hz).

현재로서는 **고칠 수 있는 설정이 없다.** 근거와 실제 해법은
[3-arm 비교](THREE_ARM_BURST_COMPARISON.md) §5.4–5.5.

### 진입 호버가 계속 실패할 때

`wait_at_entry`의 실패 메시지가 무엇이 게이트를 막았는지 직접 말한다.

* `PX4 refused to arm ... command 400 result 1` — 기체가 뜨지 못한 것이다. 위치·시야는
  증상일 뿐이다. PX4 SITL이 긴 세션에서 열화되면 arming을 거부하며, 이때 필요한 것은
  더 긴 대기가 아니라 새 시뮬레이터다.
* `offset was out of tolerance on N/N samples`이고 속도가 0이면 역시 기체가 움직이지
  않은 것이다.
* `view was out of tolerance`이고 위치·속도는 통과했다면 갑판이 화면 밖이다. 게이트는
  같은 seeded setpoint를 `entry_view_retries`회까지 다시 조준한다.

메시지 앞부분의 예산은 어느 시계가 소진되었는지 말한다.

* `within 60.0 simulated s (...)` — 기체가 시뮬레이션 60초 안에 수렴하지 못했다.
  실제 비행 문제이므로 `longest hold`와 `out of tolerance` 항목을 본다.
* `within 240.0 s wall (... simulated s; the stage is running far slower than
  real time)` — 시뮬레이터가 시각을 거의 진행시키지 못했다. 기체가 아니라 stage의
  문제이며, 렌더 부하나 멈춘 Isaac을 확인한다.

`WARNING: the Isaac/PX4 simulator was adopted, not started by this run`이 보이면
재시작은 아무것도 바꾸지 못한다. `stop`은 이 run이 띄운 프로세스만 종료하므로 이전
run이 남긴 열화된 Isaac/PX4는 그대로 채택되고 남은 재시도는 같은 실패를 반복하는 데
소모된다. 그 시뮬레이터를 run 바깥에서 직접 종료한 뒤 다시 시작해야 한다.

### 시뮬레이터가 기동 중에 죽을 때

`WARNING: simulator startup failed (... exited during startup)`은 대부분 Isaac Sim
자체의 크래시다. 관측된 형태는 기동 약 20초 지점, Ranger Mini UGV의 MDL 머티리얼을
RTX 머티리얼 DB에 올리는 중 Kit이 자기 assertion(`unlock() called by non-owning
thread`, `carb/thread/Mutex.h:158`)으로 abort하는 것이다. 벤치마크 코드 경로가 아니라
Kit 내부의 fiber/mutex 경쟁이므로 대응은 재기동뿐이며 `ExternalStack`이
`startup_attempts`(기본 3)까지 자동으로 다시 띄운다. 이 시점에는 episode·보상·라벨이
하나도 만들어지지 않았으므로 실험 계약에 영향이 없다.

abort한 Kit은 버퍼에 남은 stdout을 버리기 때문에
`/tmp/ontology_rgat_stack/isaac.log`가 비어 있는 것이 정상이다. 실패 보고는 대신
Kit 자신의 세션 로그(`~/.cache/packman/chk/kit-kernel/*/logs/Kit/*/*/kit_*.log`)에서
assertion을 읽어 출력하고 그 경로를 함께 알려준다. 직전 시도의 로그는
`isaac.previous.log`로 남는다.

### Spatial 추가 학습과 출처 보존

`run.sh spatial --stage train`의 `--evaluation-every N`은 nominal validation
간격이다. `--ppo-preset reference-v28-scratch`는 세 arm 공통으로 epoch 8,
actor LR 5e-4, entropy 0.0025, initial log-std -1.1을 사용한다. MATLAB과
RNG/모든 학습 절차가 같다는 뜻은 아니다. 공간 정책의 raw MLP는 원본의
zero bias, Gaussian fan-in scaling, 마지막 층 gain 0.1을 사용한다.
기존 2D 호출의 기본 초기화는 바꾸지 않았다.
이 preset에는 critic-only 2-iteration warmup도 포함된다. `--gae-lambda`의 기본은
0.95이며 다른 값을 쓰면 원본과 다른 학습 요인이다. `--nominal-only`는 로컬 학습도
난이도 1.0으로 고정한다(검증은 옵션과 무관하게 항상 1.0).
기존 eligible actor/critic에서 fine-tuning할 때 `--value-warmup-iterations 0`처럼
명시적으로 warmup을 바꿀 수 있다. 모든 arm에 공통 적용하고 plan/hyperparameters에
기록하며, 생략하면 기존 preset 기본값을 유지한다.

각 run의 `progress.json`은 매 iteration 원자적으로 갱신된다. 중단된 run의
진행 기록을 완료된 `summary.json`이나 selectable checkpoint로 해석하지 않는다.
`--initialize-from PREVIOUS_OUTPUT`은 동일 계약·arm·seed의 eligible PPO
checkpoint로 **새 output**을 초기화한다. 원본 SHA-256을 기록하고 optimizer를
새로 만든 fine-tuning이며, 정확한 optimizer-state resume나 교사 학습이 아니다.
nominal 에피소드를 새로 완료해야 다시 checkpoint 선택이 가능하다.

schema 3의 curriculum 승급·easy/bridge replay·속도 tolerance/unsafe penalty
ramp는 로컬 학습 전용이다. validation/test/Isaac는 항상 난이도 1.0이다.
초기 커리큘럼 SUCCESS를 정상 난이도의 학습 착륙으로 보고하지 않는다.

`--curriculum-angular-scales 2 4`는 3D 탐색을 위한 **추가 실험 요인**이다.
초기 학습 tilt/rate 허용치를 각각 2/4배로 시작하고 난이도와 함께 원래 값으로
돌린다. reference MATLAB과 동일한 절차라고 주장하지 않는다. 난이도 1.0은
옵션 유무에 관계없이 bitwise 같은 nominal 경로이며 Isaac 사용은 거절한다.
`--workers 3`은 로컬 세 arm을 별도 프로세스로 실행한다. Isaac 병렬 비행은
허용하지 않으며 interrupt는 해당 로컬 worker만 정리한다.

`tools/evaluate_spatial_checkpoint.py`는 개별 eligible 체크포인트의 진단/validation용이다.
기본 split은 validation `[2000,2001]`, 명시적 test는 `[12000,12001]`이며,
단일 arm 결과를 complete matrix로 표시하지 않는다. 실제 Isaac 전에는 nominal
로컬 unsafe preflight를 적용하고 snapshot SHA-256을 기록한다.
`checkpoint_evaluated.pt`에 실제 평가 bytes를 보존하고 `status.json`은 실행 중/
완료/중단을 구분한다. `--stage isaac`도 인프라 오류 시 `passes=false`의 불완전
acceptance를 남기며 이전 통과 파일이 남아 있지 않게 한다.
`--checkpoint-wait-seconds N`은 완료된 `summary.json`만 bounded 대기한다.
먼저 완료된 arm부터 순서대로 평가할 수 있지만 매 arm의 nominal safety preflight는
reset/arming 전에 수행한다. `progress.json`만 있는 미완료 arm은 배포하지 않는다.

현재 기본 공간 계약은 v5다. v3/v4 재현은 `--contract-version 3|4`를 명시한다.
v4 → v5 전이는 `--initialize-from OLD --initialize-v4-weights`로만 허용한다.
같은 arm/seed/나머지 config를 요구하고, source/target 서명과 원본 SHA256을 기록한다.
이전 eligible 상태는 승계하지 않으므로 v5 nominal 에피소드 완료·검증 전에는 비행하지 않는다.

실제 실행 전 gateway mirror와 running world의 YAML/source SHA256이 일치해야 한다.
Isaac 코드를 바꾼 뒤 실행 중인 이전 world를 adopt하면 arming 전에 실패한다.
자신이 소유한 비행의 종료와 disarm을 확인한 다음 그 스택만 재시작한다.
공간 경로는 PX4 wire timestamp/physics 경과를 사용하므로 앞선 역사적 legacy
wall-clock 성능 문제와 분리해서 진단한다. safety timeout이나 preflight를 해제하지 않는다.

공간 학습의 완결 에피소드 수집은 `--episodes-per-iteration 6`으로 명시한다.
이때 `--decisions` 대신 에피소드 수가 주 rollout 예산이며, 실제 steps는 progress/
summary에 누적 실측한다. `--ppo-preset reference-v28-episodic`은 upstream처럼
전체 rollout advantage를 한 번 정규화한다. 기존 preset/decision-budget 결과를
재해석하지 않는다. 세 arm의 동일 episode 수가 동일 decision 수라는 뜻은 아니다.
관계 활성화의 추가 PPO steps와 사전학습은 별도로 보고한다.

`--curriculum-loss-timeout-start 12`는 local training에서만 visual-loss abort
시작을 12→3초로 줄인다. nominal/test/Isaac에서는 항상 원래 3초이며, 기록된
커리큘럼 난이도 1에서 기존 cfg와 완전히 같아야 한다.

완료 결과의 기술 통계는 다음처럼 새 파일에 저장한다. 이 도구는 체크포인트를
선택하거나 시뮬레이션을 실행하지 않으며 기존 출력 파일을 덮어쓰지 않는다.

```bash
PYTHONPATH=python python tools/summarize_spatial_run.py \
  --run /path/to/completed-run --backend isaac --output /path/to/new-report.json
```

실제 acceptance v2는 제안 R-GAT의 착륙·비영 관계 출력까지 요구한다. 적은 횟수의
통과와 robustness/reference-parity 검증은 별개이며 보고서에 항상 구분한다.

현재 기본 reset은 Isaac world를 재시작하지 않는
`isaac-px4-ekf-cold/1`이다. 패드 꿐적·기체 pose/속도·센서를 초기화하고,
PX4를 episode별 새 rootfs/boot generation에서 재기동한다. gateway는 새
EKF의 유효 odometry가 연속 확인되기 전에는 `reset_complete`를 보내지
않는다. 따라서 일반 실제 비교에 `--fresh-stack-per-episode`는 필요하지 않다.

`--fresh-stack-per-episode`는 Isaac stage/렌더러 자체의 오염을 분리 진단하는
fallback으로만 남아 있다. 자신이 소유한 스택과 직전 flight stop 확인이
필수이며 `--adopt-stack`과 함께 사용할 수 없다. 예정된 world 분리는
`episode_isolation.jsonl`에 비용을 기록한다. `--reset-recoveries`는 여전히 별도의
정책 전 인계 오류 예산이고, 같은 시드 재시도·0 transitions 원칙을 지킨다.

실제 frozen 비교의 seed는 `--isaac-seed-start 13000 --isaac-episodes 2`처럼
명시할 수 있다. 기본은 기존 `[12000,12001,...]`이며 모든 arm에 같은 범위를 쓴다.
validation `[2000,2001]` 및 local test `[9000,9001]`는 변경하지 않는다. 평가 행에는
실제 episode seeds, 평가 checkpoint SHA256 및 cold-stack 옵션을 함께 기록한다.
새 범위를 사용했다는 사실만으로 multi-seed robustness가 증명되지는 않는다.

Direct acceleration 종료는 먼저 자신의 EKF 위치로 임시 braking hold를 만들고
NAV_LAND를 요청한다. PX4가 OFFBOARD를 벗어나면 gateway가 자체적으로 그 hold를
해제하며, learner도 모드 전환 확인 뒤 스트림을 끈다. 마지막 가속도를 남겨둔 채
먼저 heartbeat를 끊지 않는다. 공중 강제 disarm은 허용하지 않는다.
`flight_terminal.jsonl`은 task terminal 직전 상태와 cleanup 확인 결과를 따로 보존한다.
cleanup 실패는 성공/timeout 완료 에피소드로 승인되지 않는다.

각 PPO update 직후의 `checkpoint_last_update.pt`는 검증 도중 장애 분석을 위한
진단 파일이다. `eligible=false`, `phase=unvalidated_update_snapshot`이며 선택·배포
후보가 아니다. optimizer 상태도 저장하지 않으므로 exact resume 파일로 쓰지 않는다.
정상 validation/nominal eligibility를 통과한 best/final/relational 파일과 구분한다.

고정된 단일 정책의 평가 전용 시드 확대는 `tools/evaluate_spatial_checkpoint.py`
`--split test --test-seed-start 9100 --episodes 20 --backend local`로 실행한다.
기존 `--checkpoint`, `--plan`, 새로운 `--output`도 필요하다. 실제 Isaac test의
최소 시작 시드는 10000이며 validation 2000/2001은 이 옵션으로 바꾸지 못한다.
평가 bytes snapshot/SHA256과 실제 시드 목록을 저장한다. 이 명령은 학습·수정·선택을
하지 않고, 성공이 없거나 unsafe가 있으면 exit 2를 반환한다. 단일 정책 평가가
전체 3-arm acceptance를 대신하지 않으며 추가 test로 후보를 사후 선택하지 않는다.
