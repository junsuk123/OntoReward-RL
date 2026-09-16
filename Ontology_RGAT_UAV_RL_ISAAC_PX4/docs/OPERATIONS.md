# 운영: 실행 프로파일, 산출물, 상태 확인

[문서 안내](README.md) · [실험 설계](TWO_PIPELINE_COMPARISON.md) ·
[아키텍처](ARCHITECTURE.md)

## 1. 실행 프로파일

```bash
./run.sh --pipelines shin_se_fixed shin_se_onto_rgat_recovery --parallel-pairs 2
```

명시적 실행은 `config/experiments/two_pipeline_comparison.yaml`을 사용한다.
Pair 0/1은 별도 PX4 instance, ROS namespace, gateway/learner UDP port,
controller/reset state를 쓰고, 각 worker가 별도 model, rollout buffer, optimizer를
소유하며 GPU update만 lock으로 직렬화한다.

인자 없는 상위 `./run.sh`는 세미나 프로파일을 선택한다.

| 항목 | 값 |
|---|---:|
| 설정 | `config/experiments/seminar_10h_two_pipeline.yaml` |
| system 설정 | `config/seminar-fast-system.yaml` |
| pipeline / pair | 2개 / 2 pair |
| training / pipeline | 144 episodes |
| evaluation / scenario | 5 episodes |
| FOV-risk dataset | 40 episodes, 최대 120 episodes |
| R-GAT 학습 | 80 epochs |
| 결과 | `results/seminar_10h/two_pipe_parallel_144/` |

이 프로파일은 `publication_claim_allowed: false`인 예비 실행이다. 논문용 full 실행은
명시적으로 예산과 결과 디렉터리를 지정하고 manifest를 보존한다.

## 2. 실행 순서

공통 keypoint encoder 준비와 실기 카메라 검증 → baseline PPO와 FOV-risk dataset
수집 → 회귀 R-GAT validation-best 학습·동결 → proposed PPO → held-out checkpoint
선택 → paired/crossover 평가 → 표·그림 저장.

중단 후 같은 명령을 실행하면 호환되는 checkpoint와 완료된 평가 행을 재사용한다.

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
