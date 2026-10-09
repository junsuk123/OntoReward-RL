# MATLAB direct-policy 이식 및 공간 확장

새 구현은 `python/ontology_rgat/direct_policy/`에 격리되어 있다. 기존 minimal,
spatial, reference, isaac-legacy 계약과 체크포인트는 수정하지 않는다.

## 구현 흐름

센서 corner/capture stamp와 자체 navigation에서 PnP와 CV-KF를 거쳐 고정 12D
관측을 만든다. 일반 PPO는 이 벡터를 읽고, ontology arm은 같은 벡터만으로 만든
7노드 그래프를 읽는다. R-GAT은 relation별 projection과 도착 노드별 attention
softmax, 내부 local skip, 노드별 grouped readout을 사용한다. Actor와 Critic은
서로 다른 encoder를 갖는다. 출력은 latent Gaussian이고 tanh/물리 한계 변환 뒤
가속도로 적용한다.

3D profile은 고정 21D이며 x/y 값을 norm 하나로 합치지 않는다. 7개 노드의 의미와
관계는 유지하고 feature channel을 확장한다. y=0, roll=yaw=0 named projection은
12D 스케일과 부호로 환원되는 테스트가 있다. 정책 행동은 `[a_x,a_y,a_z]`이고
수직축이 마지막이다.

contract hash는 세 종류다. algorithm hash는 관측/그래프/행동/학습, task hash는
관측/행동/보상/종료/scenario, execution hash는 backend/safety/기준 SHA를 담는다.
체크포인트는 세 hash, 두 SHA, 학습 단계, optimizer 및 RNG 상태가 없거나 다르면
기본 로드를 거절한다.

## 실행

저장소 루트에서 다음처럼 실행한다. `--dry-run`은 Isaac/PX4를 시작하지 않는다.

```bash
# 기본 무인자 경로는 minimal-observation system이며 이 migration과 분리됨
./run.sh --dry-run

./run.sh matlab-port --stage parity --backend replay --dimension 2
./run.sh matlab-port --stage smoke --backend local --dimension 2
./run.sh matlab-port --stage evaluate --backend isaac --dimension 2 --control-profile direct --allow-isaac
./run.sh matlab-port --stage train --backend local --dimension 3 --methods ppo onto_rgat_ppo
./run.sh matlab-port-all --run-root results/matlab_port/my_run --headless
```

최종 Isaac 운용에는 검증 전용 하위 프로세스를 제외한 경량 경로를 사용한다.

```bash
./run.sh matlab-port-final --run-root results/matlab_port/final_run
```

이 경로는 최종 3D의 `Isaac/PX4 train -> Isaac/PX4 evaluate`만 실행한다.
학습과 평가 모두 `backend=isaac`, `control-profile=direct`를 사용한다. 2D 전체와 `audit`, MATLAB
parity, smoke, 별도 deterministic/sampled local evaluation 및 차원별 report는
생성하지 않는다. 학습 중 best checkpoint 선택에 필요한 validation은 학습
프로세스 내부에 남는다. 차원 비교가 필요하면 `--dimensions 2 3`을 명시하고,
전체 이관 검증이 필요할 때만 `matlab-port-all`을 쓴다.

최종 경로는 episode 사이에 PX4/EKF와 움직이는 pad 상태가 누적되지 않도록
각 episode를 fresh owned stack에서 시작한다. `checkpoint_last.pt`는 PPO update마다
저장된다. 중단 후 같은 run root를 계속할 때는 `--resume-training`을 명시한다.

`matlab-port-all`의 기본값은 arm별 750 update,
update당 sampled episode 6개, validation 100 seed, test 200 seed(3001--3200),
그리고 arm별 Isaac seed 12000 1회다. 기존 minimal 시스템은
`./run.sh system`으로 실행한다. 실행 루트의 `analysis.json`이 validation,
deterministic/sampled test, Isaac 결과를 합친다.

MATLAB golden은 `tools/matlab_port/export_golden_fixture.m`로 만든 뒤 parity 단계의
`--fixture`에 전달한다. MATLAB과 NumPy seed가 같은 난수열을 만든다고 가정하지
않는다.

## 검증 상태

- Python 계약/관측/그래프/gradient/PPO/reward/좌표 adapter: PASSED
- local 2D/3D smoke와 CLI dry-run: PASSED
- pinned MATLAB이 생성한 golden과의 수치 비교: PASSED (최대 절대 오차
  `1.1102230246251565e-16`)
- 전체 750-update 학습과 deterministic/sampled test: PASSED (단일 학습 seed)
- seed-only stress split: PASSED. 물리 stress 분포는 아직 구현되지 않음
- Isaac/PX4 실제 비행: PASSED. owned stack, 기존 ArUco PnP/EKF,
  acceleration-only gateway, reset/stop/cleanup을 재사용하여 2D/3D의 PPO/R-GAT
  각 1회를 비행했다. 1 seed이므로 성능 수용 게이트는 아니다.
- R-GAT 우월성: 주장하지 않음. 3D deterministic 결과는 우세했지만 sampled
  test는 모든 arm이 성공 0%이며 2D R-GAT은 unsafe 98%였음

실행 수치와 제한은 `FULL_PIPELINE_20261009_KO.md`에 기록한다.

OpenCV PnP와 MATLAB 자체 Gauss-Newton PnP의 수치 차이는 별도 gate다. 실제
Isaac 실행은 `--allow-isaac`이 있는 명시적 경로에서만 가능하고, 기존
stack을 임의로 takeover하지 않는다.

실제 Isaac PPO update 경로도 같은 CLI의 `--stage train --backend isaac
--allow-isaac`으로 연결되어 있다. 이는 구현된 opt-in 경로이지 이번 재감사에서
실행한 장기 학습 결과가 아니다. `--resume-checkpoint-root`는 동일 output root의
model/optimizer/update/RNG 상태를 검증 후 복원한다.
