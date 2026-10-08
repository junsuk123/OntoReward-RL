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
./run.sh matlab-port --stage parity --backend replay --dimension 2
./run.sh matlab-port --stage smoke --backend local --dimension 2
./run.sh matlab-port --stage evaluate --backend isaac --dimension 2 --control-profile direct --allow-isaac
./run.sh matlab-port --stage train --backend local --dimension 3 --methods ppo onto_rgat_ppo
```

MATLAB golden은 `tools/matlab_port/export_golden_fixture.m`로 만든 뒤 parity 단계의
`--fixture`에 전달한다. MATLAB과 NumPy seed가 같은 난수열을 만든다고 가정하지
않는다.

## 검증 상태

- Python 계약/관측/그래프/gradient/PPO/reward/좌표 adapter: PASSED
- local 2D/3D smoke와 CLI dry-run: PASSED
- pinned MATLAB이 생성한 golden과의 수치 비교: NOT_RUN (MATLAB fixture 없음)
- 전체 750-update 학습, validation/test/stress: NOT_RUN
- Isaac/PX4 실제 비행 및 ROS control path: NOT_RUN
- R-GAT 우월성: NOT_RUN이며 구현 또는 smoke 결과로 주장하지 않음

OpenCV PnP와 MATLAB 자체 Gauss-Newton PnP의 수치 차이는 golden fixture가 생길 때
별도 gate로 판정해야 한다. 실제 Isaac 실행은 command-owner, acceleration-only,
timestamp 기록을 연결한 다음 owned stack에서만 수행한다.
