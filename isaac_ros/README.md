# simlab — Isaac Sim 아군/적군 쿼드로터 군집

## GNSS-denied terrain visual matching

The repository now also contains a self-contained classical-CV research pipeline for
ontology-guided terrain matchability. It does not require Isaac Sim, ROS, a network
map, or GNSS input. Procedural road, grass, urban-building, and water images feed ORB,
image-quality and spatial-distribution features, Lucas–Kanade flow, external YAML
rules, global keyframe retrieval, and homography verification. Hidden coordinates are
owned exclusively by the evaluation ledger.

```bash
python -m pip install -r requirements.txt
./run_terrain_matching.sh
# or: ./run_terrain_matching.sh --config config/my_experiment.yaml
python -m unittest discover -s tests -p 'test_terrain_matching.py'
```

Use `TERRAIN_MATCHING_PYTHON=/path/to/python` to select an interpreter, or
`TERRAIN_MATCHING_CONFIG=config/my_experiment.yaml` to change the default
configuration without editing the launcher.

The experiment prints its `results/run_YYYYMMDD_HHMMSS` directory. It contains the
resolved configuration, validated frame schema in CSV/JSON, per-frame explanations,
an RDF/Turtle ontology snapshot, matching/localization tables, baseline and ablation
metrics, terrain statistics, and figures. Change terrain classes, conditions, random
seed, feature parameters, matching thresholds, and output paths in
[`config/experiment.yaml`](config/experiment.yaml); change knowledge contributions in
[`config/ontology_rules.yaml`](config/ontology_rules.yaml) without editing Python.

The simulator-facing modules remain available below and are intentionally independent
of this pipeline, allowing a future ROS camera adapter to replace only the procedural
image source.

## Accelerated temporal environment change

Run the deterministic Year0-to-Year10 batch experiment with the fast image backend:

```bash
./run_temporal_environment.sh
```

Validate the same configured epochs as mutable USD state in the installed Isaac Sim
5.1 environment and capture Replicator RGB images:

```bash
./run_temporal_environment.sh --with-isaac
```

Launch the entire workflow and keep Isaac Sim open with the validation GUI:

```bash
./run_temporal_gui.sh
```

The in-app panel provides epoch buttons that immediately reapply each deterministic
USD state, component scores, aggregate matching metrics, and a Year0/query feature
correspondence view with RANSAC inliers and outliers. Closing Isaac Sim ends the
workflow.

The default collection flight uses a procedural fixed-wing aircraft over a 300 m ×
300 m map. It flies an outward 2.5-turn spiral from 12 m to 92 m radius at 45 m
altitude, captures 32 nadir frames per epoch, and saves an MP4 for each pass. These
values are configurable under `map` and `camera` in the temporal YAML.

Isaac's verbose startup output is saved as `isaac_sim.log` inside the generated run
directory. Successful validation also writes `isaac_validation.json`,
`isaac_matching_results.csv`, and `isaac_summary_metrics.csv`. If Kit reports
many `errno=28` change-watch messages, run `./fix_inotify.sh` once; this raises the
watch and instance limits and requires `sudo`.

Virtual environment time is independent of vehicle physics time. Configuration,
events, component-score weights, fixed camera trajectory, and semantic comparison
cases live in [`config/temporal_environment.yaml`](config/temporal_environment.yaml).
See [`docs/temporal_environment_architecture.md`](docs/temporal_environment_architecture.md)
for the clock, state, USD, capture, and evaluation boundaries.

아군·적군 쿼드로터가 자동 이륙한 뒤 전방 카메라 광축의 법선 방향이자 지면과
평행한 직선에서 왕복하는 standalone 씬입니다. 매 구간의 교차 위치와 시점은
무작위로 바뀌며, 카메라 영상에서는 두 기체의 박스가 겹치도록 구성됩니다.

- 아군: Pegasus Simulator `3DR Iris` USD, 1.35배 렌더 스케일
- 적군: 공식 Isaac Sim `Quadcopter` USD, 1.5배 렌더 스케일
- 4초 자동 이륙 후 같은 순찰 링에서 반대 방향 비행
- 광축 방향 1.8 m 깊이 분리로 실제 안전거리를 유지하면서 영상상 가림 생성
- 위에서 볼 때 카메라 광축과 활동 직선이 직교하는 `ㅗ` 배치
- 드론 운동 방향과 평행한 기준선의 전방 카메라 2대 + 상공 수직 하향 카메라 1대

적군 형상과 관절 구조는 Isaac Sim 5.1 공식 자산을 참조하며, 아군 Iris는
[Pegasus Simulator](https://github.com/PegasusSimulator/PegasusSimulator)의 BSD-3-Clause
USD를 사용합니다. 비행 자세는
군집/회피 및 지각 연구의 재현성을 위해 결정론적 kinematic 제어가 담당하며, 참조
자산의 중력·충돌 물리는 로드 후 비활성화됩니다.

### 지각·트래킹 준비 구조

| 센서 ID | USD prim | 위치/방향 | 기본 해상도 |
|---|---|---|---|
| `front_near` | `/World/Sensors/FrontNear` | `(12,-1.5,4)`, `(-1,0,0)` | 1920×1080 |
| `front_far` | `/World/Sensors/FrontFar` | `(12,1.5,4)`, `(-1,0,0)` | 1920×1080 |
| `satellite_nadir` | `/World/Sensors/SatelliteNadir` | `(0,0,20)` → `(0,0,2.5)` | 2048×2048 |

전방 두 카메라는 서로 평행하게 X축 음의 방향을 바라보며, 드론 운동과 같은 Y축
방향으로 3 m 떨어져 있습니다. 각 드론 prim에는
`class`, `team`, `model`, `track_id` Replicator 레이블과 `simlab:trackId`,
`simlab:instanceId`, `simlab:assetUri`, `simlab:ontologyClassUri` 등의 USD 속성이
붙습니다. [configs/drone_ontology.yaml](configs/drone_ontology.yaml)은 위성 라벨과
전방 관측을 같은 물리 개체로 연계할 때 사용할 식별자·관계·증거 필드를 정의합니다.

## 환경

| 항목 | 값 |
|---|---|
| Isaac Sim | 5.1.0 (pip 설치, `/home/j/env_isaaclab`) |
| Python | 3.11 (`/home/j/env_isaaclab/bin/python`) |
| GPU | RTX 4060 Laptop, driver 570.172.08 / CUDA 12.8 |
| 에셋 | S3 `.../Assets/Isaac/5.1` (온라인) |
| ROS 2 | Humble, Python 3.10, `rmw_cyclonedds_cpp` |

`/home/j/isaacsim` 에도 소스 빌드(5.1.0-rc.19)가 있지만 이 프로젝트는 pip 설치본을 씁니다.
다른 인터프리터를 쓰려면 `SIMLAB_PYTHON=/path/to/python ./run.sh`.

## 실행

### 드론 지각 파이프라인 + RViz2 동시 실행

```bash
./run_pipeline.sh                         # 매번 수집 + 추가 학습 + 검증 + 시뮬레이션/RViz2
./run_pipeline.sh headless:=true seconds:=60
SIMLAB_INFERENCE_ONLY=1 ./run_pipeline.sh # 수집·학습 없이 현재 승인 모델로 즉시 실행

# 최초 데이터 수집부터 자동 라벨, 학습, 품질 게이트, 배포까지
./run_auto_pipeline.sh
```

`run_auto_pipeline.sh`는 실행할 때마다 25초간 1 Hz 카메라 샘플과 전체 TF를 zstd
rosbag으로 영구 저장하고, SAM2-tiny로 박스를 정제해 기체 모델별 누적 YOLO 데이터셋과
드론별 JSONL 인덱스에 추가합니다. 이어 같은 기체 구성의 직전 `latest` 가중치에서
YOLO26s를 추가 학습합니다. 후보 모델은 `mAP50-95 >= 0.65`이면서 기존 활성 모델보다
성능이 낮지 않을 때만 배포됩니다. 기준 미달 후보도 다음 추가 학습의 시작점으로 보존하며,
기존 활성 모델은 삭제하지 않습니다. 기체 종류나 스케일이 바뀌면 자동으로 별도 학습 계보와
누적 데이터셋을 시작합니다. 임계치와 수집/학습 설정은
[`configs/perception.yaml`](configs/perception.yaml)에 있습니다.

`./run_auto_pipeline.sh --force-retrain`은 누적 데이터는 유지하되 해당 실행만 기본
`yolo26s.pt`에서 다시 학습합니다.

기본 월드는 교차 도로·차선·보도·다층 건물로 구성된 경량 절차형 도심 환경입니다.
고정된 드론/카메라 좌표계를 유지하면서 전방/위성 영상에 도시 배경을 제공하고,
대형 외부 USD의 최초 다운로드나 누락된 하위 자산 때문에 수집이 멈추지 않습니다.

실시간 박스 영상은 활성 모델에 이름이 정확히 `drone`인 클래스만 표시합니다. COCO
기본 모델에는 드론 클래스가 없으므로 건물 등 일반 객체 예측은 표시하지 않고,
승인된 커스텀 모델이 생기기 전까지 시뮬레이터 투영 드론 박스만 사용합니다.

기본 실행은 Isaac Sim, ROS 2 카메라/TF 브리지, 프로토타입 연계 트래커, RViz2를
한꺼번에 시작합니다. RViz에는 드론 추적 ID·이동 궤적·TF, 위성 RGB와 두 전방
YOLO 박스 영상이 표시됩니다.

```text
Isaac Sim official drone USDs
  ├─ /tf: map → Ally_01..04, Enemy_01..02
  ├─ /simlab/front_near/image + camera_info
  ├─ /simlab/front_far/image + camera_info
  └─ /simlab/satellite_nadir/image + camera_info

simlab.ros.yolo_detector_node
  ├─ /simlab/front_near/detections       # YOLO 박스가 그려진 Image
  ├─ /simlab/front_far/detections
  ├─ /simlab/yolo/detections             # 카메라별 JSON 박스/클래스/신뢰도
  └─ /simlab/yolo/status
          ↓
simlab.ros.swarm_pipeline_node
  위성 GT로 ID 초기화 → ontology(team, model) hard gate
  → 모션 거리 연계 → ID switch 계수
          ↓
  /simlab/tracking/markers, /simlab/tracking/status → RViz2
```

현재 프로토타입은 실제 세 카메라 영상의 수신 여부를 계수하지만, 검출 위치는 TF 기반
대체 detector를 사용합니다. 즉 ID 생명주기와 온톨로지 연계·RViz 경로를 먼저 검증하는
단계이며, 다음 단계에서 `_detections()`를 위성 segmentation 및 전방 visual embedding
출력으로 교체하도록 경계를 분리해 두었습니다.

상태 확인:

```bash
ros2 topic echo /simlab/tracking/status
ros2 topic hz /simlab/front_near/image
ros2 topic hz /simlab/front_far/image
ros2 topic hz /simlab/satellite_nadir/image
```

### 추적 실패 시나리오 3종 (`./run_scenarios.sh`)

```bash
./run_scenarios.sh                          # 전체 계획 비행 → 추가 학습 → 배포
./run_scenarios.sh --dry-run                # 에피소드 설정만 생성하고 종료
./run_scenarios.sh --scenario sensor_dropout --no-deploy
./run_scenarios.sh --environment urban_night --episodes 2
./run_scenarios.sh --seed 7                 # 이전 실행을 그대로 재현
```

ID switch가 발생하는 세 가지 상황을 각각 독립된 **에피소드**로 비행합니다. 에피소드
하나는 시뮬레이터·수집기·rosbag이 함께 뜨고, 정해진 시간을 비행하고, **전부 종료된 뒤**
쿨다운을 거쳐 다음 에피소드가 시작합니다. 스테이지도 ROS 그래프도 DDS 디스커버리도
공유하지 않으므로 앞 에피소드가 뒤 에피소드에 남는 일이 없습니다.

| 시나리오 | 상황 | 기록되는 핵심 정보 |
|---|---|---|
| `mutual_occlusion` | 두 대 이상이 영상에서 겹치고 교차 | 이동 방향·속도, 상대 위치, 깊이 순서(`depth_rank`), 편대(`formation`), 가린 기체(`covered_by`) |
| `structural_occlusion` | 건물·컨테이너·벽 뒤로 사라졌다 재등장 | 마지막 관측 위치·속도, 가린 구조물(`blocked_by`), 예측 경로, 예상 재출현 박스(`predicted.bbox_xyxy`) |
| `sensor_dropout` | 카메라/통신이 수 프레임~수 초 두절 | 마지막 관측 시점(`seconds_since_observation`), 두절 길이(`gap.duration_s`), 예측 위치, 트랙 생존(`track_alive`) |

시나리오는 서로 섞이지 않습니다. `structural_occlusion` 에피소드에만 시선 차단
구조물이 배치되고, 나머지 두 시나리오는 시선이 열린 상태로 비행합니다. 겹침 중에
빌보드 뒤로도 사라져 버리면 "겹침을 견뎠는가"에 답할 수 없기 때문입니다.
(`world.occluder_set`을 직접 지정하면 이 규칙을 덮어쓸 수 있습니다.)

**안전거리는 규칙이 아니라 구조로 보장합니다.** 각 기체는 카메라 광축 위의 자기
레인을 갖고, 이웃 레인 간격은 항상 `drones.safety_radius` 이상입니다. 그래서 같은
픽셀을 공유해도 — 그게 목적입니다 — 3차원 거리는 절대 안전 반경 아래로 내려가지
않습니다. 회피 로직이 필요 없고, 운 나쁜 시드에 깨지지도 않습니다.

#### 비행 환경

같은 과제를 다른 조명·색·스카이라인에서 관측해야 인지 단계가 환경이 아니라 기체를
학습합니다. `world.environment`로 고르고, 계획에서는 시나리오마다 여러 환경을 지정합니다.

| 프리셋 | 특징 | 기본 차단 구조물 |
|---|---|---|
| `urban_day` | 정오 도심, 중성광 | `urban_screens` |
| `urban_overcast` | 흐림, 낮은 대비 | `gantry_wall` |
| `urban_dusk` | 저각 온광, 긴 그림자 | `urban_screens` |
| `urban_night` | 야간, 창문 띠만 밝음 | `sparse_masts` |
| `industrial_yard` | 저층 밀집 + 컨테이너 야적장 | `container_yard` |
| `desert_outpost` | 밝은 모래, 희소 구조물 | `sparse_masts` |
| `coastal_flats` | 하늘 밝은 평지 | `gantry_wall` |

환경의 모든 상자(지면·도로·건물·차단 구조물)는
[`simlab/scenarios/environments.py`](simlab/scenarios/environments.py)의 한 목록에서
나옵니다. 같은 목록이 스테이지를 세우고, 수집기의 가림 판정에도 쓰입니다. 그래서
"카메라가 못 본 것"과 "라벨에 없는 것"이 정확히 일치합니다.

#### 에피소드가 만드는 데이터

에피소드마다 `artifacts/perception/raw/<episode_id>/` 아래에 세 가지가 쌓입니다.

- `prompts.jsonl` — 저장한 프레임과 **보이는 기체만**의 박스. 구조물 뒤에 있거나 앞
  기체에 덮인 기체는 `hidden`으로 따로 남고 라벨에는 들어가지 않습니다. 가려진 것을
  라벨링하면 검출기가 빈 벽면에 반응하도록 배웁니다.
- `tracking_truth.jsonl` — 프레임 도착 여부와 무관하게 5 Hz로 흐르는 추적 정답.
  블랙아웃 순간에는 영상이 오지 않으므로, 영상에 매달린 로그로는 정작 중요한 구간을
  기록할 수 없습니다.
- `session.json` — 시나리오·환경·시드·이벤트 일정 등 세션 자체 설명.

`artifacts/scenarios/<run>/<episode_id>/episode_summary.json`에는 실제로 비행한 결과가
남습니다. 최소 3차원 이격거리, 레인별로 실제 통과한 차폐 구간, 적용된 블랙아웃 시간
등이라, 예를 들어 "아무도 가리지 못한 structural 에피소드"가 조용히 지나가지 않습니다.

#### 학습되지 않은 데이터만 추가 학습

시작할 때마다 두 가지를 순서대로 확인합니다.

1. 데이터셋이 아직 흡수하지 않은 수집 세션이 있는가 → SAM으로 정제해 누적 데이터셋에 추가
2. 흡수는 됐지만 배포 가중치가 학습한 적 없는 세션이 있는가 → 최신 가중치에서 추가 학습

둘 다 아니면 아무것도 하지 않고 그렇게 말합니다. 그래서 모든 실행 앞에 놓아도
안전합니다. `run_scenarios.sh`는 이 루틴을 **비행 전**(지난 실행의 잔여 데이터)과
**비행 후**(이번에 모은 데이터) 두 번 돌립니다.

추가 학습은 새 프레임만이 아니라 누적 데이터셋 전체로 진행합니다. 새 환경 한 번이
이전 환경들을 지워버리는 것을 막기 위한 리허설이며, 학습 트리거만 "새 세션 존재"입니다.

```bash
# 시뮬레이션 없이 밀린 데이터만 정리·학습
./scripts/setup_perception.sh >/dev/null &&   .venv-perception/bin/python -m simlab.ml.continual
```

세션 원장은 두 곳에 있습니다. `datasets/cumulative_<signature>/sessions/*.json` 이
"데이터셋에 들어간 세션", `model_lineages/<signature>/trained_sessions.json` 이
"가중치가 학습한 세션"입니다. 둘의 차이가 곧 미학습 데이터입니다.

### 전체 파이프라인 (Isaac Sim GUI + 알고리즘 노드 + RViz2)

```bash
source /opt/ros/humble/setup.bash
ros2 launch launch/simlab.launch.py                       # GUI + RViz2
ros2 launch launch/simlab.launch.py nav2:=true            # Nav2 가 주행
ros2 launch launch/simlab.launch.py headless:=true seconds:=60
ros2 launch launch/simlab.launch.py rviz:=false
ros2 launch launch/simlab.launch.py config:=configs/my_scene.yaml
```

터미널을 나눠 돌리는 편이 디버깅에 편합니다:

```bash
scripts/run_sim.sh                 # 1) Isaac Sim + ROS 2 브리지
scripts/run_controller.sh          # 2) 알고리즘 노드
scripts/run_rviz.sh                # 3) RViz2
```

### ROS 2 없이 (in-process 컨트롤러)

```bash
./run.sh                                    # GUI, 닫을 때까지
./run.sh --headless --seconds 20
./run.sh --allies 4 --enemies 2
./run.sh --set drones.activity_half_length=7.0
./run.sh --set drones.crossing_offset_limit=4.0 --set drones.shuttle_leg_duration_s=13.0
```

`--set PATH=VALUE` 는 설정 트리의 아무 leaf 나 덮어씁니다 (값은 YAML 로 파싱되므로
숫자·불리언·리스트 그대로 들어갑니다). `python -m simlab --help` 로 전체 플래그 확인.

첫 실행에는 Quadcopter USD를 Isaac Sim 자산 서버에서 받아야 하므로
네트워크 상태에 따라 시간이 걸릴 수 있으며 이후에는 로컬 캐시를 사용합니다.

핵심 비행 파라미터는 [configs/default.yaml](configs/default.yaml)의 `drones` 아래에
있습니다. `crossing_offset_limit`는 `activity_half_length`보다 작아야 하고,
광축 방향 기체 간격은 `safety_radius` 이상이어야 합니다.

아래의 UGV/People 및 ROS 2 설명은 `drones.enabled: false`로 전환할 때 사용하는
기존 호환 경로에 관한 내용입니다.

## 구조

```
isaac_ros/
├── run.sh                  ROS 2 없이 돌리는 런처
├── run_scenarios.sh        시나리오 3종 순차 비행 + 미학습 데이터 추가 학습
├── run_auto_pipeline.sh    단발 수집 + 추가 학습 + 배포
├── configs/
│   ├── default.yaml        씬 + 시나리오 계획 + ROS 2 + 센서 설정
│   ├── drone_ontology.yaml 드론·관측 식별자 및 관계 계약
│   └── nav2_params.yaml    Nav2 파라미터
├── launch/
│   ├── simlab.launch.py    전체 파이프라인
│   └── nav2.launch.py      Nav2 서버 + map->odom
├── rviz/simlab.rviz        RViz2 레이아웃
├── scripts/
│   ├── ros2_env.sh           공용 환경 (PYTHONPATH·RMW·snap 처리)
│   ├── run_sim.sh            Isaac Sim + 브리지   (Python 3.11)
│   ├── run_controller.sh     알고리즘 노드         (Python 3.10)
│   ├── run_rviz.sh           RViz2                (Python 3.10)
│   └── stop_all.sh           남은 프로세스 정리
└── simlab/
    ├── cli.py              인자 파싱 + 부팅 순서 (아래 "Import 순서" 참고)
    ├── config/             타입 있는 설정. Isaac Sim 불필요
    │   ├── schema.py         dataclass 정의, 오타 키 거부
    │   └── loader.py         YAML 로드 + deep-merge 오버라이드
    ├── algorithms/         주행 알고리즘. Isaac Sim·ROS 2 불필요 → 양쪽에서 import
    │   ├── base.py           Observation / DriveCommand / DriveController
    │   ├── patrol.py         square_loop, stand_still
    │   └── registry.py       이름 → 컨트롤러
    ├── scenarios/          시나리오·환경·가림 판정. Isaac Sim·ROS 2 불필요 → 양쪽에서 import
    │   ├── environments.py   비행 환경 프리셋 → 조명·팔레트·정적 상자 목록
    │   ├── occlusion.py      상자 기하 + 시선 차단 판정 + 차폐 구간 계산
    │   ├── projection.py     월드 좌표 → 영상 박스, 깊이 순서, 상호 가림
    │   ├── timeline.py       에피소드 이벤트 일정 (블랙아웃/프레임 드롭)
    │   ├── motion.py         시나리오별 레인·교차 그룹·속도 배정
    │   ├── scene.py          설정 하나에서 양쪽이 합의하는 값들
    │   ├── episode.py        에피소드 1회분 프로세스 기동/종료
    │   └── orchestrator.py   계획 전체 비행 → 추가 학습 → 배포
    ├── ml/                 자동 라벨링·연속 학습 (perception venv)
    │   ├── auto_label_train.py  SAM 정제, 세션 흡수, 학습, 품질 게이트
    │   ├── continual.py         미흡수/미학습 세션만 골라 따라잡기
    │   └── lineage.py           모델 계보와 세션 원장
    ├── sim/                Isaac Sim 바인딩 (Python 3.11 전용)
    │   ├── app.py            SimulationApp, 확장 활성화, 스테이지 재생성
    │   ├── assets.py         로봇/캐릭터 USD 카탈로그
    │   ├── drones.py         공식 드론 USD 스폰 + 시맨틱 ID + 군집 자세
    │   ├── cameras.py        평행 전방 카메라 2대 + 위성 카메라
    │   ├── world.py          World + ground + dome light
    │   ├── ugv.py            스폰, 휠·물리 프림 해석, 속도 명령 적용
    │   ├── people.py         omni.anim.people 캐릭터 스폰/바인딩/위치
    │   ├── ros2.py           ROS 2 브리지 OmniGraph
    │   ├── telemetry.py      주기적 포즈 출력
    │   └── runner.py         셋업 + 스텝 루프
    ├── ros/                ROS 2 노드 (Python 3.10 전용)
    │   ├── controller_node.py     알고리즘 → /cmd_vel, 마커 발행
    │   ├── yolo_detector_node.py  실시간 YOLO 추론 + 박스 영상
    │   ├── swarm_pipeline_node.py 온톨로지 게이트 연계 트래커
    │   └── dataset_collector_node.py 가림 반영 라벨 + 추적 정답 수집
    └── utils/              순수 헬퍼 (로깅, 경로)
```

경계 규칙: **`sim/` 만 `omni`/`isaacsim`, `ros/` 만 `rclpy` 를 import 합니다.**
`config`, `algorithms`, `scenarios`, `utils` 는 평범한 파이썬이라 Isaac Sim 없이
import·테스트할 수 있고, 이 덕분에 Python 3.11(Isaac)과 3.10(ROS 2) 양쪽에서 같은
코드를 씁니다. 시나리오가 이 경계에 의존합니다: 시뮬레이터는 `scenarios/`로 스테이지를
세우고 카메라를 차단하고, 다른 프로세스의 수집기는 **같은 에피소드 YAML로 같은 표를
다시 만들어** 블랙아웃 일정과 차단 구조물 위치를 알아냅니다. 서로 통신하지 않는데도
라벨과 렌더가 같은 세계를 말하는 이유입니다.

```bash
/home/j/env_isaaclab/bin/python -c "
from simlab.algorithms import build_controller, Observation
c = build_controller('square_loop', {'linear_speed': 0.5})
print(c.step(Observation(t=0.0, robot_xy=(0,0), robot_yaw=0.0)))"
```

### 데이터 흐름

```
configs/*.yaml ──> config.SceneConfig ──> cli.main
                                            │
                                   sim.app.launch (Kit 기동)
                                            │
                                   sim.runner.SimulationRunner
                                    ├─ sim.world.build_world
                                    ├─ sim.ugv.UGV.spawn
                                    ├─ sim.people.Crowd.spawn
                                    └─ 루프: observe ─> algorithms ─> ugv.apply ─> world.step
```

컨트롤러는 `Observation(t, robot_xy, robot_yaw, people_xy)` 를 받아
`DriveCommand(linear, angular)` 를 돌려줍니다. 휠 속도 변환은 `sim/ugv.py` 담당이라
알고리즘은 로봇 기구학을 몰라도 됩니다.

## ROS 2 파이프라인

```
 Isaac Sim  (Python 3.11)                    ROS 2 Humble  (Python 3.10)
 ┌───────────────────────────┐               ┌──────────────────────────────┐
 │ simlab.sim.ros2           │  /clock  ───► │                              │
 │  OmniGraph                │  /odom   ───► │ simlab.ros.controller_node   │
 │   ROS2PublishClock        │  /tf     ───► │   simlab.algorithms          │
 │   ROS2PublishOdometry     │               │        │                     │
 │   ROS2PublishRawTransform │  ◄─── /cmd_vel│        ▼ DriveCommand        │
 │   ROS2SubscribeTwist      │               │   /simlab/markers ──┐        │
 │    → DifferentialController                └─────────────────────┼───────┘
 │    → ArticulationController                                      ▼
 └───────────────────────────┘                                   RViz2
```

두 프로세스로 나뉘는 이유는 인터프리터가 다르기 때문입니다. Isaac Sim 은 Python
3.11, ROS 2 Humble 은 3.10 이라 한 프로세스에 들어갈 수 없습니다. 알고리즘 코드
(`simlab/algorithms/`)가 순수 파이썬이라 양쪽에서 그대로 import 됩니다.

### 토픽

| 방향 | 토픽 | 타입 | 내용 |
|---|---|---|---|
| Isaac → | `/clock` | `rosgraph_msgs/Clock` | 시뮬레이션 시간 (모든 노드 `use_sim_time:=true`) |
| Isaac → | `/odom` | `nav_msgs/Odometry` | UGV 위치·속도, `odom`→`base_link` |
| Isaac → | `/tf` | `tf2_msgs/TFMessage` | `odom`→`base_link`, `odom`→`person_01..03`, `base_link`→센서 |
| Isaac → | `/scan` | `sensor_msgs/LaserScan` | 전방 2D 라이다 (Nav2 입력), 3200빔 |
| Isaac → | `/front/rgb` | `sensor_msgs/Image` | 전방 카메라 640×480 |
| Isaac → | `/front/camera_info` | `sensor_msgs/CameraInfo` | |
| → Isaac | `/cmd_vel` | `geometry_msgs/Twist` | 주행 명령 (바퀴를 직접 구동) |
| 노드 → | `/simlab/markers` | `visualization_msgs/MarkerArray` | RViz 용 사람·로봇 마커 |

검증된 실측 레이트: `/odom` 58 Hz, `/scan` 7.4 Hz, `/cmd_vel` 29 Hz, `/simlab/markers` 29 Hz.

센서 프림은 `ros2.sensors` 에서 지정하고, `base_link`→센서 오프셋은 USD 에서 읽어
TF 로 냅니다 (예: `base_link`→`front_lidar` = `[0.026, 0, 0.418]`).

### 알고리즘을 ROS 2 노드로 돌리기

`simlab/ros/controller_node.py` 가 `/odom` 과 TF 를 읽어 `Observation` 을 만들고,
`configs/default.yaml` 의 `ugv.controller` 로 지정된 컨트롤러를 호출한 뒤 결과를
`/cmd_vel` 로 냅니다. 즉 **새 알고리즘을 `simlab/algorithms/` 에 추가하고 registry 에
등록하면 ROS 2 노드가 그대로 씁니다** — 노드 코드는 건드릴 필요가 없습니다.

시뮬레이터와 노드가 같은 YAML 을 읽으므로 설정이 한 곳에만 있습니다.

```bash
ros2 topic echo /cmd_vel
ros2 run tf2_ros tf2_echo odom person_02
ros2 topic hz /odom
```

수동 조종으로 파이프라인만 확인하고 싶으면 노드 대신:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

`ros2.drive_from_cmd_vel: false` 로 두면 브리지는 관측만 발행하고 주행은 다시
in-process 컨트롤러가 맡습니다.

### 환경 함정 (모두 `scripts/ros2_env.sh` 에서 처리됨)

1. **PYTHONPATH 오염** — `source /opt/ros/humble/setup.bash` 는 python3.10 경로를
   `PYTHONPATH` 에 넣습니다. 그대로 Isaac Sim(3.11)을 띄우면 cp310 `rclpy` 를 읽으려다
   죽습니다. 그래서 Isaac 프로세스에만 `python3.10` 이 들어간 항목을 전부 제거합니다
   (`/opt/ros` 뿐 아니라 사용자 워크스페이스 오버레이도 포함). C++ 브리지는
   `LD_LIBRARY_PATH`/`AMENT_PREFIX_PATH` 만 쓰므로 영향 없습니다.
2. **RMW 불일치** — 양쪽이 같은 미들웨어를 써야 합니다. 이 PC 는 bashrc 가
   `rmw_cyclonedds_cpp` + `CYCLONEDDS_URI` 를 쓰므로 그대로 따라갑니다. 다르면 토픽이
   **에러 없이** 그냥 안 붙습니다.
3. **snap 의 `GTK_PATH`** — VS Code snap 터미널에서 실행하면 `GTK_PATH` 가 snap 쪽
   GTK 모듈을 가리켜 RViz2 가 snap 의 glibc 를 끌어오고 이렇게 죽습니다:
   `symbol lookup error: /snap/core20/.../libpthread.so.0: undefined symbol: __libc_pthread_init`.
   `strip_snap_gtk_path` 가 `/snap` 항목만 제거합니다. 일반 터미널에서는 무해합니다.

### ROS 2 쪽 함정: odometry 대상 프림

`IsaacComputeOdometry` 는 **강체**를 요구합니다. 래퍼 Xform 을 주면
`"is not a valid rigid body or articulation root"` 로 실패합니다. 로봇마다 구조가
달라서 (`nova_carter` 는 관절 루트가 `chassis_link` 이자 강체, `jetbot` 은 관절 루트가
루트 Xform 이고 강체는 `chassis`) `UGV._scan_physics_prims()` 가 자동으로 찾습니다.

## 사람 회피 알고리즘 (`social_force`)

기본 컨트롤러입니다. 웨이포인트를 순회하면서 보행자를 피합니다. 순수 파이썬이라
Isaac Sim 없이 단위 테스트됩니다.

- **인력** — 현재 웨이포인트 방향 단위 벡터.
- **척력** — `influence_radius` 안의 사람마다 `1/d - 1/R` 크기. 반경 끝에서 0 으로
  매끄럽게 사라지므로 갑자기 켜지지 않습니다.
- **접선(swirl) 항** — 순수 척력만 쓰면 **정면 교착**이 생깁니다. 사람이 로봇과 목표
  사이에 정확히 서면 인력과 척력이 상쇄돼 조향이 0 이 되고 그대로 직진합니다.
  각 사람마다 목표 쪽으로 도는 접선 성분을 더해 옆으로 비껴가게 합니다.
  `swirl_gain: 0` 으로 두면 이 교착이 되살아납니다.
- **안전 버블** — `stop_radius` 안으로 들어오면 목표를 포기하고 사람 반대편으로
  돌면서 후진합니다. 감속만으로는 걸어 들어오는 보행자를 피할 수 없습니다.

```bash
./run.sh --controller square_loop        # 사람을 무시하는 개루프 순찰과 비교
./run.sh --set ugv.controller.params.swirl_gain=0   # 교착 재현
```

### 웨이포인트는 보행 레인 밖에 두세요

기본 웨이포인트가 반경 3.5 다이아몬드였을 때 목표 `(-3.5, 0)` 이 Person_02 의
레인 `x=-3.0` 에서 0.5 m 였습니다. 로봇이 그 자리에서 감속·정지하는 사이 보행자가
1.15 m/s 로 걸어 들어와 **최근접 0.33 m** (Nova Carter 폭 ~0.5 m — 실질 충돌)가
났습니다. 로봇 후진은 0.5 m/s 라 어떤 반응형 컨트롤러도 벗어날 수 없습니다.

반경 2.0 으로 옮겨 목표를 레인에서 1.0 m 이상 떼고 측정한 결과 (150 초, 0.3 초
간격 601 샘플, 65.4 m 주행):

| 지표 | 값 |
|---|---|
| 최근접 거리 | **1.01 m** |
| `influence_radius`(2.5 m) 안 | 147 샘플 — 회피가 실제로 개입 |
| `caution_radius`(1.5 m) 안 | 9 샘플 |
| `stop_radius`(0.9 m) 안 | **0 샘플** |

사람이 로봇을 피하지 않도록(`dynamic_avoidance: false`) 설정돼 있으므로,
**목표 지점이 통행로 위에 있으면 충돌은 알고리즘이 아니라 배치의 문제**입니다.

측정 간격을 넓히면 값이 낙관적으로 나옵니다 — 같은 주행을 2 초 간격으로 재면
1.55 m 로 보입니다. 안전 여유를 확인할 때는
`--set telemetry.report_every_s=0.25` 로 촘촘히 재세요.

## Nav2

```bash
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup   # 최초 1회
sudo apt install --only-upgrade ros-humble-diagnostic-updater     # 아래 함정 1
ros2 launch launch/simlab.launch.py nav2:=true
```

검증됨: 5개 서버 전부 `active`, `NavigateToPose` 목표 `(2.5, -2.5)` 에
`status: 4` (SUCCEEDED), 로봇 실제 도착 `(2.31, -2.30)` — 허용 오차 0.3 m 이내.
도중에 보행자와 최근접 1.59 m 를 유지했습니다.

RViz2 의 **2D Goal Pose** 툴로 목표를 찍으면 `/goal_pose` 로 전달됩니다.

`nav2:=true` 면 simlab 컨트롤러 노드는 자동으로 꺼집니다 — `/cmd_vel` 발행자는
하나여야 합니다.

### 지도 없이 도는 구성

바닥 평면뿐이라 AMCL 이나 SLAM 스캔 매처가 붙잡을 지형지물이 없습니다. 그래서
[configs/nav2_params.yaml](configs/nav2_params.yaml) 은 지도 없는 구성을 씁니다:

- `map`→`odom` 은 static transform 으로 고정 (localization 대역). 씬에 실제 구조물이
  생기면 여기를 `amcl` + `map_server` 나 `slam_toolbox` 로 교체하면 됩니다.
- global/local costmap 둘 다 rolling window, `/scan` 만 먹습니다. 걸어다니는 사람이
  **움직이는 장애물**로 코스트맵에 뜹니다.
- 라이다가 로봇 자기 몸체를 ~0.4 m 에서 잡으므로 `obstacle_min_range: 0.5` 로
  잘라냅니다. 안 그러면 로봇이 스스로를 벽으로 둘러쌉니다.
- 플래너 NavFn, 컨트롤러 DWB(차동 구동: `max_vel_y: 0.0`),
  `velocity_smoother` 만 `/cmd_vel` 을 냅니다.

### Nav2 함정 1: `libdiagnostic_updater.so` 누락

`ros-humble-navigation2` 를 설치해도 `nav2_lifecycle_manager` 가 **exit 127** 로
즉사하고, 모든 서버가 `unconfigured` 에 머뭅니다. 로그에는 이 한 줄뿐입니다:

```
lifecycle_manager: error while loading shared libraries: libdiagnostic_updater.so
```

원인은 `ros-humble-diagnostic-updater` 4.0.6 이 **Python 패키지만** 담고 있고 C++
공유 라이브러리가 없다는 것입니다. Nav2 1.1.20 은 그걸 링크합니다. deb 의존성에
버전 제약이 없어서 apt 는 낡은 4.0.6 을 그대로 두고 만족합니다.

```bash
sudo apt install --only-upgrade ros-humble-diagnostic-updater   # 4.0.7 에 .so 포함
ros2 lifecycle get /bt_navigator     # active [3] 이면 정상
```

[launch/nav2.launch.py](launch/nav2.launch.py) 의 `check_dependencies()` 가 실행 전에
이 상황을 감지해 해결 명령까지 출력합니다.

### Nav2 함정 2: `plugin_lib_names` 를 손으로 줄이지 마세요

목록에서 하나만 빠져도 bt_navigator 가 활성화에 실패합니다:

```
Exception when loading BT: Error at line 13: -> Node not recognized: ComputePathThroughPoses
```

기본 behavior tree 가 쓰는 노드라 `nav2_compute_path_through_poses_action_bt_node`
가 반드시 있어야 합니다. [configs/nav2_params.yaml](configs/nav2_params.yaml) 은
`/opt/ros/humble/share/nav2_bringup/params/nav2_params.yaml` 의 47개 목록을 그대로
씁니다. 직접 추리지 말고 설치본에서 복사하세요.

### 함정 3: 남은 프로세스

Nav2 서버는 노드 이름이 고정이라, 이전 실행이 안 죽으면 두 세대가 같은 이름으로
등록되고 lifecycle manager 가 먼저 응답한 쪽과 통신하면서 bringup 이 뒤죽박죽
실패합니다. 다시 띄우기 전에:

```bash
scripts/stop_all.sh
```

`pgrep -f simlab` 같은 패턴은 **그걸 실행하는 셸 자신**까지 잡으므로 쓰지 마세요.
[scripts/stop_all.sh](scripts/stop_all.sh) 는 전체 명령줄이 아니라 `comm`
(실행 파일 이름) 으로 매칭합니다.


## 확장하기

**알고리즘 추가** — `simlab/algorithms/` 에 `DriveController` 서브클래스를 만들고
`name` 을 정한 뒤 `registry.py` 의 `CONTROLLERS` 에 등록. 그러면
`--controller <name>` 또는 `ugv.controller.name` 으로 선택됩니다.
`Observation.people_xy` 로 사람 위치가 들어오므로 회피 알고리즘도 여기서 씁니다.

**로봇 추가** — `simlab/sim/assets.py` 의 `ROBOTS` 에 `RobotSpec` 추가
(USD 경로, 휠 조인트 이름, 휠 반지름, 축간거리, 스폰 높이). 휠 이름이 안 맞으면
`*wheel*left*` / `*wheel*right*` 패턴으로 자동 탐색합니다.

**ROS 2 알고리즘 교체** — 위와 동일합니다. `ugv.controller.name` 만 바꾸면
in-process 실행과 ROS 2 노드 실행 양쪽에 똑같이 반영됩니다.

**캐릭터/경로 변경** — `configs/default.yaml` 의 `people.agents`. 사용 가능한
캐릭터 키는 `simlab/sim/assets.py` 의 `CHARACTERS` 이고, `/Isaac/People/...` 로
시작하는 전체 경로도 그대로 받습니다. 실행할 때마다 `people_commands.txt` 가
다시 생성됩니다 (직접 편집해도 덮어씌워집니다).

## 함정 1: 캐릭터가 안 움직일 때

`omni.anim.graph.core` 는 **스테이지 attach 이벤트** 때 `CharacterManager` 를
초기화합니다. standalone 스크립트는 `SimulationApp()` 이 만든 스테이지가 이미
attach 된 뒤에 애니메이션 확장을 켜기 때문에, 런타임이 그 이벤트를 놓치고 캐릭터를
하나도 등록하지 않습니다. 프림·스키마·스크립트가 전부 정상으로 보이는데 사람만
제자리에 서 있고, 로그에는 이렇게 찍힙니다:

```
CharacterManager::Shutdown() called without a prior successful call to CharacterManager::Initialize().
getCharacter - /World/Characters/... is not a SkelRoot prim or does not apply AnimationGraphAPI
```

그래서 `sim/app.py` 의 `fresh_stage()` 가 확장을 모두 켠 **직후** 스테이지를 새로
만듭니다. `cli.py` 의 `launch → enable_extensions → fresh_stage → runner` 순서를
바꾸면 사람이 멈춥니다.

또한 사람 위치는 USD `xformOp` 가 아니라 anim-graph 런타임에 있습니다. USD
트랜스폼만 읽으면 항상 스폰 좌표로 보입니다 — `Crowd.positions()` 가
`omni.anim.graph.core.get_character(...).get_world_transform(...)` 을 씁니다.

### Import 순서

`SimulationApp` 이 생기기 전에는 어떤 `omni`/`isaacsim` 모듈도 import 할 수 없습니다.
`simlab.sim.app` 과 `simlab.sim.assets` 는 함수 안에서만 import 하므로 언제든
안전하고, `simlab.sim` 의 나머지 모듈은 Kit 기동 후에만 import 됩니다.
`cli.py` 가 이 순서를 지키는 유일한 곳입니다.

## 함정 2: inotify watch 부족

Kit 이 시작할 때 확장 디렉터리마다 파일 변경 감시를 겁니다. 이 PC 는 기본 한도
65536 개를 Synology `cloud-drive-daemon`(약 5만) + VS Code(약 1.5만) 가 이미 다 쓰고
있어서, 시작 로그에 `errno=28 / No space left on device` 에러가 2000개쯤 찍힙니다.
**동작에는 문제 없지만** (확장 핫리로드만 안 됨) 시작이 느려지고 로그가 지저분합니다.

해결 (sudo 필요, 1회만):

```bash
./fix_inotify.sh
```

`/etc/sysctl.d/99-isaacsim-inotify.conf` 에 `fs.inotify.max_user_watches = 524288`
을 넣고 적용합니다.

## 기타

- navmesh 를 굽지 않았으므로 사람은 직선으로 걷습니다 (`people.navmesh: false`).
  장애물 회피가 필요하면 씬에 navmesh 를 bake 하고 `navmesh: true`,
  `dynamic_avoidance: true` 로 바꿔야 합니다.
- 종료는 GUI 창을 닫거나 `Ctrl+C`. Kit 이 SIGINT 에서 파이썬 예외를 올리지 않고
  바로 셧다운하는 경우가 있어 `[scene] done` 줄이 안 남을 수 있는데, 프로세스와
  GPU 메모리는 정상적으로 정리됩니다.
