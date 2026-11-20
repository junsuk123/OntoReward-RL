# wmon-demo: 학습된 착륙 정책을 띄우고 RViz2·rqt로 보기

> **명칭 안내:** 이 문서의 WMON(Wave-to-Matter Ontological Network)은 저자가 별도 연구에서 개발한 자기 항법 추정기의 내부 명칭이다. 기술적으로는 인과적 FFT 전처리, 주파수 기반 측정 신뢰도, 운동학적 일관성 잔차를 쓰는 가중 최소제곱 상태 추정기이며, 계수는 SPSA + Adam으로 보정한다. README와 그림에서는 *FFT-Assisted Kinematic Weighted Least Squares (WLS) Estimator*(약칭 Kinematic WLS)로 표기한다. 착륙 정책 연구에서는 실험 환경의 구성 요소로만 쓰며, 본 연구의 기여로 주장하지 않는다. 이 문서는 실험 기록이므로 내부 명칭과 수치를 그대로 둔다.

한 명령으로 RViz2, rqt_plot 두 개, 비행을 함께 띄운다. 비행은 학습된 정책 하나를 개발 셀 조건 그대로
로컬 플랜트나 Isaac Sim + PX4에서 수행한다.

```bash
./run.sh wmon-demo                                        # /4 WMON 항법, rgat 시드 10, composition_id, 로컬
./run.sh wmon-demo --profile kalman --model 12/ontology_rgat_ppo --cell dropout_ood
./run.sh wmon-demo --profile wmon --model 10/bond --cell combined_id --episodes 5   # bond/1 arm
./run.sh wmon-demo --profile truth --model 11/raw_ppo --realtime 2                  # /3, 2배속
./run.sh wmon-demo --backend isaac --allow-isaac          # 자체 Isaac Sim + PX4 SITL + DDS + 게이트웨이
./run.sh wmon-demo --no-viewers --realtime 0              # 창 없이 최대 속도
```

| 옵션 | 뜻 |
|---|---|
| `--profile truth\|wmon\|kalman` | 관측 /3(참값 항법), /4 WMON 항법, /4 칼만 항법. 결과 폴더가 정해진다 |
| `--model SEED/ARM` | 그 폴더 development/1의 모델(`raw_ppo`, `semantic_mlp_ppo`, `untyped_gnn_ppo`, `ontology_rgat_ppo`) 또는 `SEED/bond`(bond/1) |
| `--cell` | 개발 셀 10개 중 하나(블랙아웃·패드 가속·광학 잡음 조건이 그 셀 정의 그대로) |
| `--episodes`, `--seeds` | 에피소드 수 / 환경 시드 직접 지정(기본: Isaac 개발 평가와 같은 시드) |
| `--backend local\|isaac` | 로컬 플랜트(수 초) / Isaac(`--allow-isaac` 필수, `--headless` 가능) |
| `--realtime X` | 로컬 재생 속도(1 = 실시간, 0 = 최대) |

## 화면

**RViz2** (`rviz/wmon_demo.rviz`, 고정 프레임 `world`)
- 파란 원판: 패드 추정(정책이 보는 21D의 근원). 초록 원판: 실제 패드를 "기체 − 실제 상대 위치"에 그린 것.
  두 원판의 차이가 정책이 받는 상대 위치 오차다.
- 노란 구와 선: 기체(항법 추정 위치)와 궤적. 자홍 화살표: 정책 가속 명령.
- HUD: 프로필·모델·셀, 에피소드, 시간, vision age, 항법 상태(OK/LOST), 착륙 수, 종료 판정(색상).
- 온톨로지 패널: 노드 8개, 빨강(0) → 초록(1)은 증거 가중치. bond arm은 학습된 c_i, 다른 arm은 유효성 0/1.
- Isaac 백엔드면 착륙 카메라 영상도 표시한다.

**rqt_plot 1:** `rel_pos_error`, `rel_vel_error`, `vision_age`, `nav_age`, `nav_valid`  
**rqt_plot 2:** `action/vector/{x,y,z}`, `wmon_confidence/vector/{x,y}`(WMON 항법일 때 입자 신뢰도)

모든 값은 `/wmon_demo/*`로 나간다. `ros2 topic echo /wmon_demo/telemetry`로 결정마다 JSON을 볼 수 있다.

## 시험 기록 (2026-11-15)

- **로컬:** 뷰어를 띄워 wmon / 10/ontology_rgat_ppo / dropout_id를 시험했다(RViz2·rqt 표시 확인).
  창 없이 다음 조합을 2개 에피소드씩 돌렸고 모두 끝까지 비행했다: truth 11/raw_ppo, truth
  12/semantic_mlp_ppo, kalman 12/ontology_rgat_ppo, kalman 14/untyped_gnn_ppo, wmon 10/bond, truth 13/bond.
- 없는 모델은 경로를 알려 주며 거부한다. 칼만 폴더에는 MLP 시드 10·13이 없다(학습 실패). bond는 /3과
  /4 WMON 폴더에만 있다.
- **Isaac:** 위 표의 명령 그대로 `--episodes 1`로 시험했다(Isaac GUI 포함). 시작 후 약 1분 만에 스택이 준비됐고,
  시드 994101이 SUCCESS로 끝났다(8.7 s). 같은 케이스의 Isaac 개발 평가 기록도 SUCCESS(9.3 s)다.
  - RViz2와 rqt_plot 두 개에 `/wmon_demo/*` 토픽 11개가 나왔다. 착륙 카메라 토픽
    `/landing_uav0/perception/landing_camera/{image_raw,annotated}`도 나왔다.
  - 비행이 끝나면 Isaac, PX4, DDS 에이전트, 게이트웨이가 모두 종료된다. Ctrl+C를 누르면 뷰어까지 닫힌다.

## 주의

- Isaac은 한 번에 하나만 띄운다. 다른 Isaac 평가가 돌고 있으면 `--backend isaac`는 거부된다.
  같은 ROS 도메인에서 Isaac 평가가 도는 중에는 데모를 띄우지 않는다. 2026-11-14 데모 시험 중 Isaac 재비행이
  게이트웨이 시간 초과로 한 번 멈춘 기록이 있다.
- VS Code(snap) 터미널에서 실행하면 snap의 GTK 경로 때문에 RViz2가 죽는다(`libpthread … __libc_pthread_init`).
  스크립트가 자동으로 정리한다.
- 창은 비행이 끝나도 열려 있다. Ctrl+C로 이 스크립트가 띄운 것을 모두 닫는다.
- 로그: `results/wmon_demo/logs/`.
