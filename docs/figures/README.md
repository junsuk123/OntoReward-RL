# 그림 모음

제안 정책 ORL(온톨로지를 정책 구조로 쓰는 R-GAT, `orl/1`)의 구조, 학습 신호, 검증 결과를 보여 주는 그림과,
실제 운용 환경(Isaac Sim + PX4 SITL + ROS 2)에서 찍은 시연 영상이다.

- 형식: IJCAS(International Journal of Control, Automation, and Systems) 지면에 맞춘다.
  - 지면은 2단(단 폭 80 mm, 간격 8 mm)이다. 구조·결과 그림 다섯 개는 모두 두 단 전체 폭인 168 mm로 넣는 것을 기준으로 그렸다.
  - 그 크기로 인쇄했을 때 가장 작은 글자는 7 pt, 본문 글자는 8 pt, 제목은 9.5 pt이다(본문 Times New Roman 10 pt에 맞춤).
  - 캡션은 그림 아래에 따로 쓰므로 그림 안에는 제목이나 설명 문장을 넣지 않는다.
  - PNG는 168 mm 폭에서 600 dpi이고, SVG에는 인쇄 크기(mm)를 적었다.
  - 저장할 때 인쇄 크기의 글자 크기, 글자끼리의 겹침, 상자 밖으로 넘친 글자를 자동으로 검사한다.
  - 흰 배경, 영문, 세리프 글꼴(Times 계열)이다. 관계 유형은 색과 선 모양을 함께 써서 흑백 인쇄에서도 구분된다.
- 다이어그램: 코드에서 직접 읽어 그렸다. 노드·간선·슬롯 수·파라미터 수는 `current_semantics/orl.py`(`orl/1`)에서 읽는다.
- 색 규칙:
  - 블록: 단계마다 파스텔 채움에 같은 색의 진한 테두리. 묶음은 파란 점선 상자로, 상세 패널은 점선으로 본 블록에 잇는다.
  - 결과 그림: 제안 정책만 강조색, 나머지 정책은 회색
  - 관계 유형 일곱 가지: 색맹 안전성을 검사한 범주형 팔레트(7칸)에 선 모양을 함께 써서 구분한다.

## 1. 전체 아키텍처

![overall architecture](overall_architecture.png)

- **위:** 장면 정보(UAV, 이동 패드, 카메라, 항법 상태(PX4 EKF2), 자세)가 ORL Actor를 거친다.
  - 순서: 의미 특징 생성기 φ → 개념별 노드 인코더 → 관계 그래프 어텐션 3홉(같은 층 공유) → 결정 판독(수평, 하강 게이트) → 거울 평균
  - 같은 가중치의 망을 네 가지 거울 반사 M_g φ에 돌리고, 출력을 S_g로 되돌려 평균한다(정확한 거울 등변성).
  - 원 관측 O_t가 그래프를 건너뛰어 Actor로 가는 경로는 없다. O_t는 학습 때 critic에만 들어간다.
  - 3축 순가속도는 자세·추력으로 바뀌어 PX4 OFFBOARD로 전달된다.
- **왼쪽 아래:** ORL 블록. 뒤의 겹친 상자는 같은 가중치로 도는 거울 복사본 4개를 뜻한다.
- **오른쪽:** 각 블록의 상세
  - 의미 특징 44개와 개념별 노드 입력. x·y 평면이 같은 인코더를 쓰고, y 평면은 선언된 부호로 x에 맞춘다.
  - 관계별 변환 W_r(관계 7개 + 자기 루프), 어텐션 점수, 들어오는 간선에 대한 softmax
  - 결정 판독: x·y가 공유하는 수평 MLP, 수직은 hold(Z) + σ(gate(E))·descend(Z), 그다음 거울 평균과 정책
  - Actor 2,747개, 전체 4,016개 파라미터(비교 정책과 같은 예산)

## 2. 시스템 파이프라인

![system pipeline](system_architecture.png)

Isaac Sim + Pegasus의 이동 패드와 PX4 SITL 쿼드로터에서 정책까지 이어지는 흐름은 다음과 같다.

1. ArUco 검출, 패드 추적, PX4 EKF2 자기 상태가 21채널 현재 관측 `O_t`를 만든다. 별도 항법 추정기는 쓰지 않는다.
2. 온톨로지 단계가 44개 의미 특징과 19개 노드의 입력을 만든다.
3. ORL 인코더, 결정 판독, 거울 평균이 3축 순가속도를 정한다.
4. 자세 변환기가 이를 PX4 설정값으로 바꾼다.

아래 띠는 오프라인 학습 흐름이다. 참값은 보상과 판정에만 들어가고, 원 관측 `O_t`는 critic에만 들어간다.

## 3. 온톨로지 그래프

![ontology graph](ontology_graph.png)

노드, 간선, 관계 유형은 `current_semantics/orl.py`(`orl/1`)에서 그대로 읽었다.

- 타원 = 노드 19개
  - 공유 상태 개념 3개(관측 품질, 수직 상태, 방위)
  - 평면마다 개념 6개(x 평면 위, 거울상인 y 평면 아래)
  - 결정 노드 4개(수평 결정 x·y, 하강 적격성, 수직 결정)
- 타원 오른쪽 위의 숫자 = 그 개념의 입력 슬롯 수(합 44). 결정 노드는 입력이 없고 학습되는 초기 상태에서 시작한다.
- 점선 고리 = 자기 루프. 같은 어텐션 층을 3번 적용한다.
- 두 평면은 인코더와 관계 가중치를 모두 공유한다.

## 4. 시나리오와 접근 속도장

![task and field](task_and_field.png)

- (a) 정책 인계 시점의 시나리오 추출. 패드 방위 A와 패드 진행 방향 B는 4개 구역씩이고, 16개 조합 중 4개는 시험용으로 학습에서 뺀다.
- (b), (c) 보상 퍼텐셜이 요구하는 드론−패드 상대 속도 `v_des`. `current_semantics/reward.py: desired_velocity`를 격자에서 직접 계산했다.
  - 이 속도장은 보상만 만든다. 정책 입력에는 없다.

## 5. 학습 속도 확인 실험 `learning-speed/1`

![learning speed](learning_speed.png)

- 판정 파일 `results/current_semantics_3d/spatial_current_semantics_v1/learning_speed/learning_speed.json`에서 그렸다.
- (a) 새 학습 시드 40–49에서 처음으로 최고 난이도(1.0)로 학습한 업데이트. 로그 축이고, 같은 시드는 회색 선으로 잇는다.
  - 굵은 가로선은 중앙값이다.
  - 빈 원은 상한 600 안에 도달하지 못한 모델이다(시드 45의 Untyped GNN과 PPO). 등록 규칙대로 600으로 센다.
- (b) 비교마다 T(ORL)/T(비교)의 기하평균과 97.5 % 짝지은 t 구간(Bonferroni)이다. 1보다 작으면 ORL이 업데이트를 덜 썼다는 뜻이다.
  - 채운 원은 등록된 판정이다. 두 구간 모두 1보다 작아 "ORL이 더 빠름"이다.
  - 빈 원은 시드 45를 뺀 탐색적 재계산이다. 판정에는 쓰지 않는다.

## 6. 실제 운용 환경 시연

![ORL demo flight](orl_isaac_px4_demo.gif)

![Isaac Sim, onboard camera and RViz2](isaac_rviz_operation.png)

- 화면 구성
  - 왼쪽: Isaac Sim 5.1 GUI
  - 오른쪽 위: ROS 2 `image_view`로 본 하방 카메라의 ArUco 검출 화면(`/landing_uav0/perception/landing_camera/annotated`)
  - 오른쪽 아래: RViz2. 기체를 따라가는 시점에서 기체와 패드의 궤적을 그린다.
- 비행
  - 모델은 `learning-speed/1` 시드 41의 ORL(`orl/1`) `final.pt`이다. 최고 난이도에 처음 도달한 업데이트에서 학습을 멈춘 모델이다.
  - 시연 전용 시드 999902–999904로 결정론적으로 비행했다. 두 번 띄운 스택에서 6회 모두 SUCCESS였다.
- 파일
  - GIF: 첫 비행, 4배속
  - [orl_isaac_px4_demo.mp4](orl_isaac_px4_demo.mp4): 세 비행 전체, 4배속, 51 s. 화면 아래에 비행 단계(PX4 호버, 정책 비행, 결과)를 적었다.
  - PNG: 첫 비행의 한 장면
- RViz의 궤적 선은 비행 사이의 리셋 이동까지 이어 그린다.
- 4배속이면 대략 시뮬레이션 시간 기준의 실시간이다. GUI 렌더링 때문에 시뮬레이션이 실제보다 약 3.8배 느리게 돌았다.
- 시연은 평가가 아니다. 성공률 주장에는 쓰지 않는다.
- 논문에 넣을 때: 사진(1866 × 1053 px)은 화면 캡처라 그 안의 GUI 글자 크기는 바꿀 수 없다.
  - 168 mm 폭이면 282 dpi로, IJCAS의 컬러 그림 기준(300 dpi 이상)에 조금 못 미친다.
  - 158 mm 이하 폭으로 넣으면 기준을 넘는다.

## 보관된 결과 그림

이전 단계의 결과 그림 두 개(기존 R-GAT 정책의 Kinematic WLS 항법 조건 Isaac 결과, 항법별 성능 비교)와 그 수치 파일
`figure_data.json`은 2026-11-20에 `archive/non_orl_20261120.tar.zst`로 옮겼다. 저자의 별도 연구인 추정기 조건에 기대는
결과라 이 연구의 주장에 쓰지 않는다.

## 다시 만들기

```bash
python3 Ontology_RGAT_UAV_RL_ISAAC_PX4/tools/make_doc_figures.py all       # 모든 그림
python3 Ontology_RGAT_UAV_RL_ISAAC_PX4/tools/make_doc_figures.py overall   # 하나만 (architecture, graph, field, speed도 같은 방식)
python3 Ontology_RGAT_UAV_RL_ISAAC_PX4/tools/make_doc_figures.py all --formats tif,eps,pdf --out OUT   # 투고용 파일
```

- IJCAS가 권하는 형식은 TIF, EPS, JPG이다.
- `--formats`로 600 dpi TIF(LZW 압축), EPS, PDF를 만든다. 이 파일들은 저장소에 두지 않는다.
- PDF와 EPS에는 글꼴이 들어간다.

시연 영상과 사진은 `tools/isaac_demo_capture.py`로 만든다.

- `capture`
  - Isaac Sim GUI, `image_view`, RViz2를 한 화면에 배치하고 비행 동안 화면 전체를 녹화한다.
  - 비행마다 리셋·인계·결정·종료 시각을 `timeline.json`에 남긴다.
  - Isaac 비행이므로 사용자 승인과 `--allow-isaac`가 필요하다.
- `media`
  - 녹화에서 세 창 영역만 잘라 이 폴더의 GIF, MP4, PNG를 만든다.
  - 전체 화면 녹화는 다른 창이 찍힐 수 있으므로 올리지 않는다.
