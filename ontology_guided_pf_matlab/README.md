# Schema/Ontology-Guided Adaptive Particle Filter MATLAB Demo

## 목적
일반 **고정 particle 수 Bootstrap Particle Filter**와 제안 방식인 **Schema/Ontology-Guided Adaptive Particle Filter**를 동일한 UGV localization 시나리오에서 비교합니다.

제안 방식은 다음 3가지를 추가합니다.

1. GNSS/LiDAR/Map schema feature를 이용한 규칙 기반 신뢰도 추론
2. 신뢰도에 따라 measurement covariance와 proposal distribution을 조절
3. ESS와 localization reliability에 따라 particle 수를 600~2000개 사이에서 적응적으로 조절

## 실행
MATLAB에서 이 폴더를 Current Folder로 연 뒤:

```matlab
run_compare_pf_ontology
```

별도 toolbox 없이 실행되도록 작성했으며 MATLAB R2021a 이상을 권장합니다.
기본 설정(`cfg.mcRuns = 30`, 2가지 Monte Carlo mode)에서 전체 실행에 약 2~4분 걸립니다.
빠르게 확인만 하려면 `defaultConfig()`에서 `cfg.runMonteCarlo = false`로 두면 됩니다.

## 실험 구성

### 1) 공정 비교 (fair comparison)
`cfg.fairResampling = true`(기본값)이면 **모든 variant가 동일한 resampling
threshold `ESS/N < 0.50`을 사용**합니다.
이전 버전은 baseline 0.50 / proposed 0.55였기 때문에 ESS 개선의 일부가
"제안 방식이 더 자주 resampling했기 때문"일 가능성이 있었습니다.
지금은 threshold가 같으므로 ESS 차이를 알고리즘 효과로 해석할 수 있습니다.
예전 설정을 재현하려면 `cfg.fairResampling = false`로 두면 됩니다.

### 2) Ablation study
각 행이 제안 요소를 정확히 하나씩 추가합니다.

| Variant | reliability -> R | guided proposal | adaptive N | N |
| ------- | :--------------: | :-------------: | :--------: | ---------- |
| PF-Base | X | X | X | 2000 고정 |
| PF-R | O | X | X | 2000 고정 |
| PF-RG | O | O | X | 2000 고정 |
| PF-Full | O | O | O | 600~2000 적응 |

네 variant 모두 `runPF()` 하나를 flag만 바꿔 호출하므로, 결과 차이는
측정 대상 요소에서만 발생합니다. `opt`의 세 flag가 모두 false면 코드가
정확히 고정-N bootstrap PF로 축약됩니다.

### 3) Monte Carlo
단일 run 결과는 PF의 random sampling 때문에 결론으로 쓸 수 없으므로 두 가지
모드로 각각 `cfg.mcRuns`회(기본 30) 반복합니다.

- **Mode A** — dataset 고정, filter seed만 변경. PF 자체의 sampling 난수 효과만 분리.
- **Mode B** — dataset seed와 filter seed 모두 변경. 시나리오/센서 잡음 변동까지 포함.

RMSE/P95는 `mean ± std`로, **runtime은 median으로** 보고합니다
(단일 `tic/toc`은 JIT warm-up과 CPU scheduling에 지배되므로 비교 불가).
추가로 같은 seed에 대한 **paired 차이**(dRMSE, dP95, 승률)를 출력합니다.
paired 비교는 run별 시나리오 난이도를 상쇄하므로 Statistics Toolbox 없이도
의미 있는 통계입니다.

## 생성되는 결과
`results/` 아래에 다음 파일이 생성됩니다.

### 표
- `summary_metrics.csv`: 4개 variant의 RMSE, P95, failure rate, 평균 particle 수, ESS/N(mean·p10), runtime, resample rate
- `context_metrics.csv`: OpenSky / UrbanCanyon / Tunnel / FeaturePoor별 variant 성능
- `ablation_metrics.csv`: ablation ladder와 각 요소의 한계 기여도
- `ontology_trace.csv`: **step별로 schema feature → 추론된 신뢰도 → 실제 적용된 σ / N_k / 발화한 rule**
- `monte_carlo_filterseed.csv`, `monte_carlo_datasetseed.csv` (+ `*_context.csv`)

### 그림
- `trajectory_and_efficiency.png`: 경로, 오차, particle budget, 센서 신뢰도
- `diagnostics.png`: ESS(5 s 이동평균), context별 RMSE, step time, error distribution
- `environment_3d.png`: **환경/센서 context의 3-D 시각화** — 경로×환경×시간, GNSS schema(numSV·C/N0), LiDAR schema(match score·feature count), 온톨로지 신뢰도, particle budget, 오차를 모두 주행 경로 위에 3-D로 표시
- `ontology_pipeline.png`: **스키마/온톨로지가 어떻게 적용되는지** — data-flow 다이어그램(schema → OWL/SWRL 스타일 rule → inferred property → PF 파라미터), feature→evidence 매핑 곡선, reliability→σ 매핑(covariance floor 표시), N_k 규칙의 3-D surface + 실제 동작점
- `ontology_applied.png`: **온톨로지가 실제로 한 일** — 추론된 신뢰도 시계열, step별로 설치된 measurement covariance, rule 발화 raster(R1~R10), 신뢰도→N_k 산점도
- `ablation.png`: RMSE / P95 / context별 RMSE / particle-set health(ESS·resample rate) / budget·runtime / accuracy-vs-cost
- `monte_carlo.png`: mean ± σ, paired 개선율 분포, median runtime
- `pf_comparison_results.mat`: 전체 실험 결과 (`runs`, `mc` 포함)

## 온톨로지가 코드에 적용되는 지점
`ontology_pipeline.png`의 다이어그램이 그대로 코드 구조에 대응합니다.

| 단계 | 함수 | 산출물 |
| ---- | ---- | ------ |
| Observed schema | 데이터셋 컬럼 | `numSV, hdop, cn0_mean, gnss_hAcc, lidar_score, lidar_feature_count, context` |
| Evidence 매핑 | `gnssEvidenceScores()` | svScore, hdopScore, cnoScore, haccScore |
| SWRL 스타일 rule R1~R8 | `ontologyReasoner()` | `r_G`, `r_L`, `r_Loc`, rule 발화 flag |
| Covariance floor axiom R9/R10 | `buildOntologyMeasurement()` | `R_G`, `R_L` (σ floor 0.90 m / 0.30 m) |
| Guided proposal | `runPF()` 의 guided 분기 | `q(x_k | x_{k-1}, u_k, z_k, O)` |
| Particle budget | `adaptiveParticleBudget()` | `N_k ∈ [600, 2000]` |

Rule 이름은 `ontologyRuleNames()`에 정의되어 있고 `ontology_trace.csv`의
`rule_R1 … rule_R10` 컬럼과 `ontology_applied.png`의 raster에 그대로 나타납니다.
odometry/IMU는 어느 variant에서도 온톨로지 게이팅을 받지 않으며 transition
model에 공통으로 들어갑니다.

## 데이터셋
기본 입력은 `data/synthetic_urbannav_like.csv`입니다.

이는 **실제 UrbanNav 원본 데이터가 아니라**, 공식 UrbanNav의 센서 구성과 urban-canyon failure mode를 참고해 만든 재현 가능한 가상 데이터입니다.

모델링한 특징:

- 10 Hz localization timeline
- GNSS nominal 5 Hz
- LiDAR pose / scan matching proxy 10 Hz
- OpenSky: 고신뢰 GNSS
- UrbanCanyon: 낮은 C/N0, 높은 HDOP, multipath/NLOS bias/outlier
- Tunnel: GNSS outage
- FeaturePoor: LiDAR matching score 및 feature count 저하
- Ground truth + wheel/IMU-like speed and yaw-rate controls

공식 UrbanNav Tokyo 센서 구성은 GNSS 5/10 Hz, IMU 50 Hz, LiDAR 10 Hz, Applanix ground truth 10 Hz로 공개되어 있습니다. 본 코드는 계산량을 줄이기 위해 IMU propagation을 10 Hz equivalent로 downsample한 형태입니다.

## 실제 UrbanNav 적용 시
공식 UrbanNav Tokyo/Hong Kong 데이터에는 GNSS RINEX, IMU CSV, LiDAR rosbag, ground truth/reference CSV가 포함됩니다. 실제 데이터 사용 시에는 다음 전처리가 추가로 필요합니다.

1. RINEX -> GNSS position / quality feature (`numSV`, DOP, C/N0, hAcc 등) 변환
2. LiDAR rosbag -> scan matching pose / matching score 추출
3. reference.csv -> local ENU ground truth 변환
4. 위 값을 `synthetic_urbannav_like.csv`와 동일한 column schema로 맞추기

그 뒤 filtering 부분은 그대로 재사용할 수 있습니다.

## 연구 해석 주의

### 1) 온톨로지 구현 수준
현재 MATLAB demo의 `ontologyReasoner()`는 OWL/SWRL 엔진을 직접 호출하지 않고,
**ontology에서 사용할 수 있는 rule을 MATLAB 함수로 명시적으로 구현한
proof-of-concept**입니다.

즉 이 실험에서 검증하려는 핵심은 "OWL reasoner 자체의 속도"가 아니라:

> semantic/context reliability를 particle proposal 및 particle budget에 주입했을 때 sampling 효율과 localization 성능이 개선되는가?

입니다.

논문화 단계에서는 ontology를 OWL/SWRL 또는 knowledge graph로 외부 구현하고, MATLAB에는 추론 결과(`GNSSReliability`, `LiDARReliability`, `LocalizationState`, `ParticleBudget`)만 전달하도록 분리하는 것이 좋습니다.

### 2) EnvironmentContext는 입력으로 주어진 값
`context` 컬럼은 map/environment schema에서 주어지는 것으로 가정합니다.
센서 feature만으로 context를 분류하는 단계는 아직 포함되어 있지 않으므로,
논문에서는 "환경 클래스는 사전 지도/의미 지도에서 제공된다"고 명시하거나
context 분류기를 추가해야 합니다.

### 3) 계산시간 표현
`N_k`가 55% 줄어들면서 median runtime도 함께 줄어들지만, 이는 이 MATLAB
구현(벡터화된 N×4 연산이 지배적)에서의 결과입니다.
Ontology reasoning과 guided proposal은 per-step 고정 비용이므로 N이 매우
작아지면 상대적 비중이 커집니다. runtime 주장은 반드시 median-over-runs로,
그리고 구현 환경을 명시해서 쓰는 것이 안전합니다.

### 4) R3는 이 데이터셋에서 발화하지 않음
Tunnel에서는 GNSS가 아예 없어 R1(`GnssObservation absent`)이 먼저 성립하므로
R3(`Tunnel GNSS denied`)는 도달하지 않습니다.
`ontology_applied.png` (c) 패널의 활성화 횟수에서 확인할 수 있으며, 이는
rule set이 중복(redundant)임을 보여주는 정상적인 관측입니다.
