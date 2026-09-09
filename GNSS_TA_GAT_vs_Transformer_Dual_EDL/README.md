# GNSS Fault Detection 비교 실험
## Transformer-Dual EDL 선행 연구 vs Ontology-Guided TA-GAT

이 패키지는 다음 두 모델을 같은 데이터 분할/윈도우/라벨 조건에서 비교하기 위한 MATLAB 실험 코드입니다.

### Model A — Prior-study baseline
- 입력: 12 GNSS features × sliding window
- Window = 30 (기본값)
- d_model = 24
- Multi-head self-attention = 4 heads
- Transformer encoder block = 2
- FFN hidden = 64
- Dual EDL heads
- Hard label + soft label을 함께 사용

### Model B — Proposed
- 동일한 12개 GNSS feature 사용
- Ontology graph prior 생성
- 각 sliding window에서
  - A_cov(t): feature correlation
  - A_dyn(t): derivative-pattern RBF dependency
  를 계산
- A_t = α_ont A_ont + α_cov A_cov(t) + α_dyn A_dyn(t)
- 2-layer Graph Attention
- 비교 공정성을 위해 baseline과 동일한 Dual EDL output head 사용

> 주의:
> 이 코드는 "비교 가능한 연구 프로토타입"입니다.
> 첨부 선행연구 자료가 제시한 구조(12 features, Window 30, d_model 24,
> 4-head attention, 2 encoder blocks, FFN 64, Dual-EDL)를 최대한 반영했습니다.
> 다만 선행연구의 soft label 생성에 사용된 정확한 LIO/RIO residual 생성 코드와
> 원 데이터 전처리 코드는 자료에 포함되어 있지 않으므로,
> `soft_fault_prob` 열이 없을 때는 GT 2D error 기반 sigmoid soft label을 fallback으로 사용합니다.

## 1. 필요한 MATLAB
- MATLAB
- Deep Learning Toolbox

## 2. 데이터 준비

`data/` 폴더에 다음 세 파일을 둡니다.

- `medium.csv`
- `harsh.csv`
- `deep.csv`

필수 feature 열:

```text
numSV
hDOP
vDOP
hAcc
vAcc
gSpeed
CN0_mean
CN0_std
CN0_gap
low_elev_ratio
PR_RMS
Fault_SVID_count
```

열 이름이 조금 달라도 `load_gnss_csv`가 아래 별칭을 자동으로 정규화합니다.

```text
CNO_mean / CNO_std / CNO_gap  -> CN0_mean / CN0_std / CN0_gap  (영문 O vs 숫자 0)
meanCN0 / stdCN0              -> CN0_mean / CN0_std
fault_SVID_count              -> Fault_SVID_count
2D_error / posError2D         -> pos_error_2d
hard_label                    -> fault_label
soft_fault_prob_surrogate     -> soft_fault_prob  (cfg.useSurrogateSoftLabel = true 일 때만)
```

라벨 생성을 위해 다음 중 하나가 필요합니다.

### 권장
```text
pos_error_2d
```

- `pos_error_2d > 3 m` → hard Fault label

### 또는
```text
fault_label
```

- 0 = Normal
- 1 = Fault

soft label은 아래 우선순위로 생성합니다.

1. `soft_fault_prob` 열이 존재하면 그대로 사용
2. 없고 `pos_error_2d`가 있으면 sigmoid(error - 3m)로 생성
3. 둘 다 없으면 hard label을 soft label로 사용

## 3. 실행

MATLAB에서 이 폴더로 이동 후:

```matlab
run_comparison
```

빠른 구조 확인:

```matlab
smoke_test
```

논문용 ablation 전체 실행:

```matlab
run_ablation
```

비교 항목:
1. Transformer-Dual EDL
2. Static Ontology-GAT
3. Dynamic GAT without Ontology
4. Full Ontology-Guided TA-GAT

## 4. 기본 데이터 분할

- Train: Middle-Class Urban + Harsh Urban
- Test / Generalization: Deep Urban

이는 선행연구 자료의 UrbanNav-HK 실험 구성과 맞춘 것입니다.

## 5. 출력

`results/`:

- `comparison_metrics.csv`
- `comparison_result.mat`
- `fault_probability_comparison.png`
- `confusion_transformer.png`
- `confusion_tagat.png`

## 6. Ontology prior 수정 위치

`build_ontology_adjacency.m`

현재 prior는 사용자가 정리한 의미 그룹과 관계를 feature-level edge로 내려서 구성한
"초기 engineering prior"입니다.

예:
- Geometry: hDOP, vDOP, low_elev_ratio
- Signal Quality: CN0_mean, CN0_std, CN0_gap
- Position Accuracy: hAcc, vAcc
- Residual: PR_RMS
- Satellite Availability: numSV
- Motion: gSpeed
- Fault Evidence: Fault_SVID_count

논문 실험에서는 반드시 ablation을 권장합니다.

1. Transformer-Dual EDL
2. Static Ontology-GAT: α_ont=1, α_cov=0, α_dyn=0
3. Dynamic GAT without Ontology: α_ont=0
4. Proposed full TA-GAT

`default_config.m`의 `cfg.graph` 값을 바꾸면 됩니다.

## 7. 해석 시 가장 중요한 비교

단순 F1만 보지 말고 아래를 같이 확인하세요.

- Deep Urban F1
- Deep Urban Recall
- False Positive Rate
- 환경 전환 직후 fault probability 안정성
- Static Ontology-GAT 대비 Full TA-GAT 향상
- No-Ontology 대비 Ontology prior 향상

이렇게 해야 "온톨로지"와 "time-adaptive graph"가 각각 실제로 기여했는지 분리해서 주장할 수 있습니다.
