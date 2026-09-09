# CICS2026 Temporal R-GAT MATLAB Package

## 목적
선행연구의 **Transformer-Dual EDL** GNSS 고장검출 구조를 기준선으로 재현하고, 제안 구조인 **Ontology-constrained Temporal R-GAT + Dual EDL**을 동일 데이터/동일 분할에서 비교 검증하기 위한 MATLAB 패키지입니다.

핵심 비교:

1. Transformer (Supervised only)
2. Transformer-EDL (Hard label only)
3. Transformer-EDL (Soft label only)
4. Residual-based Soft Label (통계 모델 출력 자체)
5. Transformer-Dual EDL (선행연구 핵심 baseline)
6. R-GAT-Dual EDL (시간축 미확장 ablation)
7. R-GAT + Transformer + Dual EDL (중간 비교)
8. **Temporal R-GAT + Dual EDL (제안)**

추가로 미래 고장 예측 horizon `k=1,5,10`에서 Transformer-Dual EDL과 Temporal R-GAT-Dual EDL을 비교할 수 있습니다.

---

## 가장 먼저 실행

MATLAB R2025b + Deep Learning Toolbox 기준:

```matlab
cd <이 폴더>
check_environment
run_all_demo
```

`run_all_demo`는 먼저 데이터/EDL/Transformer/Temporal R-GAT forward-shape 단위 테스트를 수행한 뒤, 동봉된 합성 데이터로 학습-평가-통계-시각화-export 전체가 연결되는지 확인하는 **software smoke test**입니다. 이 결과를 논문 수치로 사용하면 안 됩니다.

실제 선행연구 비교는 다음 파일을 준비한 뒤 수행합니다.

```text
data/real/medium.csv
data/real/harsh.csv
data/real/deep.csv
```

그 후:

```matlab
run_all_real            % strict 데이터가 있으면 strict 실행
run_all_real("strict")  % strict 전용. 없으면 시작을 거부
```

### strict 데이터가 없을 때

`run_all_real`을 인자 없이 실행하면, `data/real/`이 비어 있고
`data/urbannav_public/` 재구성 결과가 있는 경우 **명시적으로 non-strict로 표시된 실행**으로
자동 전환됩니다. 학습/평가/ablation/forecast/통계는 strict와 완전히 동일하며(50 epoch,
bootstrap 2000), 차이는 weak head가 `soft_fault_prob_surrogate`(PR_RMS의 logistic 함수)로
학습된다는 점 하나입니다.

이 실행이 남기는 표시:

- 콘솔 배너가 실행 시작과 종료 시점에 출력됩니다
- 결과는 `outputs/NONSTRICT_SURROGATE_<stamp>/`에 기록됩니다
- 그 폴더에 `NOT_A_REPRODUCTION.md`가 함께 기록됩니다
- `run_manifest.json`에 `strictReproduction: false`, `surrogateSoftLabel: true`
- figure 제목에 `[NON-STRICT: surrogate soft label]` 접미사

**해석 규칙:** 이 폴더의 수치를 발표자료의 F1=0.950 baseline과 비교하면 안 되며,
reproduction gate 결과도 이 모드에서는 의미가 없습니다. 반면 같은 폴더 안의
**모델 간 비교는 유효합니다** — 모든 모델이 동일한 데이터/분할/supervision을 봤기 때문입니다.
따라서 architecture 및 ablation 결론에는 쓸 수 있고, 재현 주장에는 쓸 수 없습니다.

---

## 실제 데이터 계약
각 CSV는 1 epoch = 1 row이며 다음 12개 feature가 필요합니다.

```text
numSV
hDOP
vDOP
hAcc
vAcc
gSpeed
CNO_mean
CNO_std
CNO_gap
low_elev_ratio
PR_RMS
Fault_SVID_count
```

그리고 다음 supervision이 필요합니다.

```text
hard_label          % 0 normal / 1 fault
soft_fault_prob     % [0,1]
```

`hard_label`이 없고 `pos_error_2d`가 있으면 `pos_error_2d >= 3 m`로 생성합니다.

**중요:** 선행연구 발표자료에는 원본 processed 12-feature matrix, residual→soft label 정확한 mapping, 전체 학습 hyperparameter가 공개되어 있지 않습니다. 따라서 strict mode는 `soft_fault_prob`이 없으면 실패하도록 설계했습니다. 임의의 soft label을 만들어 선행연구 재현으로 가장하지 않습니다.

---

## 데이터 분할
기본 설정은 각 시나리오 안에서 시간 순서를 보존하는 pooled chronological 분할입니다.
각 시나리오를 train/validation/test 블록으로 나누며, 겹치는 window가 경계를 넘어
누수되지 않도록 `purgeWindows = 30`을 적용합니다.

```text
모든 시나리오: chronological train/validation/test
Window = 30
Purge   = 30 windows at each boundary
```

Hard fault label 기준은 GT 기반 GNSS 2-D 위치오차 `>= 3 m`입니다.

발표자료의 기존 프로토콜인 `medium + harsh -> deep`는 `cfg.split.mode =
"cross_scenario"`로 명시적으로 선택할 수 있습니다. 이 모드에서는 시나리오별
feature-fault 상관의 방향이 달라 현재 데이터에서 세 leave-one-scenario-out
평가가 chance 이하였으므로, 구조 간 성능 비교의 기본 프로토콜로 사용하지 않습니다.

---

## 제안 모델

```text
12 Features × Window
        ↓
Ontology-Temporal Graph
        ↓
Temporal R-GAT
  - Node/Relation attention
  - Time-lag attention
  - Multi-head
  - Layer gating
        ↓
Dual EDL
  - Hard label head
  - Soft label head
        ↓
Dirichlet evidence fusion
        ↓
Fault probability + uncertainty
```

기존 R-GAT의 attention domain을 `node-relation`에서 `node-relation-time lag`로 확장합니다.

개념적으로:

```text
alpha_ijr  →  alpha_ijrΔ
```

여기서 `Δ=0...MaxLag`이며, 미래 정보가 현재 판단에 들어가지 않도록 항상 causal 조건만 사용합니다.

---

## 결과 파일
실행 후 `outputs/<run_name>/`에 생성됩니다.

- `metrics.csv`
- `paired_bootstrap_vs_prior.csv`
- `mcnemar_vs_prior.csv`
- `forecast_metrics.csv` (forecast 사용 시)
- `predictions_*.csv`
- `training_history_*.csv`
- `attention_node.csv`
- `attention_relation.csv`
- `attention_time_lag.csv`
- `attention_head.csv`
- `attention_layer.csv`
- `benchmark_summary.png`
- `fault_probability_timeseries.png`
- `attention_summary.png`
- `run_manifest.json`
- `models/*.mat`

모든 figure는 실행 시 MATLAB figure 창으로도 바로 표시됩니다.

---

## 재현 gate
선행연구 발표자료의 Proposed Dual EDL F1 = **0.950**을 기준으로 기본 허용오차 ±0.02를 사용합니다.

실제 processed data로 Transformer-Dual EDL baseline이 이 범위를 재현하지 못하면:

```text
baseline_reproduction_pass = false
```

로 기록되며, proposed-vs-baseline 차이를 논문 성능개선으로 해석하지 않도록 경고합니다.

---

## UrbanNav-HK 원자료
공식 데이터셋은 `IPNL-POLYU/UrbanNavDataset`입니다. Medium/Deep/Harsh 전체 ROS 데이터는 매우 크므로 이 패키지에 포함하지 않습니다. GNSS/GT 자료 다운로드는:

```matlab
open_urbannav_dataset
```

을 실행해 공식 페이지에서 받으세요.

선행연구의 정확한 processed table이 있다면 그것을 우선 사용해야 합니다. 원자료에서 새로 feature를 재구성하면 **동일 데이터셋을 사용하더라도 선행연구의 preprocessing 재현과는 별개**입니다.

### 원자료에서 12-feature 재구성

공개 UrbanNav 원자료(u-blox F9P NMEA + raw GT)로부터 12 feature와 `hard_label`을 다시 만들 수 있습니다.

```matlab
build_urbannav_features("all")                            % data/raw 사용
build_urbannav_features("all",rawRoot="D:\UrbanNav\raw")  % 다른 위치 사용
```

기대하는 원자료 배치:

```text
<rawRoot>/medium/gnss/UrbanNav-HK-Medium-Urban-1.ublox.f9p.nmea
<rawRoot>/medium/UrbanNav_TST_GT_raw.txt
<rawRoot>/harsh/...   <rawRoot>/deep/...
```

결과는 `data/urbannav_public/`에 기록되며 **strict slot인 `data/real/`에는 절대 쓰지 않습니다**. 14개 필요 컬럼 중 13개(12 feature + `hard_label`)를 만들지만 `soft_fault_prob`은 만들지 않습니다. 발표자료에 residual→probability mapping이 없기 때문이며, `soft_fault_prob_surrogate`라는 다른 이름으로만 기록해 strict loader가 이를 진짜 supervision으로 받아들이지 못하게 합니다. 따라서 strict gate는 이 table을 계속 거부하며, `run_all_real("strict")`도 계속 거부합니다. 인자 없는 `run_all_real`은 위에서 설명한 non-strict 실행으로 전환됩니다.

검증(이 저장소에서 실제 실행): 재구성된 epoch/fault 수는 발표자료 진단값과 거의 일치합니다.

| scenario | 재구성 (epoch / fault) | 발표자료 |
|---|---|---|
| medium | 657 / 206 | 658 / 206 |
| harsh | 2312 / 1562 | 2314 / 1569 |
| deep | 1539 / 521 | 1539 / 521 |

발표자료에 명시되지 않은 정의(low-elevation 임계, RAIM residual 임계, `CNO_gap` 정의 등)는 `data/urbannav_public/PUBLIC_RECONSTRUCTION_NOTICE.md`와 `reconstruction_manifest.json`에 기록됩니다. 논문에 쓸 때 반드시 이 값들을 다시 명시하세요.


---

## 구현 검증 상태
자세한 제한사항과 논문 수치 사용 조건은 `IMPLEMENTATION_STATUS.md`를 확인하세요.
