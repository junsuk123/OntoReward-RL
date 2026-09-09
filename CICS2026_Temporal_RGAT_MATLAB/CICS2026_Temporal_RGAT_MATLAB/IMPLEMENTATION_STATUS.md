# 구현 상태 및 연구 결과 사용 조건

## 구현 완료
- 선행연구 Transformer, TransEDL-Hard, TransEDL-Soft, residual soft-label, Transformer-Dual EDL 비교
- Ontology R-GAT, R-GAT + Transformer, 제안 Temporal R-GAT + Dual EDL
- Temporal R-GAT: node/relation/time-lag attention, causal lag, multi-head gating, layer gating
- Ablation: time encoding 제거, relation type 제거, 1-head, 1-layer
- 현재 고장 검출(k=0) 및 미래 고장 예측(k=1,5,10)
- Precision/Recall/F1/Accuracy/Specificity/AUROC/AUPRC/Brier/NLL/ECE
- paired bootstrap F1 차이, exact McNemar test
- EDL uncertainty 및 attention(node/relation/lag/head/layer) export
- 결과 CSV/JSON/MAT/PNG 및 MATLAB figure 출력
- 합성 demo 데이터와 strict real-data loader/importer
- 공개 UrbanNav 원자료(NMEA + raw GT)에서 12 feature와 hard label을 재구성하는 importer

## MATLAB R2025b에서 실제 실행하여 확인한 것
이 패키지는 MATLAB 25.2.0.3312555 (R2025b) Update 6에서 실행 검증했습니다.

- `check_environment` 통과 (Dirichlet KL autodiff 포함)
- `run_unit_tests` 전 항목 통과
- `run_all_demo` 전체 경로 통과
- `run_benchmark("real")` 전체 경로 통과: 12개 모델(ablation 포함) 학습, attention export,
  paired bootstrap, exact McNemar, k=1/5/10 forecast, 전체 CSV/JSON/PNG/MAT export까지
  exit code 0. 검증에는 아래 재구성 real 데이터를 사용했고 epoch 수만 줄였습니다.
- `build_urbannav_features("all")`로 공개 UrbanNav 원자료에서 3개 시나리오를 재구성.
  발표자료 진단값과 거의 일치했습니다.

  | scenario | 재구성 (epoch / fault) | 발표자료 |
  |---|---|---|
  | medium | 657 / 206 | 658 / 206 |
  | harsh | 2312 / 1562 | 2314 / 1569 |
  | deep | 1539 / 521 | 1539 / 521 |

### 실행 과정에서 수정한 결함
1. `check_environment`가 `dlgradient`를 찾지 못해 항상 실패했습니다. `dlgradient`는
   `dlarray` 메서드이므로 `exist`가 0을 반환합니다. `which` 확인을 추가했습니다.
2. EDL 손실이 학습 자체를 시작할 수 없었습니다. 내장 `gammaln`/`psi`가 `dlarray`를 거부해
   Dirichlet KL 항을 미분할 수 없었습니다. autodiff 가능한 `gammaln_dl`/`psi_dl`
   (shifted Stirling / asymptotic series, x>=1에서 오차 약 2e-12)로 대체했습니다.
3. `TemporalRGATLayer`가 `dlnetwork` 자동 초기화에서 오류를 냈습니다. 초기화 예제 입력에는
   시간 차원이 없어 `size(Xu)`가 2개 원소만 반환합니다.
4. `TemporalRGATLayer.predict`가 (batch, time, head, lag, relation) 전체 조합을 24x24
   행렬곱으로 순회해 real 설정에서는 사실상 학습이 불가능했습니다. batch/time을 batched
   matmul(`pagemtimes`)의 page로 벡터화했습니다. 원래 루프 수식과의 동등성을 별도
   reference 구현으로 3개 설정(maxLag=5/0/3, time encoding on/off)에서 확인했으며
   최대 상대오차는 1.5e-8입니다.
5. `make_dl_batch`가 매 minibatch마다 GPU를 조회했습니다. 세션당 1회로 캐시했습니다.
6. 사용 중단 예정인 `datestr(now,...)` 2곳을 `datetime`으로 교체했습니다.

### 성능
real 설정(batch 32, window 30, maxLag 5, 4 head, 2 layer)에서 Temporal R-GAT는
forward+backward 약 1.89 s/iter입니다. 50 epoch 기준 graph 계열 모델 1개당 약 2.4시간이며,
ablation과 3개 horizon forecast를 포함한 real 전체 실행은 이 장비에서 약 24-30시간
규모입니다. GPU는 사용하지 않습니다: 설치된 장치의 Compute Capability 12.0을 R2025b의
CUDA 라이브러리가 지원하지 않아 `gpuDeviceCount("available")`가 0을 반환하고 CPU로 동작합니다.

## strict / non-strict 두 경로
`build_urbannav_features`는 필요한 14개 컬럼 중 13개(12 feature + `hard_label`)를 만들지만
`soft_fault_prob`은 만들지 않습니다. 선행연구 발표자료에 residual→probability mapping이
없기 때문입니다. 재구성 table은 `data/urbannav_public/`에만 기록되고
`soft_fault_prob_surrogate`라는 다른 이름을 사용합니다.

- `run_all_real("strict")` : `data/real/`의 strict table을 요구하고, 없으면 시작을 거부합니다.
  발표자료 baseline과 비교할 결과는 반드시 이 경로로 만들어야 합니다.
- `run_all_real()` : strict table이 있으면 strict로, 없고 재구성 table이 있으면
  **non-strict 실행**으로 전환됩니다. 학습/평가/ablation/forecast/통계 설정은 strict와
  동일하고, weak head만 surrogate로 학습됩니다.

surrogate는 `cfg.allowSurrogateSoftLabel`이 참일 때만 메모리 상에서 사용되며 디스크의
`soft_fault_prob` 컬럼으로 기록되지 않습니다. 세 모드 모두 실제로 확인했습니다:
strict 거부 / demo 거부 / non-strict만 허용.

non-strict 실행이 스스로 남기는 표시: 콘솔 배너(시작·종료),
`outputs/NONSTRICT_SURROGATE_<stamp>/` 디렉터리명, 그 안의 `NOT_A_REPRODUCTION.md`,
manifest의 `strictReproduction:false` / `surrogateSoftLabel:true`, figure 제목 접미사,
그리고 reproduction gate가 해석 불가함을 알리는 warning.

**해석 규칙:** non-strict 결과를 발표자료 F1=0.950과 비교하면 안 됩니다. 반면 같은 실행
안의 모델 간 비교는 유효합니다(모든 모델이 동일 데이터·분할·supervision을 사용). 즉
architecture/ablation 결론에는 사용 가능하고, 재현 주장에는 사용할 수 없습니다.

## 논문 수치 사용 조건
선행연구 발표자료에는 원 processed 12-feature matrices, residual-to-soft-label 정확한 매핑,
전체 training hyperparameter가 제공되지 않습니다. 따라서 동봉 demo 결과는 software smoke
test일 뿐 연구 결과가 아니며, `data/urbannav_public/` 재구성 결과도 strict 재현이 아닙니다.
실제 논문 비교에는 원 processed `medium.csv`, `harsh.csv`, `deep.csv` 또는 원
preprocessing/soft-label 코드를 사용해야 합니다. baseline Transformer-Dual EDL
F1=0.950 ± 0.020 reproduction gate를 통과해야 proposed-vs-baseline 차이를 주 성능 결과로
해석하도록 설계했습니다.

---

# 2026-08-19 학습 결함 수정

이전까지 "전체 경로 통과(exit code 0)"는 확인했지만 **실제로 학습이 되는지는 확인하지
않았습니다.** 파이프라인은 끝까지 돌면서 결과표를 만들었지만, Dual-EDL 계열은 모두
확률 0.5에 붙어 있었고 Temporal R-GAT는 단일 클래스만 출력하고 있었습니다.
독립적인 결함 5개를 찾아 수정했습니다. 상세 근거는 `TRAINING_FIXES.md`,
분할 프로토콜 근거는 `SPLIT_PROTOCOL.md`에 있습니다.

| # | 결함 | 증상 | 수정 |
|---|---|---|---|
| 1 | EDL KL이 **정답 클래스 evidence까지** 벌점 | `CE+KL`의 최소점이 evidence=0 → `alpha→1`, `p→0.5`, `u→1` | `alpha_tilde = y+(1-y)*alpha` masking, expected-CE(digamma), hard head에도 anneal 적용 |
| 2 | Temporal R-GAT에 정규화 없음 | node 평균 + time 평균 2연속으로 신호 46배 감쇠 (window간 sd 0.012 vs Transformer 0.53) → 전 구간 단일 클래스 출력 | graph trunk 뒤에 `layerNormalizationLayer` 추가 |
| 3 | validation/model selection 없음 | 마지막 epoch을 그대로 보고. 겹치는 window 때문에 train loss 0.01인데 held-out은 chance | validation split + epoch별 val AUROC + best-epoch 선택 + early stopping |
| 4 | window 겹침 누수 | stride 1, W=30이라 분할 경계에서 최대 29 sample 공유 | `cfg.split.purgeWindows`로 경계 양쪽 window 제거 |
| 5 | threshold 0.5 고정 | 시간축 분할로 fault rate가 train 0.589 / val 0.331 / test 0.421로 이동 → AUROC 0.84인 모델이 F1 0.497(전부 positive) | validation에서만 F1 최대 threshold 선택 후 test에 그대로 적용 |

## 효과

- Temporal R-GAT 계열: 10 epoch val AUROC **0.36 → 0.84**
- Dual-EDL 확률 분포: `[0.4984, 0.5023]` → 실제 분포를 가지는 범위
- demo 데이터(신호가 있는 합성 데이터) 전체 파이프라인: test AUROC 0.88-0.94, F1 0.83-0.89

## 코드로 고칠 수 없는 것 두 가지

둘 다 **데이터 문제**이며, 이제는 결과표에 조용히 섞이는 대신 실행 시작 시점에
경고로 출력됩니다.

1. **평가 프로토콜.** medium+harsh로 학습해 deep으로 평가하는 기존 분할은
   leave-one-scenario-out 3개 fold 전부 chance 이하입니다 (AUROC 0.334 / 0.439 / 0.457).
   feature-fault 상관의 **부호가 시나리오마다 뒤집히기** 때문입니다
   (gSpeed: medium +0.33, deep -0.15). 이 분할에서는 어떤 구조를 써도 chance를
   넘을 수 없으므로 구조 비교에 쓸 수 없습니다.
   기본값을 `cfg.split.mode="pooled_chronological"`로 바꿨고, 기존 분할은
   `"cross_scenario"`로 남겨 두었습니다.

2. **weak label.** `soft_fault_prob_surrogate`(PR_RMS의 logistic)는 `hard_label`에 대한
   직접 예측기로 **AUROC 0.471**, 즉 chance 이하입니다. AUROC는 단조변환에 불변이므로
   스케일 조정으로도 복구되지 않습니다. 따라서 weak head는 noise만 학습하고,
   dual fusion은 hard-only head보다 **나빠집니다** (측정값: Transformer-DualEDL AUROC
   0.571 vs TransEDL-Hard 0.682).
   `report_label_quality.m`이 이를 매 실행 앞에서 출력하고 경고합니다.
   `cfg.weakLabel.source`로 `"pr_rms"`(정직하지만 무의미) / `"pos_error"`(선행연구가
   실제로 한 것으로 보이는 방식이지만 `hard_label`을 정의하는 바로 그 값에서 만들어져
   **순환적**, 구성상 AUROC 1.000)를 선택할 수 있습니다. **둘 다 재현 주장 근거가 될 수
   없습니다.**

선행연구의 SoftLabel baseline이 F1 0.959로 보고된 것은, 그 soft label 자체가
positioning error에서 유도되었음을 강하게 시사합니다.

## weak label을 `pos_error`로 전환 (2026-08-19)

`cfg.weakLabel.source = "pos_error"`가 non-strict 기본값입니다.

```
soft_fault_prob = sigmoid((pos_error_2d - 3.0 m) / 1.0)
hard_label      = pos_error_2d >= 3.0 m
```

**정당하게 얻는 것:** hard label은 3 m 경계를 넘었는지만 남기고 3.1 m와 30 m를 같게
취급합니다. soft label은 **오차 크기**를 보존하므로 weak head가 "얼마나 나빴는지"를
학습합니다. 선행연구도 이 방식이었을 가능성이 높습니다 — SoftLabel baseline(모델 없이
soft label을 그대로 예측으로 쓰는 것)이 F1 0.959로 보고됐다는 건 그 label이 같은
ground-truth error에서 유도됐다는 뜻입니다.

논문에서는 **"ground-truth error magnitude에 대한 auxiliary regression"** 또는
**"error threshold에 대한 label smoothing"**으로 기술해야 합니다.

**주의해야 할 것 (반드시 유지):**

1. `SoftLabel` baseline은 이제 구성상 **F1 1.000**입니다. tau=1.0에서 `soft-0.5`의 부호가
   `hard_label`과 완전히 같기 때문입니다. 이 행은 정합성 검사이지 경쟁 baseline이
   아닙니다. 결과표에 "이겨야 할 대상"으로 넣으면 안 됩니다.
2. `report_label_quality`가 AUROC 1.000과 `cics:circularWeakLabelQuality` 경고를
   출력합니다. **이 경고는 그대로 두어야 합니다.**
3. 이 설정의 Dual-EDL 결과는 "weak head가 저렴한/독립적인 출처에서 정보를 더한다"는
   주장을 뒷받침하지 **않습니다**. 뒷받침하는 것은 "이진 label과 함께 오차 크기를
   supervise하면 도움이 된다"는 label 설계 결과입니다.
4. reproduction gate는 여전히 해석 불가입니다.

**기각한 대안:** `"pr_rms"`는 ground truth와 독립이지만 AUROC 0.471로 무의미하며,
실측 결과 dual fusion을 악화시켰습니다 (TransEDL-Hard 0.760 → Transformer-DualEDL 0.661).
두 선택지가 정반대 방향으로 실패합니다 — 하나는 독립이지만 신호가 없고, 다른 하나는
신호가 있지만 독립이 아닙니다. **어느 쪽도 F1=0.950 재현 주장의 근거가 될 수 없습니다.**
자세한 내용은 `WEAK_LABEL.md`.
