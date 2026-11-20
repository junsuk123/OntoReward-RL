# bond/1: WMON의 관계 결합 평형을 온톨로지 R-GAT에 이식

> **명칭 안내:** 이 문서의 WMON(Wave-to-Matter Ontological Network)은 저자가 별도 연구에서 개발한 자기 항법 추정기의 내부 명칭이다. 기술적으로는 인과적 FFT 전처리, 주파수 기반 측정 신뢰도, 운동학적 일관성 잔차를 쓰는 가중 최소제곱 상태 추정기이며, 계수는 SPSA + Adam으로 보정한다. README와 그림에서는 *FFT-Assisted Kinematic Weighted Least Squares (WLS) Estimator*(약칭 Kinematic WLS)로 표기한다. 착륙 정책 연구에서는 실험 환경의 구성 요소로만 쓰며, 본 연구의 기여로 주장하지 않는다. 이 문서는 실험 기록이므로 내부 명칭과 수치를 그대로 둔다.

작성일: 2026-11-14 (KST)  
상태: **구현·검증·평가 완료.** `bond-development/1`(로컬 플랜트) 판정은 /3, /4 WMON 모두 INCONCLUSIVE(§5).
등록 arm·계약 해시는 HEAD와 동일.

## 1. 왜 이 부분인가 — WMON 요소 분리 결과

`python tools/wmon_ablation.py --epochs 300` — WMON 자체 벤치마크(MATLAB `simulateEpisode`, 학습 1–12,
검증 501–503, 시험 508–557). 각 변형을 **같은 데이터로 다시 학습**(SPSA+Adam, 재시작 2회)해 공정하게 비교했다.

| 변형 | 시험 손실 | 위치 RMSE | 결측 중 위치 |
|---|---:|---:|---:|
| full (L0 + L1-L2 + L4-L5) | 0.0133 | 0.104 | 0.106 |
| FFT 파동(L0) 제거 | 0.0125 | 0.101 | 0.102 |
| 적응 입자 신뢰도(L1-L2) 제거 | 0.0130 | 0.103 | 0.105 |
| **결합만 (L4-L5)** | **0.0126** | **0.102** | **0.104** |
| 결합(L4-L5) 제거 | 11.61 | 3.045 | 5.451 |

WMON의 이득은 거의 전부 **L4-L5 관계 결합 에너지**에서 나온다. 증거가 없는 원자(결측 위치)의 현장 가중치를
0으로 두면, 그 원자는 관련 원자(속도, 가속도)와의 결합으로 복원된다. L0과 L1-L2의 기여는 0 또는 음수다.

## 2. 이식한 이론

R-GAT의 노드 투영 `h0 = tanh(input(nodes))`과 관계 어텐션 2홉 사이에 다음 평형층을 넣는다.

```
E(x) = Σ_i c_i ‖x_i − h0_i‖² + Σ_(s −r→ t) k_r ‖x_t − a_r ⊙ x_s‖² + ε‖x‖²
x*   = argmin E  ⇔  (diag(c) + εI + Σ_e k_r u_e u_eᵀ) x = c ⊙ h0   (특징 차원마다, u_e = e_t − a_r e_s)
```

| WMON | bond/1 |
|---|---|
| 원자: 위치·속도·가속도 | 온톨로지 노드 8개 (Q G M A P O C L) |
| 결합: p−v, v−a (운동학) | 등록된 의미 간선 14개, 관계 유형 5개별 강성 k_r·변환 a_r 학습 |
| 결측 입자 가중치 0 | c_i = softplus(w_i)·(v_i + (1−v_i)·σ(ρ_i)), v_i는 등록된 유효성 규칙 |
| (GNSS 결측) | G·M·P·C: 추정 유효(vision age < 1 s) ∧ 항법 유효, O: front_valid, Q·A: 항상 |
| 항상 결측인 원자 없음 | LandingContext는 증거가 없음(c=0) → 오직 결합으로 정의 |
| 3×3 가중 최소제곱 | 8×8 SPD 선형계, 정확·미분 가능 |

- 유효하지 않은 노드는 잔여 신뢰도 σ(ρ_i)(초기 0.1)만 남는다. M·C 노드에는 항상 유효한 채널(drone_vz,
  기울기)도 섞여 있어서 0으로 고정하지 않았다.
- 이식하지 않은 것: FFT 파동(L0)과 적응 신뢰도(L1-L2). 이득이 없었고, 현재 관측 경로는 이력이 없다.
- 파라미터 예산: bond 57개 추가, actor head 66→64. 합계 4,007(ontology_rgat_ppo 4,016).

## 3. 파일

| 파일 | 내용 |
|---|---|
| `current_semantics/bond.py` | `RelationalBondLayer`, `BondGraphEncoder`, `BondActorCritic`, `node_validity`, 사양 해시 → arm `bond_rgat_ppo:92f2af3f95027afa` |
| `current_semantics/models.py` | `build_model`이 `bond_rgat_ppo:*`를 bond.py로 보냄(ORL과 같은 확장 방식) |
| `current_semantics/bond_development.py` | `bond-learnability` → `register-bond-development` → `bond-develop` |
| `run_current_semantics_3d.py` | 위 세 단계 |
| `tools/wmon_ablation.py` | §1 요소 분리 |
| `tests/current_semantics/test_bond_rgat.py` | 6개: 닫힌형 해 = 에너지 최솟값(밀집 기준·섭동), L은 결합만으로 정해짐, 무효 노드는 이웃 쪽으로 당겨짐, 노드 입력에서 유효성 판독, 예산·체크포인트 바인딩·기울기, 판정 규칙 |

## 4. 평가 프로토콜 `bond-development/1` (등록 전에 고정)

- 같은 결과 폴더의 **완료된 development/1**과 짝을 이룬다. 학습 시드 10–14, 학습 에피소드 시드, 커리큘럼,
  PPO, 검증 시드, 10개 셀의 평가 환경 시드가 모두 같다.
- 순서: G4 학습 가능성 파일럿(bond arm) 통과 → 등록 → 학습·평가.
- **주 판정:** 10개 셀 평균 성공에서 bond/1 − ontology_rgat_ppo를 학습 시드별로 짝지은 차이의 95 % t-구간.
  0보다 위면 SUPPORTED, 아래면 NOT SUPPORTED, 걸치면 INCONCLUSIVE.
- **보조(기술):** 증거 손실 셀(dropout_id/ood, combined_id/ood), unsafe 비율, 다른 arm과의 차이.
- 실행 순서: /4 WMON 폴더(항법 결측으로 증거 손실이 잦음) → /3 폴더. CPU 6–19만 쓴다(같은 PC의 Isaac
  재비행 보호, 학습 전후 Isaac 결정 주기 9.71 → 9.81 Hz로 변화 없음).

```bash
C=config/current_semantics/spatial_view_contact_v1_nav4_wmon.yaml
./run.sh current-semantics-3d --config $C --stage bond-learnability
./run.sh current-semantics-3d --config $C --stage register-bond-development
./run.sh current-semantics-3d --config $C --stage bond-develop
```

## 5. 결과

### 학습 가능성 파일럿 (g4-pilot/7, bond arm, 모델 시드 0·1)

| 폴더 | 판정 | 시험 결정론 / 확률 (시드 0 / 1) | 참고: raw 파일럿 |
|---|---|---|---|
| /4 WMON | 통과 | 0.96/0.88 · 0.62/0.65 | 0.54/0.58 · 0.54/0.50 |
| /3 | 통과 | 0.96/0.94 · 0.90/0.75 | 0.94/0.90 · 0.85/0.77 |

### bond-development/1 — /4 WMON (2026-11-14 20:11 등록, 11-15 06:43 완료, 10.5 h)

**판정: INCONCLUSIVE.** 주 지표 bond/1 − ontology_rgat_ppo(10개 셀 평균 성공) = **−0.069**,
95 % 구간 [−0.257, +0.118], 시드별 −0.107 / +0.081 / −0.311 / +0.004 / −0.014.

| 비교 (bond/1 −) | 10개 셀 | 증거 손실 셀 | unsafe |
|---|---|---|---|
| ontology_rgat_ppo | −0.069 [−0.257, +0.118] | −0.036 [−0.189, +0.117] | +0.060 [−0.042, +0.163] |
| raw_ppo | +0.001 [−0.157, +0.160] | +0.013 [−0.073, +0.099] | +0.026 |
| semantic_mlp_ppo | +0.036 [−0.084, +0.155] | +0.025 [−0.121, +0.171] | +0.018 |
| untyped_gnn_ppo | +0.124 [−0.138, +0.386] | +0.120 [−0.113, +0.353] | −0.052 |

- 셀별로 보면 bond는 거의 모든 셀에서 rgat보다 낮다(예: motion_ood 0.57 vs 0.73). 증거 손실 셀에서도 앞서지
  못했다(dropout_ood 0.64 vs 0.66, combined_ood 0.08 vs 0.07).
- 시드 12의 bond 모델이 약했다(검증 결정론 0.75, 확률 0.50; 셀 평균에서 rgat보다 −0.31).
- **해석:** WMON에서 이득을 만든 결합 평형을 그대로 옮겨도 학습된 R-GAT 정책은 좋아지지 않았다. WMON의 결합은
  실제 물리 관계(p−v, v−a)와 측정 단위로 정의되어 결측을 메운다. 반면 R-GAT의 노드 상태는 학습되는 8차원
  잠재값이라, "결합 = 이웃과 비슷해야 함" 제약이 보장해 주는 물리적 의미가 없다. 게다가 R-GAT의 관계 어텐션이
  이미 이웃 정보를 전달하고 있다.

### bond-development/1 — /3 (2026-11-15 08:36 등록, 13:44 완료, 5.1 h)

**판정: INCONCLUSIVE.** 주 지표 bond/1 − ontology_rgat_ppo(10개 셀 평균 성공) = **+0.007**,
95 % 구간 [−0.058, +0.072], 시드별 −0.041 / +0.027 / +0.025 / −0.051 / +0.075.
10개 셀 평균 성공은 bond 0.793, rgat 0.786(raw 0.756, mlp 0.772, untyped 0.756).

| 비교 (bond/1 −) | 10개 셀 | 증거 손실 셀 | unsafe |
|---|---|---|---|
| ontology_rgat_ppo | +0.007 [−0.058, +0.072] | +0.006 [−0.069, +0.081] | −0.018 [−0.113, +0.077] |
| raw_ppo | +0.037 [−0.039, +0.113] | +0.014 [−0.043, +0.070] | −0.037 |
| semantic_mlp_ppo | +0.020 [−0.018, +0.059] | +0.008 [−0.052, +0.068] | −0.025 |
| untyped_gnn_ppo | +0.037 [−0.016, +0.089] | +0.022 [−0.033, +0.076] | −0.016 |

- bond 모델 5개 모두 학습했다(검증 결정론 1.00, 확률 0.92–1.00).
- 셀별 bond − rgat 차이는 −0.008 ~ +0.024로 작다. 증거 손실 셀은 dropout_id 0.890 vs 0.896,
  dropout_ood 0.706 vs 0.692, combined_id 0.910 vs 0.906, combined_ood 0.070 vs 0.058이다.
  가장 큰 차이는 noise_ood(0.762 vs 0.738)다.
- bond는 /3의 다섯 arm 중 평균이 가장 높지만, 모든 비교 구간이 0을 포함한다.

### 종합

| 폴더 | bond − rgat (10개 셀) | 증거 손실 셀 | unsafe |
|---|---|---|---|
| /3 (참값 항법) | +0.007 [−0.058, +0.072] | +0.006 [−0.069, +0.081] | −0.018 [−0.113, +0.077] |
| /4 WMON (GNSS 결측) | −0.069 [−0.257, +0.118] | −0.036 [−0.189, +0.117] | +0.060 [−0.042, +0.163] |

- 두 폴더 모두 사전 규칙상 **INCONCLUSIVE**다. bond/1 이식이 R-GAT 정책을 개선한다는 근거는 나오지 않았다.
- 이식의 근거였던 "증거가 빠진 노드를 결합으로 메운다"는 효과가 기대되는 곳은 증거 손실 셀과 /4다.
  그런데 /3에서는 차이가 없었고, 증거 손실이 잦은 /4 WMON에서는 오히려 낮았다(시드 12 모델이 약했다).
- /3만 보면 해는 없다. 파라미터는 9개 적고(4,007 vs 4,016), 학습 안정성과 unsafe도 같은 수준이다.
- **결론:** WMON의 이득 메커니즘(물리 단위 원자 사이의 결합 에너지)은 상태 추정 단계에서 의미가 있다.
  그 자리에서는 같은 운동학을 쓰는 칼만이 더 낫다(`WMON_LIVE_NAVIGATION_KO.md`). 학습되는 잠재 노드
  상태에 그대로 옮기면 정책 성능 이득으로 이어지지 않았다.

