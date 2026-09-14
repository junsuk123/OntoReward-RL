# Shin baseline 대 baseline + Ontology-R-GAT FOV 비교

## 연구 질문

완전한 Shin et al. baseline을 변경하지 않고 미래 FOV-loss 위험 보상 하나를 추가하면
표적 가시성 유지와 이동 플랫폼 착륙 성능이 향상되는가?

## Pipeline

| 계약 | `shin_se_fixed` | `shin_se_onto_rgat_fov` |
|---|---:|---:|
| 6-keypoint encoder | 동일 | 동일 |
| LSTM, 6-D relative-state estimator | 사용 | 사용 |
| position/velocity auxiliary loss | 사용 | 사용 |
| PPO actor, asymmetric critic | 동일 | 동일 |
| actor/critic observation, action | 동일 | 동일 |
| Shin 5개 shaping 항과 `[1,1,0.5,1,2]` | 동일 | 동일 |
| active-perception reward | 사용 | 사용 |
| terminal reward, curriculum, environment | 동일 | 동일 |
| Ontology-R-GAT future-FOV-loss branch | 없음 | 추가 |

실행 시 `assert_primary_baseline_equivalence()`와 YAML/spec validation이 baseline flag가
달라지면 즉시 실패한다. 두 agent model은 같은 seed에서 동일 state-dict key, shape와
초기값을 가져야 한다.

## 보상

$$r_{base}(t)=r_{task}(t)+\sum_iw_i^0\rho_i(t)+r_{active}(t)$$

$$r_{active}(t)=-0.1\,clip(L_{est}(t+1)-0.01,0,1)$$

$$r_{proposed}(t)=r_{base}(t)-\lambda_{fov}p_{fov\_loss}(t)$$

`lambda_fov`의 기본값은 0.1이며 설정 가능하다. 0이면 같은 trajectory의 두 보상은
수치적으로 동일하다. 다섯 baseline 가중치는 설정 대상이 아니다.

## 평가

주 지표는 Landing Success Rate, FOV Loss Episode Rate, FOV Retention Ratio,
Mean/Maximum Continuous FOV Loss Duration다. R-GAT은 AUROC, F1, precision, recall,
confusion matrix를 보고한다. 상태추정 오차와 loss는 양쪽 공통 진단값이며 제안 기여로
해석하지 않는다.

Attention/message-passing coefficient는 relation importance나 인과 근거로 보고하지 않는다.

