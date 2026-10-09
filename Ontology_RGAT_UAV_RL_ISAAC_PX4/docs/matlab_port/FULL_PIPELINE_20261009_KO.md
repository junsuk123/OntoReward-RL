# MATLAB-port 전체 파이프라인 실행 보고서

최종 Isaac 실행은 `./run.sh matlab-port-final`을 사용한다. 이 명령은 별도
2D 전체와 2D/3D local validation을 포함한 검증 전용 단계들을 생략하고 최종
3D owned Isaac/PX4 학습과 평가만 남긴다. 아래 `matlab-port-all`은
이관/회귀 검증용이다.
최종 경로의 episode는 fresh owned stack으로 격리하며, 실제 Isaac 장기 실행을
재개할 수 있도록 매 PPO update 뒤 `checkpoint_last.pt`를 기록한다.

> 이 문서는 아래에 적힌 `ce2a3e5` / `5a12d93` pin으로 수행한 역사적 결과다.
> 현재 재감사 pin은 source `6082258`, destination baseline `3fc2a9f`이며 기존
> checkpoint는 자동 승격하지 않는다. 현재 gate는 `MIGRATION_ACCEPTANCE.json`을
> 따른다.

실행일은 2026-10-09 KST다. MATLAB 원본 SHA는
`ce2a3e5e5d8e9a95158e599a8730fe0307287ece`, 대상 기준 SHA는
`5a12d937333996a47a6e605a413eb037519a69aa`다. 로컬 학습은 arm별로
750 update, update당 6 sampled episode, 25 update마다 고정 validation seed
100개를 사용했다. 최종 test는 seed 3001--3100, seed-only stress split은
9001--9200이다. 결과는 한 학습 seed이므로 표현 우월성을 주장하지 않는다.

## 게이트 결과

- MATLAB golden 패리티: PASSED. 관측, 그래프 feature, edge source/target/relation이
  모두 일치했고 관측과 그래프의 최대 절대 오차는 각각
  `1.1102230246251565e-16`이다.
- 로컬 2D/3D 학습: PASSED. 네 arm 모두 750 update와 4,500 sampled training
  episode를 완료했다.
- deterministic test 100 seed: PASSED.
- sampled test 100 seed: PASSED.
- seed-only stress split 200 seed: PASSED. 별도 바람, 센서 열화 또는 동역학
  변경은 구현되어 있지 않으므로 물리 stress 결과로 부르지 않는다.
- Isaac/PX4 direct-policy 연결: PASSED. `run.sh matlab-port-all`이 owned
  stack을 시작·정지하고 2D/3D의 PPO/R-GAT을 seed 12000으로 각 1회
  비행했다. 관측은 기존 ArUco PnP/EKF에서 왔고, policy action은
  acceleration-only gateway로 전달되었다.
- 회귀 테스트: `1527 passed, 2 skipped, 15 warnings` (`365.39 s`). 실패는 없다.

## Validation best와 test

| 차원 | arm | best update | validation 성공/unsafe | deterministic test 성공/unsafe | 평균 return |
|---|---|---:|---:|---:|---:|
| 2D | PPO | 750 | 0% / 0% | 0% / 0% | -10.82 |
| 2D | Ontology R-GAT PPO | 25 | 1% / 99% | 2% / 98% | -40.91 |
| 3D | PPO | 200 | 13% / 87% | 15% / 85% | -32.65 |
| 3D | Ontology R-GAT PPO | 400 | 73% / 27% | 69% / 31% | +5.48 |

선택 점수는 성공률, 낮은 unsafe, 평균 return 순이다. 따라서 2D R-GAT의
`1% 성공 / 99% unsafe` 체크포인트가 `0% 성공 / 0% unsafe` 체크포인트보다
선택됐다. 이 선택 규칙 때문에 2D 결과를 안전 개선으로 해석할 수 없다.

## Sampled 정책

| 차원 | arm | sampled test 성공 | sampled test unsafe | 평균 return |
|---|---|---:|---:|---:|
| 2D | PPO | 0% | 0% | -10.85 |
| 2D | Ontology R-GAT PPO | 0% | 75% | -34.62 |
| 3D | PPO | 0% | 100% | -42.74 |
| 3D | Ontology R-GAT PPO | 0% | 100% | -42.76 |

결정론적 3D R-GAT의 69% 성공은 sampled 정책에서는 0%가 된다. 실제 sampled
training episode에서도 3D 두 arm은 각각 4,500회 중 SUCCESS가 0회였다. 2D는
PPO가 1회, R-GAT이 0회였다. 따라서 이 실행은 평균 정책이 해를 표현할 수 있음을
보이지만, 현재 분산으로 PPO가 성공 표본을 안정적으로 수집한다는 증거는 아니다.

## Seed-only stress split

| 차원 | arm | 성공 | unsafe | 평균 return |
|---|---|---:|---:|---:|
| 2D | PPO | 0% | 0% | -10.88 |
| 2D | Ontology R-GAT PPO | 1% | 99% | -41.70 |
| 3D | PPO | 10% | 90% | -35.97 |
| 3D | Ontology R-GAT PPO | 77.5% | 22.5% | +11.20 |

## 실제 Isaac/PX4 direct 비행

| 차원 | arm | 결과 | step | return | emergency rewrite |
|---|---|---|---:|---:|---:|
| 2D | PPO | TASK_TIMEOUT | 667 | -5.87 | 0 |
| 2D | Ontology R-GAT PPO | TASK_TIMEOUT | 671 | -124.15 | 0 |
| 3D | PPO | SAFETY_ENVELOPE_VIOLATION | 71 | -37.86 | 0 |
| 3D | Ontology R-GAT PPO | TASK_TIMEOUT | 672 | -87.08 | 0 |

네 비행 모두 decision id가 0부터 중복 없이 증가했고, 모든 authority
record에서 `capture <= receive <= decision = command` 순서가 성립했다.
2D의 y hold는 own navigation만 사용하며, yaw는 handover 값을 유지했다.
이는 live 연결과 명령 소유권 검증이지 성능 통계가 아니다.

## 결론과 제한

로컬 단일 seed에서는 3D R-GAT 평균 정책이 일반 PPO보다 높은 deterministic
성공률을 보였고, 2D에서는 두 방법 모두 과제를 해결하지 못했다. sampled 정책은
네 arm의 sampled local 성공이 0%이고 live 비행도 arm당 1회이므로 R-GAT
우월성이나 PPO 학습 성공을 주장하지 않는다. 다음 수용 게이트는 여러
학습/비행 seed, 실제 물리 stress 분포, sampled 성공률과 안전률이다.

원시 결과는 `results/matlab_port/full_pipeline_20261009/`에 있다. 핵심 파일은
`planar_test_seed1/evaluate.json`, `spatial_test_seed1/evaluate.json`,
`planar_test_sampled_seed1/evaluate.json`, `spatial_test_sampled_seed1/evaluate.json`,
`planar_stress_seed1/evaluate.json`, `spatial_stress_seed1/evaluate.json`,
`isaac_planar/evaluate.json`, `isaac_spatial/evaluate.json`이다. 실제 연결 재실행은
`results/matlab_port/runsh_full_20261009_causal/`이고, `pipeline_report.json`,
`2d/isaac/evaluate.json`, `3d/isaac/evaluate.json`, 및 각 `isaac_traces/*.json`이
요약과 step 단위 근거다.
