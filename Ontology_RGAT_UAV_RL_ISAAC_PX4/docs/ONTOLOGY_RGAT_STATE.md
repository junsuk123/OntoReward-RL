# 온톨로지 상황 그래프를 정책의 상태로 쓰는 제안 모델

2026-09-23부터의 제안 방법이다. 온톨로지는 더 이상 보상 가중치를 설계하지도,
보상에 가산항을 더하지도 않는다. **온톨로지 그래프가 정책의 상태 표현이다.**

은퇴한 보상항 방법은 [ONTOLOGY_RGAT_FOV_RISK.md](ONTOLOGY_RGAT_FOV_RISK.md)에
그대로 남아 있고, 그 arm(`shin_se_onto_rgat_recovery`)은 지금도 실행할 수 있다.
두 방법을 한 실행에 섞는 것은 `validate_pipeline_configuration`이 거부한다 —
요인이 둘이 되기 때문이다.

## 1. 무엇을 바꿨고 왜 바꿨는가

```
기준 모델
  이미지 + proprio(7)  →  keypoint encoder → LSTM → latent
                          →  Actor / Critic

제안 모델
  같은 이미지 + proprio
        →  같은 keypoint encoder → 같은 LSTM → 같은 latent
        →  (추가) 같은 시각 의미채널 → 온톨로지 상황 그래프 G_t
                                     → R-GAT → H_t
                                     → 그래프 수준 읽기 → g_t
        →  Actor / Critic  (입력에 g_t 이어붙임)
```

이전 설계는 온톨로지를 **보상**에 넣었다. 동결된 R-GAT이 가까운 미래의 FOV
비가용 비율을 예측하고 그 스칼라를 `-λ_fov q(G_t)`로 뺐다. 두 가지 문제가
있었고 둘 다 측정으로 확인된 것이다.

1. **온톨로지의 기여를 그것이 곱해진 가중치와 분리할 수 없다.** 이미 작동하는
   5항 shaping 합에 대해 0.1짜리 항은 작은 섭동이며, 실행 결과는 귀속하기에
   너무 작은 차이거나, 손으로 고른 상수도 만들어냈을 차이다.
2. **정책은 온톨로지를 본 적이 없다.** 그래프가 담은 구조는 상황 전체를 하나로
   평균한 스칼라 벌점으로만, 한 스텝에 하나씩 actor에 닿았다.

지금은 두 학습 arm의 **보상이 완전히 같다** — 같은 5개 shaping 항, 같은
active-perception 항, 같은 종료 보상. 유일한 실험 요인은 actor와 critic이
`G_t`를 보는지 여부다. 이것은 비교가 실제로 귀속할 수 있는 요인이다.

## 2. 무엇이 동일한가

`assert_primary_baseline_equivalence()`가 실행 시점에 강제한다.

| 항목 | 구현 위치 |
|---|---|
| 보상 함수와 계수 | `reward_modes/shin2026.py: ShinReward` |
| 행동 정의와 한계 | `controllers/planar_controller.py` |
| 환경, 카메라, 외란 | `benchmarks/live_env.py`, Isaac/PX4 |
| 인식(6-keypoint encoder, 동결) | `perception/keypoint_encoder.py` |
| LSTM, latent, 보조 상태추정 | `ppo/temporal_backbone.py`, `ppo/recurrent.py` |
| 비대칭 critic의 특권 입력 | `[u_t(7), s^rel(6)]` |
| PPO 알고리즘과 하이퍼파라미터 | `ppo/recurrent_train.py`, 실험 설정 |
| 시연 집합, 학습 예산, seed | 실험 설정 |
| 평가 시나리오와 seed | `evaluation` 블록 |

같은 seed에서 두 모델의 **공유 파라미터는 비트 단위로 동일**하다. 그래프 부호기는
생성 순서상 마지막에 만들고, actor/critic의 입력층도 마지막에 만들기 때문에 —
입력층만 `graph_dim`만큼 넓어지고 나머지 draw는 전역 난수열의 같은 지점에서
나온다. `tests/test_two_pipeline_fov.py`가 텐서 단위로 확인한다.

## 3. 최소 핵심 스키마

9 node · 4 관계 · 선언 간선 12개 + node별 self-loop 9개.

```
위험 node (6)                          지원/중간/목표 (3)
  AlignmentError  ─degrades─┐
  DescentRate     ─degrades─┤
  FOVMargin       ─degrades─┼──→ TouchdownSafety ─contributes─→ SafeLanding
  RelativeRange   ─degrades─┘                ↑
  FOVMargin       ─degrades─→ PadVisibility ─supports─┘
  TargetMotion    ─degrades─→ PadVisibility ─contributes─→ SafeLanding
  MeasurementAge  ─degrades─→ PadVisibility
  MeasurementAge  ─degrades────────────────────────────→ SafeLanding
  RelativeRange   ─degrades────────────────────────────→ SafeLanding
```

이보다 더 줄이지 않는 이유는 셋이며, 모두 축소 연구가 측정한 것이다.

* **2단 구조**(위험 → 중간 → 목표). 무너뜨리면 전달할 메시지가 없다.
* **네 관계 유형**. 하나로 합치면 "관계형" 주의는 그냥 주의가 된다.
* **모든 입력 node가 목표에 도달**. 하나라도 끊기면 스키마가 특징을 조용히 버린다.

`MeasurementAge`와 `RelativeRange`만 목표에 직행하는 간선을 함께 갖는다. 두 양은
중간 개념을 거치지 않고 결과에 직접 관계하며, 축소 연구가 남긴 지름길이다.

스키마 자체는 설정이 아니라 코드에 있다(`rgat/state_graph.py`). 스키마를 바꾸면
모든 checkpoint가 무효가 되므로, 테스트가 고정할 수 있는 곳에 두었다.

### 목표 node는 항상 0

`SafeLanding`은 2단 구조가 끝나는 node이고 값은 언제나 정확히 0이다. 결과에서
유도한 무엇이든 여기에 쓰면 정답이 상태로 새어 들어간다.

## 4. 정보경계

생성자 인자는 `SemanticObservation` 하나뿐이다. 그 필드는 동결된 keypoint
encoder의 출력과 기체 자신의 proprioception에서만 나온다. 시뮬레이터 참값, 6-D
상대상태 추정, 기하 패드중심 FOV 라벨, critic 상태가 **들어올 인자 자체가
없다**. `tests/test_ontology_graph_state.py`가 서명과 구문 트리 양쪽으로 확인한다.

| node | 출처 |
| --- | --- |
| AlignmentError | keypoint centroid의 nadir 기준 bearing |
| DescentRate | proprio 수직 속도(의 안전도 채널) |
| TargetMotion | centroid 이동률 |
| FOVMargin | centroid의 프레임 경계까지 여유 |
| MeasurementAge | 마지막 신뢰 가능한 관측 이후 경과(유계) |
| RelativeRange | apparent RMS scale로 추정한 거리 |
| PadVisibility | keypoint 신뢰도 × 가시 비율, 기억항으로 보강 |
| TouchdownSafety | 위 값들의 곱 (중간 node) |
| SafeLanding | 상수 0 (목표 node) |

## 5. 상태 표현으로 쓰기 위해 필요했던 두 가지 적응

온톨로지 의미 채널은 원래 **보상 가중치 설계**용으로 만들어진 것이다. 그대로
정책의 상태로 쓰면 두 군데에서 실패한다. 아래 수정은 node·간선·관계를 그대로 둔
채 특징 행렬에서만 했다.

### 5.1 부드러운 포화

이 양들은 두 자리 수 이상의 범위를 갖는다 — 거리는 10 m에서 0.05 m까지. 잘라내기
`min(1, x/scale)`는 기준값 하나로 양 끝을 덮을 수 없다. 축소 연구의 측정:
기준을 좁게 잡으면 거리 node가 표본의 74.6 %에서 1.0에 포화해 고도가 상태에서
사실상 사라지고 정책이 상한 고도까지 올라가 한 번도 착륙하지 못했다. 넓게 잡으면
이번에는 패드 근처가 전부 0으로 뭉개졌다.

모든 크기 항은 `x / (x + scale)`을 쓴다. `scale`은 자르는 지점이 아니라 **값이
0.5가 되는 지점**이고, 어느 구간에서도 기울기가 0이 되지 않는다.

### 5.2 방향 채널

온톨로지 node 값은 전부 "위험의 크기", 즉 절댓값이다. 가중치 설계에는 크기만
있으면 되지만, 정책의 상태로 쓰면 **앞과 뒤를, 상승과 하강을 구분할 수 없다.**
구분하지 못하는 상태 위에 제어를 세울 수는 없다.

특징 행렬 5행이 `[-1, 1]` 방향 부호다. 새 물리량이 아니라 **node가 이미
나타내는 바로 그 양의 부호**이므로, node 집합·간선·관계는 바뀌지 않는다.

`PadVisibility`의 이 자리는 부호 대신 **접근 분기**를 싣는다: +1 추종, −1 탐색,
0 종말 flare. 기준 유도 법칙의 수직 명령은 두 분기에서 정반대다 — 축소 연구는
h < 1.5 m에서 추종 분기 평균 0.024, 탐색 분기 평균 2.005(거의 상승 한계)를
측정했다. 분기를 모르는 상태로는 어느 쪽도 재현할 수 없고, 그렇게 학습한 정책은
약 7초 주기의 리밋 사이클에 갇혔다. 이 정보는 기준 arm의 actor가 이미 이미지와
자신의 recurrence로 가지고 있는 것이므로 특권 정보가 아니다.

## 6. 부호기와 읽기

```
H_t   = tanh( H_1 + RGAT_2(H_1) ),   H_1 = tanh( RGAT_1(X_t) )
g_t   = tanh( W [ mean(H_t) ; max(H_t) ] + b )
actor  입력 = latent[6:] ‖ proprio(7) ‖ g_t
critic 입력 = proprio(7) ‖ s^rel(6)  ‖ g_t
```

* **모든 node를 읽는다.** 보상 readout은 1-unit 두 번째 층의 목표 node를 취하지만,
  상태에는 9개 중 8개의 node 임베딩을 버릴 이유가 없다. 축소 연구는 평균만 쓰는
  읽기는 node별 차이가 씻겨 나가고, 목표 node 읽기는 1-node 병목임을 측정했다.
* **actor와 critic이 부호기를 하나씩 가진다.** 두 망은 이미 학습률과 Adam 상태가
  분리되어 있다. 공유 부호기는 서로 다르게 스케일된 두 기울기를 한 파라미터에
  합쳐야 한다. 둘은 같은 `G_t`를 받고 같은 구조를 쓴다.
* **PPO가 함께 학습한다.** 오프라인 단계도, 산출물도, checksum 게이트도 없다.
  기울기는 PPO 목적함수에서 관계형 커널까지 이어진다. 동결하거나 `g_t`를 미리
  계산해 detach하면 그것은 다른 방법이다 — 그래프가 학습되는 표현이 아니라 고정
  특징이 된다. 설정으로 동결하려는 시도는 검증 단계에서 거부된다.

폭은 `graph_state.hidden_dim` / `graph_dim`(기본 32)이다. 축소 연구는 시드 3개로
16 대 32를 비교해 착륙률 44 % 대 56 %, 평균 점수 −109 대 −21로 32를 택했다.
32는 학습 시간이 약 두 배다. 예산이 급하면 16으로 낮추고, 낮췄다고 적으면 된다.

## 7. 제거 실험

`graph_state.representation`으로 고른다. 셋 다 **같은 부호기 클래스와 같은 PPO**를
쓴다. 구현을 분기하지 않았다.

| 값 | arm id | 내용 |
| --- | --- | --- |
| `ontology_rgat` | `shin_se_onto_rgat_state` | 관계 유형 유지 (제안 모델) |
| `gat` | `shin_se_onto_gat_state` | 간선은 그대로, 관계 유형을 하나로 합침 |
| `node_pool` | `shin_se_node_pool_state` | self-loop만: 메시지 전달 없음 |

```bash
./run.sh --pipelines shin_se_fixed shin_se_onto_gat_state
```

## 8. 검증

| 테스트 | 내용 |
| --- | --- |
| `tests/test_ontology_graph_state.py` | 스키마 최소성·도달성·목표 node 0, 정보경계(구문 트리), 부드러운 포화의 단조성, 방향 채널이 상태를 실제로 분리하는지, 분기 채널, 부호기 기울기(중앙 차분), 모든 node가 읽기에 기여, 세 표현의 위상, PPO가 부호기를 갱신하는지, **두 arm의 보상 동일성**, 저장된 `G_t` 재생, 그래프 없는 시연의 거부 |
| `tests/test_two_pipeline_fov.py` | 공유 파라미터의 비트 동일성, 단일 요인, 두 방법 혼용 거부 |
| `tests/test_planar_envelope.py` | 액션 공간과 제약 |

## 9. 주장하지 않는 것

* **온톨로지 단독 기여가 아니다.** 비교하는 것은 "R-GAT으로 부호화한 9채널 의미
  요약을 상태에 더하면 나아지는가"이지, 온톨로지라는 형식 자체의 가치가 아니다.
* **attention 가중치는 인과 근거가 아니다.** 관계 이름의 단조성도 강제되지 않는다.
* **9개 의미 채널로 상황을 요약한다는 것 자체에 상한이 있다.** 축소 연구는 교사
  모방 손실로 이 상한을 측정했다(그래프 0.0667 대 기준 관측 벡터 0.0071). 이는
  구현 결함이 아니라 방법의 성질이며, 실험이 측정하려는 대상이기도 하다.
