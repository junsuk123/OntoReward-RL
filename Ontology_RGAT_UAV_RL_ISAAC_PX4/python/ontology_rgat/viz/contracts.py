"""Structured descriptions of the run, the proposed algorithm and the two MDPs.

These feed the dashboard's three explanatory panels. They are built from the
executable constants and the resolved configuration rather than written out by
hand, so a diagram cannot quietly drift away from what the code does: rename a
graph node or change lambda and the panel changes with it.

Nothing here reports a measurement. Live status is attached by the dashboard
from the run's own stage/phase telemetry.
"""
from __future__ import annotations

from typing import Any, Mapping

from ..controllers import PLANAR_ACTION_DIM
from ..rgat.fov_graph import (FOV_FEATURE_NAMES, FOV_GOAL_NODE,
                              FOV_GRAPH_EDGES, FOV_GRAPH_INPUT_DIM,
                              FOV_NODE_NAMES, FOV_RELATION_NAMES)
from ..rgat.state_graph import (STATE_GRAPH_EDGES, STATE_GRAPH_INPUT_DIM,
                                STATE_NODE_NAMES, STATE_RELATION_NAMES,
                                STATE_RISK_NODES)


# Stage ids are matched against the live ``stage.name`` published by the
# runner, so they must stay in step with ``monitor.stage(...)`` calls.
RUN_PIPELINE_STAGES: tuple[dict[str, Any], ...] = (
    {"id": "stack startup", "title": "스택 기동",
     "detail": "DDS · Isaac Sim · Pegasus · PX4 · gateway",
     "produces": "두 UAV/UGV pair", "kind": "infra"},
    {"id": "keypoint validation", "title": "인식 검증",
     "detail": "실 렌더 카메라 · held-out 라벨",
     "produces": "픽셀 오차 · PCK · 가시성 정확도", "kind": "gate"},
    {"id": "behavior cloning", "title": "행동 복제 워밍업",
     "detail": "두 arm 공통 teacher (프로필에 따라 생략)",
     "produces": "동일 초기 checkpoint", "kind": "train", "optional": True},
    {"id": "FOV-risk data", "title": "FOV 데이터 수집",
     "detail": "은퇴한 보상항 경로 전용 · 상태표현 실험에서는 건너뛴다",
     "produces": "episode 분리 데이터셋", "kind": "data", "optional": True},
    {"id": "collection complete", "title": "수집 단계 종료",
     "detail": "--stage collect 로 끝낸 실행에서만 표시",
     "produces": "재사용 가능한 데이터셋", "kind": "data", "optional": True},
    {"id": "R-GAT training", "title": "R-GAT 학습·동결",
     "detail": "은퇴한 보상항 경로 전용 · 상태표현의 R-GAT은 PPO가 함께 학습한다",
     "produces": "동결 스칼라 readout", "kind": "train", "optional": True},
    {"id": "parallel training", "title": "PPO 학습 (모든 arm)",
     "detail": "동결 readout 완성 후 동시 시작 · 동일 예산",
     "produces": "arm별 정책", "kind": "train"},
    {"id": "deterministic checkpoint validation", "title": "checkpoint 선택",
     "detail": "training-best 대 latest · 분리된 seed",
     "produces": "arm별 평가 대상 1개", "kind": "gate"},
    {"id": "paired evaluation", "title": "짝지은 평가",
     "detail": "동일 seed · 동일 event · 정책만 교차",
     "produces": "per-episode 기록", "kind": "eval"},
    {"id": "complete", "title": "보고",
     "detail": "실제 로그만으로 표·그림 생성",
     "produces": "결과 번들", "kind": "report"},
)

# Live stage names that mean the same step as a canonical id above.
RUN_PIPELINE_ALIASES: Mapping[str, str] = {
    "training-only teacher demonstrations": "behavior cloning",
    # One PPO box, not one per arm. ``training`` and ``parallel training`` are
    # the same step under one and many pairs, and since the stages were
    # separated (2026-09-22) every arm starts it together against the finished
    # reward design -- there is no longer an early arm to draw ahead of the
    # collection.
    "training": "parallel training",
    "parallel training preparation": "parallel training",
    "parallel crossover evaluation": "paired evaluation",
    "reporting": "complete",
}

# Stage names published only by the retired adaptive-weight / PBRS runners.
# They cannot occur in the two-pipeline experiment and are deliberately absent
# from the track rather than drawn as stages that never light up.
NON_PRIMARY_STAGES: frozenset[str] = frozenset({
    "R-GAT data", "design-source training", "reward_design",
    "adaptive reward data", "adaptive R-GAT training",
    "adaptive R-GAT retraining",
})


def run_pipeline_contract() -> dict[str, Any]:
    return {
        "stages": [dict(stage) for stage in RUN_PIPELINE_STAGES],
        "aliases": dict(RUN_PIPELINE_ALIASES),
        "note": ("현재 방법에서 온톨로지 그래프는 정책의 상태 표현이므로 "
                 "오프라인 보상 설계 단계가 없다. FOV 데이터 수집과 R-GAT "
                 "동결은 은퇴한 보상항 경로에서만 켜진다. 두 학습 arm은 같은 "
                 "시연 집합, 같은 PPO 예산, 같은 seed를 쓴다."),
    }


def algorithm_pipeline_contract(*, lambda_fov: float, horizon_seconds: float,
                                control_hz: float, hidden_dim: int = 24,
                                camera: Mapping[str, Any] | None = None
                                ) -> dict[str, Any]:
    """Sensor → ontology graph → R-GAT → reward, with the offline branch."""
    camera = dict(camera or {})
    resolution = list(camera.get("resolution", (512, 320)))
    fov_deg = float(camera.get("horizontal_fov_deg", 90.0))
    pitch_deg = float(camera.get("pitch_down_deg", 60.0))
    steps = max(1, int(round(float(horizon_seconds) * float(control_hz))))
    return {
        "chain": [
            {"id": "sensor", "title": "센서", "lines": [
                f"grayscale {int(resolution[0])}×{int(resolution[1])}",
                f"수평 FOV {fov_deg:g}° · 하향 {pitch_deg:g}°",
                f"{float(control_hz):g} Hz"]},
            {"id": "perception", "title": "인식", "lines": [
                "동결 6-keypoint encoder",
                "keypoint 좌표 + 가시성",
                "3D pose 복원 없음"]},
            {"id": "features", "title": "온톨로지 특징", "lines": [
                f"{len(FOV_FEATURE_NAMES)}개 시각·이력 특징",
                "missingness · age 포함",
                "privileged 입력 없음"]},
            {"id": "graph", "title": "온톨로지 그래프", "lines": [
                f"{len(FOV_NODE_NAMES)} node · {len(FOV_RELATION_NAMES)} relation",
                f"선언 edge {len(FOV_GRAPH_EDGES)} + self-loop",
                "모든 입력이 출력에 도달"]},
            {"id": "rgat", "title": "R-GAT", "lines": [
                f"RGAT({FOV_GRAPH_INPUT_DIM} → {int(hidden_dim)}) · tanh",
                f"RGAT({int(hidden_dim)} → 1) · sigmoid",
                "별도 readout head 없음"]},
            {"id": "scalar", "title": "스칼라 q", "lines": [
                f"{FOV_GOAL_NODE} node",
                f"향후 {steps} step({float(horizon_seconds):g}s)",
                "FOV 밖 시간 비율 기댓값"]},
            {"id": "reward", "title": "보상", "lines": [
                "r_paper(t) − λ·q(G_t+1)",
                f"λ = {float(lambda_fov):g}",
                "종료 보상은 원문 그대로"]},
        ],
        "offline": {
            "title": "오프라인 지도 (학습 전용)",
            "lines": [
                "시뮬레이터 기하: 패드 중심 frustum 가시성",
                f"y_t = Σ(1 − V[t+k]) / {steps}",
                "미관측 미래창은 mask · 0으로 채우지 않음"],
            "into": "rgat",
        },
        # How many leading chain stages survive deployment. The diagram
        # colours by this, so it must agree with ``deployment.drops``.
        "deployed_stages": 2,
        "deployment": {
            "keeps": ["인식", "LSTM", "actor", "공통 제어기"],
            "drops": ["온톨로지", "R-GAT", "보상", "critic"],
        },
        "limits": [
            "PBRS가 아니므로 최적정책 불변은 따라오지 않는다",
            "q는 시간 비율이며 이진 손실 확률이 아니다",
            "attention 자체는 인과 근거가 아니다",
        ],
    }


def graph_state_pipeline_contract(*, control_hz: float, hidden_dim: int = 32,
                                  graph_dim: int = 32,
                                  representation: str = "ontology_rgat",
                                  camera: Mapping[str, Any] | None = None
                                  ) -> dict[str, Any]:
    """Sensor -> ontology situation graph -> R-GAT -> policy state.

    The counterpart of :func:`algorithm_pipeline_contract` for the current
    method. Two differences are the whole point and are drawn as such: the
    chain ends at the actor rather than at the reward, and nothing in it is
    frozen -- PPO trains the encoder along with the policy.
    """
    camera = dict(camera or {})
    resolution = list(camera.get("resolution", (512, 320)))
    fov_deg = float(camera.get("horizontal_fov_deg", 90.0))
    pitch_deg = float(camera.get("pitch_down_deg", 60.0))
    relation_count = (len(STATE_RELATION_NAMES)
                      if representation == "ontology_rgat" else 1)
    edge_count = (len(STATE_GRAPH_EDGES) + len(STATE_NODE_NAMES)
                  if representation != "node_pool" else len(STATE_NODE_NAMES))
    return {
        "chain": [
            {"id": "sensor", "title": "센서", "lines": [
                f"grayscale {int(resolution[0])}×{int(resolution[1])}",
                f"수평 FOV {fov_deg:g}° · 하향 {pitch_deg:g}°",
                f"{float(control_hz):g} Hz"]},
            {"id": "perception", "title": "인식", "lines": [
                "동결 6-keypoint encoder",
                "keypoint 좌표 + 가시성",
                "3D pose 복원 없음"]},
            {"id": "features", "title": "의미 채널", "lines": [
                f"위험 node {len(STATE_RISK_NODES)}개 + 지원·중간·목표",
                "부드러운 포화 x/(x+scale)",
                "privileged 입력 없음"]},
            {"id": "graph", "title": "상황 그래프 G_t", "lines": [
                f"{len(STATE_NODE_NAMES)} node · {relation_count} relation",
                f"edge {edge_count} (self-loop 포함)",
                f"node 특징 {STATE_GRAPH_INPUT_DIM}차원 (방향 부호 포함)"]},
            {"id": "rgat", "title": "R-GAT 부호기", "lines": [
                f"RGAT({STATE_GRAPH_INPUT_DIM} → {int(hidden_dim)}) · tanh",
                f"RGAT({int(hidden_dim)} → {int(hidden_dim)}) · 잔차",
                "actor/critic 각각 하나씩 · PPO가 함께 학습"]},
            {"id": "readout", "title": "그래프 수준 읽기 g_t", "lines": [
                "tanh(W [mean(H) ; max(H)] + b)",
                f"{int(graph_dim)}차원",
                "특정 node를 고르지 않는다"]},
            {"id": "policy", "title": "정책 입력", "lines": [
                "actor: latent[6:] + proprio(7) + g_t",
                "critic: proprio(7) + 참 상대상태(6) + g_t",
                "보상에는 전혀 들어가지 않는다"]},
        ],
        "offline": None,
        # Every stage of this chain is deployed. There is no offline artifact
        # to drop and no reward-side branch to leave behind.
        "deployed_stages": 7,
        "deployment": {
            "keeps": ["인식", "의미 채널", "그래프", "R-GAT", "읽기", "actor"],
            "drops": ["critic"],
        },
        "limits": [
            "그래프는 상태 표현이며 보상을 바꾸지 않는다",
            "9개 의미 채널로 상황을 요약하는 표현의 상한이 존재한다",
            "attention 자체는 인과 근거가 아니다",
        ],
    }


def mdp_contract(*, control: Mapping[str, Any], reward: Mapping[str, Any],
                 lambda_fov: float, horizon_seconds: float,
                 pad_speed_range: tuple[float, float] | None = None
                 ) -> dict[str, Any]:
    """Row-by-row S/A/E/R for both arms, marking what is identical."""
    control = dict(control or {})
    reward = dict(reward or {})
    terminal = dict(reward.get("terminal") or {})
    active = dict(reward.get("active_perception") or {})
    dt = float(control.get("dt_seconds", 0.1))
    horizon = int(control.get("horizon_steps", 300))
    limit = list(control.get("max_velocity_m_s", (1.6, 0.9)))
    accel = list(control.get("max_acceleration_m_s2", (1.2, 0.8)))
    tilt = float(control.get("max_longitudinal_tilt_deg", 12.0))
    if len(limit) == 3:
        limit = [limit[0], limit[2]]
    if len(accel) == 3:
        accel = [accel[0], accel[2]]
    speed = tuple(pad_speed_range or (0.0, 0.0))
    shared = [
        {"key": "observation", "label": "Observation (S)", "identical": False,
         "value": "grayscale 이미지 + body velocity(3) + quaternion(4)",
         "delta": "여기에 온톨로지 상황 그래프 G_t 를 더한다 (유일한 실험 요인)",
         "note": "정책은 event trigger·참 가속도·미래 궤적·기하 가시성 라벨을 받지 않는다"},
        {"key": "latent", "label": "내부 표현", "identical": False,
         "value": "6-keypoint CNN → LSTM h[512] → latent y[256]",
         "delta": "R-GAT(G_t) → mean+max 읽기 → g_t 를 actor·critic 입력에 이어붙인다",
         "note": "actor는 y[6:256]+proprio(+g_t), y[0:6]은 보조 상태추정"},
        {"key": "action", "label": "Action (A)", "identical": True,
         "value": f"평면 {PLANAR_ACTION_DIM}채널 [a_fwd, a_z, tilt]",
         "note": (f"|a| ≤ {accel[0]:g}/{accel[1]:g} m/s² · "
                  f"|v| ≤ {limit[0]:g}/{limit[1]:g} m/s · "
                  f"|tilt| ≤ {tilt:g}° · {dt:g} s 주기 · "
                  "횡방향 속도와 요레이트는 항상 0")},
        {"key": "environment", "label": "Environment (E)", "identical": True,
         "value": (f"Isaac Sim + PX4 · 이동 패드 {speed[0]:g}–{speed[1]:g} m/s"
                   if speed[1] else "Isaac Sim + PX4 · 이동 패드"),
         "note": f"horizon {horizon} step · 동일 seed·동일 외란·동일 카메라"},
        {"key": "reward", "label": "Reward (R)", "identical": True,
         "value": "원문 Table III 5개 항 + active perception "
                  f"(α={float(active.get('alpha', 0.1)):g}, "
                  f"β={float(active.get('beta', 1.0)):g}, "
                  f"τ={float(active.get('tau', 0.01)):g})",
         "note": ("두 arm이 같은 보상을 쓴다. 5번째 항은 요레이트 채널이 없어져 "
                  "같은 가중치의 −2|tilt| 로 대체되었다. 온톨로지는 보상에 "
                  "관여하지 않는다")},
        {"key": "terminal", "label": "종료 보상", "identical": True,
         "value": (f"성공 +{float(terminal.get('success', 10.0)):g} · "
                   f"충돌/과도 이탈 {float(terminal.get('failure', -10.0)):g}"),
         "note": "종료 보상은 shaping을 대체한다"},
        {"key": "termination", "label": "종료 조건", "identical": True,
         "value": "접촉 · workspace 이탈 · horizon · 배터리 고갈",
         "note": "일시적 FOV 상실은 종료가 아니며 LSTM hidden도 초기화하지 않는다"},
        {"key": "critic", "label": "Critic", "identical": True,
         "value": "비대칭 · privileged [proprio(7), 참 상대상태(6)]",
         "note": "배포 시 제거"},
        {"key": "deployment", "label": "배포", "identical": False,
         "value": "인식 · LSTM · actor · 공통 제어기",
         "delta": "제안 arm은 그래프 생성과 R-GAT 부호기까지 함께 배포된다",
         "note": "critic과 참 상대상태는 어느 쪽에도 배포되지 않는다"},
    ]
    return {
        "arms": [
            {"id": "shin_se_fixed", "label": "Baseline",
             "role": "관측 벡터만 쓰는 PPO"},
            {"id": "shin_se_onto_rgat_state", "label": "Proposed",
             "role": "관측에 온톨로지 상황 그래프를 더한 PPO"},
        ],
        "rows": shared,
        "identical_count": sum(1 for row in shared if row["identical"]),
        "total_count": len(shared),
    }
