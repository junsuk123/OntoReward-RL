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

from ..rgat.fov_graph import (FOV_FEATURE_NAMES, FOV_GOAL_NODE,
                              FOV_GRAPH_EDGES, FOV_GRAPH_INPUT_DIM,
                              FOV_NODE_NAMES, FOV_RELATION_NAMES)


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
    {"id": "training", "title": "Baseline PPO",
     "detail": "보상 artifact에 의존하지 않아 즉시 시작",
     "produces": "shin_se_fixed 정책", "kind": "train"},
    {"id": "FOV-risk data", "title": "FOV 데이터 수집",
     "detail": "동결 baseline 정책 · 시각 전용 그래프 + 기하 라벨",
     "produces": "episode 분리 데이터셋", "kind": "data"},
    {"id": "R-GAT training", "title": "R-GAT 학습·동결",
     "detail": "Huber 회귀 + 규약 손실 R-04 · validation-best",
     "produces": "동결 스칼라 readout", "kind": "train"},
    {"id": "parallel training", "title": "Proposed PPO",
     "detail": "동일 예산 · 동결 readout을 보상에만 사용",
     "produces": "shin_se_onto_rgat_recovery 정책", "kind": "train"},
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
        "note": ("Baseline PPO와 FOV 데이터 수집은 서로 다른 물리 pair에서 동시에 "
                 "진행된다. 두 arm은 같은 초기 checkpoint와 같은 PPO 예산을 쓴다."),
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
    limit = list(control.get("max_velocity_m_s", (2.0, 2.0, 1.0)))
    yaw = float(control.get("max_yaw_rate_deg_s", 60.0))
    speed = tuple(pad_speed_range or (0.0, 0.0))
    shared = [
        {"key": "observation", "label": "Observation (S)", "identical": True,
         "value": "grayscale 이미지 + body velocity(3) + quaternion(4)",
         "note": "정책은 event trigger·참 가속도·미래 궤적·기하 가시성 라벨을 받지 않는다"},
        {"key": "latent", "label": "내부 표현", "identical": True,
         "value": "6-keypoint CNN → LSTM h[512] → latent y[256]",
         "note": "actor는 y[6:256]+proprio, y[0:6]은 보조 상태추정"},
        {"key": "action", "label": "Action (A)", "identical": True,
         "value": "heading frame [vx, vy, vz, ωz]",
         "note": (f"|v| ≤ {limit[0]:g}/{limit[1]:g}/{limit[2]:g} m/s · "
                  f"|ωz| ≤ {yaw:g}°/s · {dt:g} s 주기")},
        {"key": "environment", "label": "Environment (E)", "identical": True,
         "value": (f"Isaac Sim + PX4 · 이동 패드 {speed[0]:g}–{speed[1]:g} m/s"
                   if speed[1] else "Isaac Sim + PX4 · 이동 패드"),
         "note": f"horizon {horizon} step · 동일 seed·동일 외란·동일 카메라"},
        {"key": "reward", "label": "Reward (R)", "identical": False,
         "value": "원문 Table III 5개 항 + active perception "
                  f"(α={float(active.get('alpha', 0.1)):g}, "
                  f"β={float(active.get('beta', 1.0)):g}, "
                  f"τ={float(active.get('tau', 0.01)):g})",
         "delta": (f"비종료 step에 −λ·q(G_t+1) 추가 (λ={float(lambda_fov):g}, "
                   f"H={float(horizon_seconds):g}s). λ=0이면 baseline과 동일"),
         "note": "공통 5개 계수와 active perception은 그대로 둔다"},
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
        {"key": "deployment", "label": "배포", "identical": True,
         "value": "인식 · LSTM · actor · 공통 제어기",
         "note": "온톨로지/R-GAT/보상/critic 의존성 없음"},
    ]
    return {
        "arms": [
            {"id": "shin_se_fixed", "label": "Baseline", "role": "원문 재현 대상"},
            {"id": "shin_se_onto_rgat_recovery", "label": "Proposed",
             "role": "학습 전용 온톨로지 보상 추가"},
        ],
        "rows": shared,
        "identical_count": sum(1 for row in shared if row["identical"]),
        "total_count": len(shared),
    }
