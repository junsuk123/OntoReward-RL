"""세미나 슬라이드용 결과 지표와 16:9 그림을 실제 산출물에서 생성한다.

완료되지 않은 평가를 최종 결과처럼 표시하지 않는 것이 이 모듈의 가장 중요한
계약이다. 실행 중에는 현재 실행의 학습 CSV만 사용해 ``예비 학습 결과``를
만들고, manifest가 완료된 뒤에만 paired evaluation을 ``최종 검증``으로 쓴다.
"""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


METHODS = (
    "shin_se_fixed",
    "shin_se_onto_rgat_fov",
)
METHOD_LABELS = {
    "shin_se_fixed": "Baseline · Shin SE fixed",
    "shin_se_onto_rgat_fov": "Proposed · Shin + Ontology-R-GAT FOV",
}
METHOD_COLORS = {
    "shin_se_fixed": "#D95319",
    "shin_se_onto_rgat_fov": "#77AC30",
}
COMPONENT_LABELS = ("수평 접근", "수직 접근", "하강 안전", "미달 방지", "요 안정")
COMPONENT_COLORS = ("#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    except (OSError, csv.Error):
        return []


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    names = list(fields or dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Mapping[str, Any]):
    def safe(item):
        if isinstance(item, Mapping):
            return {str(key): safe(entry) for key, entry in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(entry) for entry in item]
        if isinstance(item, (float, np.floating)) and not math.isfinite(float(item)):
            return None
        if isinstance(item, np.generic):
            return item.item()
        return item
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe(value), ensure_ascii=False, indent=2),
                    encoding="utf-8")


def _number(row: Mapping[str, Any], key: str, default=math.nan) -> float:
    try:
        text = row.get(key, "")
        if isinstance(text, str) and text.lower() in {"true", "false"}:
            return 1.0 if text.lower() == "true" else 0.0
        return default if text in (None, "") else float(text)
    except (TypeError, ValueError):
        return default


def _wilson(successes: int, count: int, z=1.959963984540054) -> tuple[float, float]:
    if count <= 0:
        return math.nan, math.nan
    p = successes / count
    denominator = 1.0 + z * z / count
    centre = (p + z * z / (2.0 * count)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / count + z * z / (4.0 * count * count)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def _configure_matplotlib():
    import warnings
    import matplotlib as mpl
    from matplotlib import font_manager

    mpl.use("Agg")
    cjk_font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    font_family = "DejaVu Sans"
    if cjk_font.is_file():
        font_manager.fontManager.addfont(str(cjk_font))
        font_family = font_manager.FontProperties(fname=str(cjk_font)).get_name()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font_family, "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.edgecolor": "#7F7F7F",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#D9D9D9",
        "grid.linestyle": ":",
        "grid.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    with warnings.catch_warnings():
        # 설치된 system/user Matplotlib의 3D 선택 모듈 충돌은 본 2D 그림과 무관하다.
        warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
        import matplotlib.pyplot as plt
    return plt


def _current_training_rows(results_dir: Path, manifest: Mapping[str, Any]):
    rows: dict[str, list[dict[str, str]]] = {}
    specs = dict(manifest.get("pipeline_specs") or {})
    for method in METHODS:
        spec = dict(specs.get(method) or {})
        if spec.get("fov_risk_reward_enabled") and not manifest.get(
                "fov_risk_design_id"):
            rows[method] = []
            continue
        primary = results_dir / "models" / method / f"{method}_training.csv"
        candidate = _read_csv(primary)
        if not candidate:
            candidate = _read_csv(results_dir / "training" / f"{method}.csv")
        rows[method] = [row for row in candidate
                        if row.get("optimization_phase", "ppo") == "ppo"]
    return rows


def _evaluation_is_complete(rows, manifest: Mapping[str, Any]) -> bool:
    if not str(manifest.get("execution_status", "")).startswith("complete"):
        return False
    required_per_method = sum(int(value) for value in dict(
        manifest.get("evaluation") or {}).values())
    selected = dict(manifest.get("selected_checkpoints") or {})
    if required_per_method <= 0 or not all(method in selected for method in METHODS):
        return False
    for method in METHODS:
        chosen = selected[method]
        digest = str(chosen.get("sha256", ""))
        matching = [row for row in rows if row.get("pipeline", row.get("method")) == method
                    and (not digest or row.get("selected_checkpoint_sha256") == digest)]
        if len(matching) < required_per_method:
            return False
    return True


def _metric_rows(grouped: Mapping[str, Sequence[Mapping[str, Any]]], source: str):
    summaries = []
    distributions: dict[str, list[float]] = {}
    for method in METHODS:
        rows = list(grouped.get(method, ()))
        successes = sum(_number(row, "strict_success", _number(
            row, "paper_success", 0.0)) >= 0.5 for row in rows)
        low, high = _wilson(successes, len(rows))
        contact_rows = [row for row in rows if _number(row, "pad_contact", 0.0) >= .5]
        lateral = [_number(row, "touchdown_lateral_error") for row in contact_rows]
        lateral = [value for value in lateral if math.isfinite(value)]
        distributions[method] = lateral
        fov = [_number(row, "fov_loss_fraction") for row in rows]
        fov = [value for value in fov if math.isfinite(value)]
        unsafe = sum(_number(row, "unsafe_pad_contact", 0.0) >= .5 for row in rows)
        summaries.append({
            "method": method,
            "label": METHOD_LABELS[method],
            "source": source,
            "episodes": len(rows),
            "safe_landings": int(successes),
            "safe_landing_rate": successes / len(rows) if rows else math.nan,
            "safe_landing_ci95_low": low,
            "safe_landing_ci95_high": high,
            "pad_contacts": len(contact_rows),
            "unsafe_contact_rate": unsafe / len(rows) if rows else math.nan,
            "touchdown_lateral_error_median_m": (
                float(np.median(lateral)) if lateral else math.nan),
            "touchdown_lateral_error_mean_m": (
                float(np.mean(lateral)) if lateral else math.nan),
            "fov_loss_fraction_mean": float(np.mean(fov)) if fov else math.nan,
        })
    return summaries, distributions


def _load_performance_source(results_dir: Path, manifest: Mapping[str, Any]):
    evaluation = _read_csv(results_dir / "evaluation" / "per_episode.csv")
    if _evaluation_is_complete(evaluation, manifest):
        grouped = {method: [row for row in evaluation
                            if row.get("pipeline", row.get("method")) == method]
                   for method in METHODS}
        return "최종 paired 검증", "final_evaluation", grouped
    return "진행 중 학습(예비 결과)", "training_preliminary", _current_training_rows(
        results_dir, manifest)


def _save_performance_figures(output_dir: Path, summaries, distributions,
                              grouped, title_prefix: str, status: str,
                              training_grouped=None):
    plt = _configure_matplotlib()
    labels = [METHOD_LABELS[name] for name in METHODS]
    colors = [METHOD_COLORS[name] for name in METHODS]
    rates = np.asarray([row["safe_landing_rate"] for row in summaries], dtype=float)
    lows = np.asarray([row["safe_landing_ci95_low"] for row in summaries], dtype=float)
    highs = np.asarray([row["safe_landing_ci95_high"] for row in summaries], dtype=float)
    counts = [int(row["episodes"]) for row in summaries]

    def success_chart(ax):
        x = np.arange(len(METHODS))
        valid = np.isfinite(rates)
        ax.bar(x[valid], rates[valid], color=np.asarray(colors)[valid], width=.62,
               edgecolor="white", linewidth=.8)
        if valid.any():
            ax.errorbar(x[valid], rates[valid],
                        yerr=np.vstack((rates[valid] - lows[valid],
                                        highs[valid] - rates[valid])),
                        fmt="none", ecolor="#333333", capsize=5, linewidth=1.2)
        for index in range(len(METHODS)):
            if valid[index]:
                ax.text(index, min(1.07, rates[index] + .055),
                        f"{100*rates[index]:.1f}%\n({int(round(rates[index]*counts[index]))}/{counts[index]})",
                        ha="center", va="bottom", fontsize=9)
            else:
                ax.text(index, .08, "대기", ha="center", va="center",
                        color="#777777", fontsize=10)
        ax.set_xticks(x, labels, rotation=9, ha="right")
        ax.set_ylim(0, 1.16)
        ax.set_ylabel("안전 착륙률")
        ax.set_title("안전 착륙률과 Wilson 95% 신뢰구간", fontweight="bold")

    def lateral_chart(ax):
        present = [(index + 1, distributions[name]) for index, name in enumerate(METHODS)
                   if distributions[name]]
        if present:
            boxes = ax.boxplot([values for _, values in present],
                               positions=[position for position, _ in present],
                               widths=.55, patch_artist=True, showmeans=True,
                               meanprops={"marker": "D", "markerfacecolor": "white",
                                          "markeredgecolor": "#333333", "markersize": 4})
            for patch, (position, _) in zip(boxes["boxes"], present):
                patch.set_facecolor(colors[position - 1])
                patch.set_alpha(.78)
        for index, name in enumerate(METHODS, start=1):
            if not distributions[name]:
                ax.text(index, .04, "접촉 표본 없음", ha="center", va="bottom",
                        color="#777777", fontsize=8)
        ax.axhline(.35, color="#A2142F", linestyle="--", linewidth=1.3,
                   label="성공 한계 0.35 m")
        ax.set_xlim(.5, len(METHODS) + .5)
        ax.set_xticks(range(1, len(METHODS) + 1), labels, rotation=9, ha="right")
        ax.set_ylabel("접촉 순간 수평 오차 [m]")
        ax.set_title("패드 접촉 시 수평 오차 분포", fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)

    def metric_table(ax):
        ax.axis("off")
        columns = ("방법", "N", "안전 착륙률 [95% CI]", "위험 접촉률",
                   "수평 오차 중앙값", "시야 상실률")
        cells = []
        for row in summaries:
            if not row["episodes"]:
                cells.append((row["label"], "—", "평가 대기", "—", "—", "—"))
                continue
            cells.append((
                row["label"], str(row["episodes"]),
                f"{100*row['safe_landing_rate']:.1f}% "
                f"[{100*row['safe_landing_ci95_low']:.1f}, "
                f"{100*row['safe_landing_ci95_high']:.1f}]",
                f"{100*row['unsafe_contact_rate']:.1f}%",
                (f"{row['touchdown_lateral_error_median_m']:.3f} m"
                 if math.isfinite(row["touchdown_lateral_error_median_m"]) else "—"),
                (f"{100*row['fov_loss_fraction_mean']:.1f}%"
                 if math.isfinite(row["fov_loss_fraction_mean"]) else "—"),
            ))
        table = ax.table(cellText=cells, colLabels=columns, cellLoc="center",
                         colLoc="center", bbox=[0, .11, 1, .82],
                         colWidths=[.22, .07, .25, .14, .17, .15])
        table.auto_set_font_size(False)
        table.set_fontsize(8.7)
        for (row_index, _), cell in table.get_celld().items():
            cell.set_edgecolor("#B7B7B7")
            if row_index == 0:
                cell.set_facecolor("#E7E6E6")
                cell.set_text_props(weight="bold")
        ax.text(0, .01,
                "성공 = 패드 접촉 ∧ 위치·수직속도·상대수평속도·자세·각속도 기준 모두 통과",
                transform=ax.transAxes, fontsize=8.5, color="#444444")

    fig = plt.figure(figsize=(13.333, 7.5))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, .56), hspace=.42, wspace=.28)
    success_chart(fig.add_subplot(grid[0, 0]))
    lateral_chart(fig.add_subplot(grid[0, 1]))
    metric_table(fig.add_subplot(grid[1, :]))
    fig.suptitle(f"안전 착륙 성능 — {title_prefix}", fontsize=16, fontweight="bold")
    fig.text(.995, .007, "※ episode return은 방법 간 성능지표에서 제외",
             ha="right", fontsize=8, color="#666666")
    fig.subplots_adjust(left=.07, right=.98, top=.90, bottom=.06)
    composite = output_dir / "slide13_safe_landing_performance.png"
    fig.savefig(composite, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.15))
    success_chart(ax)
    fig.tight_layout()
    success_path = output_dir / "slide13_success_rate_ci.png"
    fig.savefig(success_path, dpi=220)
    page14_success_path = output_dir / "page14_success_rate_ci.png"
    fig.savefig(page14_success_path, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.15))
    lateral_chart(ax)
    fig.tight_layout()
    lateral_path = output_dir / "slide13_touchdown_lateral_error.png"
    fig.savefig(lateral_path, dpi=220)
    page14_lateral_path = output_dir / "page14_touchdown_lateral_error.png"
    fig.savefig(page14_lateral_path, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    progress_grouped = grouped if training_grouped is None else training_grouped
    for method in METHODS:
        rows = list(progress_grouped.get(method, ()))
        values = [_number(row, "strict_success", _number(row, "paper_success", 0.0))
                  for row in rows]
        if not values:
            continue
        rolling = [float(np.mean(values[max(0, index - 4):index + 1]))
                   for index in range(len(values))]
        ax.plot(np.arange(1, len(values) + 1), rolling, linewidth=2,
                color=METHOD_COLORS[method], label=f"{METHOD_LABELS[method]} (N={len(values)})")
    ax.set(xlabel="PPO 에피소드", ylabel="최근 5회 안전 착륙률",
           ylim=(-.03, 1.03), title="학습 진행률 — PPO checkpoint 기록")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    progress_path = output_dir / "training_safe_landing_progress.png"
    fig.savefig(progress_path, dpi=220)
    page16_progress_path = output_dir / "page16_training_safe_landing_progress.png"
    fig.savefig(page16_progress_path, dpi=220)
    plt.close(fig)
    return [
        composite, success_path, page14_success_path,
        lateral_path, page14_lateral_path,
        progress_path, page16_progress_path,
    ]


def _artifact_diagnostics(results_dir: Path, dataset_manifest: Mapping[str, Any]):
    rgat_dir = results_dir / "rgat"
    candidates = []
    accepted = rgat_dir / "adaptive_reward_weights.manifest.json"
    if accepted.is_file():
        candidates.append((accepted, "accepted"))
    candidates.extend((path, "archived") for path in sorted(
        rgat_dir.glob("adaptive_reward_weights.manifest.rejected-*.json"),
        key=lambda path: path.stat().st_mtime, reverse=True))
    current_sha = str(dataset_manifest.get("dataset_sha256", ""))
    selected = None
    for path, kind in candidates:
        data = _read_json(path)
        exact = bool(current_sha and data.get("dataset_sha256") == current_sha)
        if exact:
            selected = (path, kind, data, True)
            break
        if selected is None:
            selected = (path, kind, data, False)
    if selected is None:
        return {}, None, "모델 학습 대기", False
    path, kind, data, exact = selected
    status = ("현 데이터 검증 완료" if exact and kind == "accepted" else
              "현 데이터 artifact(품질 게이트 미통과)" if exact else
              "이전 데이터 진단 — 현 데이터 재학습 대기")
    suffix = path.name.removeprefix("adaptive_reward_weights.manifest").removesuffix(".json")
    history_name = ("adaptive_training_history.csv" if not suffix else
                    f"adaptive_training_history{suffix}.csv")
    return data, rgat_dir / history_name, status, exact


def _save_reward_diagnostics(output_dir: Path, results_dir: Path,
                             dataset_manifest: Mapping[str, Any]):
    plt = _configure_matplotlib()
    artifact, history_path, artifact_status, exact = _artifact_diagnostics(
        results_dir, dataset_manifest)
    metrics = dict(artifact.get("metrics") or {})
    gate = dict(artifact.get("quality_gate") or {})
    thresholds = dict(gate.get("thresholds") or {})
    history = _read_csv(history_path) if history_path else []
    artifact_dataset = dict(artifact.get("dataset_manifest") or {})
    model_episode_count = int(artifact_dataset.get("episodes", 0) or 0)
    current_episode_count = int(dataset_manifest.get("episodes", 0) or 0)

    means = list(metrics.get("mean_weights") or [math.nan] * 5)
    cvs = list(metrics.get("weight_coefficient_of_variation") or [math.nan] * 5)
    diagnostic_specs = (
        ("검증 정확도", metrics.get("validation_accuracy"),
         thresholds.get("minimum_validation_accuracy", .50), "higher"),
        ("순위 일치율", metrics.get("validation_ranking_accuracy"),
         thresholds.get("minimum_validation_ranking_accuracy", .50), "higher"),
        ("평균 가중치 CV", metrics.get("mean_weight_coefficient_of_variation"),
         thresholds.get("minimum_mean_weight_cv", .003), "higher"),
        ("관측 방향 일치율", metrics.get("potential_observability_monotonic_compliance"),
         thresholds.get("minimum_potential_monotonic_compliance", .55), "higher"),
    )
    validation_rows = []
    for name, value, threshold, direction in diagnostic_specs:
        numeric = float(value) if value is not None else math.nan
        limit = float(threshold)
        validation_rows.append({
            "metric": name, "value": numeric, "threshold": limit,
            "direction": direction,
            "passed": bool(math.isfinite(numeric) and numeric >= limit),
            "artifact_status": artifact_status,
            "artifact_dataset_episodes": model_episode_count,
            "current_dataset_episodes": current_episode_count,
            "current_dataset_match": exact,
        })
    _write_csv(output_dir / "slide14_reward_model_validation.csv", validation_rows)

    strata = dict(dataset_manifest.get("outcome_strata") or {})
    composition = [
        {"stratum": "success", "episodes": int(strata.get("success", 0))},
        {"stratum": "failure", "episodes": int(strata.get("failure", 0))},
        {"stratum": "unsafe_pad_contact", "episodes": int(strata.get("unsafe_pad_contact", 0))},
        {"stratum": "near_miss", "episodes": int(strata.get("near_miss", 0))},
        {"stratum": "risky_failure", "episodes": int(strata.get("risky_failure", 0))},
    ]
    _write_csv(output_dir / "slide14_reward_dataset_composition.csv", composition)

    fig = plt.figure(figsize=(13.333, 7.5))
    grid = fig.add_gridspec(2, 2, hspace=.42, wspace=.38)
    ax = fig.add_subplot(grid[0, 0])
    x = np.arange(5)
    valid_means = np.asarray(means, dtype=float)
    baseline = np.asarray((1.0, 1.0, .5, 1.0, 2.0))
    ax.bar(x - .18, baseline, width=.36, color="#A6A6A6", label="고정 기준")
    if np.isfinite(valid_means).any():
        ax.bar(x + .18, valid_means, width=.36, color=COMPONENT_COLORS,
               edgecolor="white", label="R-GAT 평균")
    ax.set_xticks(x, COMPONENT_LABELS, rotation=10, ha="right")
    ax.set_ylabel("보상 가중치")
    ax.set_title("5개 보상 성분의 평균 가중치", fontweight="bold")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(grid[0, 1])
    y = np.arange(len(validation_rows))
    for index, row in enumerate(validation_rows):
        value, threshold = row["value"], row["threshold"]
        if math.isfinite(value):
            ax.barh(index, value, color="#77AC30" if row["passed"] else "#A2142F",
                    height=.52)
            ax.text(value, index, f" {value:.3f}", va="center", fontsize=8)
        else:
            ax.text(.02, index, "산출 대기", va="center", color="#777777", fontsize=8)
        ax.plot([threshold, threshold], [index - .34, index + .34],
                color="#000000", linewidth=2)
    ax.set_yticks(y, [row["metric"] for row in validation_rows])
    ax.invert_yaxis()
    ax.set_xlim(0, max(1.02, max([row["value"] for row in validation_rows
                                 if math.isfinite(row["value"])] + [1.0]) * 1.13))
    ax.set_xlabel("측정값 (검은 선 = 통과 기준)")
    ax.set_title("보상 모델 검증 지표", fontweight="bold")

    ax = fig.add_subplot(grid[1, 0])
    if history:
        epochs = [_number(row, "epoch", index + 1) for index, row in enumerate(history)]
        for key, label, color in (
                ("total_loss", "학습 총손실", "#0072BD"),
                ("validation_objective", "검증 목적함수", "#D95319"),
                ("validation_bce", "검증 BCE", "#7E2F8E")):
            values = [_number(row, key) for row in history]
            if np.isfinite(values).any():
                ax.plot(epochs, values, color=color, linewidth=1.8, label=label)
        ax.legend(fontsize=8)
    else:
        ax.text(.5, .5, "R-GAT 학습 이력 생성 대기", ha="center", va="center",
                transform=ax.transAxes, color="#777777")
    ax.set(xlabel="R-GAT epoch", ylabel="손실", title="학습·검증 목적함수")
    ax.set_title("학습·검증 목적함수", fontweight="bold")

    ax = fig.add_subplot(grid[1, 1])
    names = ["성공", "실패", "위험 접촉", "근접 실패", "위험 실패"]
    values = [row["episodes"] for row in composition]
    ax.bar(names, values, color=("#77AC30", "#A6A6A6", "#A2142F", "#4DBEEE", "#EDB120"))
    for index, value in enumerate(values):
        ax.text(index, value + max(values + [1]) * .025, str(value), ha="center", fontsize=9)
    ax.set_ylabel("에피소드 수")
    ax.tick_params(axis="x", labelrotation=18, labelsize=8)
    ax.set_title("실제 Isaac/PX4 보상 설계 데이터", fontweight="bold")
    ax.text(.01, .98,
            f"전이 {int(dataset_manifest.get('transitions', 0) or 0):,} · "
            f"검증 episode {int(dataset_manifest.get('validation_episodes', 0) or 0)}",
            transform=ax.transAxes, va="top", fontsize=8.5, color="#444444")

    fig.suptitle("R-GAT 적응 보상 모델 진단", fontsize=16, fontweight="bold")
    mismatch = (f"{artifact_status} (모델 데이터 N={model_episode_count}, "
                f"현재 데이터 N={current_episode_count})")
    fig.text(.995, .007, mismatch, ha="right", fontsize=8,
             color="#A2142F" if not exact else "#444444")
    fig.subplots_adjust(left=.08, right=.98, top=.90, bottom=.075)
    path = output_dir / "slide14_rgat_reward_validation.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)

    # The current seminar deck uses two independent image placeholders on
    # page 15.  Keep the composite diagnostic for documentation, and also save
    # slide-ready charts that can be inserted without cropping or rescaling a
    # four-panel figure.
    fig, ax = plt.subplots(figsize=(7.2, 4.15))
    x = np.arange(5)
    ax.bar(x - .18, baseline, width=.36, color="#A6A6A6", label="고정 기준")
    if np.isfinite(valid_means).any():
        ax.bar(x + .18, valid_means, width=.36, color=COMPONENT_COLORS,
               edgecolor="white", label="R-GAT 평균")
    ax.set_xticks(x, COMPONENT_LABELS, rotation=10, ha="right")
    ax.set_ylabel("보상 가중치")
    ax.set_title("5개 보상 성분의 평균 가중치", fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    weights_path = output_dir / "page15_reward_weights_comparison.png"
    fig.savefig(weights_path, dpi=220)
    plt.close(fig)

    # Page 15 explicitly reports the weight-model gate only.  The semantic
    # potential observability check remains in the composite diagnostic above,
    # but is intentionally excluded from this slide-specific chart.
    weight_validation_rows = validation_rows[:3]
    fig, ax = plt.subplots(figsize=(7.2, 4.15))
    y = np.arange(len(weight_validation_rows))
    finite_values = []
    for index, row in enumerate(weight_validation_rows):
        value, threshold = row["value"], row["threshold"]
        if math.isfinite(value):
            finite_values.append(value)
            ax.barh(index, value, color="#77AC30" if row["passed"] else "#A2142F",
                    height=.52)
            ax.text(value, index, f" {value:.3f}", va="center", fontsize=9)
        else:
            ax.text(.02, index, "산출 대기", va="center", color="#777777", fontsize=9)
        ax.plot([threshold, threshold], [index - .34, index + .34],
                color="#000000", linewidth=2)
    ax.set_yticks(y, [row["metric"] for row in weight_validation_rows])
    ax.invert_yaxis()
    ax.set_xlim(0, max(1.02, max(finite_values + [1.0]) * 1.13))
    ax.set_xlabel("측정값 (검은 선 = 통과 기준)")
    ax.set_title("보상 가중치 모델 검증 지표", fontweight="bold")
    fig.tight_layout()
    validation_path = output_dir / "page15_reward_model_validation.png"
    fig.savefig(validation_path, dpi=220)
    plt.close(fig)
    return [path, weights_path, validation_path], validation_rows, artifact_status, exact


def _save_fov_risk_diagnostics(output_dir: Path, manifest: Mapping[str, Any]):
    model = dict(manifest.get("fov_risk_model") or {})
    validation = dict(model.get("validation_metrics") or {})
    rows = [{
        "metric": key,
        "value": validation.get(key),
        "split": "whole held-out episodes",
    } for key in ("auroc", "f1", "precision", "recall", "best_validation_bce")]
    confusion = validation.get("confusion_matrix")
    if confusion:
        rows.append({"metric": "confusion_matrix", "value": str(confusion),
                     "split": "whole held-out episodes"})
    _write_csv(output_dir / "fov_risk_validation.csv", rows)
    status = ("validation-best frozen classifier" if model.get("frozen")
              else "FOV-risk model training pending")
    figures = []
    finite = [(row["metric"], float(row["value"])) for row in rows
              if row["metric"] != "confusion_matrix"
              and row["value"] is not None
              and math.isfinite(float(row["value"]))]
    plt = _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 4.15))
    if finite:
        names, values = zip(*finite)
        ax.bar(names, values, color="#77AC30")
        ax.set_ylim(0, max(1.0, max(values) * 1.1))
        ax.set_ylabel("Metric value")
    else:
        ax.text(.5, .5, "FOV-risk model training pending", ha="center",
                va="center", transform=ax.transAxes, color="#777777")
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("Future FOV-loss R-GAT validation")
    fig.tight_layout()
    path = output_dir / "fov_risk_model_validation.png"
    composite_path = output_dir / "slide14_rgat_reward_validation.png"
    fig.savefig(path, dpi=220)
    fig.savefig(composite_path, dpi=220)
    plt.close(fig)
    figures.extend((path, composite_path))
    return figures, rows, status, bool(model.get("frozen"))


def write_presentation_results(results_dir, output_dir=None) -> dict[str, Any]:
    """발표용 지표 CSV/JSON과 슬라이드용 PNG를 생성하고 경로를 반환한다."""
    results_dir = Path(results_dir).resolve()
    output_dir = Path(output_dir or (results_dir / "presentation")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(results_dir / "manifest.json")
    title, status, grouped = _load_performance_source(results_dir, manifest)
    summaries, distributions = _metric_rows(grouped, status)
    _write_csv(output_dir / "slide13_safe_landing_metrics.csv", summaries)
    figures = _save_performance_figures(
        output_dir, summaries, distributions, grouped, title, status,
        training_grouped=_current_training_rows(results_dir, manifest))

    reward_figures, validation, artifact_status, exact = (
        _save_fov_risk_diagnostics(output_dir, manifest))
    figures.extend(reward_figures)
    catalog = (
        {"slide": 14, "metric": "안전 착륙률", "definition":
         "strict_success 평균; 접촉·수평오차·수직속도·상대수평속도·자세·각속도 동시 통과", "uncertainty": "Wilson 95% CI"},
        {"slide": 14, "metric": "접촉 수평 오차", "definition":
         "pad_contact가 참인 episode의 touchdown_lateral_error", "uncertainty": "box plot"},
        {"slide": 14, "metric": "위험 접촉률", "definition":
         "unsafe_pad_contact 평균", "uncertainty": "표본 수 N 병기"},
        {"slide": 14, "metric": "시야 상실률", "definition":
         "episode별 fov_loss_fraction 평균", "uncertainty": "표본 수 N 병기"},
        {"slide": 15, "metric": "FOV-risk AUROC/F1", "definition":
         "held-out episode의 1초 내 FOV-loss 이진 분류", "uncertainty": "episode 단위 분할"},
        {"slide": 16, "metric": "최근 5회 학습 안전 착륙률", "definition":
         "방법별 PPO training checkpoint에 기록된 strict_success의 5-episode 이동평균", "uncertainty": "학습 추세 진단"},
    )
    _write_csv(output_dir / "presentation_metric_catalog.csv", catalog)
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "results_dir": str(results_dir),
        "performance_status": status,
        "performance_title": title,
        "reward_artifact_status": artifact_status,
        "reward_artifact_matches_current_dataset": exact,
        "safe_landing_metrics": summaries,
        "reward_validation_metrics": validation,
        "figures": [str(path) for path in figures],
        "data_policy": (
            "최종 manifest가 완료되고 모든 방법의 paired evaluation 표본이 채워진 "
            "경우에만 최종 검증으로 표시한다. 그 전에는 현재 학습 CSV만 예비 결과로 표시한다."),
    }
    _write_json(output_dir / "presentation_results_summary.json", summary)
    (output_dir / "README.md").write_text(
        "# 세미나 발표용 결과 그림\n\n"
        f"- 성능 자료 상태: **{title}**\n"
        f"- R-GAT 보상 모델 상태: **{artifact_status}**\n"
        "- `slide13_safe_landing_performance.png`: 13쪽 전체 패널\n"
        "- `page14_success_rate_ci.png`: 14쪽 안전 착륙률과 95% 신뢰구간\n"
        "- `page14_touchdown_lateral_error.png`: 14쪽 접촉 수평 오차 분포\n"
        "- `fov_risk_model_validation.png`: R-GAT AUROC/F1/precision/recall\n"
        "- `page16_training_safe_landing_progress.png`: 16쪽 PPO 학습 진행 추세\n"
        "- `slide13_safe_landing_performance.png`, `slide14_rgat_reward_validation.png`: 복합 진단용\n\n"
        "> 완료 전 수치는 예비 학습 결과입니다. 최종 평가와 혼용하지 마십시오. "
        "동일 명령을 다시 실행하면 최신 산출물로 갱신됩니다.\n",
        encoding="utf-8")
    return summary
