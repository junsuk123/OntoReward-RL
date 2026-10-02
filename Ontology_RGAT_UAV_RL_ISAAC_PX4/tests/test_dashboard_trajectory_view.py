"""The trajectory panels: stable identity and comparable absolute geometry.

Two defects, both reported from watching a live four-pair run on 2026-09-22.

*The panels looked as though their agent assignment kept swapping.* They never
did -- a panel is a physical pair and its canvas, its series key and its index
are fixed for the run. The title was written from ``active_method``, the policy
currently flying that pair, which legitimately rotates: during collection every
pair flies the reward-design source policy, and crossover evaluation cycles the
arms across pairs seed by seed. The pair cards already separated the standing
assignment from the borrowed policy; this panel did not.

The four canvases must use one fixed-size world-ENU box and one metres-to-pixels
factor. Its X origin is the episode's initial UGV position, because the UGV
continues from its previous world position between episodes. Auto-fitting each
flight made identical motion look different between pairs and made the axes
jump whenever a new extreme sample arrived.
"""
from __future__ import annotations

from ontology_rgat.viz.dashboard import PAGE


def _function(name: str) -> str:
    body = PAGE[PAGE.index(f"function {name}"):]
    return body[:body.index("\nfunction ")]


# --------------------------------------------- stable panel <-> pair identity

def test_a_panel_is_a_physical_pair_and_says_so():
    body = _function("drawTrajectories")
    # The series a panel draws is keyed by the panel index, never by a method.
    assert "state.series['benchmark_step_pair_'+i]" in body
    assert "pairs.find(p=>Number(p.index)===i)" in body


def test_the_title_leads_with_the_standing_assignment():
    body = _function("drawTrajectories")
    assert "배정 ${methodLabel(assigned)}" in body
    assert "pair.assigned_method||pair.method" in body


def test_a_borrowed_policy_is_marked_as_borrowed_not_substituted():
    """The old title silently replaced the arm name with whoever was flying."""
    body = _function("drawTrajectories")
    assert "active!==assigned?` · 현재 비행" in body
    # the bare active-method title must not survive
    assert "물리 Pair ${i+1} · ${methodLabel(pair.active_method" not in PAGE


def test_the_trajectory_panel_matches_how_the_pair_cards_name_things():
    """Both read the same two fields, so they cannot disagree on screen."""
    for name in ("drawTrajectories", "pairPanel"):
        body = _function(name)
        assert "assigned_method" in body and "active_method" in body, name


# -------------------------------- fixed absolute X, nonnegative altitude view

def test_the_plot_uses_absolute_world_x_and_nonnegative_altitude():
    body = _function("drawTrajectory")
    assert "U=uav.map(p=>[p[0],p[2]])" in body
    assert "P=pad.map(p=>[p[0],p[2]])" in body
    assert "world ENU X (m)" in body and "altitude / world ENU Z (m)" in body
    assert "sideViewAxis" not in PAGE


def test_x_bound_is_anchored_to_the_episode_initial_ugv_position():
    assert "const TRAJECTORY_VIEWPORT=Object.freeze" in PAGE
    assert "xLength:170,yMin:0,yMax:10,xTick:20,yTick:1" in PAGE
    body = _function("drawTrajectory")
    assert "const initialUgvX=P.length?P[0][0]" in body
    assert "xMin:initialUgvX,xMax:initialUgvX+TRAJECTORY_VIEWPORT.xLength" in body
    assert "a0=Infinity" not in body and "z0=Infinity" not in body


def test_reverse_shuttle_anchors_the_initial_ugv_at_the_right_boundary():
    body = _function("drawTrajectory")
    assert "const directionSample=P.find" in body
    assert "const ugvDirection=" in body
    assert "xMin:initialUgvX-TRAJECTORY_VIEWPORT.xLength,xMax:initialUgvX" in body


def test_every_pair_uses_the_same_fixed_size_x_and_altitude_scales():
    body = _function("drawTrajectory")
    assert "const scaleX=availableW/xSpan,scaleY=availableH/ySpan" in body
    assert "const px=v=>ox+(v-bound.xMin)*scaleX" in body
    assert "const py=v=>oy+H-(v-bound.yMin)*scaleY" in body
    assert "에피소드 초기 UGV 기준" in PAGE
    assert "모든 pair 동일 축척" in PAGE


def test_current_altitude_above_the_pad_is_drawn():
    body = _function("drawTrajectory")
    assert "const altitudeGap=u[1]-p[1]" in body
    assert "Δh ${altitudeGap.toFixed(2)} m" in body
    assert "패드 위 ${(last[2]-pz).toFixed(2)} m" in body


def test_four_pair_cards_are_stacked_to_make_x_wide():
    assert ".traj-grid{display:grid;grid-template-columns:1fr" in PAGE
    assert ".traj-plot canvas{height:260px;aspect-ratio:auto}" in PAGE


def test_both_tracks_are_still_drawn_with_their_start_and_current_markers():
    body = _function("drawTrajectory")
    assert "poly(P,PALETTE[1],[4,3])" in body and "poly(U,PALETTE[0],[])" in body
    assert "○ 시작" in PAGE and "● 현재" in PAGE
