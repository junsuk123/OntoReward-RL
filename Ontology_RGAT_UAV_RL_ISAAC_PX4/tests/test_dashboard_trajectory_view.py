"""The trajectory panels: stable identity, and a view that shows altitude.

Two defects, both reported from watching a live four-pair run on 2026-09-22.

*The panels looked as though their agent assignment kept swapping.* They never
did -- a panel is a physical pair and its canvas, its series key and its index
are fixed for the run. The title was written from ``active_method``, the policy
currently flying that pair, which legitimately rotates: during collection every
pair flies the reward-design source policy, and crossover evaluation cycles the
arms across pairs seed by seed. The pair cards already separated the standing
assignment from the borrowed policy; this panel did not.

*The view could not answer the landing question.* It was a top-down plot of
world ENU x/y, so whether the vehicle was descending onto the deck or holding
above it -- the thing the whole experiment is about -- was invisible.
"""
from __future__ import annotations

import re

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


# ------------------------------------------------- altitude-bearing side view

def test_the_plot_is_a_side_elevation_with_altitude_up():
    body = _function("drawTrajectory")
    # vertical axis is the z component of the stored 3-D points
    assert "proj=arr=>arr.map(p=>[along(p),p[2]])" in body
    assert "고도 기준 측면 뷰" in PAGE
    # and the retired top-down framing is gone
    assert "world ENU top-down" not in PAGE
    assert "축 world ENU x/y (m)" not in PAGE


def test_the_view_axis_is_the_track_direction_not_the_instantaneous_bearing():
    """A vehicle directly overhead has no bearing, and that is the flare."""
    assert "function sideViewAxis" in PAGE
    body = _function("sideViewAxis")
    assert "Math.atan2(2*sxy,sxx-syy)" in body
    assert "return [1,0]" in body, "a hover with no extent still needs an axis"


def test_the_two_axes_are_scaled_independently_and_the_legend_says_so():
    body = _function("drawTrajectory")
    assert "aSpan" in body and "zSpan" in body
    assert re.search(r"const aSpan=.*\n?.*const aMid", body) or "aMid" in body
    # the equal-aspect square this replaced would flatten the descent
    assert "const span=Math.max(x1-x0,y1-y0,2.0)*1.15" not in PAGE
    assert "두 축의 축척은 다름" in PAGE


def test_the_altitude_gap_and_the_deck_level_are_drawn():
    body = _function("drawTrajectory")
    assert "(u[1]-p[1]).toFixed(2)" in body, "the altitude gap is the number"
    assert "const deck=P[P.length-1][1]" in body, "the deck level is the target"


def test_both_tracks_are_still_drawn_with_their_start_and_current_markers():
    body = _function("drawTrajectory")
    assert "poly(P,PALETTE[1],[4,3])" in body and "poly(U,PALETTE[0],[])" in body
    assert "○ 시작" in PAGE and "● 현재" in PAGE
