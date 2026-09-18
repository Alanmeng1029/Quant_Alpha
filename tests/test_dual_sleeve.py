import pytest

from a_share_data.dual_sleeve import CSI500, CSI1000, assign_sleeves, blend_joint_scores, decide


def test_joint_score_blend_standardizes_each_horizon_before_weighting():
    scores = blend_joint_scores(["A", "B", "C"], [1.0, 2.0, 3.0], [1000.0, 0.0, -1000.0])
    assert scores["A"] == pytest.approx(0.0)
    assert scores["B"] == pytest.approx(0.0)
    assert scores["C"] == pytest.approx(0.0)


def test_joint_score_blend_excludes_non_finite_pairs():
    scores = blend_joint_scores(["A", "B", "C"], [1.0, float("nan"), 3.0], [1.0, 2.0, 5.0])
    assert set(scores) == {"A", "C"}


def test_overlap_is_csi500_and_forced_refill_consumes_shared_slots():
    memberships = assign_sleeves(["A", "B", "C"], ["B", "D", "E"])
    assert memberships["B"] == CSI500
    target, actions = decide({"A": 1, "B": .9, "C": .8, "D": .7, "E": .6}, memberships,
                             {CSI500: {"A"}, CSI1000: {"D"}}, {"A"}, shared_limit=1)
    assert "A" not in target[CSI500]
    assert any(x.reason == "forced_refill" for x in actions)
    assert sum(x.action == "buy" for x in actions) == 1


def test_three_shared_replacements_cap_buys():
    memberships = assign_sleeves([f"A{i}" for i in range(100)], [f"B{i}" for i in range(30)])
    scores = {code: float(1000 - n) for n, code in enumerate(memberships)}
    held = {CSI500: {f"A{i}" for i in range(4, 84)}, CSI1000: {f"B{i}" for i in range(4, 24)}}
    _, actions = decide(scores, memberships, held, shared_limit=3)
    assert sum(x.action == "buy" for x in actions) <= 3


def test_per_sleeve_replacement_limits_are_independent():
    memberships = assign_sleeves([f"A{i}" for i in range(100)], [f"B{i}" for i in range(30)])
    scores = {code: float(1000 - n) for n, code in enumerate(memberships)}
    held = {CSI500: {f"A{i}" for i in range(4, 84)}, CSI1000: {f"B{i}" for i in range(4, 24)}}
    _, actions = decide(scores, memberships, held, shared_limit=99, sleeve_limits={CSI500: 1, CSI1000: 2})
    assert sum(x.action == "buy" and x.sleeve == CSI500 for x in actions) <= 1
    assert sum(x.action == "buy" and x.sleeve == CSI1000 for x in actions) <= 2
