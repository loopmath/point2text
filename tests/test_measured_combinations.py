"""Measured edit composition remains an externally verified approximation."""

from dataclasses import replace
import math

import pytest

from point2text.diffusion.combinations import MeasuredCombinationPlanner
from point2text.diffusion.refinement_contracts import TextProposal, VerifiedText


SOURCE = "Alpha beta gamma delta epsilon zeta. The remaining text stays intact."
WORDS = ("Alpha", "beta", "gamma", "delta", "epsilon", "zeta")


def observation(text, point=(0.0, 0.0), target=(1.0, 0.0)):
    return VerifiedText(text, *point, math.hypot(point[0] - target[0], point[1] - target[1]))


def edit(source, word, replacement):
    start = source.index(word)
    span = (start, start + len(word))
    return TextProposal(source[:start] + replacement + source[span[1]:], span,
                        {"source_spans": [list(span)], "replacements": [replacement]})


def measured(deltas, *, source=SOURCE, planner=None):
    planner = planner or MeasuredCombinationPlanner()
    proposals = [edit(source, word, word.upper() if word != "Alpha" else "ALPHA")
                 for word in WORDS[:len(deltas)]]
    values = [observation(p.text, delta) for p, delta in zip(proposals, deltas)]
    base = observation(source)
    planner.record(base, proposals, values)
    return planner, base, proposals, values


def reconstruct(source, proposal):
    result = source
    for (start, end), replacement in reversed(list(zip(proposal.provenance["source_spans"],
                                                       proposal.provenance["replacements"]))):
        result = result[:start] + replacement + result[end:]
    return result


def test_individually_worse_edits_cancel_using_measured_two_dimensional_motion():
    planner, source, singles, verified = measured([(0.6, 3.0), (0.4, -3.0)])
    assert all(value.distance > source.distance for value in verified)
    proposals = planner.propose(source, (1, 0))
    assert len(proposals) == 1
    best = proposals[0]
    assert best.text.startswith("ALPHA BETA gamma")
    assert best.text.endswith("The remaining text stays intact.")
    assert best.provenance["measured_sum_prediction"] == [1.0, 0.0]
    assert best.provenance["atomic_measured_deltas"] == [[0.6, 3.0], [0.4, -3.0]]
    assert best.provenance["predicted_distance"] == 0.0
    assert best.provenance["canonical_verified"] is False
    assert "canonical verification required" in best.provenance["nonlinear_disclaimer"]
    assert reconstruct(source.text, best) == best.text
    assert all(best.text != p.text for p in singles)


def test_nonlinear_fake_map_does_not_turn_predicted_hit_into_canonical_claim():
    # This artificial verifier models a contextual interaction invisible in singles.
    def fake_map(text):
        a, b = float("ALPHA" in text), float("BETA" in text)
        return 0.6 * a + 0.4 * b + 0.25 * a * b, 3 * a - 3 * b + 0.4 * a * b

    planner = MeasuredCombinationPlanner()
    source = observation(SOURCE, fake_map(SOURCE))
    singles = [edit(SOURCE, "Alpha", "ALPHA"), edit(SOURCE, "beta", "BETA")]
    planner.record(source, singles, [observation(p.text, fake_map(p.text)) for p in singles])
    proposal = planner.propose(source, (1, 0))[0]
    assert proposal.provenance["predicted_distance"] == 0
    actual = observation(proposal.text, fake_map(proposal.text))
    assert actual.distance == pytest.approx(math.hypot(0.25, 0.4))
    assert actual.distance > 1e-4
    assert not proposal.provenance["canonical_verified"]


def test_meet_search_finds_four_edit_cancellation_and_offset_preservation():
    planner, source, _, _ = measured([(3, 7), (-2, 5), (4, -8), (-4, -4)])
    proposals = planner.propose(source, (1, 0))
    best = proposals[0]
    assert best.provenance["edit_count"] == 4
    assert best.provenance["predicted_distance"] == 0
    assert best.text.startswith("ALPHA BETA GAMMA DELTA epsilon")
    assert reconstruct(source.text, best) == best.text
    assert best.provenance["planner_costs"] == planner.last_stats
    assert planner.last_stats["meet_queries"] > 0


def test_optional_six_edit_search_uses_bounded_triples():
    planner, source, _, _ = measured([(0.1, 7), (0.1, 11), (0.1, 17),
                                      (0.1, -2), (0.1, -5), (0.5, -28)],
                                     planner=MeasuredCombinationPlanner(max_edits=6))
    proposals = planner.propose(source, (1, 0))
    assert proposals[0].provenance["edit_count"] == 6
    assert proposals[0].provenance["predicted_distance"] == pytest.approx(0, abs=1e-14)
    assert planner.last_stats["triple_pool_size"] > 0
    assert all(reconstruct(source.text, p) == p.text for p in proposals)


def test_matches_verified_subset_by_text_and_ignores_distance_and_gradient_prediction():
    planner, source, singles, verified = measured([(0.6, 3), (0.4, -3)])
    other = MeasuredCombinationPlanner()
    singles = [replace(p, provenance=dict(p.provenance, predicted_map_delta=[999, 999])) for p in singles]
    stats = other.record(source, singles, [replace(verified[1], distance=999), replace(verified[0], distance=999)])
    assert stats["accepted"] == 2
    assert other.propose(source, (1, 0))[0].provenance["predicted_distance"] == 0
    partial = MeasuredCombinationPlanner()
    assert partial.record(source, singles, verified[1:])["unmatched"] == 1
    assert partial.propose(source, (1, 0)) == []
    assert planner.cache_info()["atoms"] == 2


@pytest.mark.parametrize("provenance,span,text", [
    ({}, None, "BETA"),
    ({"source_spans": [[0, 5], [6, 10]], "replacements": ["ALPHA", "BETA"]}, (0, 10), "ALPHA BETA"),
    ({"source_spans": [[0, 5]], "replacements": ["ALPHA"]}, (1, 5), None),
    ({"source_spans": [[-1, 5]], "replacements": ["ALPHA"]}, None, None),
    ({"source_spans": [[0, len(SOURCE) + 1]], "replacements": ["ALPHA"]}, None, None),
    ({"source_spans": [[False, 5]], "replacements": ["ALPHA"]}, None, None),
    ({"source_spans": [[0.0, 5]], "replacements": ["ALPHA"]}, None, None),
    ({"source_spans": [[0, 5]], "replacements": [123]}, None, None),
    ({"source_spans": [[0, 5]], "replacements": ["ALPHA"]}, None, "ALPHA BETA"),
    ({"source_spans": [[0, 5]], "replacements": ["Alpha"]}, None, SOURCE),
])
def test_rejects_mismatched_provenance_offsets_noops_and_unsupported_multiedits(provenance, span, text):
    text = text if text is not None else "ALPHA" + SOURCE[5:]
    proposal = TextProposal(text, span, provenance)
    planner = MeasuredCombinationPlanner()
    stats = planner.record(observation(SOURCE), [proposal], [observation(text, (1, 2))])
    assert stats["rejected"] == 1
    assert planner.cache_info()["atoms"] == 0


def test_overlapping_alternatives_never_combine_but_adjacent_edits_can():
    source = observation("abc def")
    proposals = [edit(source.text, "abc", "ABC"), edit(source.text, "abc", "Abc"),
                 edit(source.text, "bc d", "BC D"), edit(source.text, "def", "DEF")]
    planner = MeasuredCombinationPlanner()
    planner.record(source, proposals, [observation(p.text, (i, -i)) for i, p in enumerate(proposals)])
    outputs = planner.propose(source, (1, 1))
    assert outputs
    for output in outputs:
        spans = output.provenance["source_spans"]
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
        assert reconstruct(source.text, output) == output.text
    adjacent = observation("abcd")
    a = TextProposal("ABcd", (0, 2), {"source_spans": [[0, 2]], "replacements": ["AB"]})
    b = TextProposal("abCD", (2, 4), {"source_spans": [[2, 4]], "replacements": ["CD"]})
    planner.record(adjacent, [a, b], [observation(a.text, (0.5, 1)), observation(b.text, (0.5, -1))])
    assert planner.propose(adjacent, (1, 0))[0].text == "ABCD"


def test_insertion_deletion_and_same_boundary_insertion_conflicts():
    source = observation("abc def tail")
    proposals = [TextProposal("prefix abc def tail", (0, 0),
                              {"source_spans": [[0, 0]], "replacements": ["prefix "]}),
                 TextProposal("other abc def tail", (0, 0),
                              {"source_spans": [[0, 0]], "replacements": ["other "]}),
                 TextProposal("abc tail", (4, 8),
                              {"source_spans": [[4, 8]], "replacements": [""]})]
    planner = MeasuredCombinationPlanner()
    planner.record(source, proposals, [observation(p.text, (i + 1, i)) for i, p in enumerate(proposals)])
    outputs = planner.propose(source, (1, 0))
    assert len(outputs) == 2
    assert {p.text for p in outputs} == {"prefix abc tail", "other abc tail"}
    assert all(reconstruct(source.text, p) == p.text for p in outputs)


def test_budget_caps_cover_search_memory_work_and_text_materialization():
    planner = MeasuredCombinationPlanner(max_edits=6, max_pair_attempts=7, max_pair_pool=3,
                                         max_triple_attempts=5, max_triple_pool=2,
                                         max_meet_queries=5, max_meet_node_visits=9,
                                         max_nodes_per_query=3, neighbors_per_query=2,
                                         max_candidate_sets=4, max_materializations=2)
    planner, source, _, _ = measured([(i, i * i) for i in range(6)], planner=planner)
    outputs = planner.propose(source, (2, 1), max_proposals=50)
    costs = planner.last_stats
    assert len(outputs) <= 2
    for actual, cap in (("pair_attempts", 7), ("pair_pool_size", 3), ("triple_attempts", 5),
                        ("triple_pool_size", 2), ("meet_queries", 5), ("meet_node_visits", 9),
                        ("candidate_sets_retained", 4), ("text_materializations", 2)):
        assert costs[actual] <= cap
    assert costs["meet_candidates"] <= costs["meet_queries"] * 2


def test_stable_determinism_across_record_and_verified_order_and_repeated_queries():
    planner, source, singles, values = measured([(i / 4, (-1) ** i * i) for i in range(6)])
    other = MeasuredCombinationPlanner()
    other.record(source, list(reversed(singles)), list(reversed(values)))
    expected = planner.propose(source, (1, 0), max_proposals=7)
    assert expected == planner.propose(source, (1, 0), max_proposals=7)
    assert expected == other.propose(source, (1, 0), max_proposals=7)
    assert len({p.text for p in expected}) == len(expected)
    assert expected == sorted(expected, key=lambda p: (p.provenance["predicted_distance"],
                                                      p.provenance["edit_count"],
                                                      p.provenance["source_spans"]))


def test_cache_is_bounded_per_source_and_sources_are_lru():
    planner = MeasuredCombinationPlanner(max_sources=2, max_atoms_per_source=2,
                                         max_cached_texts_per_source=3)
    planner, source, singles, values = measured([(1, 0), (2, 0), (3, 0)], planner=planner)
    assert planner.last_record_stats["evicted_atoms"] == 1
    assert planner.cache_info()["texts"] == 3
    stats = planner.record(source, singles[1:], values[1:])
    assert stats["duplicates"] == 2
    second, third = observation("second source"), observation("third source")
    planner.record(second, [], [])
    planner.propose(source, (1, 0))  # Touch first; second should be evicted next.
    planner.record(third, [], [])
    assert planner.cache_info()["sources"] == 2
    assert source.text in planner._sources and second.text not in planner._sources
    assert planner.propose(second, (0, 0)) == []
    assert planner.cache_info()["sources"] == 2  # A read miss must not allocate a source.
    assert planner.cache_info()["texts"] <= 6


def test_zero_budget_and_unknown_source_do_not_allocate_or_materialize():
    planner, source, _, _ = measured([(0.6, 3), (0.4, -3)])
    assert planner.propose(source, (1, 0), max_proposals=0) == []
    assert planner.last_stats["text_materializations"] == 0
    assert MeasuredCombinationPlanner().propose(source, (1, 0)) == []
    assert planner.propose(source, (1, 0), max_proposals=1)


@pytest.mark.parametrize("target", [(float("nan"), 0), (0, float("inf")), (1,), (1, 2, 3), (True, 0), ("1", 0)])
def test_rejects_nonfinite_or_malformed_target(target):
    with pytest.raises(ValueError, match="finite"):
        MeasuredCombinationPlanner().propose(observation(SOURCE), target)


def test_conflicting_and_nonfinite_coordinates_fail_closed():
    planner, source, singles, values = measured([(0.6, 3), (0.4, -3)])
    with pytest.raises(ValueError, match="conflicting"):
        planner.propose(replace(source, x=0.01), (1, 0))
    with pytest.raises(ValueError, match="conflicting"):
        planner.record(source, singles[:1], [replace(values[0], x=9)])
    with pytest.raises(ValueError, match="conflicting"):
        planner.record(source, singles[:1], [values[0], replace(values[0], x=9)])
    with pytest.raises(ValueError, match="finite"):
        planner.record(source, singles[:1], [replace(values[0], y=float("nan"))])
    with pytest.raises(ValueError, match="finite"):
        planner.record(replace(source, x=float("inf")), [], [])


@pytest.mark.parametrize("options", [{"max_edits": 1}, {"max_edits": 7}, {"max_pair_pool": 0},
                                    {"max_meet_queries": True}, {"max_materializations": 1.5},
                                    {"max_atoms_per_source": 5, "max_cached_texts_per_source": 5}])
def test_rejects_invalid_configuration(options):
    with pytest.raises(ValueError):
        MeasuredCombinationPlanner(**options)


def test_rejects_invalid_proposal_budgets():
    for value in (-1, True, 1.5):
        with pytest.raises(ValueError, match="nonnegative"):
            MeasuredCombinationPlanner().propose(observation(SOURCE), (0, 0), max_proposals=value)
