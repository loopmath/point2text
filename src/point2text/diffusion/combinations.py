"""Combine canonical-measured single edits without making new encoder calls.

Single-edit displacements are observations. Their sum is still an approximation
for a multi-edit text: contextual encoder interactions need not be additive.
Every returned proposal requires independent canonical and quality verification.
"""

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
import heapq
from itertools import combinations, islice
import math
from numbers import Real
import random
from typing import Any

from .refinement_contracts import TextProposal, VerifiedText


@dataclass(frozen=True)
class _Atom:
    start: int
    end: int
    replacement: str
    text: str
    dx: float
    dy: float


@dataclass
class _Source:
    point: tuple[float, float]
    atoms: OrderedDict[tuple[int, int, str], _Atom] = field(default_factory=OrderedDict)
    coordinates: OrderedDict[str, tuple[float, float]] = field(default_factory=OrderedDict)


@dataclass(frozen=True)
class _Partial:
    indices: tuple[int, ...]
    dx: float
    dy: float


@dataclass(frozen=True)
class _Node:
    partial: _Partial
    axis: int
    left: "_Node | None"
    right: "_Node | None"


def _point(values: Sequence[float], name: str) -> tuple[float, float]:
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain two finite coordinates") from exc
    try:
        valid = len(values) == 2 and all(not isinstance(v, bool) and isinstance(v, Real)
                                         and math.isfinite(v) for v in values)
        if valid:
            return float(values[0]), float(values[1])
    except (OverflowError, TypeError, ValueError):
        pass
    raise ValueError(f"{name} must contain two finite coordinates")


def _conflict(a: _Atom, b: _Atom) -> bool:
    # Zero-width insertions at the same boundary have ambiguous composition.
    if a.start == a.end:
        return b.start <= a.start <= b.end
    if b.start == b.end:
        return a.start <= b.start <= a.end
    return a.start < b.end and b.start < a.end


def _compatible(indices: tuple[int, ...], other: tuple[int, ...], atoms: list[_Atom]) -> bool:
    return not any(i == j or _conflict(atoms[i], atoms[j]) for i in indices for j in other)


def _merge(a: _Partial, b: _Partial) -> _Partial | None:
    dx, dy = a.dx + b.dx, a.dy + b.dy
    if not math.isfinite(dx) or not math.isfinite(dy):
        return None
    return _Partial(tuple(sorted((*a.indices, *b.indices))), dx, dy)


def _tree(values: list[_Partial], depth: int = 0) -> _Node | None:
    if not values:
        return None
    axis = depth % 2
    values.sort(key=lambda p: ((p.dx, p.dy)[axis], p.indices))
    middle = len(values) // 2
    return _Node(values[middle], axis, _tree(values[:middle], depth + 1),
                 _tree(values[middle + 1:], depth + 1))


def _nearest(root: _Node | None, location: tuple[float, float], left: _Partial,
             atoms: list[_Atom], count: int, budget: int) -> tuple[list[_Partial], int]:
    """Bounded 2-D tree search; exclusions apply before nearest-neighbor pruning."""
    best: list[tuple[float, tuple[int, ...], _Partial]] = []
    visited = 0

    def visit(node: _Node | None) -> None:
        nonlocal visited
        if node is None or visited >= budget:
            return
        visited += 1
        part = node.partial
        distance = math.hypot(part.dx - location[0], part.dy - location[1])
        if math.isfinite(distance) and _compatible(left.indices, part.indices, atoms):
            item = (-distance, tuple(-i for i in part.indices), part)
            if len(best) < count:
                heapq.heappush(best, item)
            elif item[:2] > best[0][:2]:
                heapq.heapreplace(best, item)
        separation = location[node.axis] - (part.dx, part.dy)[node.axis]
        near, far = (node.left, node.right) if separation < 0 else (node.right, node.left)
        visit(near)
        if len(best) < count or abs(separation) <= -best[0][0]:
            visit(far)

    visit(root)
    return [item[2] for item in sorted(best, key=lambda item: (-item[0], item[2].indices))], visited


class _Pool:
    """Keep nearby partials and a deterministic reservoir of distant partials."""

    def __init__(self, limit: int, desired: tuple[float, float], seed: int):
        self.near_limit = max(1, limit // 2)
        self.diverse_limit = limit - self.near_limit
        self.desired = desired
        self.near: list[tuple[float, tuple[int, ...], _Partial]] = []
        self.diverse: list[_Partial] = []
        self.seen: set[tuple[int, ...]] = set()
        self.random = random.Random(seed)

    def add(self, partial: _Partial) -> None:
        if partial.indices in self.seen:
            return
        self.seen.add(partial.indices)
        distance = math.hypot(partial.dx - self.desired[0], partial.dy - self.desired[1])
        item = (-distance, tuple(-i for i in partial.indices), partial)
        if len(self.near) < self.near_limit:
            heapq.heappush(self.near, item)
        elif item[:2] > self.near[0][:2]:
            heapq.heapreplace(self.near, item)
        if len(self.diverse) < self.diverse_limit:
            self.diverse.append(partial)
        elif self.diverse_limit:
            index = self.random.randrange(len(self.seen))
            if index < self.diverse_limit:
                self.diverse[index] = partial

    def values(self) -> list[_Partial]:
        values = {p.indices: p for p in (*self.diverse, *(item[2] for item in self.near))}
        return [values[key] for key in sorted(values)]


class MeasuredCombinationPlanner:
    """Rank 2--4 (optionally up to 6) nonoverlapping measured atomic edits.

    ``record`` matches canonical observations to proposal text, never by list
    position. Only one explicitly declared source span that reconstructs the
    complete proposal text is eligible. ``propose`` uses the measured 2-D deltas,
    retaining distant partials so individually worse edits can cancel. Bounded
    nearest-neighbor searches make this a heuristic, not an exhaustive search.
    No encoder, tokenizer, verifier, or quality policy is invoked here.
    """

    def __init__(self, *, max_edits: int = 4, max_atoms_per_source: int = 256,
                 max_sources: int = 8, max_cached_texts_per_source: int = 512,
                 max_pair_attempts: int = 32768, max_pair_pool: int = 4096,
                 max_triple_attempts: int = 16384, max_triple_pool: int = 2048,
                 max_meet_queries: int = 4096, max_meet_node_visits: int = 262144,
                 max_nodes_per_query: int = 128, neighbors_per_query: int = 4,
                 max_candidate_sets: int = 1024, max_materializations: int = 256):
        limits = locals().copy()
        limits.pop("self")
        for name, value in limits.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, value)
        if not 2 <= max_edits <= 6:
            raise ValueError("max_edits must be between 2 and 6")
        if max_cached_texts_per_source < max_atoms_per_source + 1:
            raise ValueError("Text cache must fit max_atoms_per_source plus the source")
        self._sources: OrderedDict[str, _Source] = OrderedDict()
        self.last_record_stats: dict[str, int] = {}
        self.last_stats: dict[str, int] = {}

    def cache_info(self) -> dict[str, int]:
        return {"sources": len(self._sources),
                "atoms": sum(len(s.atoms) for s in self._sources.values()),
                "texts": sum(len(s.coordinates) for s in self._sources.values()),
                "max_sources": self.max_sources,
                "max_atoms_per_source": self.max_atoms_per_source,
                "max_cached_texts_per_source": self.max_cached_texts_per_source}

    @staticmethod
    def _atomic(source: str, proposal: TextProposal) -> tuple[int, int, str] | None:
        provenance = proposal.provenance
        if not isinstance(provenance, dict):
            return None
        spans, replacements = provenance.get("source_spans"), provenance.get("replacements")
        if (not isinstance(spans, (list, tuple)) or len(spans) != 1
                or not isinstance(replacements, (list, tuple)) or len(replacements) != 1):
            return None
        span, replacement = spans[0], replacements[0]
        if not isinstance(span, (list, tuple)) or len(span) != 2 or not isinstance(replacement, str):
            return None
        start, end = span
        if (any(isinstance(v, bool) or not isinstance(v, int) for v in span)
                or not 0 <= start <= end <= len(source)
                or (proposal.edit_span is not None and proposal.edit_span != (start, end))
                or proposal.text == source
                or source[:start] + replacement + source[end:] != proposal.text):
            return None
        return start, end, replacement

    def _source(self, source: VerifiedText, *, create: bool) -> _Source | None:
        if not isinstance(source.text, str) or not source.text:
            raise ValueError("Source must have nonempty text")
        point = _point((source.x, source.y), "Source")
        state = self._sources.get(source.text)
        if state is not None:
            if state.point != point:
                raise ValueError("Source text has conflicting canonical coordinates")
            self._sources.move_to_end(source.text)
        elif create:
            state = _Source(point)
            state.coordinates[source.text] = point
            self._sources[source.text] = state
            while len(self._sources) > self.max_sources:
                self._sources.popitem(last=False)
        return state

    def record(self, source: VerifiedText, proposals: Sequence[TextProposal],
               verified: Sequence[VerifiedText]) -> dict[str, int]:
        """Record separately verified atoms; malformed/unmatched proposals are skipped.

        Conflicting coordinates for the same text, or nonfinite coordinates,
        raise ``ValueError``. Distances may refer to different targets and are
        deliberately never used to infer an edit's displacement or eligibility.
        """
        state = self._source(source, create=True)
        assert state is not None
        points: dict[str, tuple[float, float]] = {}
        for value in verified:
            point = _point((value.x, value.y), "Verified text")
            if value.text in points and points[value.text] != point:
                raise ValueError("Verified text has conflicting canonical coordinates")
            points[value.text] = point
        stats = {"proposals": len(proposals), "verified": len(verified), "accepted": 0,
                 "rejected": 0, "unmatched": 0, "duplicates": 0, "evicted_atoms": 0}
        for proposal in proposals:
            key = self._atomic(source.text, proposal)
            if key is None:
                stats["rejected"] += 1
                continue
            if proposal.text not in points:
                stats["unmatched"] += 1
                continue
            point = points[proposal.text]
            previous = state.coordinates.get(proposal.text)
            if previous is not None and previous != point:
                raise ValueError("Cached text has conflicting canonical coordinates")
            dx, dy = point[0] - state.point[0], point[1] - state.point[1]
            if not math.isfinite(dx) or not math.isfinite(dy):
                raise ValueError("Measured displacement must contain two finite coordinates")
            if key in state.atoms:
                stats["duplicates"] += 1
                state.atoms.move_to_end(key)
                continue
            state.atoms[key] = _Atom(*key, proposal.text, dx, dy)
            state.coordinates[proposal.text] = point
            state.coordinates.move_to_end(proposal.text)
            stats["accepted"] += 1
            while len(state.atoms) > self.max_atoms_per_source:
                state.atoms.popitem(last=False)
                stats["evicted_atoms"] += 1
            # Cache only source and still-retained atoms; both limits stay hard.
            retained = {source.text, *(atom.text for atom in state.atoms.values())}
            for text in list(state.coordinates):
                if text not in retained:
                    del state.coordinates[text]
        self.last_record_stats = stats
        return dict(stats)

    def propose(self, source: VerifiedText, target: tuple[float, float], *,
                max_proposals: int = 64) -> list[TextProposal]:
        if isinstance(max_proposals, bool) or not isinstance(max_proposals, int) or max_proposals < 0:
            raise ValueError("max_proposals must be a nonnegative integer")
        location = _point(target, "Target")
        state = self._source(source, create=False)
        source_x, source_y = float(source.x), float(source.y)
        desired = (location[0] - source_x, location[1] - source_y)
        if not all(math.isfinite(v) for v in desired):
            raise ValueError("Target displacement must contain two finite coordinates")
        atoms = [] if state is None else sorted(state.atoms.values(), key=lambda a: (a.start, a.end, a.replacement))
        stats = {"atomic_count": len(atoms), "pair_attempts": 0, "pair_pool_size": 0,
                 "triple_attempts": 0, "triple_pool_size": 0, "meet_queries": 0,
                 "meet_node_visits": 0, "meet_candidates": 0, "candidate_sets_considered": 0,
                 "candidate_sets_retained": 0, "text_materializations": 0, "returned": 0}
        self.last_stats = stats
        if max_proposals == 0 or len(atoms) < 2:
            return []
        singles = [_Partial((i,), atom.dx, atom.dy) for i, atom in enumerate(atoms)]
        best: list[tuple[float, int, tuple[int, ...], _Partial]] = []
        retained: set[tuple[int, ...]] = set()

        def consider(part: _Partial | None) -> None:
            if part is None or part.indices in retained:
                return
            stats["candidate_sets_considered"] += 1
            distance = math.hypot(part.dx - desired[0], part.dy - desired[1])
            if not math.isfinite(distance):
                return
            item = (-distance, -len(part.indices), tuple(-i for i in part.indices), part)
            if len(best) < self.max_candidate_sets:
                heapq.heappush(best, item)
                retained.add(part.indices)
            elif item[:3] > best[0][:3]:
                old = heapq.heapreplace(best, item)
                retained.remove(old[3].indices)
                retained.add(part.indices)

        pair_builder = _Pool(self.max_pair_pool, (desired[0] / 2, desired[1] / 2), 0)
        for i, j in islice(combinations(range(len(atoms)), 2), self.max_pair_attempts):
            stats["pair_attempts"] += 1
            if _compatible((i,), (j,), atoms):
                part = _merge(singles[i], singles[j])
                if part is not None:
                    pair_builder.add(part)
                    consider(part)
        pairs = pair_builder.values()
        stats["pair_pool_size"] = len(pairs)
        pools = {1: singles, 2: pairs}
        if self.max_edits >= 5 and pairs:
            builder = _Pool(self.max_triple_pool, (desired[0] / 2, desired[1] / 2), 1)
            total = len(pairs) * len(singles)
            for flat in random.Random(2).sample(range(total), min(total, self.max_triple_attempts)):
                stats["triple_attempts"] += 1
                a, b = pairs[flat // len(singles)], singles[flat % len(singles)]
                if _compatible(a.indices, b.indices, atoms):
                    part = _merge(a, b)
                    if part is not None:
                        builder.add(part)
                        consider(part)
            pools[3] = builder.values()
            stats["triple_pool_size"] = len(pools[3])
        passes = [(a, b) for a, b in ((1, 2), (2, 2), (2, 3), (3, 3))
                  if a + b <= self.max_edits and pools.get(a) and pools.get(b)]
        trees = {size: _tree(list(pools[size])) for size in {b for _, b in passes}}
        for step, (a_size, b_size) in enumerate(passes):
            remaining_passes = len(passes) - step
            query_budget = (self.max_meet_queries - stats["meet_queries"]) // remaining_passes
            visit_budget = (self.max_meet_node_visits - stats["meet_node_visits"]) // remaining_passes
            lefts = pools[a_size]
            queries = min(len(lefts), query_budget)
            used_visits = 0
            for index in range(queries):
                if used_visits >= visit_budget:
                    break
                left = lefts[index * len(lefts) // queries]
                residual = (desired[0] - left.dx, desired[1] - left.dy)
                if not all(math.isfinite(v) for v in residual):
                    continue
                nearby, visits = _nearest(trees[b_size], residual, left, atoms,
                                           self.neighbors_per_query,
                                           min(self.max_nodes_per_query, visit_budget - used_visits))
                stats["meet_queries"] += 1
                stats["meet_node_visits"] += visits
                used_visits += visits
                stats["meet_candidates"] += len(nearby)
                for right in nearby:
                    consider(_merge(left, right))
        stats["candidate_sets_retained"] = len(best)
        selected: list[tuple[str, _Partial, float]] = []
        seen_text = {source.text, *(atom.text for atom in atoms)}
        for item in sorted(best, key=lambda item: (-item[0], -item[1], item[3].indices)):
            if len(selected) >= max_proposals or stats["text_materializations"] >= self.max_materializations:
                break
            part = item[3]
            text = source.text
            for i in reversed(part.indices):
                atom = atoms[i]
                text = text[:atom.start] + atom.replacement + text[atom.end:]
            stats["text_materializations"] += 1
            if text in seen_text:
                continue
            seen_text.add(text)
            px, py = source_x + part.dx, source_y + part.dy
            if math.isfinite(px) and math.isfinite(py):
                selected.append((text, part, -item[0]))
        stats["returned"] = len(selected)
        proposals = []
        for text, part, distance in selected:
            edits = [atoms[i] for i in part.indices]
            provenance: dict[str, Any] = {
                "method": "canonical-measured-combinations-v1", "kind": "measured-disjoint-combination",
                "source_spans": [[a.start, a.end] for a in edits],
                "replacements": [a.replacement for a in edits],
                "source_map_location": [source_x, source_y],
                "measured_map_delta": [part.dx, part.dy],
                "measured_sum_prediction": [source_x + part.dx, source_y + part.dy],
                "predicted_distance": distance, "edit_count": len(edits),
                "atomic_measured_deltas": [[a.dx, a.dy] for a in edits],
                "prediction_kind": "sum-of-separately-canonical-measured-single-edit-deltas",
                "nonlinear_disclaimer": "Contextual multi-edit interactions are not measured; canonical verification required.",
                "quality": "inherits input edits only; canonical and semantic verification required",
                "canonical_verified": False, "planner_costs": dict(stats),
            }
            proposals.append(TextProposal(text, (edits[0].start, edits[-1].end), provenance))
        return proposals
