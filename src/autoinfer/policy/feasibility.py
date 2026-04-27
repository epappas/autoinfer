"""Feasibility classifier for constrained Bayesian optimization (P9 extension).

Learns ``P(success | config)`` from observed ``(config, FailureRecord | None)``
pairs so the surrogate can reject candidates whose nearest-neighbor history
is dominated by typed failures, instead of relying on hand-authored catalog
rules.

Why this exists: the 2026-04-25 full L1×L2×L3 campaign showed
``OptunaSurrogate`` with penalty-encoded failures ("score=0 on fail")
hits 0/4 on infeasibility-rich surfaces (L1 reserve trials kept proposing
``kv_cache_dtype=fp8_e5m2`` on A100, all STARTUP-failing). Penalty=0
collapses every failure mode into one scalar; TPE's KDE can't extract
"this knob class is structurally infeasible" from "these points scored 0."

The fix is constrained BO: a separate feasibility model alongside the
perf model. Sample candidates from TPE; gate by feasibility before
emission. The model learns the constraint surface FROM TYPED FAILURES,
no manual rules required — adding a new model/hardware combo "just
works" once a few failures are recorded.

Implementation: nearest-neighbor classifier with a per-knob mixed-type
distance. No sklearn dep — k-NN with k=3 weighted by inverse distance
is sufficient for the 5-50 trials/layer regime autoinfer operates in,
and stays interpretable.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from autoinfer.harness.failure import FailureKind


@dataclass(frozen=True)
class _Observation:
    """One (config, outcome) data point."""

    config: dict[str, Any]
    success: bool
    failure_kind: FailureKind | None


def _knob_distance(
    a: Any, b: Any, *, class_map: dict[str, str] | None = None
) -> float:
    """Distance between two knob values, normalised to [0, 1].

    - Booleans / strings / None: 0 if equal, else 1 (Hamming).
    - Numerics: absolute relative difference, clipped to [0, 1].
      ``|a - b| / max(|a|, |b|, 1e-9)``.
    - Mixed types: 1 (treat as fully different — bool vs int is the
      common case, and treating ``True == 1`` as equal would lose
      structural information about the knob's type).

    When a ``class_map`` (value -> class label) is provided AND both
    values are strings, two distinct values that map to the same class
    compare as distance 0. This lets the classifier collapse e.g.
    ``{fp8, fp8_e4m3, fp8_e5m2}`` into one structural region so a single
    failure generalises across all variants. T-26: campaign 01 evidence.
    Values not appearing in the class_map fall back to per-value Hamming.
    """
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return 1.0
    a_is_bool = isinstance(a, bool)
    b_is_bool = isinstance(b, bool)
    if a_is_bool != b_is_bool:
        return 1.0
    if a_is_bool and b_is_bool:
        return 0.0 if a == b else 1.0
    if isinstance(a, str) and isinstance(b, str):
        if a == b:
            return 0.0
        if class_map is not None:
            ca = class_map.get(a)
            cb = class_map.get(b)
            if ca is not None and ca == cb:
                return 0.0
        return 1.0
    if isinstance(a, str) or isinstance(b, str):
        return 0.0 if a == b else 1.0
    try:
        af = float(a)
        bf = float(b)
    except (TypeError, ValueError):
        return 0.0 if a == b else 1.0
    denom = max(abs(af), abs(bf), 1e-9)
    return min(1.0, abs(af - bf) / denom)


def _config_distance(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    knob_classes: dict[str, dict[str, str]] | None = None,
    knob_weights: dict[str, float] | None = None,
) -> float:
    """Weighted-average per-knob distance over the union of keys, in [0, 1].

    Missing keys on either side count as distance 1 — penalises configs
    that explore knobs the other side doesn't set, since a structurally-
    different config-space region isn't comparable.

    ``knob_classes`` (knob_name -> value -> class_label) routes per-knob
    class taxonomies into ``_knob_distance`` so structurally-equivalent
    values collapse to distance 0.

    ``knob_weights`` (knob_name -> weight) lets some knobs dominate the
    distance computation. Default per-knob weight is 1.0 (= the legacy
    plain-average behaviour). T-26b: campaign 02 evidence showed plain
    averaging dilutes the per-knob class signal — the fp8-cluster
    distance on ``kv_cache_dtype`` collapses to 0 via class collapse,
    but averaging with 11 unrelated knobs at distance ~0.5 keeps the
    overall distance at ~0.45, which doesn't reliably push P(success)
    below ``feasibility_threshold=0.4``. Weighting knobs that
    deterministically predict feasibility (e.g. ``kv_cache_dtype`` on
    A100) ~10x higher restores the signal: weighted_avg(10*0 + 11*0.5)
    / (10 + 11) ≈ 0.26, comfortably below threshold.
    """
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    weighted_total = 0.0
    weight_sum = 0.0
    for k in keys:
        w = knob_weights.get(k, 1.0) if knob_weights else 1.0
        weight_sum += w
        if k not in a or k not in b:
            weighted_total += w * 1.0
        else:
            cm = knob_classes.get(k) if knob_classes else None
            weighted_total += w * _knob_distance(a[k], b[k], class_map=cm)
    if weight_sum == 0.0:
        return 0.0
    return weighted_total / weight_sum


@dataclass
class FeasibilityModel:
    """Online k-NN feasibility classifier.

    Records ``(config, success, failure_kind)`` triples on every trial.
    Predicts ``P(success)`` for a candidate config from the inverse-
    distance-weighted vote of the ``k`` nearest observations.

    With fewer than ``k`` observations, ``predict_proba`` returns 1.0
    (no constraint signal yet — let the surrogate explore freely). Once
    enough data accumulates, the model gates the sampler.
    """

    k: int = 3
    """Number of nearest neighbors used for prediction."""

    min_observations: int = 4
    """Below this many recorded trials, predict_proba returns 1.0 — the
    classifier needs some baseline signal before it can usefully filter.
    Default 4 matches the typical warmstart batch size."""

    distance_floor: float = 1e-6
    """Smallest allowable distance for the inverse-distance weight, so
    an exact-match neighbor doesn't divide-by-zero into infinite weight."""

    knob_classes: dict[str, dict[str, str]] = field(default_factory=dict)
    """Per-knob value-to-class taxonomies. ``{knob_name: {value: class}}``.
    Two values mapped to the same class compare at distance 0 within that
    knob, so a single failure of e.g. ``kv_cache_dtype=fp8_e4m3``
    generalises to ``fp8`` and ``fp8_e5m2`` when all three live in one
    class. Empty by default — the classifier falls back to plain
    per-value Hamming distance for unclassed knobs. T-26."""

    knob_weights: dict[str, float] = field(default_factory=dict)
    """Per-knob weights for the config-distance computation. Default
    weight = 1.0 (plain-average behaviour). Knobs with weight > 1
    dominate the weighted average so structural knobs (e.g.
    ``kv_cache_dtype`` on A100, where fp8 is deterministically
    incompatible with sm_80) carry their failure signal even when the
    other ~11 knobs vary across surrogate proposals. T-26b: Campaign
    02 evidence showed plain averaging diluted T-26's class collapse;
    weighting catalog-rule knobs ~10x restores the reject signal."""

    _history: list[_Observation] = field(default_factory=list)

    def record(
        self,
        config: dict[str, Any],
        *,
        success: bool,
        failure_kind: FailureKind | None = None,
    ) -> None:
        """Append a trial outcome to the history."""
        if success and failure_kind is not None:
            raise ValueError("success=True with non-None failure_kind is contradictory")
        if not success and failure_kind is None:
            raise ValueError("success=False requires a failure_kind")
        self._history.append(
            _Observation(config=dict(config), success=success, failure_kind=failure_kind)
        )

    def predict_proba(self, config: dict[str, Any]) -> float:
        """Return the model's estimate of ``P(success | config)``.

        Inverse-distance-weighted vote of the k nearest neighbors. Output
        is in ``[0.0, 1.0]``: 1.0 = "every nearby observation succeeded",
        0.0 = "every nearby observation failed."
        """
        if len(self._history) < self.min_observations:
            return 1.0
        return self._weighted_vote(config, predicate=lambda o: o.success)

    def predict_kind_proba(self, config: dict[str, Any]) -> dict[FailureKind, float]:
        """Per-failure-kind probability among the k nearest failed neighbors.

        Useful for diagnostics ("this region tends to OOM" vs "tends to
        fail QUALITY_KL"); the surrogate doesn't have to use this — it
        can route on aggregate ``predict_proba`` alone.
        """
        if len(self._history) < self.min_observations:
            return {}
        out: dict[FailureKind, float] = {}
        for kind in FailureKind:
            out[kind] = self._weighted_vote(
                config,
                predicate=lambda o, _k=kind: o.failure_kind == _k,
            )
        return out

    def n_observations(self) -> int:
        return len(self._history)

    def n_successful(self) -> int:
        return sum(1 for o in self._history if o.success)

    def history(self) -> Iterable[_Observation]:
        return tuple(self._history)

    def _weighted_vote(
        self,
        config: dict[str, Any],
        predicate: Any,
    ) -> float:
        """Inverse-distance-weighted fraction of k-NN matching ``predicate``.

        Algorithm:
        1. Compute distance from ``config`` to every observation.
        2. Pick the ``k`` smallest.
        3. Weight = 1 / max(distance, distance_floor).
        4. Return sum(weight where predicate) / sum(weight).
        """
        scored = [
            (
                _config_distance(
                    config,
                    obs.config,
                    knob_classes=self.knob_classes,
                    knob_weights=self.knob_weights,
                ),
                obs,
            )
            for obs in self._history
        ]
        scored.sort(key=lambda x: x[0])
        topk = scored[: self.k]
        total_weight = 0.0
        match_weight = 0.0
        for dist, obs in topk:
            w = 1.0 / max(dist, self.distance_floor)
            total_weight += w
            if predicate(obs):
                match_weight += w
        if total_weight == 0.0:
            return 1.0
        return match_weight / total_weight


__all__ = ["FeasibilityModel"]
