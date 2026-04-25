"""Classical-surrogate samplers (Optuna TPE / CMA-ES / random).

Implements the bulk of the hybrid policy (P7) after the LLM warm-start.
Per C6 evidence (arxiv 2603.24647), this surrogate is expected to
dominate the search at ~100-500 evaluation budgets; the LLM operator
only re-enters on stall.

Failure handling (P9): a typed ``FailureRecord`` is converted to a
penalty objective value (worst-possible for the configured direction)
and reported via ``study.tell`` with ``state=COMPLETE`` rather than
``state=FAIL``. Optuna's TPE excludes ``FAIL`` trials from its KDE,
which produces 0% hit rates on surfaces where most configs are
infeasible — see iteration-zero L1 50-trial run + 2026-04-25 joint run
(both 0/8 surrogate hit rate). Penalty-encoded failures let the
sampler learn to avoid the infeasible region by modelling it as
"low value" instead of "missing data".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple, Protocol, runtime_checkable

from autoinfer.harness.failure import FailureKind
from autoinfer.harness.ledger import Measurement
from autoinfer.policy.feasibility import FeasibilityModel


class Suggestion(NamedTuple):
    trial_id: str
    config: dict[str, Any]


@runtime_checkable
class Surrogate(Protocol):
    def suggest(self) -> Suggestion: ...
    def record(
        self,
        trial_id: str,
        measurement: Measurement | None,
        failure: FailureKind | None,
    ) -> None: ...


_FAILURE_PENALTY = 0.0
"""Worst-possible objective value for a maximize direction; for
minimize it's negated to a large positive value via
``OptunaSurrogate._penalty_for_failure``. The exact magnitude doesn't
matter for TPE — what matters is that it's strictly worse than any
real observation the search could produce."""


class OptunaSurrogate:
    """Optuna-backed surrogate; supports ``tpe``, ``cmaes``, ``random``.

    Failures land as ``state=COMPLETE`` with a penalty value rather than
    ``state=FAIL`` so the sampler models the infeasible region instead
    of ignoring it. See module docstring.
    """

    def __init__(
        self,
        kind: str,
        seed: int,
        surface: dict[str, dict[str, Any]],
        objective_axis: str,
        maximize: bool,
    ) -> None:
        import optuna
        from optuna import samplers

        samplers_by_kind: dict[str, Any] = {
            "tpe": samplers.TPESampler(seed=seed),
            "cmaes": samplers.CmaEsSampler(seed=seed),
            "random": samplers.RandomSampler(seed=seed),
        }
        if kind not in samplers_by_kind:
            raise ValueError(f"unknown surrogate kind: {kind!r}")

        self._optuna = optuna
        self._study = optuna.create_study(
            sampler=samplers_by_kind[kind],
            direction="maximize" if maximize else "minimize",
        )
        self._surface = surface
        self._axis = objective_axis
        self._maximize = maximize
        self._pending: dict[str, Any] = {}

    def suggest(self) -> Suggestion:
        trial = self._study.ask()
        config = {name: self._sample(trial, name, spec) for name, spec in self._surface.items()}
        tid = f"t{trial.number:06d}"
        self._pending[tid] = trial
        return Suggestion(trial_id=tid, config=config)

    def record(
        self,
        trial_id: str,
        measurement: Measurement | None,
        failure: FailureKind | None,
    ) -> None:
        trial = self._pending.pop(trial_id, None)
        if trial is None:
            raise KeyError(f"no pending trial {trial_id!r}")
        if failure is not None:
            # Penalty-encode the failure as state=COMPLETE so the sampler's
            # KDE includes it (see module docstring for rationale).
            self._study.tell(trial, self._penalty_for_failure(failure))
            return
        if measurement is None:
            raise ValueError("measurement required when failure is None")
        self._study.tell(trial, measurement.value(self._axis))

    def prune(self, trial_id: str) -> None:
        """Discard a pending trial without modeling it.

        Used by the constrained wrapper to drop rejected resample
        candidates so they don't pollute the perf model's KDE — Optuna's
        ``state=PRUNED`` is the canonical way to mark a trial as
        "ignore for model fitting" without leaving it dangling.
        """
        trial = self._pending.pop(trial_id, None)
        if trial is None:
            raise KeyError(f"no pending trial {trial_id!r}")
        self._study.tell(trial, state=self._optuna.trial.TrialState.PRUNED)

    def _penalty_for_failure(self, failure: FailureKind) -> float:
        """Worst-possible objective value for the configured direction.

        ``failure`` kind is ignored today (every typed failure earns the
        same penalty). A future extension could differentiate hard
        failures (STARTUP/HANG/OOM) from soft (QUALITY_KL) so quality
        regressions don't earn the same penalty as structural infeasibility.
        """
        del failure
        return _FAILURE_PENALTY if self._maximize else 1e9

    @staticmethod
    def _sample(trial: Any, name: str, spec: dict[str, Any]) -> Any:
        kind = spec["type"]
        if kind == "int":
            return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
        if kind == "float":
            return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
        if kind == "categorical":
            return trial.suggest_categorical(name, spec["values"])
        raise ValueError(f"unknown surface type for {name!r}: {kind!r}")


@dataclass
class ConstrainedOptunaSurrogate:
    """``OptunaSurrogate`` + ``FeasibilityModel`` — constrained BO.

    Wraps the perf surrogate with a feasibility classifier so the
    sampler avoids structurally-infeasible regions instead of just
    being penalised inside them. Built to address the 0/4 L1 reserve
    hit rate in the 2026-04-25 full L1xL2xL3 campaign, where TPE
    kept proposing ``kv_cache_dtype=fp8_e5m2`` on A100 even after
    multiple penalty signals — the penalty encoding doesn't extract
    structural rules; the feasibility classifier does.

    Sampling protocol:
    1. Ask the perf surrogate for a candidate.
    2. Score its feasibility via the classifier.
    3. If P(success) >= ``threshold``, accept and return.
    4. Otherwise, remember it as fallback; ask again.
    5. After ``max_resamples`` attempts, accept the highest-scoring
       fallback and prune the rest (discarded from KDE via
       ``OptunaSurrogate.prune``).

    On record, both models learn: the perf surrogate via existing
    penalty-or-score logic; the feasibility model via the typed
    success/failure outcome.

    First ``min_observations`` trials skip the feasibility filter (the
    classifier returns 1.0 there) so the search isn't gated by no-data
    pessimism.
    """

    inner: OptunaSurrogate
    feasibility: FeasibilityModel
    threshold: float = 0.4
    """Minimum P(success) to accept a candidate without resampling."""

    max_resamples: int = 8
    """How many times to ask the inner surrogate before falling back to
    the highest-feasibility candidate seen so far."""

    _suggested_configs: dict[str, dict[str, Any]] = field(default_factory=dict)

    def suggest(self) -> Suggestion:
        best_sugg: Suggestion | None = None
        best_score: float = -1.0
        # Trials suggested but not chosen this round; pruned at the end
        # so Optuna's KDE doesn't see them.
        to_prune: list[str] = []
        for _ in range(max(1, self.max_resamples)):
            sugg = self.inner.suggest()
            score = self.feasibility.predict_proba(sugg.config)
            if score >= self.threshold:
                # Accept: prune everything we rejected before this.
                for tid in to_prune:
                    self.inner.prune(tid)
                self._suggested_configs[sugg.trial_id] = sugg.config
                return sugg
            # Below threshold; track the best fallback.
            if score > best_score:
                if best_sugg is not None:
                    to_prune.append(best_sugg.trial_id)
                best_sugg = sugg
                best_score = score
            else:
                to_prune.append(sugg.trial_id)
        # All attempts below threshold — return the least-bad. Prune
        # the rest so the perf KDE only sees the one we'll actually run.
        for tid in to_prune:
            self.inner.prune(tid)
        # ``best_sugg`` is non-None as long as max_resamples >= 1.
        assert best_sugg is not None
        self._suggested_configs[best_sugg.trial_id] = best_sugg.config
        return best_sugg

    def record(
        self,
        trial_id: str,
        measurement: Measurement | None,
        failure: FailureKind | None,
    ) -> None:
        # Recover the config we associated with this trial at suggest
        # time so the feasibility model sees the actual run config.
        config = self._suggested_configs.pop(trial_id, None)
        if config is None:
            raise KeyError(f"no suggested config for trial {trial_id!r}")
        # Perf surrogate learns first (existing penalty/score logic).
        self.inner.record(trial_id, measurement, failure)
        # Feasibility classifier learns next.
        self.feasibility.record(
            config,
            success=(failure is None),
            failure_kind=failure,
        )

    def prune(self, trial_id: str) -> None:
        """Pass-through to the inner surrogate."""
        self._suggested_configs.pop(trial_id, None)
        self.inner.prune(trial_id)
