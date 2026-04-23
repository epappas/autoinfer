"""Classical-surrogate samplers (Optuna TPE / CMA-ES / random).

Implements the bulk of the hybrid policy (P7) after the LLM warm-start.
Per C6 evidence (arxiv 2603.24647), this surrogate is expected to
dominate the search at ~100-500 evaluation budgets; the LLM operator
only re-enters on stall.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Protocol, runtime_checkable

from autoinfer.harness.failure import FailureKind
from autoinfer.harness.ledger import Measurement


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


class OptunaSurrogate:
    """Optuna-backed surrogate; supports ``tpe``, ``cmaes``, ``random``."""

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
            self._study.tell(trial, state=self._optuna.trial.TrialState.FAIL)
            return
        if measurement is None:
            raise ValueError("measurement required when failure is None")
        self._study.tell(trial, measurement.value(self._axis))

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
