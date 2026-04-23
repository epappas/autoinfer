"""Continuous outer loop (P10 frozen/mutable boundary).

Composes the shared substrate (ledger + quality gate, etc. — the gate
is injected by the caller; see ``QualityHook``) with the hybrid policy
(warmstart + surrogate + operator) against one or more ``LayerAdapter``
instances selected by the ``LayerScheduler``.

The runner never raises from a trial: adapters convert all internal
errors to ``FailureRecord`` (P9). The loop terminates on:

1. every layer exhausting its ``max_trials`` budget, OR
2. the caller-supplied stop callable returning ``True``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from autoinfer.controller.stale import LayerScheduler, LayerSpec, history_projection
from autoinfer.harness.ledger import Entry, Ledger
from autoinfer.layers import TrialInput
from autoinfer.policy.operator import Operator

StopCallable = Callable[[Ledger], bool]


@dataclass
class StallTracker:
    """Tracks the longest run of non-improving trials per layer."""

    threshold: int
    best: dict[str, float] = field(default_factory=dict)
    counter: dict[str, int] = field(default_factory=dict)

    def record(self, layer: str, value: float | None) -> None:
        if value is None:
            self.counter[layer] = self.counter.get(layer, 0) + 1
            return
        prev = self.best.get(layer)
        if prev is None or value > prev:
            self.best[layer] = value
            self.counter[layer] = 0
        else:
            self.counter[layer] = self.counter.get(layer, 0) + 1

    def stalled(self, layer: str) -> bool:
        return self.counter.get(layer, 0) >= self.threshold

    def reset(self, layer: str) -> None:
        self.counter[layer] = 0


@dataclass
class ContinuousRunner:
    scheduler: LayerScheduler
    ledger: Ledger
    objective_axis: str
    maximize: bool = True
    stall_threshold: int = 8
    operator: Operator | None = None
    stop: StopCallable | None = None

    def __post_init__(self) -> None:
        self._tid_idx: dict[str, int] = dict.fromkeys(self.scheduler.specs, 0)
        self._stall = StallTracker(self.stall_threshold)
        self._trials_since_operator: dict[str, int] = dict.fromkeys(self.scheduler.specs, 0)

    def run(self) -> list[Entry]:
        while True:
            layer = self.scheduler.pick_layer(self.ledger)
            if layer is None:
                break
            if self.stop is not None and self.stop(self.ledger):
                break
            self._step(layer)
        return self.ledger.pareto_front()

    def _step(self, layer: str) -> None:
        spec = self.scheduler.specs[layer]
        if self.scheduler.is_warmstart_needed(layer):
            self._run_warmstart(layer, spec)
            self.scheduler.mark_warmstart_done(layer)
            return
        if self._should_invoke_operator(layer):
            self._run_operator_batch(layer, spec)
            return
        self._run_surrogate_trial(layer, spec)

    def _run_warmstart(self, layer: str, spec: LayerSpec) -> None:
        configs = spec.warmstart.propose_configs(
            surface=spec.adapter.surface(),
            n=min(spec.warmstart_n, spec.max_trials),
            prior_notes=spec.warmstart_prior,
            history=history_projection(self.ledger, layer),
        )
        for cfg in configs:
            if not self.scheduler.has_budget(layer):
                return
            tid = self._next_tid(layer, prefix="w")
            self._execute(layer, spec, tid, cfg, surrogate_tid=None)

    def _run_operator_batch(self, layer: str, spec: LayerSpec) -> None:
        assert self.operator is not None
        configs = self.operator.propose(
            surface=spec.adapter.surface(),
            n=1,
            prior_notes=spec.warmstart_prior,
            history=history_projection(self.ledger, layer),
        )
        for cfg in configs:
            if not self.scheduler.has_budget(layer):
                return
            tid = self._next_tid(layer, prefix="o")
            self._execute(layer, spec, tid, cfg, surrogate_tid=None)
        self._trials_since_operator[layer] = 0
        self._stall.reset(layer)

    def _run_surrogate_trial(self, layer: str, spec: LayerSpec) -> None:
        sugg = spec.surrogate.suggest()
        tid = self._next_tid(layer, prefix="s")
        self._execute(layer, spec, tid, sugg.config, surrogate_tid=sugg.trial_id)

    def _execute(
        self,
        layer: str,
        spec: LayerSpec,
        tid: str,
        cfg: dict[str, object],
        surrogate_tid: str | None,
    ) -> None:
        out = spec.adapter.run(TrialInput(trial_id=tid, config=cfg))
        entry = Entry(
            trial_id=tid,
            layer=layer,
            config=cfg,
            measurement=out.measurement,
            failure=out.failure,
        )
        self.ledger.record(entry)
        if surrogate_tid is not None:
            failure_kind = out.failure.kind if out.failure is not None else None
            spec.surrogate.record(surrogate_tid, out.measurement, failure_kind)
        self.scheduler.notify_trial_done(layer)
        self._trials_since_operator[layer] = self._trials_since_operator.get(layer, 0) + 1
        self._stall.record(layer, self._score(entry))

    def _score(self, entry: Entry) -> float | None:
        if entry.measurement is None:
            return None
        return entry.measurement.value(self.objective_axis)

    def _should_invoke_operator(self, layer: str) -> bool:
        if self.operator is None:
            return False
        return self.operator.should_propose(
            trials_since_last=self._trials_since_operator.get(layer, 0),
            stalled=self._stall.stalled(layer),
        )

    def _next_tid(self, layer: str, prefix: str) -> str:
        idx = self._tid_idx[layer]
        self._tid_idx[layer] += 1
        return f"{layer}_{prefix}{idx:04d}"
