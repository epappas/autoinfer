# CLAUDE.md

Guidance for Claude Code working on the autoinfer repository.

## What this project is

Three-layer search over LLM inference optimization:

- **L1** engine config — vLLM `EngineArgs` + runtime flags
- **L2** hardware topology — Basilica deployment (GPU class, count, TP×PP×EP, PD-disagg)
- **L3** kernel — vLLM custom ops (AutoKernel-style)

over a **shared substrate** (workload driver, quality gate via live
reference replica, Pareto keep-discard ledger, hybrid LLM+surrogate
policy). Layers are adapters onto the substrate, not separate subprojects.

## Read this first

- `docs/research/references/00-hypothesis-seed.md` — thesis, C1–C9 claims
  with evidence status, P1–P12 design principles. Every architectural
  decision should trace to a principle or a claim.
- `docs/research/raw/references-*.md` — evidence backing C1–C9.

## Invariants

Enforced by code review; violations are bugs:

- **P1** Three layers from day one. `src/autoinfer/layers/` always has
  l1_engine / l2_topology / l3_kernel directories even if stubbed.
- **P2** Shared substrate before layer work. No adapter ships before the
  harness module it depends on.
- **P3** Layers are adapters. Each implements `LayerAdapter` Protocol in
  `src/autoinfer/layers/__init__.py`.
- **P4** Cross-layer stale-signal invalidation. `Ledger.mark_stale()`
  is the only path to invalidate cached results.
- **P7** Hybrid policy. Pure LLM-guided search is banned by C6 evidence.
- **P8** Live reference replica for quality. No cached-logit gates.
- **P9** Failure is a first-class signal — typed `FailureRecord`, fed
  to the surrogate, never a zero-reward hole.
- **P10** Frozen/mutable boundary. `harness/` is frozen per run;
  layers/policy are mutable. Policy never edits the harness.
- **P11** Typed (mypy strict), modular, SOLID. Functions ≤50 LoC.
  Assert-and-fail-fast over nested if/else. Pydantic at config
  boundaries.

## Build & test

```bash
# install (uv is the primary tool)
uv sync --extra dev

# optional extras
uv sync --extra dev --extra policy           # Optuna
uv sync --extra dev --extra vllm             # vLLM runtime
uv sync --extra dev --extra basilica         # Basilica SDK

# run tests
uv run pytest -q                             # all CPU tests
uv run pytest -m "not gpu and not basilica"  # skip heavy
uv run pytest tests/test_ledger.py -q        # single file

# lint + type
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run mypy src/
```

## Architecture (current state)

Scaffolded session one — architectural foundation only. No GPU code yet.

```
src/autoinfer/
├── __init__.py
├── cli.py                 (stub — next session)
├── config.py              pydantic RunConfig + sub-models
├── controller/
│   ├── continuous.py      (stub — outer loop)
│   └── stale.py           (stub — cross-layer scheduler)
├── harness/               shared substrate
│   ├── __init__.py        exports core types
│   ├── failure.py         typed failure outcomes (P9)
│   ├── ledger.py          Pareto frontier + stale-signal (P4)
│   ├── driver.py          (stub — vllm bench serve wrapper)
│   ├── gate.py            (stub — logit divergence + batch invariance)
│   └── replica.py         (stub — FP16 reference replica lifecycle)
├── policy/                hybrid stack (P7)
│   ├── warmstart.py       (stub — LLM initial design)
│   ├── surrogate.py       (stub — Optuna TPE / CMA-ES)
│   ├── fidelity.py        (stub — Hyperband/BOHB)
│   └── operator.py        (stub — LLM proposal operator)
├── layers/                per-layer adapters (P3)
│   ├── __init__.py        LayerAdapter Protocol, TrialInput/Output
│   ├── l1_engine/         (stub — real in next session)
│   ├── l2_topology/       (stub)
│   └── l3_kernel/         (stub; kernel sandbox subdir)
├── target/
│   ├── local.py           (stub)
│   └── basilica.py        (stub; reuse autoresearch-rl pattern)
└── telemetry/
    ├── run.py             (stub)
    └── trace.py           (stub)
```

## Working conventions

- Prefer editing existing files over creating new ones.
- Never add comments that explain what code does — let names carry that.
  Only comment *why* when the reason is non-obvious.
- No emojis in code, commits, or comments.
- Never write `git add -A`. Stage specific paths.
- Do not skip tests or delete code to make tests pass.
- Do not fake, mock, or stub what the user asked to be real.
- Never claim a task complete without running and verifying it.

## Backlog discipline

`TODO.md` at repo root is the **single source of truth** for open
tasks, tracked corner-cuts, and research extensions. Bands by
priority (P0 / P1 / P2). Update as part of every commit that opens
or closes an item — the commit that fixes T-XX moves the row to the
"Closed" section and references the closing commit hash.

Don't track open work in commit messages, code TODOs, or scratch
files. If it's worth tracking, it goes in `TODO.md`.

## Campaign discipline

**No campaign launches without a pre-registration document on
`main`.** Pre-registration goes in
`docs/research/campaigns/NN-<name>-<date>.md` using the structure in
`TEMPLATE.md`:

1. Goal — questions to answer with measurable success criteria.
2. Pre-flight changes — concrete commit references; the run starts
   from a known commit.
3. Configuration — exact RunConfig + launch command + overrides.
4. Expected timeline + cost.
5. Expected outcomes (predictions written *before* data is seen).
6. Decision tree from the data.

After the run, the "Outcome" section reconciles predictions with
reality, including cases where predictions were wrong. The campaign's
analysis writeup goes into `docs/research/references/NN-<analysis>.md`
as before; the pre-registration is the *prediction* document, the
reference is the *reconciliation* document.

## Thesis → code traceability

Every non-trivial PR should cite the principle (P#) or claim (C#) it
implements. Example commit trailers:

```
Implements: P4 (cross-layer stale-signal invalidation)
Evidence:   C9 (reference replica required)
```
