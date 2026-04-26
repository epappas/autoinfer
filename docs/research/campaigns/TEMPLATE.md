# Campaign N — <name> (YYYY-MM-DD)

**Status:** PLANNED | RUNNING | COMPLETE | ABANDONED

Pre-registration of an experiment campaign. Written and committed
**before** launch. The "Outcome" section at the bottom is filled in
after the run, reconciling predictions with reality.

Discipline:

1. No campaign launches without a pre-registration doc on `main`.
2. Pre-flight changes listed below land in earlier commits, not in
   the campaign run itself, so the run is reproducible from a known
   commit.
3. Predictions are written before data is seen. Outcome reconciles
   honestly — including cases where predictions were wrong.

---

## Goal — questions to answer

What this campaign is designed to answer. Each Q is a question that
the run's data will resolve, with a measurable success criterion.

| # | Question | Mechanism | Success criterion |
|---|---|---|---|
| Q1 | … | … | … |

## Pre-flight changes

Code or config changes that must land before launch. Each item is a
concrete commit reference; the run uses the resulting state.

| Change | Why | Commit |
|---|---|---|
| … | … | `…` |

## Configuration

The exact RunConfig the campaign uses, plus any campaign-runner /
orchestrator flags. Either inline or by reference to
`examples/<config>.yaml` at a specific commit.

```
config:           examples/...
launch:           ./scripts/launch_joint_campaign.sh --mode full --yes
overrides:        ...
```

## Expected timeline

| Phase | Trials | Per-trial wall | Total |
|---|---|---|---|
| … | … | … | … |
| **Total** | | | **~X h** |

## Expected cost

- Compute: $…
- LLM API: $…
- Total: ~$…

## Expected outcomes

For each likely outcome bucket, what the data would look like and
what it would mean.

### Outcome A — …
*Probability:* low | medium | high

What the artifacts would show. What it answers about Q1/Q2/Q3.

### Outcome B — …
…

## Decision tree from the data

```
If <observation>:
    → <next-step>
If <observation>:
    → <next-step>
```

---

## Outcome (filled in after the run)

**Status:** COMPLETE | ABANDONED

### Headline numbers

### Reconciliation with predictions

| Prediction | Actual | Match? |
|---|---|---|
| Outcome A: … | … | yes/no |

### What the data tells us about each Q

### Bugs surfaced and their fixes

### What's still open after this run

### Cost actually spent

### Artifacts

- `docs/research/raw/<campaign-dir>/`
- `docs/research/references/<NN>-<analysis>.md`
- Commits: `…`
