# autoinfer

Discover the most optimal way to fine-tune inference engines so a given model
is served on the right hardware at its optimal operating point. "Optimal" is
open across four first-class axes (tokens/s, memory, context length, quality)
with tokens/\$ as the derived economic axis.

Autoinfer is a **three-layer search system** — engine config, hardware
topology, and kernel implementation — over a shared substrate (workload
driver, quality gate via live reference replica, keep-discard ledger, hybrid
policy). Layers are adapters, not phases; they come online in parallel once
the substrate exists. See `docs/research/references/00-hypothesis-seed.md`
for the thesis, C1–C9 claims with evidence status, and design principles
P1–P12.

Engine substrate: **vLLM**. Deployment substrate: **Basilica** (heterogeneous
decentralized GPU fleet). Iteration-zero model: **Qwen3-8B**; Covenant-72B
enters later as validator.

## Relationship to sibling projects

| Project | Loop | Artifact |
|---------|------|----------|
| `autoresearch-rl` | LLM proposes params / code diffs → train → eval → keep/discard | trained model |
| `AutoKernel` (RightNow AI) | LLM proposes kernel → benchmark → keep/discard | optimized kernel |
| `autoinfer` (this) | Hybrid LLM+surrogate proposes engine / topology / kernel changes → serve → bench → keep/discard | inference deployment (config + topology + kernels) |

Same frozen/mutable boundary as autoresearch-rl. Same LLM-agent-loop pattern
as AutoKernel. Extended to **three layers with cross-layer stale-signal
invalidation** — the contribution no existing system makes.

## Layout

```
docs/research/raw/          # per-source notes, immutable after write
docs/research/references/   # compiled references + thesis
src/autoinfer/              # package (planned — see thesis §9)
  harness/                    shared substrate (driver, gate, replica, ledger)
  policy/                     LLM warm-start + TPE/CMA-ES + Hyperband + operator
  layers/{l1_engine,l2_topology,l3_kernel}/
                              per-layer adapters
  controller/                 outer loop + cross-layer stale-signal scheduler
  target/{local,basilica}/    deployment backends
examples/                   # per-slice configs (first: qwen3-8b-l1-slice)
```

External knowledge lives in alexandria (llmwiki) under the `global` workspace.

## Status

Iteration zero seeded 2026-04-22. Thesis and evidence review complete
(four research tracks, ~9.4k words, 100+ primary sources). Next: scaffold
the Python package per thesis §9 and spin up the shared substrate before
any adapter ships.
