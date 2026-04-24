# Cloudflare Omni — multi-model GPU multiplexing (raw, 2026-04-23)

Primary: Cloudflare engineering blog, *"How Cloudflare runs more AI
models on fewer GPUs: A technical deep-dive"*. Authors: Sven Sauleau
(@svensauleau), Mari Galicer. Published 2025-08-27, 8-minute read.
URL: https://blog.cloudflare.com/how-cloudflare-runs-more-ai-models-on-fewer-gpus/

Scope of this note: a scholarly extraction of the architecture, numbers,
techniques, and caveats described in the post, then a mapping to the
autoinfer thesis (P-principles, C-claims). Source material is a vendor
engineering blog, not a peer-reviewed paper — quantitative claims are
the company's own and are not independently reproduced.

---

## 1. Problem statement (the WHY)

Cloudflare serves many models through Workers AI. Two observations make
single-model-per-GPU deployment inefficient:

- **Traffic is non-uniform.** Not every model is equally utilised —
  popular models saturate their GPUs, long-tail models sit idle but
  still own the device's HBM.
- **Edge placement amplifies the waste.** Cloudflare's stated goal is
  "place GPUs as close as we possibly can to people and applications
  that are using them." Edge POPs multiply the per-node cost of
  underutilisation; shaving one GPU per site matters.

The traditional "one model per container/VM" pattern is described in
the post as "simple. But it's also heavy-handed — because it requires
managing the entire stack from provisioning the VM, installing GPU
drivers, downloading model weights, and managing the Python
environment." Omni's justification: *"Efficiency is a core value at
Cloudflare, and with GPUs being the scarce commodity they are, we
realized that we needed to build something to fully maximize our GPU
usage."*

Omni's stated purpose: *"allow us to run models more efficiently by
spawning them from a single control plane and implementing lightweight
process isolation."*

---

## 2. System architecture (the WHAT)

Omni is a **node-local orchestration layer** underneath the inference
frameworks, not a serving framework itself. Inference frameworks (vLLM,
plain Python handlers, Cloudflare's own **Infire** engine) plug in as
backends.

### 2.1 Deployment modes

Two modes are described:

1. **Container with multiple models** — isolation mechanisms used to
   pack several models onto one container / GPU.
2. **Bare metal with one model** — isolation used only for process +
   Python-venv hygiene; overcommit mechanics still apply.

### 2.2 Topology

- **One scheduler per node** ("single control plane").
- **Per-model child processes**, each with its own CUDA context.
- **IPC** between scheduler and children for request dispatch.
- **Workers KV** stores per-model configuration, loaded by the
  scheduler at startup / model provision time.

### 2.3 Scheduler responsibilities (all reported in the post)

- Receive inference requests, route to the right model process.
- Distribute load across multiple GPUs on the node.
- Ensure model processes are running; restart on failure (including
  OOM kills) without disturbing siblings.
- Roll out new model versions.
- Collect billing metrics.
- Emit logs.
- For feature-laden requests (prompt templating, tool calling) the
  scheduler **buffers** and transforms before dispatch.
- For large binary payloads, the scheduler **hands the TCP connection
  directly** to the child process (zero-copy path).

### 2.4 Request lifecycle

The post describes the global and local paths:

**Global routing (Workers AI plane):**
1. Request enters Workers AI.
2. Model config is fetched from Workers KV.
3. Routing layer forwards to the **nearest** Omni instance **with
   available capacity**.
4. For async batch requests: the router targets **idle** Omni
   instances, typically at night-time POPs (batch is time-shifted to
   underutilised regions).

**Node-local execution:**
1. Omni runs pre-checks on the request.
2. Model-specific pre-processing runs.
3. Request is passed to the model process over IPC (or TCP handoff
   for binaries).
4. Inference executes (vLLM / Python / Infire).
5. Model-specific post-processing.
6. Response is returned as one of: JSON object, SSE stream (text
   generation), binary (image generation).

---

## 3. Isolation mechanisms (the HOW — part 1)

### 3.1 Process + namespace isolation

Each model runs as a **separate OS process**, each with **its own mount
namespace**. The post explicitly demonstrates mount-namespace entry
with `nsenter -t <pid> -m` when showing the faked `/proc/meminfo`.

Why separate processes (not threads / not a single serving process):

- **Error recovery.** If a model's CUDA kernel faults, the process
  (and its CUDA context) can be torn down and restarted without
  affecting co-tenants.
- **Different CUDA contexts.** One context per process naturally
  partitions GPU state.
- **Independent Python venvs.** Each model ships its own dependency
  closure.

### 3.2 cgroups for host RAM

Each model is placed in a cgroup with a **CPU memory limit**. Example
from the post: a container with 15 GiB total memory hosts a model
capped at 7 GiB. The cgroup enforces the cap at the kernel level; OOM
kills are scoped to the offending model.

### 3.3 `uv` venvs with symlinked dependency sharing

- Dependency manager: **`uv`** (Astral; https://docs.astral.sh/uv/).
- Each model has its own venv.
- Dependencies are declared in `requirements.txt` (example shown:
  `cowsay==6.1`).
- An internal Python registry **mirrors the public registry** so that
  installs at model startup are fast and survive public registry
  outages.
- Common dependencies are **symlinked** across venvs, so the same
  wheel on disk is shared rather than copy-duplicated per model.

### 3.4 FUSE-backed virtual `/proc/meminfo`

This is the single most technically specific trick in the post. The
problem chain:

1. Python code (and, critically, `psutil.virtual_memory()`) reads
   `/proc/meminfo` to decide how much RAM exists.
2. `psutil` **does not consult cgroup v2 limits**; it reports the
   host's figures.
3. Serving frameworks pre-allocate against that figure and OOM the
   cgroup before they ever trigger a Python-level guard.

Omni's fix:

- A FUSE (filesystem-in-userspace) mount exposes a **custom
  `/proc/meminfo`** whose numbers reflect the model's cgroup limit
  and current usage.
- This custom file is bind-mounted at `/proc/meminfo` inside the
  model's mount namespace.
- The post shows the effect directly: `nsenter -t <pid> -m cat
  /proc/meminfo` reports 7 GiB `MemTotal` for the 7 GiB-capped model
  even though the container has 15 GiB.

The WHY: *"Python doesn't account for cgroups memory limits"*, so the
only tractable workaround without patching every framework is to lie
to `/proc/meminfo` at the namespace boundary.

---

## 4. GPU memory overcommit (the HOW — part 2)

This is the trick that delivers the headline numbers.

### 4.1 CUDA Unified Memory as the substrate

NVIDIA's **Unified Memory** gives GPU and CPU a **single shared address
space**; pages are demand-migrated between HBM and host DRAM by the
CUDA driver. Reference the post cites: the NVIDIA dev-blog post
"Maximizing Unified Memory Performance in CUDA".

Omni exploits unified memory as a **paging tier**: models that are
active reside in HBM, idle models migrate to host DRAM, and PCIe moves
pages back on demand when traffic arrives.

### 4.2 CUDA stub / allocator interception

Omni **injects a CUDA stub library** (mechanism not named in the post
but conventionally `LD_PRELOAD`) that **intercepts allocation calls**:

- `cuMalloc*` (CUDA driver API)
- `cudaMalloc*` (CUDA runtime API)

All intercepted allocations are **rewritten to unified-memory mode**
(equivalent of `cudaMallocManaged`). The frameworks above (vLLM,
PyTorch caching allocator, Infire) believe they are getting device
memory; they are actually getting managed memory that the driver can
page.

### 4.3 Per-model memory view

Frameworks commonly pre-reserve "free GPU memory" at startup
(e.g. vLLM's `gpu_memory_utilization` × `cuMemGetInfo(free)`). If Omni
did not intervene, the first model to start would claim the whole card.

Omni **overrides** the introspection APIs:

- **Runtime:** `cudaMemGetInfo`
- **Driver:** `cuMemGetInfo`

Each model sees **only the subset of GPU memory allocated to it**,
which prevents greedy pre-allocation. The CUDA stub thus does two
things simultaneously: rewrite allocations to unified-memory, and lie
about free memory.

### 4.4 The paging narrative

Worked example from the post (models A, B, C; C larger than A+B
combined):

1. Initial state: A and B resident in HBM; C resident in CPU memory.
2. Request for C arrives → A and B pages migrate out; C migrates in.
3. Request for B arrives → part of C migrates out; B migrates back in.
4. Request for A arrives → A migrates back in; C fully migrates out.

Active set churns; inactive set waits on the CPU side.

### 4.5 Headline numbers

- **~400% GPU memory overcommit** on a single GPU.
- **13 models** running simultaneously on one GPU.
- **~4 GPUs saved** vs. single-model-per-GPU.

### 4.6 The cost: cold-start migration latency

PCIe 4.0 ×16 peak bandwidth is **32 GB/s**. The post derives a first-
token latency floor for cold models:

- A **5 GiB** model migrating cold from DRAM to HBM takes
  **~156 ms** over PCIe 4.0 (32 GB/s).
- *"PCIe 4.0 transfer penalty [is] minimal for small models."*

This is a latency tax, not a correctness tax — logits are unchanged.
The post does not report a percentile distribution, only the
point estimate.

---

## 5. Developer-facing interface

### 5.1 Handler pattern

The user-facing unit of deployment is a **Python handler function**,
explicitly compared to the JavaScript Workers pattern:

```python
from omni import Response

def handle_request(request, context):
    try:
        json = request.body.json()
        text = json["text"]
    except Exception as err:
        return Response.error(...)
    return [model_logic_result]
```

Characteristics from the post:

- Signature `handle_request(request, context)`.
- **May be `async`**.
- **May return Pydantic objects**; Omni serialises them to Workers AI
  responses automatically.
- Error path returns `Response.error(...)`.

### 5.2 The `omni` Python package

Injected at runtime into the venv; also **published as a regular
Python package** so the same code can run under pytest.

Testing pattern shown in the post:

```python
from omni import Context, Request
from model import handle_request

def test_basic():
    ctx = Context.inactive()
    req = Request(json={"text": "..."})
    out = handle_request(req, ctx)
```

### 5.3 Built-in Workers AI features passed through

- Batching (incl. async Batch API).
- Function calling.
- Prompt templating (applied by the scheduler before dispatch).
- Per-model customisation hooks (pre/post processing).

### 5.4 Response types

- JSON objects.
- SSE streams for text generation.
- Binary for image generation.

---

## 6. Failure handling

- **Per-model OOM kill → scheduler restarts the model**, siblings
  untouched (cgroup-scoped kill + separate CUDA context).
- **Model process crash** is isolated by the separate CUDA context; no
  cascade.
- **Scheduler self-restarts** on its own error conditions.
- **Version rollout** is scheduler-driven — implies health-checked,
  gradual.

The post does not describe: how GPU-level ECC / XID errors are
handled, whether a single GPU fault drains the whole node, or the
behaviour of unified-memory thrashing at the limit.

---

## 7. Current deployment status (as of 2025-08-27)

- Omni runs in production for *"a handful"* of Workers AI models.
- *"More every week."*
- Not all models are equally utilised; the overcommit design is
  motivated by this variance.

---

## 8. Mapping to autoinfer — WHY this matters for us

### 8.1 Orthogonal L2 axis

The existing L2 corpus (`docs/research/raw/references-L2-topology.md`)
— Splitwise, DistServe, Mooncake, Sarathi-Serve, SPAD, HexGen-2,
Hetis, Dynamo, llm-d — optimises **one model across many GPUs**:
prefill/decode disaggregation, heterogeneous pools, KV-cache tiers.

Omni optimises **one GPU across many models**. These are complementary
axes of L2. Autoinfer's claim to cover L2 is weaker if it only handles
the first. Concretely: the topology adapter today assumes sole tenancy
per device. A **co-tenancy / residency adapter** is the missing
sibling.

### 8.2 The three techniques are independently reusable

All three are usable **inside the autoinfer harness itself**, not just
as objects of study:

- **CUDA allocator stub for memory isolation.** Lets a tuning run
  cap a trial's footprint without frameworks pre-reserving the
  whole card. Multiple trials on one device becomes feasible.
- **FUSE `/proc/meminfo` shim.** Silently fixes a bug class for any
  Python tuner running in cgroups-v2 containers: `psutil` otherwise
  reports host memory, memory-aware schedulers mis-trigger,
  OOM-guards fire at the wrong thresholds.
- **Unified-memory fallback.** A concrete, named failure mode (P9)
  for trials that overflow: latency spike + PCIe saturation, not
  OOM kill.

### 8.3 Principle alignment

- **P1 / P3 (three layers, adapter shape).** Omni describes the host-
  policy layer beneath per-model topology. `l2_topology` today has
  no co-tenancy concept; this is a gap.
- **P4 (cross-layer stale-signal invalidation).** Overcommit makes
  "free memory" a time-varying, neighbour-dependent quantity. Unified-
  memory migration triggered by a neighbour wake-up invalidates the
  current trial's `gpu_memory_utilization` assumption and its TTFT
  measurement. This is exactly what `Ledger.mark_stale()` exists for —
  provided the harness can observe co-tenancy events.
- **P8 / C9 (live reference replica).** Unified-memory migration is
  correctness-preserving and latency-destroying. A logit-parity gate
  stays green while goodput collapses. Argues that the replica
  comparison must also report **percentile latency**, not just
  sample equivalence.
- **P9 (typed failure).** Two new `FailureRecord` categories:
  - `UnifiedMemoryThrash` — PCIe-bound migration dominating step
    time.
  - `CgroupMemMisreport` — Python / `psutil` sees host memory; OOM
    mid-trial without a Python-level warning.

### 8.4 Evidence bearing on C-claims

- **C1 (real knobs worth searching):** **+1.** Overcommit factor,
  residency policy, per-model memory cap are ordinal/continuous knobs
  with a measurable Pareto of goodput vs. cold-start P99.
- **C4 (topology Pareto-dominates):** **neutral.** Same direction
  (heterogeneity wins) on a different axis (tenancy, not P/D split).
  Extends C4's scope rather than confirming it.
- **C6 (pure LLM search insufficient):** **+1.** No LLM invents
  "inject a CUDA stub that rewrites allocations to unified memory"
  from first principles. Once the mechanism exists, tuning the
  overcommit factor is a clean surrogate problem (P7).
- **C9 (reference replica required):** **+1.** Overcommit is
  correctness-preserving, latency-destroying; only a latency-tracking
  gate catches it.

---

## 9. Open questions

1. Does the CUDA allocator stub compose with **vLLM's PagedAttention
   + CUDA graph capture**? Graph capture against a stub-controlled
   allocator is non-trivial; the post implies "yes" for vLLM as a
   backend but shows no numbers.
2. What is the **actual first-token penalty distribution** (not just
   the 5 GiB / 156 ms point estimate) under realistic mixed traffic?
   This is precisely the empirical curve the autoinfer harness can
   produce if given the mechanism.
3. Can per-model CUDA contexts **share a KV cache tier** (Mooncake-
   style global KV across co-located models on the same node)?
   Orthogonal to the post but implied by the architecture.
4. Is the FUSE `/proc/meminfo` shim **upstreamable** (to the kernel,
   `psutil`, or cgroups tooling) or is this a permanent Cloudflare
   patch? Affects whether external tuners can adopt it cheaply.
5. GPU-level failure behaviour: does an **XID / ECC event** stay
   confined to the offending model's CUDA context, or does the node
   drain? Post claims isolation but does not benchmark.
6. Behaviour at the **overcommit ceiling** — at what aggregate
   working-set : HBM ratio does unified-memory thrash collapse
   throughput? The 400% headline is a successful point; the failure
   boundary is unstated.
7. **Scheduler fairness** under sustained co-tenancy — does Omni
   priority-serve hot models, FIFO, weighted fair? Not described.
8. **Security posture** of the CUDA stub: does allocator rewriting
   survive an attacker-controlled workload inside one tenant, or is
   per-tenant CUDA context enough?

---

## 10. Reference list

### 10.1 Primary

- Sauleau, S. & Galicer, M. "How Cloudflare runs more AI models on
  fewer GPUs: A technical deep-dive." Cloudflare Blog, 2025-08-27.
  https://blog.cloudflare.com/how-cloudflare-runs-more-ai-models-on-fewer-gpus/

### 10.2 Cited or directly implied by the post

- Cloudflare, "Infire — Cloudflare's most efficient AI inference
  engine." https://blog.cloudflare.com/cloudflares-most-efficient-ai-inference-engine/
- Workers AI docs. https://developers.cloudflare.com/workers-ai/
- Workers AI models catalog.
  https://developers.cloudflare.com/workers-ai/models/
- Workers AI Batch API.
  https://developers.cloudflare.com/workers-ai/features/batch-api/
- Workers KV docs. https://developers.cloudflare.com/kv/
- Astral `uv`. https://docs.astral.sh/uv/
- NVIDIA, "Maximizing Unified Memory Performance in CUDA."
  https://developer.nvidia.com/blog/maximizing-unified-memory-performance-cuda/
- FUSE (Wikipedia).
  https://en.wikipedia.org/wiki/Filesystem_in_Userspace
- cgroups (Wikipedia).
  https://en.wikipedia.org/wiki/Cgroups
- `psutil`. https://github.com/giampaolo/psutil
- Pydantic. https://docs.pydantic.dev/

### 10.3 Autoinfer cross-references

- `docs/research/references/00-hypothesis-seed.md` — C1, C4, C6, C9;
  P1, P3, P4, P7, P8, P9.
- `docs/research/raw/references-L2-topology.md` — the
  "one-model-many-GPUs" corpus this note is orthogonal to.
- `docs/research/raw/references-L1-engine-config.md` — vLLM
  `gpu_memory_utilization` knob interacts with Omni-style overcommit.
