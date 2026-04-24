# vLLM Router — multi-replica serving dataplane (raw, 2026-04-23)

Primary sources (both public, Apache-2.0):

- **Upstream**: [`vllm-project/router`](https://github.com/vllm-project/router) — the vLLM project's official Rust dataplane. ~200 stars, 71 forks, 31 open issues at snapshot time (2026-04-22). Default branch `main`, last push 2026-04-17. Rust crate `vllm_router_rs` v0.1.22 (crate `name=vllm_router_rs`, binary `name=vllm-router`).
- **PrimeIntellect fork**: [`PrimeIntellect-ai/router`](https://github.com/PrimeIntellect-ai/router) — managed-inference tenancy overlay. +24 / -10 vs upstream at snapshot time; last push 2026-04-16. Apache-2.0.
- **PyPI**: `pip install vllm-router` (Python launcher wrapping the PyO3 extension module).

This note is the authoritative source of truth inside autoinfer for what this dataplane actually does, so that later work can reference it without re-reading the Rust. Everything below was cross-checked against the source at fork-`HEAD` 2026-04-23.

## 1. WHAT it is — and what it is not

### 1.1 What it is

A **stateless request-forwarding dataplane** (OpenAI-compatible reverse proxy) that sits in front of a fleet of vLLM workers and decides, per request, which worker handles it. Compiled to a single Rust binary (`vllm-router`) and simultaneously exposed as a PyO3 extension module (`vllm_router_rs`) for a Python launcher (`pip install vllm-router`). Apache-2.0.

Features it provides:

1. **Five named routing policies** over a shared `LoadBalancingPolicy` trait (`src/policies/mod.rs`).
2. **Two dispatch modes**: Regular (single-worker) and Prefill/Decode-disaggregated (two-worker, NIXL- or NCCL-connector aware).
3. **DP-aware fan-out**: one worker URL expands into N DP-aware ranks when `--intra-node-data-parallel-size > 1`.
4. **Kubernetes-native service discovery** (`kube-rs`, label-selector based, PD-mode aware via distinct prefill/decode selectors).
5. **Reliability primitives**: circuit breaker (Closed→Open→HalfOpen state machine), retry with exponential backoff + jitter, per-worker health checking, pod-unhealthy eviction.
6. **Auth** (pluggable): upstream API-key validation via round-trip to an external URL, or — in the PrimeIntellect fork only — local RS256 JWT verification with run-scoped claim allowlisting.
7. **Observability**: Prometheus exporter (default `127.0.0.1:29000`), structured `tracing` logs, OpenTelemetry-compatible trace-header forwarding (upstream kept full OTel spans; fork removed span emission and kept passthrough).
8. **Request-ID propagation** across configurable headers (defaults `x-request-id`, `x-correlation-id`, `x-trace-id`, `request-id`).

### 1.2 What it is NOT

- **Not a serving engine.** It does not run models. It dispatches HTTP (and streaming SSE) requests to vLLM workers that do.
- **Not a scheduler in the vLLM-internal sense.** It makes coarse "which worker" decisions; vLLM's own V1 scheduler still owns within-worker batching, chunked prefill, KV eviction, etc.
- **Not a config optimiser.** No search, no ledger, no quality gate. Autoinfer's controller is the consumer, not the peer.
- **Not a cluster controller like llm-d.** No CRDs, no Inference Resilience Operator, no planner. Just the dataplane. Meant to be *light-weight* relative to llm-d.

### 1.3 Architectural placement

```
           clients (OpenAI-compatible HTTP + SSE)
                         │
                         ▼
                ┌────────────────────┐
                │   vllm-router      │  ← this project
                │  (axum + tower)    │
                │  policy registry   │
                │  PD dispatcher     │
                │  K8s watcher       │
                │  circuit breaker   │
                └────────┬───────────┘
                         │
        ┌────────────────┼───────────────────┐
        ▼                ▼                   ▼
    vLLM worker      vLLM worker         vLLM worker
    (prefill)        (decode)            (regular / DP rank)
     │   │                │
     └───┴── NIXL / NCCL KV transfer ──┘
```

## 2. WHY it is relevant to autoinfer

Autoinfer's L2 search produces deployment topologies of the form `(gpu_class, count, TP×PP×EP, P/D split, connector)` plus per-role replica counts (thesis §4.2). Any topology with more than one replica per role, or any split prefill/decode topology, requires a dispatching component between the workload driver and the workers. The default (direct-to-worker) routing is:

- **Not realistic** — DistServe, Mooncake, Splitwise baselines all assume a router-mediated path.
- **Not measurable on interesting axes** — without cache-aware dispatch, prefix-cache hit rate is determined by workload arrival order, not by policy.
- **Wrong for PD-disagg** — `vllm-project/router` is the implementation that invokes the NIXL/NCCL connector split; without it, P/D trials can't even be wired up.

So autoinfer does not *depend on* this router the way it depends on vLLM itself, but the moment L2 trials go beyond a single replica or collocated baseline, *some* equivalent must be present. Documenting this router lets us treat "router policy" as a first-class search axis rather than a silently-pinned default.

## 3. HOW it is built — technical deep-dive

### 3.1 Binary and library layout

From `Cargo.toml`:

- Crate: `vllm_router_rs` v0.1.22, `crate-type = ["cdylib", "rlib"]` — same crate powers the `vllm-router` binary (`src/main.rs`) and the Python extension module (via `pyo3 = 0.26`, `extension-module`).
- Runtime stack: `axum 0.8.4` (handlers, WS, tracing), `tower 0.5` + `tower-http 0.6` (trace/gzip/CORS/timeout/limit/request-id), `tokio 1.42.0 (full)`, `reqwest 0.13` (streaming/blocking/JSON), `hyper` via axum, `rustls 0.23 (ring, std)` TLS.
- Concurrency primitives: `dashmap 6.1`, `parking_lot 0.12.4`, `tokio-stream`.
- K8s integration: `kube 1.1.0` + `k8s-openapi 0.25.0 (v1_33)`.
- Observability: `tracing`, `tracing-subscriber (env-filter, json, chrono)`, `metrics 0.24.2`, `metrics-exporter-prometheus 0.17.0`, `opentelemetry 0.27` + `opentelemetry-otlp (trace, grpc-tonic)`.
- Tokenisation: `tokenizers 0.22.2`, `tiktoken-rs 0.7.0`, `minijinja 2.0` (chat templates), `hf-hub 0.4.3 (tokio)` — so the router can tokenize/template server-side when needed for fair routing decisions.
- Release artefacts: `manylinux_2_28` wheels (PR #8 fixed build container), `Dockerfile.router`.

### 3.2 Source-tree map (fork `main` HEAD 2026-04-23)

```
src/
├── main.rs                        — CLI entry (clap derive)
├── lib.rs                         — PyO3 module
├── server.rs                      — axum server, auth middleware wiring
├── handler.rs                     — request handlers (chat/completions/embeddings/etc)
├── auth.rs                        — fork-only: JWT verifier (see §3.7.2)
├── middleware.rs                  — middleware stack
├── metrics.rs                     — Prometheus metric registry
├── service_discovery.rs           — K8s pod watcher (upstream path)
├── tree.rs                        — approximate radix tree for cache-aware (§3.5.5)
├── types.rs                       — shared request/response types
├── otel_http.rs / otel_trace.rs   — OpenTelemetry trace-header handling
├── logger.rs / logging.rs         — tracing setup
├── policies/
│   ├── mod.rs                     — `LoadBalancingPolicy` trait + `CacheAwareConfig`
│   ├── random.rs
│   ├── round_robin.rs
│   ├── power_of_two.rs
│   ├── consistent_hash.rs         — Facebook mcrouter MurmurHash64A port
│   ├── cache_aware.rs             — prefix-match + shortest-queue hybrid
│   ├── factory.rs                 — CLI-string → policy object
│   └── registry.rs                — policy lifecycle
├── routers/
│   ├── router_manager.rs          — top-level router selection
│   ├── factory.rs
│   ├── grpc/                      — gRPC router path (feature-gated)
│   └── http/
│       ├── router.rs              — regular single-worker router
│       ├── openai_router.rs       — OpenAI-compat surface
│       ├── pd_router.rs           — generic PD router
│       ├── vllm_pd_router.rs      — vLLM-specific PD (NIXL/NCCL)
│       ├── pd_types.rs            — PD error types + bootstrap wrappers
│       ├── dp_utils.rs            — DP-rank fan-out
│       ├── logprobs_merge.rs      — merge logprobs across PD legs
│       ├── usage_metrics.rs       — fork-only: per-run token counting (§3.8.2)
│       └── vllm_service_discovery.rs — ZMQ/NCCL bootstrap discovery
├── routes/                        — route tree builder (§3.6)
├── proto/                         — protobuf definitions (grpc)
├── config/                        — config parsing
├── core/                          — `Worker` trait, worker registry
├── data_connector/                — KV-connector abstractions
└── tokenizer/                     — tokenizer helpers
```

### 3.3 The `Worker` and `LoadBalancingPolicy` abstractions

`LoadBalancingPolicy` (`src/policies/mod.rs`) is the only surface a new policy has to implement:

```rust
pub trait LoadBalancingPolicy: Send + Sync + Debug {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<usize>;

    fn select_worker_pair_with_headers(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<(usize, usize)> { /* default: independent select from each pool */ }

    fn on_request_complete(&self, _worker_url: &str, _success: bool) { /* no-op default */ }

    fn update_loads(&self, _loads: &HashMap<String, isize>) { /* no-op default */ }

    fn name(&self) -> &'static str;
    fn needs_request_text(&self) -> bool { false }
    fn needs_headers(&self) -> bool { false }
    fn requires_initialization(&self) -> bool { false }
    fn init_workers(&self, _workers: &[Arc<dyn Worker>]) {}
    fn reset(&self) {}
    fn as_any(&self) -> &dyn std::any::Any;
}
```

The contract is explicit about **what a policy may depend on**: the workers list, the request body text (for prefix matching), and HTTP headers (for session-affinity keys like `X-Session-ID`). Nothing else. This is important for autoinfer: policies cannot depend on the *model*'s internal state, so we can A/B-compare policies on identical engine/topology configurations without touching vLLM.

Healthy-worker filtering is shared: `get_healthy_worker_indices` (`src/policies/mod.rs`) filters by `w.is_healthy() && w.circuit_breaker().can_execute()`. Circuit-breaker-opened workers are invisible to all policies, which means a circuit-broken worker is indistinguishable from a scaled-out-and-deleted worker from the policy's perspective. This matters for fairness in benchmark runs: flaky workers mask good routing decisions.

Model normalisation: `normalize_model_key(model_id)` returns `"default"` for empty / `"unknown"`, otherwise the raw model id. The **cache-aware tree is keyed per model**, so multi-model deployments don't pollute each other's prefix trees.

### 3.4 Dispatch modes

Three modes are built from the two policy entrypoints above:

- **Regular / single-worker**: one worker pool, `select_worker` → one URL. Used for standard fan-out and for DP-aware fan-out.
- **PD-disaggregated**: two disjoint pools (`--prefill` / `--decode` URLs), `select_worker_pair` → `(prefill_idx, decode_idx)`. Optionally each pool has its own policy via `--prefill-policy` / `--decode-policy`.
- **DP-aware fan-out**: `--intra-node-data-parallel-size N` expands each registered worker URL into N DP-rank workers (see `src/routers/http/dp_utils.rs`). This lets one physical vLLM process with `--data-parallel-size N` be addressed as N logical ranks.

### 3.5 Load-balancing policies in detail

#### 3.5.1 `random`

Baseline. Uniform random over healthy indices. `src/policies/random.rs`. Exists to be a control in A/B experiments.

#### 3.5.2 `round_robin`

`src/policies/round_robin.rs`. One `AtomicUsize` counter, `fetch_add(1, Relaxed)`, index modulo `healthy_indices.len()`. Resettable via `reset()`. No per-request state, no headers, no body.

**WHY it matters for autoinfer**: the zero-intelligence baseline. An L2 trial's "worst realistic" TTFT/tail number at fixed topology. Any other policy must beat it or justify its complexity.

#### 3.5.3 `power_of_two` (choices)

`src/policies/power_of_two.rs`. Classic Mitzenmacher-style load-aware policy:

1. Sample two distinct random healthy indices.
2. Query each worker's load via `Worker::load()` (falls back from an internal `cached_loads: RwLock<HashMap<String, isize>>` that can be pushed from external monitoring via `update_loads()`).
3. Route to the lower-loaded one.

Key design note: load can be *pushed in* from outside. This is the hook by which the router can be driven by a vLLM `/metrics` scraper — useful if the signal we care about is in-flight tokens rather than request count.

**WHY it matters for autoinfer**: `power_of_two` is theoretically proven to keep tails tight under Poisson-ish arrival patterns (max load `O(log log n / log 2)` rather than `O(log n / log log n)` for round-robin with infinite buffers). Under bursty real workloads the guarantee weakens. This is exactly the kind of regime a surrogate model can characterise empirically.

#### 3.5.4 `consistent_hash`

`src/policies/consistent_hash.rs`. Session-affinity routing.

- **Hash function**: MurmurHash64A, ported byte-for-byte from **Facebook's mcrouter** (`mcrouter/lib/fbi/hash.c`). Constants `M = 0xc6a4a7935bd1e995`, `R = 47`. Seed is configurable.
- **Hash ring**: `BTreeMap<u64, String>` with `VIRTUAL_NODES_PER_WORKER = 160` virtual nodes per physical worker — a standard ketama-style choice to reduce variance on small N.
- **Hash key selection** (from the README): `X-Session-ID` → `X-User-ID` → `Authorization` → client IP → request-body-derived key. Order matters for workload reproducibility.
- **Rebalance behaviour**: when a worker joins/leaves, only `~1/N` of keys migrate — the property that makes consistent hashing useful in the first place.

**WHY it matters for autoinfer**: `consistent_hash` is the "deterministic prefix affinity without a KV index" option. At moderate workload diversity it approximates cache-aware routing without the radix-tree overhead. Useful as a surrogate-friendly midpoint between `round_robin` (no cache) and `cache_aware` (full tree).

#### 3.5.5 `cache_aware`

`src/policies/cache_aware.rs` + `src/tree.rs`. The most interesting policy for autoinfer experimentation.

**Mechanism** (verbatim from the policy header comment, confirmed against source):

1. Maintain an **approximate radix tree per model** (tenant = worker URL). Each tree stores raw text characters, not token ids, to avoid tokenization overhead.
2. For each request, find the worker with the highest prefix match.
3. If `match_rate > cache_threshold` (default 0.5), route there.
4. Else route to the worker with the smallest tree size (proxy for most-available cache capacity).
5. Even in load-balancing mode, *still update* the chosen worker's tree — i.e. the tree tracks actual placements, not aspirational ones.
6. Background thread evicts LRU leaves periodically to bound memory.

**Load-imbalance override**: the tree guides routing *only when the system is balanced*. It flips to shortest-queue if **both**:

- `(max_load − min_load) > balance_abs_threshold` (default 32), **and**
- `max_load > balance_rel_threshold × min_load` (default 1.1).

This "balance gate" is why `cache_aware` does not pathologically pile onto one worker: a popular prefix raises load, the gate trips, shortest-queue takes over until balance is restored.

**Default configuration** (`CacheAwareConfig::default`):

| Knob | Default | Meaning |
|---|---|---|
| `cache_threshold` | `0.5` | Fraction of prompt that must already be present to use cache affinity. |
| `balance_abs_threshold` | `32` | Load delta above which the gate trips. |
| `balance_rel_threshold` | `1.1` | Load ratio above which the gate trips (multiplicative). |
| `eviction_interval_secs` | `30` | Background LRU cycle. |
| `max_tree_size` | `10000` | Per-model node cap. |

**Tree internals** (`src/tree.rs`):

- `type TenantId = Arc<str>` — interned worker URLs, cheap to clone/compare.
- Root node uses `DashMap` with `ROOT_SHARD_COUNT = 32` shards; child nodes use `NODE_SHARD_COUNT = 8`. The asymmetry matters: *every* request passes through the root (high contention), few requests reach a given leaf (low contention). The default DashMap sharding (`num_cpus × 4`, ~256 on a 64-core host) is replaced with this tighter layout. Claimed memory reduction: ~90%.
- Custom `CharHasher` uses golden-ratio multiplication (`0x9E3779B97F4A7C15`) on single-char keys — pure identity hashing would cluster ASCII; this mix distributes.
- `advance_by_chars(s, n)` has a fast ASCII path using direct byte slicing, fallback via `char_indices` for UTF-8. Concrete optimisation: prompt-text traversal is the hot loop.

**WHY it matters for autoinfer**:

1. It is the canonical *upper bound* on prefix-cache utilisation at the dispatch layer. The delta between `cache_aware` and `round_robin` on a given trial tells us how much of the prefix-cache win is routing-dependent vs engine-internal.
2. The `cache_threshold × balance_*` triple is **itself a tunable surface** — a policy, not a constant. Autoinfer could legitimately search it, though that pushes us into router-parameter tuning which is a layer below §4.2 L2.
3. `Tree::evict_tenant_by_size` is a background thread — evictions happen between trials, which means a freshly-restarted router has an empty tree and a cold-start regime that distorts the first N requests of a trial. Trial harness must warm the tree (or reset it uniformly) for A/B fairness.

#### 3.5.6 Policy-state invalidation contract

Note the `reset()` and `init_workers()` calls on the trait. Autoinfer's driver **must** call `reset()` between trials of different policies (round-robin counter carries across, tree state carries across). This is currently implicit — no flag announces "new trial starting". Issue tracked below.

### 3.6 The PD-disaggregated dispatcher

`src/routers/http/vllm_pd_router.rs` extends `pd_router.rs` with vLLM-specific request handling (the generic `pd_router` supports other engines; the vLLM variant knows about NIXL/NCCL and vLLM's bootstrap wire format).

**Request ID format**. Every PD request is stamped with a router-generated id:

```
___prefill_addr_{prefill_host_port}___decode_addr_{decode_host_port}_{uuid_no_dashes}
```

This id is what the prefill and decode workers use to pair their KV handoff. Any log-correlation or failure-attribution across legs must parse this format.

**Bootstrap wrapper** (`src/routers/http/pd_types.rs`). For NIXL transfers the router wraps the original request body with three additional fields before forwarding to prefill:

```rust
struct RequestWithBootstrap<'a, T: Serialize> {
    #[serde(flatten)] original: &'a T,
    bootstrap_host: String,
    bootstrap_port: Option<u16>,
    bootstrap_room: u64,   // rand u63 to match Python's random.randint(0, 2**63 - 1)
}
```

The `bootstrap_room` is deliberately constrained to `[0, 2^63 − 1]` so it round-trips through Python's signed-int random without overflow. Also `BatchRequestWithBootstrap` for batch.

**Discovery**. Two transport variants:

- **NIXL** — explicit `--prefill` / `--decode` URLs, no ZMQ, one-pool-per-role. Example in `scripts/llama3.1/`.
- **NCCL** — ZMQ-based discovery: router listens on `--vllm-discovery-address 0.0.0.0:30001`, workers register their ZMQ endpoints there. The `VllmPDRouter::get_zmq_address` helper resolves HTTP → ZMQ via the `ServiceRegistry`; falls back to the HTTP address if no ZMQ registration. Bootstrap port comes from the pod annotation `vllm.ai/bootstrap-port` (default; overridable).

**Policy composition**. `--prefill-policy` and `--decode-policy` can be set independently (e.g. `consistent_hash` for prefill to bind sessions to a KV-cached worker, `round_robin` for decode because decode workers are more symmetric). The default `select_worker_pair_with_headers` implementation selects independently from each pool; custom policies can override to correlate choices.

**Profiling hooks**. `start_profiling` / `stop_profiling` (PyTorch profiler on a vLLM worker) with a per-worker abort-handle map (`profiling_tasks: Arc<Mutex<HashMap<String, AbortHandle>>>`) and a configurable timeout. **Relevant for autoinfer**: a trial can request a profiler trace from a specific worker without racing the main request path.

### 3.7 Authentication

#### 3.7.1 Upstream (both repos): API-key validation round-trip

Configured via `API_KEY_VALIDATION_URLS=<csv>` env or `--api-key-validation-urls`. When set, every endpoint (except `/liveness` / `/readiness`, exempted in fork PR #10) requires `Authorization: Bearer <token>`. The router forwards the token to each validation URL; any HTTP 200 accepts the request. No local verification, no cryptographic validation of the token itself — the upstream service is the source of truth.

**Cost**: one round-trip per request. Upside: pluggable into any existing auth infrastructure (GitLab example in the README: `https://codebase.helmholtz.cloud/api/v4/user`).

#### 3.7.2 Fork-only: local RS256 JWT with run-scoped allowlisting

Files: `src/auth.rs`, wired via `server.rs::authorize_request`.

**Why added** (PR #6): the PrimeIntellect platform signs a per-run JWT with an RSA private key, the router carries the public key, and verification is **local** (no network round-trip per request — a big deal at multi-QPS benchmark load). Companion: `PrimeIntellect-ai/platform#1150`.

**Claim structure** (`RftClaims`):

```rust
pub struct RftClaims {
    pub sub: String,            // user id
    pub run_id: String,         // RFT run id
    pub team_id: String,        // optional
    pub model: Option<String>,  // base model name
    pub lora: Option<String>,   // allowed LoRA adapter name
}
```

**Verification** (`JwtVerifier::new` → `verify`):

- Algorithm pinned to `RS256`.
- `Validation::leeway = 60` (60s clock-skew tolerance between platform and router).
- `validate_aud = false` — audience check disabled intentionally.
- Invalid PEM causes startup failure (fail-fast).

**Scope enforcement** (`RftClaims::allows_model`, PR #12 — upstream fork marks this "High Risk"):

A run-scoped JWT may target:

1. The base model named in `model`.
2. The LoRA named in `lora`.
3. Any LoRA whose name is `<lora>-<suffix>` — forward-compat for step-versioned adapter names like `rft-<run_id>-step-42`. The `run_id` portion is the unguessable security boundary.

Empty `lora` claim rejects unconditionally — explicit guard to prevent `""` prefix-matching every requested name. Empty `requested` also rejects — the caller must resolve `model` before calling.

**Extra enforcements in PR #12** (applied to chat/completions, embeddings, `/v1/responses`, transparent proxy):

- If `model` is missing/empty in the request body, **pin** it to the JWT's base model.
- If `model` is outside the allowlist, **reject**.
- `lora_path` override (the vLLM bypass route) is **rejected** when a JWT is present.
- `/v1/models` response is filtered to only models the JWT permits; fail-closed on unexpected upstream shapes.

**API-key auth** is unrestricted by design — the scope enforcement only fires when a JWT is presented. This means the router supports mixed tenancy (API-key-users see the full surface; JWT-authenticated runs see only their scope).

**WHY this matters for autoinfer**: zero direct relevance *unless* autoinfer ever productionises trials on a shared pool where multiple tenants must be isolated. But it is relevant as a **design reference** for how per-trial identity could be carried as a capability rather than a URL. An autoinfer trial could plausibly be represented as a JWT with `run_id = trial_id`, and the router's existing usage-metrics pipeline would attribute tokens to the trial for free (see §3.8.2).

### 3.8 Observability

#### 3.8.1 Structured logging + traces

Upstream emits full OpenTelemetry spans (`opentelemetry-otlp` export). The PrimeIntellect fork (PR #1) **removed OTel span emission**, keeping only passive trace-header forwarding. The `tracing` crate stays; `tracing-opentelemetry` is kept in `Cargo.toml` but not wired for span export post-PR-#1.

**Why the removal matters**: the fork is optimising for low overhead under run-scoped traffic where the platform already emits canonical traces upstream. Autoinfer trials that need spans should use the *upstream* router, or re-wire OTel export in a local build.

#### 3.8.2 Prometheus metrics (`src/metrics.rs`)

Default endpoint: `127.0.0.1:29000`; configurable via `--prometheus-host` / `--prometheus-port`. Confirmed metric families (subset; list from PR #6 and source references):

- `vllm_router_run_prompt_tokens_total{run_id,model}` — fork-only, per-run (PR #6).
- `vllm_router_run_completion_tokens_total{run_id,model}` — fork-only.
- `vllm_router_run_requests_total{run_id,model}` — fork-only.
- Policy decision counter (`record_policy_decision(policy, worker_url)`).
- Load balancing event counter (`record_load_balancing_event()`) — fires whenever `cache_aware` trips the imbalance gate.
- Load range gauge (`set_load_range(max, min)`) — observable on Prometheus.
- Per-worker processed counter (`record_processed_request(worker_url)`).

**Cardinality hazard** (PR #6 risk note): `run_id` is unbounded — long-lived Prometheus processes will accumulate memory proportional to (runs × models). Relevant for autoinfer if we scrape this dataplane with its built-in PI attribution enabled; typically we would not.

**Streaming token counting** (`src/routers/http/usage_metrics.rs`, added by PR #6): parses token counts from both non-streaming JSON bodies and SSE streaming chunks. The SSE parser is how per-run counts stay accurate for streaming chat completions. Autoinfer's driver could cheaply reuse this module as a library to extract token counts from the router's own observations rather than re-parsing at the harness layer.

### 3.9 Reliability primitives

#### 3.9.1 Circuit breaker (per worker)

State machine:

- `Closed → Open` after `cb-failure-threshold` (default 5) consecutive failures.
- `Open → HalfOpen` after `cb-timeout-duration-secs` (default 30).
- `HalfOpen → Closed` after `cb-success-threshold` (default 2) consecutive successes.
- `cb-window-duration-secs` (default 60) bounds the failure-counting window.

Fork-only refinement (PR #24): vLLM input-validation errors return HTTP 500 by default; the fork downgrades those to 400-equivalent *for circuit-breaker purposes*, so a bad client doesn't trip the breaker against an otherwise healthy worker. **Autoinfer-relevant**: a config trial that emits a malformed request must not poison circuit-breaker state for subsequent trials on the same worker — the fork behaves correctly here; upstream may not.

#### 3.9.2 Retry

Retries on HTTP status codes 408, 429, 500, 502, 503, 504. Configurable:

- `retry-max-retries` (default 3)
- `retry-initial-backoff-ms` (default 100)
- `retry-max-backoff-ms` (default 10000)
- `retry-backoff-multiplier` (default 2.0)
- `retry-jitter-factor` (default 0.1)

Exponential backoff with multiplicative jitter.

#### 3.9.3 Phantom-load fix (fork PR #23)

The upstream `cache_aware` tracker double-decremented the load counter on retryable failures, leading to "phantom" negative load that locked workers out of rotation. Fork fix: remove the double-decrement on retry path, add a 300s per-chunk timeout on the streaming load-tracking task (sends error to client on timeout), add 9 tests asserting load counters return to zero across success/failure/streaming/repeated-request scenarios.

**Direct relevance for autoinfer**: trials that use streaming (which is the natural mode for chat workloads) must run on a router that has this fix, or the phantom-load drift *will* silently confound results within hours of runtime. Pin the fork or wait for upstream merge.

### 3.10 Kubernetes service discovery

`src/service_discovery.rs`. `kube-rs` watcher over Pods with label-selector matching.

Config (`ServiceDiscoveryConfig`):

- `selector: HashMap<String, String>` — regular mode.
- `prefill_selector` / `decode_selector` — PD mode (exact label sets that distinguish roles).
- `check_interval: Duration` — default 60s.
- `port: u16` — default 8000.
- `namespace: Option<String>` — `None` watches all.
- `bootstrap_port_annotation: String` — default `vllm.ai/bootstrap-port`. The mooncake-connector convention.
- `pd_mode: bool`.

Pod-unhealthy eviction (fork PR #7): pods that fail readiness are **removed from the router's worker list** rather than left behind as circuit-broken shadows. Relevant for autoinfer: a trial that spins up K8s pods and tears them down gets clean worker-set transitions — no stragglers.

### 3.11 CLI flag surface (the autoinfer-visible tuning surface)

All flags from README + inferred from `clap` derive usage in source. The autoinfer-relevant tunables are starred.

- `--worker-urls <URL...>` — static worker list.
- `--host <HOST>` / `--port <PORT>` — listen address.
- `--policy <NAME>` ★ — `random | round_robin | power_of_two | consistent_hash | cache_aware`.
- `--intra-node-data-parallel-size <N>` ★ — DP rank expansion.
- `--vllm-pd-disaggregation` ★ — enable PD mode.
- `--prefill <URL...>` ★ / `--decode <URL...>` ★ — explicit PD pools (NIXL).
- `--prefill-policy <NAME>` ★ / `--decode-policy <NAME>` ★ — per-pool policy.
- `--vllm-discovery-address <ADDR>` — ZMQ discovery (NCCL connector).
- `--service-discovery` / `--service-discovery-port` / `--service-discovery-namespace` / `--selector`.
- `--api-key-validation-urls <CSV>` — API-key validation.
- `--jwt-public-key-path <PATH>` or `JWT_PUBLIC_KEY` env — fork-only JWT.
- `--prometheus-host` / `--prometheus-port` (default `127.0.0.1:29000`).
- `--retry-max-retries` / `--retry-initial-backoff-ms` / `--retry-max-backoff-ms` / `--retry-backoff-multiplier` / `--retry-jitter-factor`.
- `--cb-failure-threshold` / `--cb-success-threshold` / `--cb-timeout-duration-secs` / `--cb-window-duration-secs`.
- `--request-id-headers <NAME...>`.

Everything starred is a candidate L2 sub-axis for autoinfer. Everything unstarred is an ops/reliability flag that should be held constant across trials so it does not contaminate measurements.

### 3.12 API surface exposed to clients

OpenAI-compatible (confirmed from source: `src/routers/http/openai_router.rs`, `src/routes/`):

- `POST /v1/chat/completions`
- `POST /v1/chat/completions/tokens` — fork-added alias (PR #1).
- `POST /v1/completions`
- `POST /v1/embeddings`
- `POST /v1/responses`
- `GET /v1/models` (filtered by JWT scope in fork).
- Transparent-proxy routes (generic passthrough).
- `GET /liveness`, `GET /readiness` (auth-exempt in fork).
- `/metrics` (Prometheus).

## 4. The PrimeIntellect fork — what it adds and WHY

Snapshot: +24 / −10 vs upstream (2026-04-16). Read in chronological order, the fork story is: *add local auth + model-awareness + per-run observability, strengthen tenant isolation, fix production bugs upstream hasn't shipped yet*.

| PR | What | Why | Risk | Notes |
|---|---|---|---|---|
| `#1` | Model-aware routing; `extra` fields passthrough; remove OTel span emission; `/v1/chat/completions/tokens` alias | Workers auto-discover their served models via `/v1/models` at startup + periodic refresh; registry re-indexes on model change. Passive OTel header forwarding kept; full span export removed for overhead. | Medium | Touches worker registration + health checker behaviour. Concurrency gate added around health-check + idle-traffic reset. |
| `#2` | Preserve extra request fields; merge `prompt_token_ids` in P/D disagg | Drops token-id info in PD was causing decode-side prompt drift when upstream client passed `prompt_token_ids` to skip re-tokenise. | Low | Correctness fix. |
| `#3` | Docker image build workflow | CI plumbing. | Low | |
| `#4` | Stop renaming wheels; keep standard naming | Packaging correctness. | Low | |
| `#7` | Evict workers from router when pods become unhealthy | K8s pods that flip ready→not-ready were leaving stale worker entries. Eviction drains them properly. | Low | Operational. |
| `#6` | **JWT auth verification + per-run usage metrics** | See §3.7.2. Per-run accounting at the dataplane without a round-trip per request. | Medium | `run_id` unbounded cardinality — Prometheus memory. Companion: platform#1150. |
| `#10` | Exempt `/liveness` and `/readiness` from auth | K8s probes were being rejected by the JWT middleware. | Low | |
| `#8` | Build GH Actions wheels in `manylinux_2_28` containers | Distribution. | Low | |
| `#12` | **Enforce JWT model scope per run** | Lock each run JWT to its own model surface; reject `lora_path` override; filter `/v1/models`; pin missing model to base. Prevents one run's leaked token from touching another run's adapter. | **High** | Changes auth behaviour across chat/completions/embeddings/`/v1/responses`/transparent-proxy. May reject previously-accepted requests (empty model, out-of-scope model, any `lora_path`). |
| `#14` | Failing tests for LoRA adapter routing (issue #1092) | Worker registry supported one model per worker → LoRA adapters never indexed. Autoscale replicas started with only base model; `/v1/models` discovery took only the first entry; `get_router_for_model` returned nothing; fallback routed to any backend. | Medium | Companion upstream: hosted-rl#1092. |
| `#15` | LoRA adapter routing fix | Workers indexed under **multiple** model ids; `WorkerRegistry::sync_worker_models` syncs membership over time; health checker indexes *all* models returned by `/v1/models`; regression tests. | Medium | Distribution correctness. |
| `#16` | Strip `v` prefix from branch name before adding it back | Release scripting. | Low | |
| `#17` | Fix TITO route silently rewritten to `/v1/chat/completions` | Transparent-proxy was over-eager. | Low | |
| `#19` | Trigger wheel + Docker builds after tag creation | Release. | Low | |
| `#23` | **Phantom request accumulation in `cache_aware` load tracking** | See §3.9.3. Double-decrement on retry path → phantom negative load → workers locked out. | Medium | Streaming inactivity timeout (300s) added. 9 invariant tests. |
| `#24` | Treat vLLM input-validation 500s as 400 for circuit breaker | Upstream vLLM returns HTTP 500 for bad input; the fork stops counting these as worker failures for circuit-breaker purposes. | Low | Correctness — prevents bad clients from tripping breakers. |
| `#18 / #13 / #11 / #9 / #15 / #20 / #22 / #25` | Release bumps | v0.1.14 through v0.1.22. | Low | |

**Two themes**:

1. **Multi-tenant isolation via JWT** — this is an RFT (reinforcement fine-tuning) productisation story. Each training run gets a JWT and a model/adapter scope. The router is the enforcement point.
2. **Cache-aware correctness under production load** — the phantom-request bug (#23) and circuit-breaker fix (#24) are the sort of subtle issues that only surface at scale. This is a vote of confidence in the fork over upstream for production runs.

## 5. HOW this plugs into autoinfer

### 5.1 Integration points, ordered by urgency

| File | Current state | What changes |
|---|---|---|
| `src/autoinfer/target/basilica.py` | stub | When `L2TopologyConfig.replicas > 1` or `pd_disagg=True`, must provision N workers via Basilica SDK, launch `vllm-router` in front with the configured policy, and return the **router URL** — not per-worker URLs — to the driver. |
| `src/autoinfer/harness/driver.py` | stub | `vllm bench serve` takes a single URL, so the change is cosmetic. The subtlety: prompt-stream determinism is required across policy A/B, otherwise `cache_aware` and `round_robin` aren't comparable on the same trace. |
| `src/autoinfer/layers/l2_topology/surface.py` | stub | Add `router_policy: Literal["cache_aware", "power_of_two", "consistent_hash", "round_robin", "random"]` to the topology schema and the policy-visible search space. Implicitly add `prefill_policy`, `decode_policy` when `pd_disagg=True`. |
| `src/autoinfer/harness/failure.py` | stub | Add typed failures: `RouterPoolExhausted` (all workers circuit-broken or removed; retries exhausted), `PDConnectorTimeout` (NIXL/NCCL KV handoff timed out across pools), `RouterStartupFailed` (binary not on PATH, port conflict, invalid JWT public key). All P9-typed, all fed to the surrogate. |
| `src/autoinfer/harness/ledger.py` | stub | Every ledger row for `replicas > 1` must carry the router config alongside engine args. Comparing L1 configs across different router policies without this would silently mis-attribute. |
| `src/autoinfer/controller/stale.py` | stub | Router-policy change invalidates cached L1 numbers on the same engine config (P4): publish a stale flag. |

### 5.2 Policy-fairness hazards to defend against in the driver

1. **Warm vs cold tree (cache_aware)**. Per-trial router restart gives a cold tree; mid-run policy switch gives a warm tree. Either is fine as long as the trial *declares* which regime it's in. Recommended default: restart router per trial, warm with N = 100 priming requests before the measured window.
2. **Round-robin counter carryover**. The `AtomicUsize` counter is process-lifetime. `reset()` exists on the trait but nothing calls it between trials. Driver must call it explicitly.
3. **Circuit-breaker carryover**. A worker that tripped open in trial K starts trial K+1 in `Open`. Either wait `cb-timeout-duration-secs` across trials, or reset the router.
4. **Consistent-hash key selection**. The built-in key priority is `X-Session-ID > X-User-ID > Authorization > client IP > body-derived`. `vllm bench serve` does not set session headers by default, so the effective key is client IP or a body hash. Trials should pin a deterministic key (e.g. request-index-modulo-N passed as `X-Session-ID`) to make consistent-hash routing reproducible.
5. **Power-of-two `cached_loads` staleness**. The pushable load table is process-local. If autoinfer's metrics scraper is not pushing updates, `power_of_two` falls back to `Worker::load()` — a local counter of in-flight requests, not vLLM's actual queue depth. That may or may not be the signal you want; either is fine, but declare it in the ledger.
6. **PD-disagg connector asymmetry**. Over NVLink vs RoCE vs TCP the NIXL handoff latency varies by *orders of magnitude*. The headline DistServe 7.4× and Mooncake 59–498% numbers assume NVLink-class fabric. Basilica's heterogeneous pool includes TCP paths — C4 evidence on those will not match the published numbers.

### 5.3 Flag-pinning policy for iteration-zero trials

Until an L2 adapter exists and L2 trials go multi-replica, the iteration-zero L1 slice (§8 of `00-hypothesis-seed.md`) stays single-replica and this router is not loaded. The moment that changes:

**Pin these** (hold constant; not in the search space):

- `retry-*` — reliability, not performance.
- `cb-*` — reliability.
- `request-id-headers` — observability only.
- `api-key-validation-urls` / `jwt-public-key-path` — off for internal trials.
- `service-discovery` — static URL list preferred for reproducibility of pod identity.

**Search over these** (L2 sub-axes):

- `policy` ∈ {round_robin, random, power_of_two, consistent_hash, cache_aware}.
- `vllm-pd-disaggregation` ∈ {false, true}; if true then `prefill-policy`, `decode-policy` independently.
- `intra-node-data-parallel-size` ∈ {1, 2, 4, 8} (compat-gated by the engine config's `--data-parallel-size`).

**Derived, not search axes** (intrinsic to the trial's topology):

- `--worker-urls` / `--prefill` / `--decode` — determined by the L2 topology.
- `--vllm-discovery-address` — determined by whether the chosen connector is NIXL or NCCL (engine-config coupling).

### 5.4 Mapping to principles and claims

- **P1 (three layers from day one)**. Router is *not* a new layer. It is part of L2's deployment output. Adding "L4 routing" would break the three-layer invariant.
- **P3 (layers are adapters)**. `l2_topology/adapter.py::run` must produce `(vllm_config, basilica_allocation, router_config)` atomically. Router config is part of the deployment artefact.
- **P4 (cross-layer stale-signal)**. Policy flip at L2 invalidates cached L1 results. Today there is no way to express that in the ledger; the policy axis literally does not exist in the schema yet.
- **P8 / C9 (reference replica for quality)**. Router has no effect on logit-level correctness — all policies forward bytes faithfully. The gate stays green across policies. Latency-percentile reporting is therefore the only signal that discriminates policies.
- **P9 (typed failure)**. New failure categories listed in §5.1. `RouterPoolExhausted` is especially important because it is *not* a worker failure — every worker is healthy, the router just ran out of retry budget because the *aggregate* was saturated. If we don't type it distinctly, the surrogate attributes it to the engine config and we waste iterations tuning irrelevant knobs.
- **P10 (frozen/mutable boundary)**. Router is mutable (it's part of the deployment the policy produces). It is *not* part of the harness — the harness is the driver + gate + ledger. Mutable side.

### 5.5 Evidence bearing on C-claims

- **C1 (engine surface has slack)**. Neutral. Router lives above `EngineArgs` and does not refute C1. But any "L1 beats defaults" claim at `replicas > 1` is under-specified without stating the router policy — adds a precondition to the evidence report.
- **C4 (PD-disagg Pareto-dominates)**. **Positive on feasibility, negative on defensibility without router config**. The router's PD mode is the reference implementation needed to run PD-vs-collocated fairly. No C4 measurement without it is comparable to DistServe / Mooncake baselines.
- **C6 (hybrid policy required)**. Mild +1. Router policy is a categorical with five values and strong interactions with prompt distribution. Surrogate-friendly. An LLM operator would pick `cache_aware` from a tuning guide; a surrogate discovers when `consistent_hash` or `power_of_two` actually wins on tails under the observed prompt entropy.
- **C7 (joint vs layered)**. Neutral. Router × engine-args is a plausible "well-scoped pair" for joint search (Ansor / DistServe shape). No published joint study on this pair exists.
- **C9 (reference replica required)**. +1 indirect. Router flips are *latency-visible only* — batch-invariance and logit divergence are unchanged by policy. Argues for percentile-based latency as the discriminator.

## 6. Open questions (tracked in the issue below)

1. Does `cache_aware`'s router-side tree double-count against vLLM's in-engine prefix cache? Two caches, one request. Under heavy churn the router tree's stale state may misroute; vLLM's internal state is authoritative but private.
2. PD-disagg TTFT penalty across Basilica's interconnect classes (NVLink / RDMA / IB / RoCE / TCP)? The NIXL path is the reference, but its latency is fabric-dominated — headline numbers do not transfer.
3. Is the PrimeIntellect fork's LoRA-aware multi-model worker indexing upstreamable, or permanently a fork delta? Determines whether an autoinfer "serve N adapters on M replicas" L2 axis can rely on upstream.
4. What signal does `power_of_two` use on Basilica? Pushed from `/metrics`? Polled? Local-counter fallback? Different signals produce different tail behaviour at identical topology.
5. Does the router support **heterogeneous engine-config across workers** (prefill FP16 + decode FP8/AWQ; different `max_num_batched_tokens` per role)? Required for asymmetric-quantization L2 experiments named in `00-hypothesis-seed.md §4.2`.
6. Mid-trial reconfiguration — is policy-flip or pool-resize an online operation, or does it require a router restart? Affects how the controller schedules router-axis perturbations.
7. The `/v1/models` scope filter (PR #12) changes the answer to "what models does this endpoint serve". If autoinfer queries the endpoint to auto-discover topology, run-scoped JWTs will lie by omission. Pin API-key auth for trials.

## 7. Deliberate non-overlaps

- **Not a replacement for `llm-d`**. `llm-d` is a cluster controller: CRDs, inference-resilience operator, planner, autoscaler. `vllm-project/router` is only the request dataplane. `00-hypothesis-seed.md §4.2` names `llm-d` for PD-disagg + routing when the full operator surface is needed; this router is the lighter alternative when it is not.
- **Not a replacement for an autoscaler**. K8s discovery + pod-eviction is reactive, not proactive. Autoinfer's scale decisions are made out-of-band by the L2 adapter; the router just dispatches to whatever exists.
- **Not a quality gate or observability pipeline**. Prometheus counters are operational, not scientific. Trials must carry their own ledger + reference replica.

## 8. References

### Source (code, commits, PRs)

- Upstream router: https://github.com/vllm-project/router
- PrimeIntellect fork: https://github.com/PrimeIntellect-ai/router
- Compare fork → upstream: https://github.com/PrimeIntellect-ai/router/compare/vllm-project:main...PrimeIntellect-ai:main
- Load-balancing docs (upstream): https://github.com/vllm-project/router/blob/main/docs/load_balancing/README.md
- `src/policies/cache_aware.rs`, `src/policies/consistent_hash.rs`, `src/policies/power_of_two.rs`, `src/policies/round_robin.rs`, `src/policies/mod.rs`
- `src/tree.rs` — approximate radix tree.
- `src/routers/http/vllm_pd_router.rs`, `src/routers/http/pd_types.rs` — PD-disagg dispatcher.
- `src/auth.rs` — fork-only JWT.
- `src/service_discovery.rs` — K8s pod watcher.

### Fork PRs (technical context)

- #1  Model-aware routing + extra-fields passthrough + OTel removal — https://github.com/PrimeIntellect-ai/router/pull/1
- #2  PD disagg: preserve extra fields + merge `prompt_token_ids` — https://github.com/PrimeIntellect-ai/router/pull/2
- #6  JWT auth + per-run usage metrics — https://github.com/PrimeIntellect-ai/router/pull/6
- #7  Evict workers on pod-unhealthy — https://github.com/PrimeIntellect-ai/router/pull/7
- #12 Enforce JWT model scope per run (**High Risk**) — https://github.com/PrimeIntellect-ai/router/pull/12
- #14 Failing tests for LoRA adapter routing — https://github.com/PrimeIntellect-ai/router/pull/14
- #15 LoRA adapter routing fix — https://github.com/PrimeIntellect-ai/router/pull/15
- #23 Phantom request accumulation in `cache_aware` — https://github.com/PrimeIntellect-ai/router/pull/23
- #24 vLLM 500-as-400 for circuit breaker — https://github.com/PrimeIntellect-ai/router/pull/24

### External algorithmic references

- **Consistent hashing (MurmurHash64A)** — Facebook mcrouter, `mcrouter/lib/fbi/hash.c`. Austin Appleby's MurmurHash family, original spec: https://github.com/aappleby/smhasher/wiki/MurmurHash3.
- **Ketama-style virtual nodes** — 160 virtual nodes per physical worker. Original ketama (last.fm): https://github.com/RJ/ketama.
- **Power-of-two choices** — Mitzenmacher 2001, "The Power of Two Choices in Randomized Load Balancing", IEEE TPDS. https://www.eecs.harvard.edu/~michaelm/postscripts/tpds2001.pdf
- **Radix / prefix trees for LLM serving** — RadixAttention / SGLang, Zheng et al. 2023: https://arxiv.org/abs/2312.07104. The cache-aware tree in this router is an *approximate* per-worker variant of the same idea, pushed one layer up into the dispatcher.

### vLLM and disaggregation references (already ingested as autoinfer notes)

- vLLM Disaggregated Prefilling (docs): https://docs.vllm.ai/en/latest/features/disagg_prefill/
- LMCache PD bench: https://blog.lmcache.ai/2025-04-29-pdbench/
- DistServe: https://arxiv.org/abs/2401.09670
- Mooncake: https://arxiv.org/abs/2407.00079
- Splitwise: https://arxiv.org/abs/2311.18677
- vLLM V1 anatomy: https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html
- Cross-ref: `docs/research/raw/07-vllm-v1-architecture.md` (internal).
- Cross-ref: `docs/research/raw/06-cloudflare-omni-gpu-multiplexing.md` (orthogonal "one GPU, many models" axis).
- Cross-ref: `docs/research/raw/references-L1-engine-config.md` (scope note added explaining why routing is not L1).

### K8s / operational

- `kube-rs`: https://github.com/kube-rs/kube
- NIXL connector (vLLM): https://docs.vllm.ai/en/latest/features/disagg_prefill/
- Mooncake bootstrap-port annotation convention (`vllm.ai/bootstrap-port`): referenced in `src/service_discovery.rs`.

### Related autoinfer PRs and tickets (pending; see §9)

- Ticket to be opened: "L2 multi-replica / PD-disagg trials must drive through vllm-router (or llm-d)".

## 9. Implementation checklist — what autoinfer needs to do

Kept terse; tracked as the GitHub issue filed alongside this note.

1. Add `router_policy`, `prefill_policy`, `decode_policy`, `intra_node_data_parallel_size`, `pd_disagg`, `connector ∈ {nixl, nccl}` to `l2_topology/surface.py`.
2. Implement router lifecycle in `target/basilica.py::run`: provision → launch router → return router URL → teardown in `teardown()`. Pin `--policy` from `TrialInput`. Emit `RouterStartupFailed` as a `FailureRecord` on binary-missing / port-in-use / PEM-invalid.
3. In `harness/driver.py`, inject `X-Session-ID` as `f"autoinfer-{trial_id}-{request_index}"` for `consistent_hash` reproducibility.
4. Add router-reset semantics between trials: restart the router process per trial (simplest) or call `reset()` on the policy (requires a control endpoint upstream doesn't expose today — restart is safer).
5. Ledger: persist `(policy, intra_node_dp, pd_disagg, prefill_policy, decode_policy, connector)` on every multi-replica row.
6. Typed failures added to `harness/failure.py`: `RouterPoolExhausted`, `PDConnectorTimeout`, `RouterStartupFailed`.
7. `controller/stale.py`: publish cross-layer stale flag when any router-axis changes, so cached L1 points on the same engine config are re-evaluated.
8. Decide build provenance: `pip install vllm-router` (upstream, no phantom-load fix until merged) vs pin to the PrimeIntellect fork (phantom-load fix + LoRA-indexing fix, but carries JWT code we don't need). Recommend: pin to the **fork** for iteration-one trials because of #23 / #24, turn off JWT (`--jwt-public-key-path` unset), use API-key auth off (trials are internal).
