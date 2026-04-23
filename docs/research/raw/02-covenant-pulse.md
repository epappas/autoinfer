# Covenant AI, Templar, and "Pulse" (raw, 2026-04-22)

## Status of "Pulse"
User referenced "Pulse by Covenant AI". I cannot confirm this product exists:

- [covenant.ai](https://www.covenant.ai/) lists Templar, Basilica, and GRAIL
  (coming soon). No Pulse.
- [github.com/one-covenant](https://github.com/one-covenant) lists 33 repos —
  templar, basilica, grail, SparseLoCo, crusades, bittensor-rs, hermes-agent,
  autoresearch-rl, openclaw. No Pulse.
- The only web hit for "pulse" in a distributed-inference context is an HN
  thread about an unnamed Apple-Silicon inference engine where a developer says
  "the pulse is still a placeholder. You can ignore that." Different project,
  not Covenant.

**Action**: ask user to clarify — is Pulse (a) an internal/unreleased Covenant
project, (b) a different org's product, (c) GRAIL renamed, or (d) a
misrecollection?

## What Covenant actually ships today

### Templar — decentralized pretraining
Permissionless pretraining across commodity hardware on Bittensor. First major
result: **Covenant-72B**, 72B-param model trained on ≥1.2T tokens across 70+
independent contributors over the public internet. Paper:
arxiv [2603.08163v2](https://arxiv.org/html/2603.08163v2). First checkpoint
("Checkpoint One") on Hugging Face, documented in
[Templar Research blog](https://templarresearch.substack.com/p/checkpoint-one).

### SparseLoCo / CCLoco — gradient-compression optimizer
Repo: `one-covenant/SparseLoCo` — "CCLoco: Scaling Up Top-K Error Feedback with
Local Optimizers". This is the compression primitive that makes 70+ internet-peer
training feasible: Top-K gradient compression + error-feedback + local-step
optimizers.

### Basilica — decentralized GPU compute substrate (Rust). Used as a target in
sibling project `autoresearch-rl`.

### Covenant exited Bittensor post-Covenant-72B — announced moving the
decentralized-training effort outside Bittensor.
Sources: [The Block](https://www.theblock.co/post/396959/covenant-ai-exits-bittensor-tao),
[PANews](https://www.panewslab.com/en/articles/019d7670-51db-736f-82c2-a2b49f1536e4).

## Relevance to autoinfer
The Covenant/Templar stack is **training-side**. Its relevance to inference is
indirect: the communication-compression and heterogeneous-peer coordination
primitives (SparseLoCo, task scheduling across unreliable peers) are analogous
problems to **disaggregated inference serving** — especially prefill/decode
disaggregation across heterogeneous GPUs, which llm-d already explores.

Concrete questions to bring back into autoinfer:
1. Does Top-K error-feedback translate to **KV-cache delta compression** for
   disaggregated serving (prefill node → decode node transfer)?
2. Are there **local-step** analogs for inference (staggered batching,
   micro-batched decode)?
3. Does Basilica's heterogeneous-compute routing fit as a target for our
   search loop — "find the best (hardware, engine-config) pairing for this model"?
