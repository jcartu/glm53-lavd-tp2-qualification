# GLM-5.3-Flash NVFP4 (QAD step 3500) — Lavd's 2-GPU recipe, independently qualified

Independent qualification of **Lavd's TP2/DCP2 serving recipe** for
`local-inference-lab/GLM-5.3-Flash-NVFP4` revision `qad_tvn_step_3500`,
run on 4× RTX PRO 6000 Blackwell (PCIe, single root complex) on 2026-10-02.

Everything here is measured. No cell was rerun until it matched an expectation;
failures are kept as failures. Each arm directory retains boot logs, benchmark
JSON with per-request records, telemetry, and command receipts with exit codes.

## Headline results (decode tok/s, ctx 0, median of 3×30 s)

| Setup | C1 | C4 | C8 | KV tokens | MMLU-Pro 200 (paired) | Lavd ledger |
|---|---:|---:|---:|---:|---:|---|
| **Lavd TP2/DCP2 (his exact config)** | 207 | 488 | 672 | 1,093,386 @1M | 168/200 (84.0 %) | 8★ 2☆ 0✕ |
| Same QAD checkpoint, TP4 | 306 | 694 | 962 | 6,416,584 @1M | 168/200 (84.0 %) | 9★ 1☆ 0✕ |
| Spark preset TP2 (Spark checkpoint) | 214 | 507 | 695 | 1,015,018 | 170/200 (85.0 %) | 10★ 0☆ |
| NVIDIA NVFP4 TP4 (existing prod) | 262 | 668 | 924 | — | 170/200 (85.0 %) | 10★ 0☆ |

Paired MMLU-Pro: TP2 vs TP4 identical at 168/168 (McNemar p = 1.0); Spark +1 pp
(p = 0.79, not significant). The unchanged-reference repeat moved 84 % → 87 %,
so ±3 pp is this sample's noise floor. Nothing here separates these setups on
quality at n=200.

**Two GPUs buy you ~68 % of the four-GPU decode speed at identical paired
quality — and the full 1M context genuinely works.**

## Long context (cold, salted, single request)

Both planted facts retrieved at every length, 5/5 including a **1,040,355-token**
prompt (249 s wall): 8K/131K/524K/770K/1.04M all pass. With the 786,432 cap
uncommented, the KV pool lands on exactly **1,066,052 tokens** — Lavd's own
reported number — vs 1,093,386 at the 1M default. Concurrent 700K + 420K
pressure pair: both complete, both retrieve.

Estonia 30-run: 29 pass, 1 decoy. Queue control (12 cold 30K prompts into 8
slots, default `max_parallel_prefills=1`): 12/12 complete, TTFT p50 37 s.

## Tuning A/Bs (vs his config, same checkpoint + image)

- **MTP drafter B12X vs Marlin** (new `20261001-73bcfb99` image): C1 212 vs 212,
  C8 667 vs 675 — parity. Ledger 10★/0☆ vs 9★/1☆. No reason to switch.
- **NCCL 4ch/2 MiB**: 209/478/674 — parity with his 2ch/1 MiB (207/488/672).
  His tuning is confirmed; there is nothing left on the table.
- **NCCL 16ch (launcher default)**: server boots and answers, but the scheduler
  metrics never came up and every benchmark cell aborted. On a 2-GPU
  `lo`-bound pair the launcher default breaks this config; his override is
  load-bearing, not cosmetic.

## Scheduler stall repro (vllm #959 family)

Pinned image `…20260930-9cde0ace` with `--max-parallel-prefills 2` and twelve
cold 30K prompts into 8 running slots: **all 12 streams died at ~30 s**
(`Response ended prematurely`), 0/12 complete. Same workload with the default
`max_parallel_prefills=1`: 12/12 complete. The failure netwalker4370 reported on
DeepSeek reproduces on GLM TP2/DCP2 with the trigger knob off-default. The
fixed-image stress arm was cut short when the GPUs were reclaimed; not claimed
as verified here.

## What was cancelled and why

`dual-tp2` (two instances on pairs 0/1 + 2/3) and `accuracy-off` never ran, and
`scheduler-fixed-stress` was still in flight when the GPUs were reallocated to
another project (~14:20 UTC). Recorded in `execution-scope.json`. No numbers
from those arms appear anywhere in the findings.

## Repository layout

- `arms/<name>/` — per-configuration evidence: compose, boot log, capacity line,
  benchmark JSON (per-request records), command receipts, telemetry
- `charts/` — publication figures (PNG + SVG + phone-size PNG)
- `control/` — coordinator receipts (production pause/restore, phase plan)
- `preflight/` — topology, image digests, checkpoint identity (config/quant/
  materialization SHA256 vs HF revision `1ebb2e01…`), frozen benchmark source,
  vendor launcher/preset sources, planned composes for every arm
- `publications/` — Discord post requests/receipts/verifications (message IDs,
  attachment SHA256)
- `public-summary.json` — one-file summary of every measured cell
- `docs-GAME-PLAN.md` — the execution plan with its 2026-10-02 amendment
- `docs-lavd-original-transcript.yaml` — Lavd's compose exactly as received
- `campaign.py`, `scenarios.py`, `report.py`, `await-result.py` — the drivers
- `originals-manifests/SHA256SUMS.originals` — SHA256 of files that were
  gzipped for GitHub's 100 MB limit; `.gz` siblings contain the identical bytes

Excluded: `cache/` (rebuildable JIT/kernel cache, 496 MB) and `fonts/` (licensed
Inter binary, redistribution not permitted).

## Environment

- 4× RTX PRO 6000 Blackwell Workstation, PCIe Gen5 x16, NODE topology, P2P OK
  on every pair (recorded per-run; idle links downshift to Gen1)
- Image: `ghcr.io/local-inference-lab/vllm@sha256:04bdd08b…7077`
  (`karmic-kraken-beta-20260930-9cde0acefeade5bb`), pinned by digest
- Benchmark: `llm_decode_bench.py` 0.7.5 (frozen copy in `preflight/`),
  `lil-bench` 1.3.2 standard profile (local, not uploaded)
- Production container paused only after request drain; original container ID
  restored unchanged afterwards
