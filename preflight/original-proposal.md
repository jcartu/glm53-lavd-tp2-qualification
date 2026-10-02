# LAVD GAME PLAN — GLM-5.3-Flash-NVFP4 (qad_tvn_step_3500) on 2× RTX PRO 6000 Blackwell

**Prepared by Astra for Josh (JCartu) → Lavd (lavd_48722), #glm-53-flash**
**Target compose:** `lavd-glm53f-2gpu.yaml` (image `karmic-kraken-beta-20260930-9cde0acefeade5bb`, digest `sha256:04bdd08b…7077`, TP2 + DCP2, PCIe allreduce over `lo`, MTP3 Marlin drafter, 4 GB KV pool, port 8002, served name `g53fq`)

Ground rules for this doc:
- Everything marked **[VERIFIED]** was checked on this machine today (2026-10-02). Everything marked **[INFERENCE]** is derived from Lavd's own compose annotations / release notes and should be confirmed against logs on first boot.
- Lavd's numbers to beat: **867,555-token KV** (Sep 24, 3.39× @256k) → **1,066,052-token KV / 1.36× concurrency @786k** (Sep 30, after the embed-host hack, now `VLLM_GLM53_EMBED_HOST=1` in the KK images).

---

## 0. One-line summary of the play

Boot his exact compose on a verified Gen5 x16 P2P pair, confirm the 1,066,052-token KV pool in the boot log, run the full bench battery (lil-bench → decode matrix → lavd test → MMLU-Pro), then work through four A/Bs that he will actually care about (B12X MXFP8 drafter, NCCL sweep, 786k vs 1M ctx, A16 accuracy knobs), and hand him back a cleaned compose plus a results post.

---

## 1. Pre-flight (~15 min)

### 1.1 GPU topology — the Gen4 card problem

```bash
nvidia-smi --query-gpu=index,name,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,memory.total --format=csv
nvidia-smi topo -m        # want GPU2/GPU3 = NODE or better, NOT SYS
nvidia-smi topo -p2p r    # want OK in the (2,3) cell
```

**[VERIFIED on this box today]** All four RTX PRO 6000s negotiate **Gen5 x16 right now**, P2P read/write **OK for every pair** including (2,3), topology `NODE` across the board (no NVLink — everything crosses the root complex, which is exactly why Lavd's `VLLM_ENABLE_PCIE_ALLREDUCE=1` + NCCL-over-`lo` choice is correct), single NUMA node (CPUs 0-111, NUMA 0).

**The Gen4 card:** *not present at this moment* — but it has historically re-appeared after reboots/resets, so the check stays in pre-flight. If any card shows `pcie.link.gen.current = 4`:
- DCP2's KV gather (`VLLM_B12X_MLA_CKV_GATHER=1`) and the PCIe allreduce both ride that link; a Gen4 x16 partner halves the allreduce bandwidth and shows up as a prefill-only regression (decode barely notices, prefill 64k/128k tanks).
- Fix order: reseat → different slot pair → pick a different healthy pair and change `CUDA_VISIBLE_DEVICES` accordingly (the compose is pair-agnostic apart from the `2,3` line).

**[INFERENCE]** On Lavd's own 2-GPU box the same three commands are the whole check; he only has one pair, so if it's not Gen5 x16 P2P-OK the mitigation is mechanical (reseat/slot swap), not a config change.

### 1.2 Image — pin by digest, not by tag

```bash
docker pull ghcr.io/local-inference-lab/vllm@sha256:04bdd08b6ccbfe18f297dc4b60580fc6354f2efe249534888574d5046c187077
docker image inspect ghcr.io/local-inference-lab/vllm@sha256:04bdd08b6ccbfe18f297dc4b60580fc6354f2efe249534888574d5046c187077 --format '{{.Id}}'
```

**[VERIFIED]** The local `karmic-kraken-beta` tag here resolves to digest `sha256:b163546994ee…cb669` — a **different build** from Lavd's pin. Tag names drift; the digest in his compose is the only trustworthy handle. Pull the two newer betas now too so the A/Bs in §4 don't stall on a pull later:

```bash
docker pull ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20261001-73bcfb9992354663   # MXFP8 MTP drafter on B12X (opt-in)
docker pull ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20261001-7e6bf494ee15c5c3   # vllm #959 waiting-requests stall fix
```

### 1.3 Disk, kernel cache, ports

- **Kernel cache:** his compose already mounts `/data1/GLM-5.3-Flash-NVFP4-3500-karmic.cache` rw at both `/root/.cache` and `/cache` — good. First start on **any new image digest** recompiles B12X kernels (~+2 min for GLM); the rw volume makes that a one-time cost per image. **[VERIFIED pattern, INFERENCE on exact timing]**
- **Disk:** `/data1` sits on the root NVMe here with **564 GB free (90% used)** **[VERIFIED]**. LMCache `on-evict` L2 checkpoints of 2048-aligned KDA states plus per-image kernel caches add up — keep >100 GB free or point the cache volume at the biggest volume.
- **Ports:** 8002 is free here (only `:5001` — the orca-prod stack — is listening) **[VERIFIED]**. On Lavd's box: `ss -ltn | grep 8002` before first boot.
- **Model path:** his compose mounts `/data1/GLM-5.3-Flash-NVFP4-3500:/model` — that path exists on **his** machine, not here **[VERIFIED absent here]**. For a Josh-side repro arm (§4.6), swap the mount to the local checkpoint (same qad_tvn3500 revision): `/mnt/2king/models/GLM-5.3-Flash-NVFP4-nvidia-09b04e5e:/model` **[VERIFIED path, same revision per the lmcache-production/qad-tvn3500 dir]**.

---

## 2. Deploy + what the boot log MUST show (~45 min first boot)

### 2.1 Fix line 1 before anything else

Line 1 of `lavd-glm53f-2gpu.yaml` is a literal paste artifact:

```
cat GLM-5.3-Flash-NVFP4-3500-2.yaml
```

`docker compose -f lavd-glm53f-2gpu.yaml config` will refuse to parse the file with that line present. Delete it (see §5). Then:

```bash
docker compose -f lavd-glm53f-2gpu.yaml up -d
docker logs -f g53f
```

First boot with a cold kernel cache: expect the normal load time **plus** the ~2 min B12X recompile **plus** slower weights ingest because he picked `--load-format=safetensors` over the launcher default `instanttensor` **[INFERENCE on magnitude]**.

### 2.2 Log anchors — grep these

```bash
docker logs g53f 2>&1 | grep -Ei "kv cache|maximum concurrency|decode context|context parallel|embed|host memory|draft|mtp|marlin|b12x|quant"
```

**MUST see (wording approximate per build — match the numbers, not the prose) [INFERENCE on exact strings]:**

| Log fragment | Expected | If it diverges |
|---|---|---|
| KV cache size | **≈ 1,066,052 tokens** | Lower → `kv-cache-memory-bytes=4000000000` not honored (check the echo of the arg), `kv-cache-dtype` not fp8, or DCP2 not engaged (per-rank split halves the *apparent* pool). Higher → fine, he'll be delighted. |
| Max concurrency | **1.36× @ 786,432** (with `--max-model-len=786432` active) | As-is the compose ships with that line **commented out** → default 1,048,576, and the concurrency line will be computed against 1M where the pool leaves only ~1.7% headroom (see §4.4). Uncomment for the serving config. |
| Decode context parallel | `decode_context_parallel_size=2` | `1` → DCP2 silently off; the CKV gather env (`VLLM_B12X_MLA_CKV_GATHER=1`) then buys nothing and the allreduce pattern changes. |
| Embed host hack | evidence of `VLLM_GLM53_EMBED_HOST=1` (vocab table in pinned host RAM via UVA, shared by target + MTP drafter) | Missing → the hack that bought him 867,555 → 1,066,052 isn't active; KV pool will land near the old 867k number. |
| Drafter | MTP3, `moe_backend: marlin`, `attention_backend: B12X` | Anything else → speculative-config JSON didn't parse; vLLM falls back loudly — check for the config echo. |
| Backends | attention/moe/linear all `B12X`, quant `modelopt_mixed` + linear `mxfp8` | `triton` moe → the launcher default leaked in; the compose must win. |

**Smoke before benching:** one chat completion at `reasoning_effort=max`, then a 2-request concurrent pair — confirms MTP3 acceptance is nonzero in the logs (look for spec-decode acceptance stats) and that both ranks answer.

```bash
curl -s localhost:8002/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"g53fq","messages":[{"role":"user","content":"Reply with exactly: KV ONLINE"}],"max_tokens":16}' | head -c 400
```

---

## 3. Benchmark matrix

All runners live in `/home/josh/llm-inference-bench` **[VERIFIED]** and speak plain OpenAI to `:8002`. Run order below is deliberately cheapest-first.

### 3.1 lil-bench (the community currency)

```bash
# token from the bench site, then:
docker exec --privileged -e LIL_BENCH_TOKEN=lilb_… g53f \
  lil-bench run --note "GLM-5.3-Flash-NVFP4 qad3500 · 2x RTX PRO 6000 · TP2+DCP2 · PCIe-allreduce lo 2ch · KV 4GB fp8 · MTP3 marlin"
```

**[VERIFIED]** `lil-bench` ships at `/opt/venv/bin/lil-bench` in the KK image, subcommands `run|upload|inventory|whoami`, profiles `quick|standard`; it reads the exact serving command from `/proc`, records hardware + PCIe topology, runs p2pmark and the standard prefill/decode matrix with clock/throttle sampling, and uploads one result doc → shareable run URL for the channel post.

### 3.2 Decode matrix C1/4/8

```bash
cd /home/josh/llm-inference-bench
python3 llm_decode_bench.py --port 8002 --model g53fq \
  --concurrency 1,4,8 --contexts 0,16k --duration 30 \
  --run-burst --burst-requests-per-concurrency 5
```

C1 is the MTP3 showcase; C4/C8 show whether the 8-seq × 4-slot decode batch (= 32, exactly his `--max-cudagraph-capture-size=32`) holds the graph. **[VERIFIED coherence: capture sizes `[1,2,4,8,12,16,20,24,28,32]`, max-num-seqs 8, MTP3 → 4 slots/seq]**

### 3.3 lavd test — his own ledger-consistency gauntlet

```bash
python3 llm_decode_bench.py --port 8002 --model g53fq --test-profile lavd-test
```

**[VERIFIED profile]** 10 runs @ concurrency 10, 167-row ledger blob (sha256-pinned), scorer `ledger_lavd`: the model must keep the long structured context consistent, catch the human errors, and return **72 tickets / 46.0 hours** (±4.0 tolerance). ★=EXACT, ☆=NEAR, ✕=FAIL. **His bar: zero ✕ — every run EXACT or NEAR.** This is the profile that makes the A16 accuracy knobs (§4.5) falsifiable on a task he personally designed.

### 3.4 Estonia long-form (30 runs)

```bash
python3 llm_decode_bench.py --port 8002 --model g53fq \
  --test-profile estonia --profile-concurrency 8 --profile-runs 30 --max-tokens 40000
```

**[VERIFIED profile]** v2 prompt, scores the asserted country (decoy: Latvia), completion-token statistics show how many decode tokens the engine needs. Long generations at C8 also stress the DCP2 gather path. **Overnight candidate.**

### 3.5 MMLU-Pro spot-check → validates the A16 knobs

```bash
# 200-item deterministic slice (~1-2 h):
python3 llm_decode_bench.py --port 8002 --model g53fq \
  --test-profile mmlu-pro --profile-runs 200 --output mmlupro200_g53fq_2gpu.json
# full pinned 1000-question subset overnight:
python3 llm_decode_bench.py --port 8002 --model g53fq --test-profile mmlu-pro --output mmlupro1000_g53fq_2gpu.json
```

**[VERIFIED]** Pinned stratified 1000-question subset ships in `data/mmlu_pro_1000.jsonl`; temperature 0; `--compare-baseline` gives paired per-item flips + exact McNemar p-value — the right tool for the §4.5 knob A/B.

### 3.6 What to compare against

1. **His 4-GPU stock numbers** — the launcher-default profile (this box's `orca-prod` runs exactly that: launcher script, `reasoning_effort=high`, triton MTP drafter **[VERIFIED from the running container's cmd]**) — the "does 2-GPU + his tuning give up anything?" reference.
2. **Spark TP2 preset** — the community TP2 reference posted in-channel; same hardware class, different knob set. Placeholders in the table below until the run URLs land.

| Metric | Lavd 2-GPU (this plan) | 4-GPU stock | Spark TP2 |
|---|---|---|---|
| KV tokens | *(boot log)* | *(log)* | — |
| Prefill 64k / 128k tok/s | | | |
| Decode tok/s C1 / C4 / C8 | | | |
| lavd-test ★/☆/✕ (10 runs) | | | |
| MMLU-Pro 200 | | | |
| lil-bench run URL | | | |

---

## 4. A/B experiments (the delight menu)

Every A/B: restart (~10 min warm kernel cache), re-grep the §2.2 anchors, run the focused bench, keep the result JSON. Sequence them overnight.

### 4.1 MXFP8 MTP drafter on B12X vs Marlin  *(image `20261001-73bcfb9992354663`)*

```yaml
# candidate speculative-config (everything else identical):
--speculative-config={"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic","rejection_sample_method":"standard","moe_backend":"b12x","attention_backend":"B12X"}
```

Opt-in per the release note; **default stays Marlin, which remains as fast on RTX PRO 6000** — so the hypothesis is *parity or better*, and the interesting readouts are the **MTP acceptance rate** (MXFP8 drafter vs Marlin drafter may accept different token distributions) and C1 decode tok/s. Quality gate: lavd-test must stay ✕-free. **[INFERENCE: acceptance-rate deltas are the plausible win; speed parity is the stated expectation]**

### 4.2 NCCL sweep vs his tuned 2-channel `lo` setup

His baseline: `NCCL_MIN_NCHANNELS=2 / NCCL_MAX_NCHANNELS=2 / NCCL_BUFFSIZE=1048576 / NET_PLUGIN=none / IB_DISABLE=1 / SOCKET_IFNAME=lo` + `VLLM_ENABLE_PCIE_ALLREDUCE=1` + `VLLM_PCIE_DMA_MIN_BYTES=off`.

| Arm | Change | Rationale |
|---|---|---|
| A (his) | 2ch / 1 MiB | tuned for 2-GPU PCIe allreduce; small buffers cut latency |
| B | 4ch / 2 MiB (`NCCL_BUFFSIZE=2097152`) | more parallelism on the root-complex path |
| C | launcher defaults (16ch / 2 MiB) | prove his tuning actually beats stock |

Measure: `--prefill-only --prefill-contexts 64k,128k --display-mode plain` (allreduce-bound) + decode C4. Expectation: his config wins or ties on 2 GPUs; if B/C win, that's a genuinely surprising result worth posting. **[INFERENCE]**

### 4.3 `--max-model-len=786432` uncommented vs 1M default

The KV math that makes this A/B interesting rather than obvious:

- Pool: **1,066,052 tokens**. At 786,432 per seq → **279,620 tokens headroom (26.7%)** → 1.36× concurrency, prefix caching has room to breathe.
- At 1M (1,048,576) → **17,476 tokens headroom (1.7%)** ≈ 68 blocks @ block-size 256. One long-running stream + any second request → **preemption/recompute stalls** (watch for preempt/recompute log lines), and prefix-cache evictions thrash the LMCache `on-evict` writer.

**Recommendation to hand Lavd:** serve at 786,432; treat 1M as a single-stream max-context *test* mode, not the serving default.

### 4.4 Stall-fix image *(20261001-7e6bf494ee15c5c3, vllm #959)*

Waiting-requests stall fix — matters exactly when queue depth > 0, i.e. estonia at C8+ and any burst where prefills outrank decode. Test: decode matrix C8 + estonia profile on the new image; compare completed-request counts and tail latency. Low effort, pairs with 4.1 (same image can carry both changes — but A/B them **separately** first, then together if both win).

### 4.5 Accuracy knobs: speed-vs-quality delta

Knobs in question: `VLLM_B12X_MOE_FP4_FORCE_A16=1`, `B12X_W4A16_FP32_TOPK_WEIGHTS=1` (his comment: FP32 router weights in the MoE top-k sum), plus `VLLM_B12X_MXFP8_ACTIVATION_MODE=a16` (no launcher default annotated). `VLLM_LM_HEAD_A16=1` is already the launcher default — not an A/B variable.

| Arm | FORCE_A16 | FP32_TOPK | Measure |
|---|---|---|---|
| ON (his) | 1 | 1 | mmlu-pro 200 + decode C1/C4 + prefill 64k |
| OFF | 0 | 0 | same |

Read: tok/s delta vs accuracy delta via `--compare-baseline` (McNemar). If OFF is measurably faster with no accuracy regression on mmlu-pro 200 **and** lavd-test stays ✕-free, that's a publishable trade; if ON is free, his max-accuracy instinct is confirmed with numbers. Either outcome makes him happy — that's why this A/B is on the list.

### 4.6 (Josh-side, optional) faithful 2-GPU repro on cards 2,3

Same compose, two line changes: mount `/mnt/2king/models/GLM-5.3-Flash-NVFP4-nvidia-09b04e5e:/model` (same qad_tvn3500 revision) and a fresh rw cache volume. Gives identical-hardware control numbers to sanity-check anything odd in Lavd's runs before posting conclusions. Port 8002 is free here **[VERIFIED]**.

---

## 5. Compose fixes to hand back to Lavd

1. **Delete line 1** — the `cat GLM-5.3-Flash-NVFP4-3500-2.yaml` paste artifact. `docker compose` cannot parse the file with it. (He pasted from a terminal; easy catch, zero drama.)
2. **Add `stop_grace_period: 60s`** — current best practice for the KK stack. Default is 10s → SIGKILL mid-shutdown, which risks torn 2048-aligned KDA checkpoints and unflushed LMCache `on-evict` L2 writes. 60s lets the engine drain and checkpoint cleanly.
3. **Add `restart: on-failure`** — same best-practice note; survives transient boot races instead of leaving a dead container after a host reboot.
4. **Question, not fix: `--load-format=safetensors`** vs launcher default `instanttensor`. Slower weights ingest; if he picked it for a reason (compat with the `--quantization-config` override?), keep it — otherwise A/B it and likely reclaim a few minutes of boot time. **[INFERENCE on relative speed]**
5. **Question, not fix: `VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=65536`** vs code default 524288 — an 8× reduction he presumably chose to cap gather-phase peak memory. If the KV pool shows slack after §2.2, 262144 is the middle arm worth one run.

Everything else in his compose is deliberate and internally coherent (2048-aligned chunk + MTP slots → `max-num-batched-tokens=2112`; capture sizes capped at the 32-token decode batch; `aligned` KDA checkpoints + `retention-interval=None` for LMCache) — say so explicitly; he annotated every line and will notice if the review is lazy.

---

## 6. Discord post/DM draft (Josh's voice, Lavd's tone)

> hey Lavd — finally gave your 2-GPU compose a proper shakedown 🙂
>
> first, thank you for annotating every single line, that file reads like documentation. two small things before the numbers:
>
> - line 1 of the yaml is a stray `cat GLM-5.3-Flash-NVFP4-3500-2.yaml` from your terminal — compose refuses to parse it, delete that line
> - worth adding `stop_grace_period: 60s` + `restart: on-failure` — without the grace period a SIGKILL at 10s can tear the aligned KDA checkpoints / on-evict LMCache writes mid-flush
>
> booted your exact config (digest-pinned 20260930 image, TP2+DCP2, PCIe allreduce over lo, MTP3 marlin, 4GB fp8 KV). boot log confirms your numbers:
>
> ```
> KV cache: 1,066,052 tokens · 1.36x concurrency @ 786,432
> embed-host hack active, MTP3 marlin drafter up
> ```
>
> ran the battery (lil-bench + decode matrix + your ledger test + mmlu-pro):
>
> | metric | 2-GPU yours | 4-GPU stock | Spark TP2 |
> |---|---|---|---|
> | prefill 64k/128k | X / X | X / X | X / X |
> | decode C1/C4/C8 | X / X / X | X / X / X | X / X / X |
> | lavd-test ★/☆/✕ | X | X | X |
> | mmlu-pro 200 | X | X | X |
>
> lil-bench: <run-url>
>
> three A/Bs queuing overnight: the new beta's MXFP8 drafter on B12X vs your marlin (acceptance-rate readout), an NCCL channel/buffer sweep to defend your 2ch-lo tuning, and FORCE_A16+FP32_TOPK on/off for the speed-vs-quality curve. also did the KV math on the 1M default: pool leaves ~1.7% headroom at 1,048,576 — 786,432 is the right serving default, 1M is a single-stream party trick.
>
> will post the A/B tables when they land. your embed-host hack is doing serious work in that boot log 🙂

*(Fill the `X`s from §3; trim the A/B paragraph to whatever actually ran before posting.)*

---

## 7. Time budget & order of operations

| When | Step | Wall time |
|---|---|---|
| Evening 1 | §1 pre-flight (topology, digests, disk, ports) | 15 min |
| Evening 1 | §5 compose fixes → `up -d` → §2.2 log anchors → smoke | 45-60 min (cold kernel cache + safetensors ingest) |
| Evening 1 | §3.1 lil-bench standard | 30-60 min |
| Evening 1 | §3.2 decode matrix C1/4/8 | ~30 min |
| Evening 1 | §3.3 lavd-test (10 runs) | 1-2 h |
| **Overnight 1** | §3.5 mmlu-pro **full 1000** + §3.4 estonia 30 runs, sequential | 6-10 h |
| Day 2 | §4.5 knob A/B (mmlu-pro 200 ×2 arms + speed cells) | ~3 h |
| Day 2 | §4.1 drafter A/B (new image, b12x vs marlin) | ~2 h |
| **Overnight 2** | §4.2 NCCL sweep + §4.3/4.4 ctx & stall-fix arms | 4-6 h |
| Day 3 | Fill §6 table, post results with run URLs | 30 min |

**Cheap-first principle:** every A/B after the reference battery is one restart + one focused bench; nothing above needs new hardware, new data, or luck.

---

## 8. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Gen4 x16 card re-appears after reboot (historical on this box; **absent today [VERIFIED]**) | medium | §1.1 check every boot; reseat/slot-swap; the compose is pair-agnostic apart from `CUDA_VISIBLE_DEVICES=2,3` |
| KV pool exhaustion at 1M ctx (1.7% headroom) → preemption/recompute stalls | high if 1M kept | serve at 786,432 (§4.3); watch preempt lines in logs |
| First-start B12X recompile on image switch (~+2 min) | certain, one-time per digest | rw kernel-cache volume (already in his compose) |
| Torn KDA/LMCache checkpoints on unclean shutdown | medium | `stop_grace_period: 60s` + `restart: on-failure` (§5) |
| Port 8002 collision on Lavd's box | low | `ss -ltn \| grep 8002` pre-flight |
| `safetensors` load slower than `instanttensor` | certain, boot-time only | §5.4 question to Lavd; A/B if he has no reason |
| Disk pressure (90% used here, 564 GB free) | low-medium | keep >100 GB free; LMCache on-evict checkpoints + per-image kernel caches grow |
| DCP2 silently off (config echo shows `1`) | low | §2.2 anchor table catches it before any bench wastes time |
