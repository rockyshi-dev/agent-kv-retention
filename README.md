# agent-kv-retention

Measures what it costs to resume a coding-agent session on one 24 GB GPU when the session's KV
cache was kept on the GPU, offloaded to host memory by one of vLLM's offload connectors, or
dropped and recomputed. The engine is an unmodified vLLM v0.30.0 serving Qwen2.5-7B-Instruct in
bf16; the reference machine is one RTX 4090 in a two-socket server.

A coding agent alternates model turns with tool calls. While a tool runs, the session's KV cache
sits idle; if other work needs the GPU memory meanwhile, the cache is evicted. When the tool
result comes back, the engine either still has the cache on the GPU, restores it from host memory,
or recomputes it. This harness measures the time to first token (TTFT) of that next turn, and
counts how many of its tokens were served from the GPU cache, restored from host memory, or
recomputed.

A second, CPU-only count reads a public trace of real coding-agent sessions (TraceLab) and reports
what share of tool calls have a context this model can serve at all, and how many of those would
not fit beside k other such sessions on the card.

## What it measures

**Configurations ("arms").** Every server uses vLLM's defaults; the arms differ only in
`--kv-transfer-config`.

| Arm | Configuration | What happens to an evicted session |
|---|---|---|
| KEEP | defaults; the session is not evicted | its KV is still on the GPU |
| DROP | defaults; the session is evicted | recomputed (no offload path by default) |
| NATIVE | `OffloadingConnector`, `cpu_bytes_to_use` 16 GiB (its defaults: eager store, `offload_prompt_only: true`) | restored from pinned host memory for the blocks that were stored: prompt blocks only |
| NATIVE-ALL | NATIVE plus `"offload_prompt_only": false` | restored for prompt and generated blocks |
| LAZY | `SimpleCPUOffloadConnector`, `cpu_bytes_to_use` 16 GiB, `"lazy_offload": true` | blocks copied to host only when about to be evicted, then restored |

NATIVE minus NATIVE-ALL is the cost of the documented default (only prompt blocks are offloaded);
NATIVE-ALL minus LAZY is eager against lazy copying. KEEP and DROP share one server start. The
OffloadingConnector selects the `LBHNC` KV-cache layout, the other servers use `LBNHC`; this comes
with the connector as shipped and is part of what NATIVE and NATIVE-ALL measure.

**A session.** Requests go to `/v1/completions` as token ids, so nothing is re-tokenized.

- Turn 1: a prompt of P = N - 768 tokens (a chat rendered with the model's template; system text
  and an earlier turn cut from the vLLM source tree; it ends with the assistant header),
  `max_tokens` = `min_tokens` = 512, `ignore_eos`, temperature 0. The turn is fixed in length,
  not in content: each arm's own server generates its 512 ids, which are kept exactly as
  returned. Separately compiled servers need not produce the same greedy output; no check or
  count compares the ids across arms, only their number.
- Eviction (every arm except KEEP): distinct 4,096-token prompts with no shared first cache block,
  `max_tokens` 1, sent one after another until their total exceeds 1.5 times the GPU KV pool.
- A 2 s wait.
- Resume: the P prompt ids, the 512 generated ids, and 256 ids that close the assistant turn and
  add a user message (the tool result): exactly N tokens. `max_tokens` 8, streamed, with usage.
  TTFT is measured by the client, from writing the request to the first stream chunk that carries
  a choice.

N is 2,048, 8,192 and 28,672. The KV of the last generated token is never computed in turn 1 and
prefix caching keeps only full 16-token blocks, so the most any arm can reuse is
C = floor((P + 511) / 16) x 16 tokens (1,776, 7,920 and 28,400), and every arm computes at least
N - C = 272 tokens.

**Counted on each resume request**, from `/metrics` before and after it, with nothing else in
flight (the client checks that exactly one request completed and none is running or waiting):

- G: GPU prefix-cache hits (`vllm:prefix_cache_hits`);
- Hh: tokens served by the connector (`vllm:external_prefix_cache_hits`); for NATIVE and
  NATIVE-ALL, `vllm:kv_offload_load_bytes` / 57,344 is recorded as a cross-check (57,344 bytes is
  the model's KV per token);
- R = N - G - Hh, recomputed tokens;
- TTFT.

**Procedure.** Five rounds. Each round starts one server per arm (four starts) in a random order
drawn from `--seed` and the round, and, for each N, runs one warm-up session and four measured
sessions per arm, each session with its own text. The server runs on the CPUs of the GPU's socket;
the client and filler on the first four physical cores (with their SMT siblings) of the other
socket. A 1 Hz monitor records the GPU clock, power, temperature and utilisation, per-CPU times,
the CPU time of the run's own processes per socket, and the affinity of every thread.

## The checks

| Check | Pass condition | If it fails |
|---|---|---|
| V start | The arm starts; 16-token blocks and prefix caching are on; the logged GPU KV pool matches `/metrics`. NATIVE and NATIVE-ALL log the OffloadingConnector, LAZY logs `SimpleCPUOffloadConnector ... mode=lazy`. The host buffer holds the largest session plus the filler: 28,672 + 1.5 x pool < 0.9 x 299,593 tokens (16 GiB / 57,344 B). For NATIVE and NATIVE-ALL, a separate untimed start with `VLLM_LOGGING_LEVEL=DEBUG` (the start test) must log `cudaHostRegister ... 17.18 GB`, and no timed start may log a failed registration | the arm is not measured; the error is recorded |
| pool | every start of a round logged the same GPU KV pool | the round is invalid for every arm |
| L length | turn 1 returns 512 ids; the resume request reports `prompt_tokens` = N | the session is dropped |
| K keep | in KEEP, G >= C - 16 | the level is invalid for every arm; no usable KEEP session also counts as a failure |
| E evicted | in the evicted arms, G <= 16 | the session is dropped |
| drops | at most 1 of the 20 measured sessions of an arm and level dropped by L and E together | that arm and level is invalid |
| H host copy | NATIVE: Hh >= floor(P / 16) x 16 - 16; NATIVE-ALL and LAZY: Hh >= C - 16, in every kept session | that arm and level is "host copy incomplete" and enters no decision |
| Q quiet | no other compute process on the GPU; CPU time of processes not started by the run at most 2% of each of the two sockets; GPU clock below 90% of its median in at most 10% of the samples; no thread outside its socket | the round is rerun, at most twice; then the result is NOT-MEASURABLE |

A failed warm-up or an engine failure ends that arm for that round (its raw records stay on
disk); the other arms continue. A server that cannot be stopped, with its GPU memory released,
within 60 s stops the run.

The generated turn is fixed in length and generated by each arm's own server. For information,
every session records which arm ran it first in the round (the base arm) and the position of
the first token at which its ids differ from that arm's, and the report prints, per round and
level, how many sessions agree with the base arm and how many had prefix-cache hits in turn 1
(expected: 0). None of this enters a check or the decision.

## Decision rule

For each offload arm that passed H, each level and each round: r = median TTFT(arm) / median
TTFT(DROP), the medians over that round's kept sessions. Rows are checked top to bottom; the first
that applies is the result.

| Row | Condition |
|---|---|
| NOT-MEASURABLE | Q ran out of reruns, or K failed at N = 28,672 |
| RESTORE-WINS | some offload arm passed H at 8,192 and 28,672 and has r <= 0.8 at both levels in every round |
| NO-WIN | at least one offload arm passed H at 28,672, and every such arm has r > 0.8 there in every round |
| INCONCLUSIVE | anything else, including no offload arm passing H |

The 0.8 threshold means a 20% saving in resume time. Reported with every outcome, entering no row:
r at 2,048; the cost of the default, R(NATIVE) - R(NATIVE-ALL) and the TTFT difference, per level;
eager against lazy, TTFT(NATIVE-ALL) - TTFT(LAZY).

**Trace count.** TraceLab release v0.0.1 (`uw-syfi/TraceLab`, file `syfi_coding_trace.jsonl.gz`,
sha256 `9d265eae69a31cae203848bea936f018148eed7ca8bf56050c5abe96da0b4e6b`, CC BY 4.0). For every tool record, the idle context is
`input_tokens_total + output_tokens` of the LLM round that issued the call. Reported: the share of
tool records whose idle context is at most 32,768 tokens (the model's limit); among those, for
k = 1, 2, 4, 8, the share whose bytes (context x 57,344) exceed pool / k, where pool is the KEEP
arm's logged KV pool in bytes; overall and for tool calls longer than 1 s. Records without a
duration count overall and are left out of the longer-than-1-s subgroup (the number is reported).
Token counts are the providers' (Claude, Codex), not Qwen's.

## Requirements

- Linux on x86_64, two CPU sockets, an NVIDIA GPU with 24 GB and `nvidia-smi`, `lscpu`,
  `taskset`. About 17 GB of free host memory for the pinned 16 GiB buffer of the offload arms.
- vLLM 0.30.0, CUDA 12.9 build (`vllm==0.30.0+cu129`, commit `ced6857a`), with its torch built for
  CUDA 12.9, in the Python environment that runs this package (its `vllm` executable is used).
  Other versions refuse to run.
- `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`, in a local
  directory; it is served by its path, offline. Its `config.json`, `generation_config.json`,
  `tokenizer.json` and `tokenizer_config.json` must have the sha256 of that revision (stored in
  `akvr/spec.py`); `generation_config.json` matters because it sets the server's default sampling
  parameters. Another copy is refused unless `--allow-other-model` is given, which is recorded in
  `manifest.json` and printed by `report`. The timed run also refuses a sessions file built with
  other tokenizer files.
- For `sessions`: a git checkout of vLLM at v0.30.0 (`ced6857afa0ea7b2e3f0846a62e1394e90f15607`)
  and `transformers` (any recent version; no torch needed).
- An otherwise idle machine. Inherited `VLLM_*`, `PYTORCH_*`, `TORCH*` (except `TORCH_HOME`),
  `CUDA_*` (except `CUDA_DEVICE_ORDER=PCI_BUS_ID`), `NCCL_*`, `OMP_*`, `TRITON_*`, `FLASHINFER_*`,
  `CUBLAS_*`, `CUBLASLT_*`, `CUDNN_*`, `NVIDIA_TF32_*`, `MKL_*`, `OPENBLAS_*`, `LD_PRELOAD`, `LD_LIBRARY_PATH`
  and `PYTHONPATH` stop the run, because each would change what is measured. A busy GPU or port
  stops a start.

No other Python package is needed; the tests need none at all.

## Run

```bash
# 1. sessions (CPU, about half a minute); prints ids_sha256 and whether they are the reference sessions
python -m akvr sessions --model /models/Qwen2.5-7B-Instruct --vllm-source ~/src/vllm \
    --output runs/sessions.json

# 2. start test: each configuration started once, untimed; the two native arms with DEBUG logging
python -m akvr start-test --gpu 4 --model /models/Qwen2.5-7B-Instruct --sessions runs/sessions.json \
    --output runs/start-test

# 3. the five timed rounds (about four hours on one RTX 4090), detached from the terminal
setsid python -m akvr run --gpu 4 --model /models/Qwen2.5-7B-Instruct --sessions runs/sessions.json \
    --start-test runs/start-test/start-tests.json --output runs/run-01 > run-01.log 2>&1 < /dev/null &

# 4. every number and the decision row, recomputed from the raw records
python -m akvr report runs/run-01 --json runs/run-01/report.json

# 5. trace count (CPU only); with --pools the capacity shares use the run's KEEP pool
python -m akvr trace --trace /data/tracelab/syfi_coding_trace.jsonl.gz --pools runs/run-01/summary.json \
    --output runs/trace
```

Options of `start-test` and `run`:

| Option | Default | Meaning |
|---|---|---|
| `--gpu` | required | GPU index in PCI bus order |
| `--port` | 8066 | server port |
| `--seed` | 603 | seed of the arm order |
| `--cache-dir` | `$AKVR_CACHE_DIR`, else `$XDG_CACHE_HOME/agent-kv-retention`, else `~/.cache/agent-kv-retention` | vLLM, Triton and CUDA compile caches of the servers |
| `--lock-dir` | `$AKVR_LOCK_DIR`, else `/tmp` | holds `agent-kv-retention-gpu<N>.lock` |
| `--lock` | none | an extra lock file held for the whole run (repeatable), e.g. one that other jobs on the machine also take |

**Use the same `--cache-dir` for the start test and the run.** vLLM measures the memory left for
the KV pool around the first forward pass, where a start without a cached compile compiles the
model. On an RTX 4090 a cold start gets a pool of 84,864 tokens and a start that loads a cached
compile 118,080. The start test compiles every configuration once, so all timed starts load a
cached compile and get the same pool; a round whose starts logged different pools is invalid.

The timed run accepts only a start test made by the same code with the same sessions file, on
the same GPU, model directory, `--cache-dir` and Python interpreter (it reads the start test's
`manifest.json`); it re-reads the DEBUG starts' logs and launch records (by path and sha256), so
keep the start-test directory in place. An arm whose start test failed is not measured.

## Files written

`sessions` writes one JSON file: every session's text, token ids, source offsets and hashes, the
filler vocabulary, the tokenizer file hashes and `ids_sha256`, a hash of all ids.

`start-test` writes, in `--output`: `manifest.json` (topology; model files and whether they are the reference; GPU, model
directory, cache directory and interpreter; code and sessions
hashes, options), `start-tests.json` (check V per arm), and one directory per server start with
`server.log`, `launch.json` (argv, environment with secrets redacted, versions of every installed package, GPU
placement, the listening socket, the default sampling parameters the server logged) and
`start.json` (check V and the first `/metrics` scrape). It exits 1 if any arm failed V.

`run` writes, in `--output`:

- `manifest.json`;
- `raw.json`: every round attempt with its server starts, the arm order, the quiet check, and
  every kept session: turn 1 (request, response, ids, their sha256, the base arm and the first
  position differing from it, turn-1 prefix-cache hits; the base arm written here is the arm that
  ran the session first, while the report derives the base from the rows it keeps, so after an arm
  failed the two can name different arms), the resume request (every stream event
  with its arrival time, usage, the full `/metrics` text before and after, the counter deltas),
  the gates; rewritten after every round attempt;
- `summary.json`: the checks per arm and level, the per-round medians and ratios, the
  differences, the decision row, a list of points to examine, and the KEEP pool per round for the
  trace count;
- per server start, `round-<r>-attempt-<a>-<group>/`: `server.log`, `launch.json`, `start.json`,
  `shutdown.json`, `telemetry.jsonl` (the 1 Hz monitor), `sessions.jsonl` and `filler.jsonl`
  (every request as it happened, including those of an arm that later failed), and `failure.json`
  when an arm failed.

`trace` writes `histogram.json` (tool records per idle-context length) and `report.json`.

`launch.json` records the server's environment with the values of secret-like variables
replaced by `<redacted>` (names containing KEY, TOKEN, SECRET, PASSWORD, CREDENTIAL or AUTH, and
names starting with SSH_, in any case); the server itself receives the real values. The records
still contain absolute paths, GPU UUIDs and process lists; look through them before sharing.

## How to read the result

`python -m akvr report RUN` recomputes the counter deltas from the stored `/metrics` text, TTFT
from the stored stream arrival times, and check Q from the telemetry files, then every check, ratio
and the decision row. It warns, and exits 1, where a recomputed counter delta, TTFT or quiet check
differs from the one stored in `raw.json`, and where any value of the stored `summary.json`
differs from the recomputed summary (it names each place). It prints:

- whether the sessions and the model are the reference ones;

- per round, check V and the GPU KV pool of every arm, and check Q (other processes' CPU share
  per socket, share of low-clock samples);
- per arm and level, whether it is valid, the L and E drops, K, the host-copy status, and the
  median G, Hh, R and load cross-check; on a correct run these are fixed numbers: KEEP R = 272,
  DROP R = N, NATIVE Hh = floor(P / 16) x 16 and R = 768, NATIVE-ALL and LAZY Hh = C and R = 272;
- the median TTFT per arm, level and round, and the ratios r;
- the differences reported with every outcome;
- per round and level, how many sessions generated the same turn 1 as the base arm ("agree N of
  M", M counting every measured session once per arm other than the base), the position of the
  first differing token where they differ, and how many sessions had turn-1 prefix-cache hits;
  information only;
- the decision row, followed by every point to examine: any invalid arm and level, an incomplete
  host copy, host-load bytes that disagree with Hh by more than one block (Hh is counted when the
  request is admitted and is not corrected if a load fails later), and a row other than
  RESTORE-WINS.

| Row | What it licenses |
|---|---|
| RESTORE-WINS | Restoring from host memory saves at least 20% of resume time at 8K and 28K tokens on this card: publish the curve and the token counts |
| NO-WIN | Restoring does not beat recomputing even at 28K tokens on this card; examine before explaining, then publish with the restore path's timing |
| INCONCLUSIVE | No row's condition holds; publish all numbers per round, without a conclusion |
| NOT-MEASURABLE | The machine was not quiet, or KEEP did not keep the session at 28K tokens; report why |

Scope: one session in flight at a time; no claim about concurrent sessions, scheduling policies,
other engines or other models.

## Tests

```bash
python -m unittest
```

The tests use synthetic records, recorded log lines and fake servers; they need no GPU, no network
and no package outside the standard library.

## License

MIT
