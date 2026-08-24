# DARS — Dynamic Adaptive Retention Scoring

A four-layer memory governance framework for agentic large language models.

DARS addresses a limitation common to flat retrieval stores used as agent memory: every stored item
is treated as equally valuable at retrieval time, regardless of how recently it was used, how often
it has helped, or whether it was ever correct. DARS assigns each memory a single bounded composite
score and uses that one score to govern ingestion, retrieval, compression and deletion through a
consistent interface.

> **Status.** This repository accompanies a manuscript currently **under revision at *Scientific
> Reports*** (Nature Portfolio). It has not yet been accepted for publication. See
> [Citation](#citation).

---

## The composite score

Every memory carries a score in $[0, 1]$:

$$S = w_r R + w_f F + w_u U + w_p P$$

| Term | Meaning | Definition | Default weight |
|------|---------|------------|----------------|
| $R$ | Recency | $e^{-\lambda \Delta t_{\text{hours}}}$, Ebbinghaus decay, $\lambda = 0.005\ \text{hr}^{-1}$ | $w_r = 0.30$ |
| $F$ | Frequency | $\min\left(\frac{\log(1+f)}{\log(1+F_{\text{cap}})},\ 1\right)$, $F_{\text{cap}} = 50$ | $w_f = 0.20$ |
| $U$ | Utility | $\frac{s+1}{s+f+2}$, Laplace-smoothed success rate | $w_u = 0.30$ |
| $P$ | Predictive relevance | cosine similarity between the memory embedding and a goal vector | $w_p = 0.20$ |

Laplace smoothing in $U$ prevents a single early failure from collapsing a memory's score to zero.
Weights are validated to sum to 1.0 at startup (`config/settings.py`).

## Architecture

| Layer | Package | Responsibility |
|-------|---------|----------------|
| **D — Memory Vault** | `core/layer_d/` | Qdrant persistence, 384-dim cosine index, atomic optimistically-locked metadata updates, DARS score computation, hybrid retrieval |
| **A — Cognitive Gateway** | `core/layer_a/` | Async query reformulation (Gemini), DARS reranking, XML-encapsulated injection-safe prompt construction |
| **B — Learning Engine** | `core/layer_b/` | LLM-as-a-Judge binary verdict → Laplace utility update → atomic metadata patch |
| **C — Maintenance Manager** | `core/layer_c/` | Volume-triggered triage, three-tier retain/compress/delete, 24-hour grace period, semantic compression with shadow indexing |

**Retention policy.** $S \ge 0.70$ retain · $0.30 \le S < 0.70$ compress · $S < 0.30$ delete.
Memories created within the last 24 hours are exempt from triage.

**Shadow indexing.** Semantic compression rewrites a memory's text but never its embedding vector,
so a compressed memory stays discoverable at its original position in the retrieval geometry.

**Retrieval fusion.** Two modes are implemented in `MemoryVault.search_and_rerank()`: Reciprocal
Rank Fusion over the similarity and DARS rankings ($k = 60$, the default and the mode used for all
reported results), and a weighted blend $\alpha \hat{s}_i + (1-\alpha) S_i$ with variance-aware
min-max normalisation. Under RRF the blend factor $\alpha$ is not applied.

## Repository layout

```
config/settings.py            All tunable parameters (weights, thresholds, endpoints)
core/layer_a/                 Cognitive Gateway  — reformulator, reranker, prompt constructor
core/layer_b/                 Learning Engine    — evaluator, calculator, engine
core/layer_c/                 Maintenance Mgr    — triage, janitor, compressor
core/layer_d/                 Memory Vault       — storage, schema, embedding
core/gemini_transport.py      Governed Gemini transport (rate limiting, key pool, retries)
benchmarks/memory_agent_bench/  MemoryAgentBench evaluation driver
benchmark_runs/paper_final/   Run artifacts backing the reported results
third_party/memoryagentbench_eval/  Vendored upstream metric utilities (see VENDOR.md)
data/groupA/                  Multi-Session Chat pipeline (temporal learning)
data/groupB/                  ALFWorld pipeline (strategic learning)
tests/                        Automated test suite
docs/                         Implementation guide, test documentation, audits
```

## Installation

Requires **Python 3.14**, a Qdrant instance (local or Qdrant Cloud), and a Gemini API key.

```bash
git clone https://github.com/HemanthBangera/Dynamic-Adaptive-Retention-Framework.git
cd Dynamic-Adaptive-Retention-Framework

python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS

pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and fill in your own values:

```ini
QDRANT_URL=
QDRANT_API_KEY=
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
TRAINING_GROUP=ALFWorld          # selects the GOAL_VECTOR preset for the P component
```

`.env` and `keys.txt` are gitignored and must never be committed.

`TRAINING_GROUP` selects the goal description used to build `GOAL_VECTOR` for the predictive term.
It should be matched to your deployment domain — see [Scope and limitations](#scope-and-limitations-of-the-reported-evaluation).

## Reproducing the reported results

List the available data sources for a split:

```bash
python -m benchmarks.memory_agent_bench list-sources --split Accurate_Retrieval
```

**EventQA-65K** (5 contexts × 5 QA pairs = 25 scored episodes):

```bash
python -m benchmarks.memory_agent_bench run \
  --split Accurate_Retrieval --source eventqa_65536 \
  --max-samples 5 --max-qa 5 --chunk-size 4096 --seed 42 \
  --path b --fetch-k 15 --top-n 3 --alpha 0.5 --baseline normal \
  --gemini-sleep 4.0 --gemini-retries 4 --gemini-min-interval 4.0 \
  --run-label framework_final --audit-jsonl audit.jsonl \
  --output-dir ./benchmark_runs/paper_final/accurate_retrieval_eventqa_65536_path_b
```

**RulerQA1-197K** (this source exposes a single context row; 5 QA pairs on it):

```bash
python -m benchmarks.memory_agent_bench run \
  --split Accurate_Retrieval --source ruler_qa1_197K \
  --max-samples 20 --max-qa 5 --chunk-size 4096 --seed 42 \
  --path b --fetch-k 15 --top-n 3 --alpha 0.5 --baseline normal \
  --gemini-sleep 4.0 --gemini-retries 4 --gemini-min-interval 4.0 \
  --run-label framework_final --audit-jsonl audit.jsonl \
  --output-dir ./benchmark_runs/paper_final/accurate_retrieval_ruler_qa1_197k_path_b
```

Each run directory contains `run_manifest.json` (full configuration, environment and seed),
`metrics_summary.json` (aggregate means and standard deviations), `results.json` and
`per_sample.jsonl` (per-episode outputs), `audit.jsonl` (per-query token accounting and DARS score
statistics) and `summary.md`.

`--path a` routes queries through the full Layer A gateway instead; see the note below on which
configuration produced the reported numbers.

## Results

MemoryAgentBench, **Accurate Retrieval** split, direct-retrieval configuration (`--path b`),
`gemini-2.5-flash` reader, `all-MiniLM-L6-v2` embeddings.

| Source | n | Exact Match | Token F1 | ROUGE-L F1 | Context → retrieved | Reduction |
|--------|---|-------------|----------|------------|---------------------|-----------|
| EventQA-65K | 25 | 0.640 | 0.7635 | 0.7880 | 65,536 → 12,484 | 80.95% |
| RulerQA1-197K | 5 | 0.200 | 0.3747 | 0.3489 | 202,010 → 12,266 | 93.93% |

These two operating points characterise an accuracy–efficiency trade-off rather than a uniform
gain: at moderate reduction the retrieved subset preserves most answer-bearing content, while at
94% reduction the probability of retaining the exact answer span falls sharply.

> **Two token-savings conventions appear in this repository.** The table above uses
> $1 - \texttt{input\_len}/\texttt{context\_tokens}$, where `input_len` is the retrieved memories
> plus the query as presented to the reader. The `token_savings_ratio` field logged in
> `audit.jsonl` uses the retrieved memories alone, giving 81.29% and 93.95% respectively. The two
> differ by the length of the query.

## Scope and limitations of the reported evaluation

Stated plainly, because it affects how the numbers above should be read:

- **Direct-retrieval configuration.** All reported runs used `--path b`, which exercises the Memory
  Vault and its hybrid ranking. The Cognitive Gateway, Learning Engine and Maintenance Manager were
  not invoked. The `reformulate_s` field is null throughout `audit.jsonl`.
- **Single-session protocol.** Each context was ingested into a freshly created collection and
  evaluated in one session, so no access history, feedback signal or decay interval accumulated.
  Under those conditions $R$, $F$ and $U$ take near-identical values across all stored items and the
  composite score is close to uniform — the logged `dars_mass_ratio` (retrieved mean ÷ vault mean)
  is 1.02 and 0.99. Semantic similarity is therefore the dominant ranking signal in these runs, and
  they do not establish the discriminative power of the scoring function.
- **Small samples, single seed.** n = 25 and n = 5, seed 42. Means and standard deviations are
  reported without confidence intervals.
- **No baseline or ablation.** No store-all, recency-only or per-component comparison was run, so
  the incremental contribution of each scoring term is unquantified.
- **Goal-vector domain mismatch.** $P$ was computed against the ALFWorld preset while evaluating
  EventQA and RulerQA1, yielding small, near-constant values (mean $P \approx 0.06$ and $0.03$).
- **One competency split.** Exploratory runs on the remaining MemoryAgentBench splits produced
  empty reader outputs under the current answer-formatting configuration and are not reported.

## Tests

```bash
pytest tests/ -v                                    # full suite
pytest tests/test_verifier_research_audit.py -v     # cross-layer contract tests
```

The suite requires a reachable Qdrant instance and valid credentials in `.env`; tests will error at
fixture setup without them. `tests/test_verifier_research_audit.py` holds six contract tests
covering the A→B→C→D handoff invariants; see `docs/verifier_research_audit.md` for what each one
checks and the three issues the suite originally exposed.

## Datasets

| Dataset | Source | Use |
|---------|--------|-----|
| MemoryAgentBench | [`ai-hyz/MemoryAgentBench`](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench) | Primary benchmarking (`Accurate_Retrieval`: `eventqa_65536`, `ruler_qa1_197K`) |
| Multi-Session Chat | [`nayohan/multi_session_chat`](https://huggingface.co/datasets/nayohan/multi_session_chat) | Group A — temporal learning (recency, frequency) |
| ALFWorld | [`awawa-agi/alfworld-raw`](https://huggingface.co/datasets/awawa-agi/alfworld-raw) | Group B — strategic learning (utility, predictive) |

All three are third-party public datasets. Metric implementations under
`third_party/memoryagentbench_eval/` are vendored from the upstream MemoryAgentBench evaluation
utilities; see `third_party/memoryagentbench_eval/VENDOR.md` for provenance.

## Citation

The manuscript is under revision and not yet accepted. Please do not cite it as published. Once a
decision is issued this section will carry the final reference.

```
Pushpalatha M N, Harshendra M, and Hemanth L Bangera.
"Dynamic Adaptive Retention Scoring (DARS): A Layered Memory Governance Framework
for Agentic Large Language Models."
Under revision, Scientific Reports (Nature Portfolio), 2026.
```

Department of Information Science and Engineering, Ramaiah Institute of Technology, Bengaluru, India.

## License

<!-- TODO: choose a license (MIT and Apache-2.0 are the common choices for research code)
     and add the corresponding LICENSE file at the repository root. Until then, no licence
     is granted and reuse rights are undefined. -->

No licence has been specified yet. Until a `LICENSE` file is added, all rights are reserved and no
permission to reuse is granted.
