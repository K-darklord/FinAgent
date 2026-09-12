# Finance Agent Evaluation Pipeline

A financial Agent evaluation pipeline: **real benchmark data + tool-calling agents + trajectory recording + 3-tier scoring + error attribution**.

> Design goal: run the full data line end-to-end. No training needed; default uses a local rule-based agent that still fetches real financial report URLs.

## Directory structure

```
FinAgent/
├── benchmark.py   # Task schema + live mini benchmark + FAB public dataset loader
├── agent.py       # Agents: RuleBased + FinGPT + OpenAI + HuggingFace (ReAct + EDGAR tools)
├── runner.py      # Main loop: run agent on tasks -> trajectories.jsonl + run_summary.csv
├── evaluator.py   # Unified 3-tier scoring + analysis -> results.csv + error_report.json + plots
├── config.py      # API keys, HF/FinGPT model, FAB data, scoring tolerance config
├── requirements.txt
├── data/          # Downloaded FAB public.csv (auto-created)
├── output/        # Run artifacts (auto-created)
└── README.md
```

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the mini benchmark (3 tasks, rule-based agent)
python runner.py

# Score and analyze results
python evaluator.py
```

Outputs:
- `output/trajectories.jsonl` — full trajectory per task (step / thought / tool / observation / latency)
- `output/run_summary.csv` — per-task aggregate metrics
- `output/results.csv` — scored results (3-tier breakdown + error_type)
- `output/error_report.json` — error distribution / accuracy / latency / cost
- `output/accuracy_by_category.png` — bar chart of accuracy by FAB category
- `output/accuracy_by_difficulty.png` — bar chart of accuracy by difficulty

## Agents

### 1. Rule-based agent (default, no API key needed)

Local rule-based agent with hardcoded answers for the 3 mini benchmark questions. Useful for pipeline validation.

```bash
python runner.py
```

### 2. HuggingFace Inference agent (baseline, free with HF account)

Uses HuggingFace Inference API with ReAct loop and EDGAR search tools. Default model: `deepseek-ai/DeepSeek-V4-Flash`.

```bash
# Set HF token (get free at huggingface.co/settings/tokens)
export HF_TOKEN="hf_xxxxx"

# Run FAB 50-question public set
FINAGENT_AGENT=hf FINAGENT_BENCH=fab python runner.py

# Quick test with 5 questions
FINAGENT_AGENT=hf FINAGENT_BENCH=fab FINAGENT_NUM_TASKS=5 python runner.py

# Score with 3-tier evaluation
python evaluator.py
```

Configuration (env vars, see `config.py`):
- `HF_TOKEN` — HuggingFace access token
- `HF_MODEL` — model for agent inference (default: `deepseek-ai/DeepSeek-V4-Flash`)
- `HF_JUDGE_MODEL` — model for LLM-as-Judge T3 scoring (default: same as agent)

### 3. FinGPT baseline agent (requires local GPU + model download)

FinGPT is an open-source financial LLM (LoRA adapter on falcon-7b). It is a **deliberately weak baseline**.

```bash
FINAGENT_AGENT=fingpt FINAGENT_BENCH=fab python runner.py
```

Configuration:
- `FINGPT_BASE_MODEL` — base model (default: `tiiuae/falcon-7b`)
- `FINGPT_PEFT_MODEL` — LoRA adapter (default: `FinGPT/fingpt-mt_falcon-7b_lora`)
- `FINGPT_DEVICE` — device (default: `cuda`)

### 4. OpenAI agent (upper bound, requires API key)

```bash
export OPENAI_API_KEY="sk-xxxxx"
FINAGENT_AGENT=openai FINAGENT_BENCH=fab python runner.py
```

## Tools (ReAct agent)

The HuggingFace and OpenAI agents use a ReAct loop with the following tools:

| Tool | Description |
|------|-------------|
| `edgar_search` | Search SEC EDGAR full-text search for filings |
| `fetch_url` | Fetch a URL and return clean text (strips HTML tags) |
| `parse_html` | Fetch + parse HTML page, return structured text |
| `retrieve_information` | Retrieve specific information from collected text |

Max tool calls per task: 10 (configurable via `MAX_TOOL_CALLS_PER_TASK` in `config.py`).

## Finance Agent Benchmark (FAB)

The full 537-question FAB dataset is split into:
- **Public validation (50 questions)** — auto-downloaded, CC BY 4.0
- **Private validation (150 questions)** — requires license from Vals AI (contact antoine@vals.ai)
- **Test set (337 questions)** — permanently private (leaderboard only)

```bash
# Run FAB public subset
FINAGENT_BENCH=fab python runner.py
```

FAB question types: Quantitative Retrieval, Qualitative Retrieval, Numerical Reasoning, Market Analysis, Trends, Beat or Miss, Complex Retrieval, Financial Modeling Projections, Adjustments.

## 3-Tier Evaluation

`evaluator.py` uses a 3-tier scoring system:

| Tier | Method | Description |
|------|--------|-------------|
| T1 | Exact match | Normalized string comparison (after answer normalization) |
| T2 | Numeric/Rubric | Numeric tolerance comparison + rubric keyword coverage |
| T3 | LLM-as-Judge | LLM evaluates each rubric criterion (YES/NO), coverage >= 60% = correct |

Final `is_correct = T1 OR T2 OR T3`.

Error taxonomy (6 labels):
- `retrieval_failure` — did not retrieve / retrieved wrong source
- `numeric_error` — wrong number, unit, or rounding
- `citation_missing` — missing source citation
- `tool_error` — tool call itself errored
- `qualitative_incomplete` — qualitative answer missing key points
- `correct` — passed

Scoring config (env vars in `config.py`):
- `SCORING_NUMERIC_TOL` — relative tolerance for numeric scoring (default: `0.05`)
- `SCORING_QUANTITATIVE_TOL` — looser tolerance (default: `0.5`)
- `SCORING_RUBRIC_COVERAGE` — coverage threshold for rubric/T3 scoring (default: `0.6`)

## Git rollback

The `v0.1-skeleton` tag marks the pre-FinGPT baseline:
```bash
git checkout main              # back to skeleton
git reset --hard v0.1-skeleton # discard everything after skeleton
```
