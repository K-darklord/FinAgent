# CHANGELOG

All notable changes to this project will be documented in this file.
I follow the principle of recording what changed, why, and what remains.

---

## 2026-09-13 — Bug fixes + ReAct tool improvements

### Fixed
- **ARGS parsing bug**: LLM outputs for tool arguments were nesting redundant `query:` prefixes inside the query value. I added a 3-layer fallback: JSON parse -> ast.literal_eval -> regex key-value extraction. Tool arguments now arrive clean.
- **EDGAR URL format**: `edgar_search` returned directory listing pages instead of specific filing documents. I now build the filing URL from the `_id` field (contains the exact document path like `tm242389d18_defa14a.htm`), so `fetch_url` can retrieve actual filing content.
- **User-Agent compliance**: SEC EDGAR requires a contact email in the User-Agent header. I was using `FinAgentEval/1.0` without an email, causing 403/503 errors. Fixed to include `yuan.kevin.wang@connect.hku.hk`.
- **fetch_url HTML stripping**: `fetch_url` was returning raw HTML tags. I added script/style stripping and tag cleanup logic, matching `parse_html` behavior.
- **max_steps increased**: ReAct loop max steps raised from 5 to 10. Previously 21/50 questions hit the step limit before the agent could finish searching -> parsing -> answering.

### Changed
- `config.py`: Added HuggingFace agent configuration (HF_TOKEN, HF_MODEL, HF_JUDGE_MODEL, HF_BASE_URL).
- `requirements.txt`: `openai` package moved from commented-out optional to required (T3 LLM-as-Judge depends on the OpenAI client library to call HF router).
- `README.md`: Full rewrite -- documented 4 agent types, ReAct tool table, 3-tier scoring explanation, FAB dataset structure (50 public / 150 private / 337 test).

### T3 (LLM-as-Judge) diagnosis
- I confirmed T3 is **not a bug**. It works correctly: it calls the LLM for each rubric criterion and checks YES/NO.
- T3=0 on the 5-question test is because model answers genuinely do not cover enough rubric criteria (e.g., fab_001: answer covers 1/8 criteria = 12%, threshold is 60%).
- T3 is an OR condition in `is_correct = T1 OR T2 OR T3`, so it can only increase accuracy, never decrease it.

### Test results
- 5-question quick test (post-fix): 2/5 = 40% accuracy (T1=2, T2=1, T3=0).
- Full 50-question run scheduled for tomorrow 07:00 Beijing time.

### Remaining tasks
- [High] Run full 50-question FAB baseline with fixed code.
- [High] Email antoine@vals.ai to request FAB 150-question private validation set.
- [High] Commit + push all changes to GitHub.
- [Medium] Add `google_search` tool (DuckDuckGo free API) for non-SEC queries.
- [Medium] T3 optimization: batch all criteria into one LLM call (N API calls -> 1).
- [Medium] Run 3-5 trials for variance estimation.
- [Medium] Run upper bound model (GPT-4o-mini or DeepSeek-V4-Pro).
- [Low] Upgrade error taxonomy from hardcoded to configurable.
- [Low] Trajectory clustering with sentence-transformers.

---

## 2026-09-12 — FinGPT baseline + FAB integration + Evaluator merge

### Added
- **FinGPTAgent**: Lazy-loading LoRA model agent with graceful degradation (returns empty answer if torch/model unavailable).
- **HuggingFaceAgent**: ReAct-style agent using HF Inference API with tool-calling loop. Supports `edgar_search`, `fetch_url`, `parse_html`, `retrieve_information`.
- **FAB public dataset loader**: `benchmark.py` auto-downloads `public.csv` from Vals AI GitHub (50 questions, CC BY 4.0).
- **3-tier scoring** in `evaluator.py`:
  - T1 (exact match): Normalized string comparison after answer normalization.
  - T2 (numeric/rubric): Numeric tolerance comparison + rubric keyword coverage.
  - T3 (LLM-as-Judge): LLM evaluates each rubric criterion, coverage >= 60% = correct.
- **Answer normalization**: `_normalize()` strips commas, percent signs, unifies case. `_extract_numbers()` picks the closest number to gold.
- **EDGAR search tool**: `edgar_search()` queries SEC full-text search API, returns formatted filing results.
- **Visualization**: `plot_accuracy_by()` generates bar charts by category and difficulty.

### Changed
- `scorer.py` and `analysis.py` **merged** into `evaluator.py` (unified scoring + analysis class).
- `runner.py`: Added environment variable control (`FINAGENT_AGENT`, `FINAGENT_BENCH`, `FINAGENT_NUM_TASKS`) and metadata persistence.
- `config.py`: Added FinGPT, FAB, and scoring tolerance configurations.
- `requirements.txt`: Added `transformers`, `peft`, `pandas`, `matplotlib`, `sentence-transformers`.

### Test results
- Mini benchmark (3 questions, rule agent): 3/3 = 100%.
- FAB 50 questions (rule agent): 0/50 = 0% (expected -- rule agent only handles live_001/002/003).
- FAB 50 questions (HF agent, no tools): 0/50 = 0% (HF free quota exhausted after 9 questions).
- FAB 50 questions (HF agent + EDGAR tools, pre-ARGS-fix): 25/50 = 50%.
- FAB 50 questions (HF agent + normalize only, no tools): 16/50 = 32%.

---

## 2026-09-11 — Project initialization + Git setup

### Added
- Project skeleton: `benchmark.py`, `agent.py`, `runner.py`, `scorer.py`, `analysis.py`, `config.py`.
- 3 mini benchmark questions (NVIDIA/Apple/Microsoft financial data).
- `RuleBasedFinanceAgent` with hardcoded answers for mini benchmark.
- `fetch_url` tool with local fallback for offline environments.
- Trajectory recording: each step logs thought/tool_name/tool_input/tool_output/latency.
- Error taxonomy: 6 labels (retrieval_failure, numeric_error, citation_missing, tool_error, qualitative_incomplete, correct).
- `.gitignore` for Python/IDE/output/model artifacts.

### Git
- Initialized repository, pushed to `github.com/K-darklord/FinAgent` (private).
- Tag `v0.1-skeleton` marks the initial commit.

### Issues found and fixed
- `score_numeric` regex extracted "2024" from "FY2024" instead of "60,922" -- fixed with `_extract_numbers()`.
- `is_correct` and `classify_error` produced inconsistent results -- unified to derive from `classify_error`.
- `analysis.py` naive string comparison caused 0/3 accuracy -- replaced with `scorer.classify_error()`.
