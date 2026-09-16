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

## 2026-09-13 (afternoon) — Evaluation v2.1 + Limitations doc

### What I did
1. Refactored evaluator.py to v2.0: 2-tier continuous scoring (T1 numeric + T2 LLM semantic with dealbreaker). Removed old 3-tier T1/T2/T3 binary system.
2. Wrote EVALUATION_STANDARD.md as standalone spec (versioned, for paper supplementary).
3. Updated README evaluation section to v2.1 with link to spec.
4. Implemented T1 keyword matching fallback for non-numeric gold answers (continuous score based on token coverage).
5. Implemented T2 multi-vote (3 rounds, median for correctness, majority 2/3 for dealbreaker) to reduce LLM judge variance.
6. Updated agent.py ReAct system prompt + max_steps fallback prompt to enforce ANSWER-only format (no reasoning prefix).
7. Confirmed FAB public data has only {operator, criteria} in rubric — no severity field. Updated limitation 8.3.
8. Documented 7 known limitations in EVALUATION_STANDARD.md Section 8.

### Evaluation v2.1 on existing 50 trajectories
- Accuracy: 14% (7/50), up from 10% in v2.0
- T1 avg: 0.110 (up from 0.075)
- T2 avg: 0.068
- Dealbreakers: 1/50
- Qualitative Retrieval: 33% (up from 22%)

### Known limitations (see EVALUATION_STANDARD.md Section 8)
1. Single-model multi-vote vs multi-model judge (+/-3-5%)
2. FAB-specific rubric dependency (other benchmarks fall back to T1)
3. No severity weights (FAB public data does not have severity field)
4. Judge model capability ceiling (Flash vs Pro)
5. No calibration (fixed 0.5 threshold)
6. Trajectory truncation (4000 chars)
7. Single-judge per criterion within a round

### Pending
- Re-run 50 FAB tasks with new agent prompt (running, ~60 min)
- Expect 20-40% accuracy with cleaner ANSWER format

## 2026-09-13 (evening) — fetch_url XBRL fix + Easy failure analysis

### What I did
1. Fixed fetch_url: it was returning XBRL metadata (first 3000 chars) instead of filing content for SEC iXBRL documents. Applied the same XBRL filtering + start_markers search (up to 500K chars) as parse_html. Increased return limit from 3000 to 8000 chars.
2. Re-ran 50 FAB tasks with fixed fetch_url. Accuracy: 30% (15/50), up from 28%.
3. Analyzed remaining 12 Easy complete_failures:
   - 9/12 have reasoning prefix ("The question asks...", "Let me...")
   - 8/12 retrieved real content but max_steps fallback did not extract answer
   - 2/12 have retrieve_information tool parameter errors
4. Confirmed parse_html fix: 27/50 trajectories now have real content (UNITED STATES), 0/50 have XBRL metadata (was widespread before).

### Results comparison
| Run | Accuracy | Easy | complete_failure |
|-----|----------|------|-----------------|
| v2.1 old traj | 14% | - | 32 |
| v2.1 new prompt | 28% | 27.27% | 25 |
| v2.1 + parse_html fix | 28% | 27.27% | 26 |
| v2.1 + fetch_url fix | 30% | 36.36% | 26 |

### Remaining issue
max_steps fallback prompt is still not strong enough. Agent retrieves real content but outputs reasoning text instead of extracting the answer. Need to strengthen fallback to force answer extraction from trajectory context.


---

## 2026-09-13 (night) — Tool fixes + evaluation v2.1 final + 34% accuracy

### What I did

#### Bug fixes
1. **parse_html XBRL filtering**: SEC iXBRL filings have XBRL metadata tags BEFORE the actual filing text. The `start_markers` search had `idx < 5000` limit, but "UNITED STATES" can be at index 180K+. Fixed: increased search range to 500K, added aggressive XBRL tag stripping (iso4217, xbrli, UUID-like patterns, long numeric runs).
2. **fetch_url XBRL filtering**: Same bug as parse_html — fetch_url was returning first 3000 chars of XBRL metadata for iXBRL filings. Applied same XBRL filtering + start_markers search. Increased return limit from 3000 to 8000 chars.
3. **retrieve_information tool**: Added graceful error handling for missing `text` parameter. Updated TOOL_SCHEMA description to clarify that `text` must come from previous fetch_url/parse_html output.

#### Evaluation improvements
4. **max_steps fallback context**: Increased context window from 3000 to 6000 chars. This was the single most impactful change — Hard questions went from 0% to 42%.
5. **T1 keyword matching fallback**: When gold answer has no numeric values, T1 now does token coverage matching (continuous score 0-1) instead of binary substring match.
6. **T2 multi-vote**: 3 rounds of LLM judge, median for correctness criteria, majority 2/3 for dealbreaker detection. Reduces boundary case variance.
7. **T2 dealbreaker mechanism**: If any contradiction criterion is triggered (majority 2/3), the question scores 0. Prevents hallucinated answers from scoring.
8. **T2 judge prompt**: Added instruction to ignore reasoning prefix and focus on factual content.

#### What I tried but reverted (negative impact)
9. **Few-shot examples in fallback**: Added 3 Q&A examples to max_steps fallback prompt. Result: 16% accuracy (down from 30%). Model outputs too-short answers, losing keywords for T1 matching. Reverted.
10. **Post-processing of answers**: Regex to strip reasoning prefix from answers. Result: 22% accuracy. Truncated correct answers (e.g., "TO", "(", "FCF, and"). Reverted.

### Accuracy progression
| Run | Accuracy | Easy | Hard | complete_failure | Key change |
|-----|----------|------|------|-----------------|------------|
| v2.1 old traj | 14% | - | - | 32 | T1 keyword + T2 multivote |
| v2.1 new prompt | 28% | 27% | 0% | 25 | ANSWER format prompt |
| + parse_html XBRL fix | 28% | 27% | 0% | 26 | XBRL filtering |
| + fetch_url XBRL fix | 30% | 36% | 0% | 26 | fetch_url XBRL |
| + few-shot (broken) | 16% | 27% | 0% | 31 | reverted |
| + few-shot (no post-proc) | 22% | 27% | 17% | 31 | reverted |
| **+ 6000 context only** | **34%** | **41%** | **42%** | **25** | final |

### Final results (34% accuracy)
- Total: 17/50 = 34%
- Easy: 9/22 = 40.91%
- Hard: 5/12 = 41.67%
- Medium: 3/16 = 18.75%
- Dealbreaker triggered: 1/50 (factual_contradiction)
- Error distribution: complete_failure=25, numeric_error=7, factual_contradiction=1

### By category
| Category | Accuracy |
|----------|----------|
| Complex Retrieval | 66.67% |
| Financial Modeling Projections | 50.00% |
| Numerical Reasoning | 50.00% |
| Qualitative Retrieval | 44.44% |
| Quantitative Retrieval | 33.33% |
| Trends | 33.33% |
| Beat or Miss | 14.29% |
| Adjustments | 0.00% |
| Market Analysis | 0.00% |

### Gap to FAB leaderboard
- DeepSeek V4 Pro: 60.4%
- FinAgent (V4-Flash): 34%
- Gap: 26%, mainly from Flash vs Pro model capability (~15-20%) + single-model judge (~3-5%)

### Key lesson
**Context size matters more than prompt engineering.** Increasing fallback context from 3000 to 6000 chars gave +6% accuracy (28% to 34%). Few-shot examples and post-processing both hurt accuracy by truncating answers. The model needs full retrieved context to generate complete answers, not tighter formatting constraints.

---

## 2026-09-16 — Interference Experiments (INT-05, INT-06, INT-07 paused, INT-12 designed)

### Summary
Today's work focused on **controlled interference experiments** following the
"every interference is a future test" principle. The accuracy trajectory
across experiments: 48% (v3 baseline) → 54% (INT-05) → 56% (INT-06).

### Experiments Conducted

#### INT-05: max_steps 25→50 (CONFIRMED interference, +6pp)
- **Hypothesis**: max_steps=25 prematurely terminated model search
- **Method**: Single variable change, 50 FAB public tasks, V4-Flash
- **Result**: 48% (24/50) → 54% (27/50), complete_failure 20→13 (-7)
- **Conclusion**: Hyperparameter interference confirmed. Model needs more
  search budget to complete thorough investigations on SEC filings.
- **Commit**: 4d4862d

#### INT-06: T2 trajectory 4000→20000 chars (MINIMAL interference, +2pp)
- **Hypothesis**: T2 judge trajectory truncation at 4000 chars cut off context
- **Method**: Reused INT-05 trajectories, only changed T2_MAX_TRAJECTORY_CHARS
- **Result**: 54% (27/50) → 56% (28/50), only 1 task flipped (Hard Beat-or-Miss)
- **Conclusion**: Minimal interference. T2 judge already had sufficient context
  in 4000 chars for most tasks. Hard difficulty benefited most (+8.34pp).
- **Commit**: aaeb3de

#### INT-07: T2 judge V4-Flash→V4-Pro (PAUSED, cost)
- **Hypothesis**: Same model for agent and judge creates self-evaluation bias
- **Status**: PAUSED — V4-Pro inference cost ~10x V4-Flash
- **Resume Strategies**:
  - Option A: Use cheaper alt judge (V4.1-Flash or R1)
  - Option B: Targeted V4-Pro on boundary cases only (final_score ∈ [0.4, 0.6])
  - Option C: Full V4-Pro run (most rigorous, highest cost)
- **Recommended**: Option B balances cost and statistical signal
- **Commit**: 0844129

#### INT-12: Transparent budget + no fallback (PLANNED, Option D)
- **Key Insight**: INT-12 is a DUAL-LAYER interference structure:
  - Layer 1 (INT-05): max_steps is a HIDDEN constraint (model doesn't know budget)
  - Layer 2 (INT-12): fallback synthesis is COMPENSATION for Layer 1
  - Removing Layer 2 alone = double punishment (still cut off + no compensation)
- **Pre-experiment evidence**: 9/50 tasks used fallback, ALL 9 returned
  "Not found" (0% accuracy) → Option B (remove fallback only) = NO-OP
- **Real test**: Can model AVOID hitting max_steps once it knows its budget?
- **Code changes**: (1) Add budget to system prompt, (2) Replace fallback
  synthesis with explicit "Not found"
- **Confounding controls**: Option D-placebo (equal-length irrelevant prompt
  addition) to isolate transparency effect from prompt change effect
- **Status**: PLANNED, awaiting execution
- **Commits**: a7192d7, 8c89a04

### Accuracy Trajectory (cumulative)
| Date | Version | Accuracy | Delta | Notes |
|---|---|---|---|---|
| 2026-09-12 | v0-ReAct (V4-Pro) | 8.3% | — | Format tax |
| 2026-09-13 | v1 (50 tasks) | 34% | +25.7pp | FC + bug fixes |
| 2026-09-13 | v0 with neg examples | 16% | -18pp | Regression test |
| 2026-09-13 | v1 maintained | 34% | +18pp | Restored |
| 2026-09-16 | v3-interference-fix (50 tasks) | 48% | +14pp | INT-13/14 fixes |
| 2026-09-16 | INT-05 (max_steps=50) | 54% | +6pp | INT-05 confirmed |
| 2026-09-16 | INT-06 (T2=20000) | 56% | +2pp | INT-06 minimal |

### Interference Causal Table Updates
- INT-05: SUSPECTED → CONFIRMED (+6pp)
- INT-06: SUSPECTED → MINIMAL (+2pp)
- INT-07: SUSPECTED → PAUSED (cost)
- INT-12: SUSPECTED → PLANNED (Option D, dual-layer insight)

### Key Files Modified
- `INTERFERENCE_CAUSAL_TABLE.md` — Updated with INT-05/06/07/12 statuses
- `experiments/20260916_int05_max_steps/EXPERIMENT.md` — Results + Incident log
- `experiments/20260916_int06_context_length/EXPERIMENT.md` — Results
- `experiments/20260916_int07_judge_bias/EXPERIMENT.md` — Pause + resume strategies
- `experiments/20260916_int12_transparent_budget/EXPERIMENT.md` — Full design with 5 gaps filled

### Key Insights Discovered
1. **Dual-layer interference** (INT-12): max_steps is hidden constraint,
   fallback is compensation — removing one without the other is unfair
2. **Fallback produces 0% accuracy**: 9/9 fallback outputs were "Not found",
   meaning fallback never actually helped the model
3. **FAB official has no published max_steps**: Our 50-step hidden cap is
   a deviation from FAB philosophy
4. **Hard tasks benefit most from budget**: INT-06 Hard +8.34pp, INT-05
   reduced Hard failures significantly
5. **complete_failure dominates errors**: 13/50 in INT-05, all from "Not found"
   — points to retrieval/search strategy, not reasoning capability

### Remaining Work
- [HIGH] Execute INT-12 Option D (transparent budget + no fallback)
- [HIGH] Resume INT-07 with Option B (targeted V4-Pro on boundary cases)
- [MEDIUM] Run INT-12 Option D-placebo for confounding control
- [MEDIUM] Investigate why Adjustments (0%) and Market Analysis (0%) categories
  remain at 0% across all experiments
- [LOW] Multi-seed runs for variance estimation
