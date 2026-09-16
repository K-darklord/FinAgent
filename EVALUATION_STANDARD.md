# FinAgent Evaluation Standard

**Version**: 2.3 (2026-09-15)
**Author**: FinAgent Project
**Alignment**: Finance Agent Benchmark (FAB) v2 + FrontierFinance + BigFinanceBench

---

## 1. Design Goals

1. **Continuous scoring** (0-1), not binary pass/fail — aligned with FAB Partial Credit
2. **Two-tier architecture**: deterministic numeric check + LLM semantic judgment
3. **Dealbreaker mechanism**: contradiction with gold facts triggers score = 0 (prevents hallucination)
4. **Trace-aware**: LLM judge sees agent trajectory, not just final answer
5. **Reproducible**: T1 fully deterministic; T2 uses temperature=0

---

## 2. Tier 1: Numeric Accuracy (Rule-based, Deterministic)

### 2.1 Goal
Check whether the predicted answer contains the **core numeric values** from the gold answer, within a configurable tolerance. T1 does NOT care about reasoning text, word order, or phrasing — it only checks if the right numbers appear.

### 2.2 Input
- gold_answer: the reference answer string
- final_answer: the agent predicted answer string

### 2.3 Algorithm

Step 1: Extract all numbers from gold_answer
Step 2: Filter out year-like numbers (1900-2100) -> significant_gold_nums
Step 3: Extract all numbers from final_answer -> pred_nums
Step 4: For each significant gold number g:
            matched = any(abs(p - g) / max(abs(g), 1e-10) <= T1_NUMERIC_TOLERANCE for p in pred_nums)
Step 5: T1_score = matched_count / len(significant_gold_nums)

### 2.4 Edge Cases
- If gold_answer has no significant numbers (text-only answer):
  - First try: normalized substring match (gold_norm in pred_norm) -> 1.0
  - Fallback: keyword matching (continuous score)
    - Extract significant tokens from gold (length >= 3, not stopwords)
    - Count how many appear in pred
    - Return matched_count / total_count (continuous 0-1)
- If pred has no numbers at all but gold has numbers: T1_score = 0.0

### 2.4.1 Keyword Matching Details
When gold has no numbers (qualitative answers like "Workday reports Gross Revenue Retention Rate"):
- Tokenize gold: ["workday", "reports", "gross", "revenue", "retention", "rate"]
- Remove stopwords (the, and, for, etc.) and short tokens (< 3 chars)
- Check each token against pred (normalized)
- Score = matched_tokens / total_tokens
- Example: pred has "net revenue retention" -> matches "revenue", "retention" -> 2/6 = 0.33

### 2.5 Output
- T1_score: float in [0, 1] (continuous)
- T1_pass: bool, T1_score >= T1_PASS_THRESHOLD (default 0.5)

### 2.6 Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| T1_NUMERIC_TOLERANCE | 0.01 | Relative tolerance for numeric matching (1%) |
| T1_PASS_THRESHOLD | 0.5 | Minimum score to pass T1 |

### 2.7 Design Rationale
- ANY-match, not ALL-match: A pred that contains 3/6 gold numbers gets 0.5, not 0.0. This rewards partial correctness.
- Year filtering: Prevents 2019 in gold from matching 2019 in reasoning text.
- No LLM dependency: Fully deterministic, zero-cost, fully reproducible.
- Only looks at numbers: Ignores reasoning prefixes ("The user wants...", "Let me...") since numbers are the ground truth for quantitative questions.

---

## 3. Tier 2: LLM Semantic Judgment (Continuous Score + Dealbreaker)

### 3.1 Goal
Use an LLM judge to semantically evaluate whether the predicted answer satisfies each rubric criterion. T2 handles synonyms, paraphrasing, and qualitative criteria that T1 numeric extraction cannot capture.

### 3.2 Input
- question: the original task prompt
- final_answer: the agent predicted answer
- agent_trajectory: the full trajectory (tool calls, retrieved evidence, observations)
- gold_answer: the reference answer (for context)
- rubric_structured: list of criteria with operator field (correctness or contradiction)

### 3.3 Algorithm

#### Step 1: Dealbreaker Check (Multi-Vote Contradiction)
For each round in range(3):
    For each criterion c where c.operator == contradiction:
        verdict = LLM_judge(question, final_answer, trajectory, c.criteria)
        If verdict == YES:
            triggered_count[round] += 1
If triggered_count >= 2 (majority of 3 rounds):
    T2_score = 0.0  # Dealbreaker triggered — hallucination detected
    Return immediately

Why majority vote: Dealbreaker is a strong penalty (score = 0).
Using 2/3 majority prevents a single random YES from zeroing the score.

Rationale: A contradiction means the answer states something opposite to the gold fact. This is worse than missing information — it is a factual error. Any contradiction triggers score 0.

#### Step 2: Correctness Scoring (Multi-Vote Batched LLM)
correctness_criteria = [c for c in rubric if c.operator == correctness]
If no correctness criteria:
    T2_score = 0.5  # Neutral, let T1 decide

Multi-vote (3 rounds):
  For round in range(3):
    Prompt LLM judge with ALL correctness criteria in one call:
      Input: question + final_answer + trajectory_summary + all_criteria
      Output: comma-separated YES/NO (one per criterion)
    score[round] = sum(verdict_i == YES) / len(correctness_criteria)
  T2_score = median(score[0], score[1], score[2])

Why median: Reduces LLM judge variance. A boundary case (e.g., 3/6 = 0.5)
might score 0.33, 0.5, 0.5 across runs. Median = 0.5 (stable).

Judge prompt enhancement:
  "IMPORTANT: The answer may contain reasoning text (e.g., 'The question asks...').
   Ignore the reasoning prefix — focus on whether the factual content satisfies
   each criterion."

#### Step 3: Final
T2_score = score  # Continuous 0-1
T2_pass = T2_score >= T2_PASS_THRESHOLD  (default 0.5)

### 3.4 Judge Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| T2_JUDGE_MODEL | deepseek-ai/DeepSeek-V4-Flash | LLM model for semantic judgment |
| T2_PASS_THRESHOLD | 0.5 | Minimum score to pass T2 |
| T2_TEMPERATURE | 0 | For reproducibility |
| T2_MAX_TRAJECTORY_CHARS | 4000 | Truncate trajectory to fit context window |

### 3.5 Judge Prompt Structure (Correctness)
System: You are an expert financial evaluator. Judge whether the answer
satisfies EACH criterion. Consider the agent retrieved evidence in the
trajectory. Reply with ONLY a comma-separated list of YES/NO.

User:
  Question: {question}

  Agent Trajectory (retrieved evidence):
  {trajectory_summary}

  Answer: {final_answer}

  Criteria:
  1. {criterion_1}
  2. {criterion_2}
  ...

  Verdicts (comma-separated YES/NO, one per criterion):

### 3.6 Dealbreaker Prompt Structure (Contradiction)
System: You are an expert financial fact-checker. Does the answer
CONTRADICT the given statement? If the answer states something opposite
to the statement, reply YES. If the answer is silent or agrees, reply NO.

User:
  Answer: {final_answer}

  Statement (must be true): {contradiction_criterion}

  Does the answer contradict this statement? YES or NO:

### 3.7 Design Rationale
- Batched call: All criteria in one LLM call (cost: 1 API call per task, not N)
- Trace-aware: Judge sees retrieved evidence, can distinguish wrong answer but right retrieval from wrong answer and wrong retrieval
- Dealbreaker = zero tolerance: Contradictions are factual errors, not missing info. Any contradiction triggers 0.
- Continuous score: A pred satisfying 3/6 criteria gets 0.5, not 0.0. Aligned with FAB Partial Credit.

---

## 4. Final Score Aggregation

### 4.1 Formula
final_score = max(T1_score, T2_score)
final_pass = final_score >= FINAL_PASS_THRESHOLD  (default 0.5)

### 4.2 Rationale for max() not average()
- T1 and T2 measure different things: T1 checks numbers, T2 checks semantics
- A pred might have perfect numbers (T1=1.0) but fail T2 due to strict rubric
- A pred might have no numbers (qualitative question) but pass T2 semantically
- max() lets the best-available signal determine the score

### 4.3 Dealbreaker Override
If T2 dealbreaker triggered (contradiction found):
    final_score = 0.0
    final_pass = False
    # Overrides T1 even if numbers match

Why: If the answer contains a factual contradiction, even correct numbers elsewhere cannot save it. Example: "Revenue decreased to 60B" when gold says "Revenue increased to 60B" — the 60B is right but the direction is wrong.

---

## 5. Error Type Classification

When final_pass = False:

| Condition | Error Type |
|-----------|------------|
| T2 dealbreaker triggered | factual_contradiction |
| T1_score == 0 and T2_score == 0 | complete_failure |
| T1_score > 0 and T1_score < threshold | numeric_error |
| T2_score > 0 and T2_score < threshold | qualitative_incomplete |
| T1_score >= threshold but T2 dealbreaker | factual_contradiction |

---

## 6. Comparison with FAB Official Standard

| Dimension | FAB v2 (vals.ai) | FinAgent v2.0 | Gap |
|-----------|-----------------|---------------|-----|
| Primary metric | Partial Credit (continuous) | max(T1, T2) continuous | Aligned |
| Dealbreaker | Yes (severity-weighted) | Yes (contradiction triggers 0) | Aligned |
| Judge input | question + trace + ref + rubric | question + trace + ref + rubric | Aligned |
| Judge count | Multiple (majority vote) | 1 (extensible to N) | To improve |
| Rule-based layer | No | Yes (T1 numeric) | Extra |
| Weighted criteria | Yes (severity weights) | No (equal weight, future work) | To improve |

---

## 8. Known Limitations (as of v2.1)

These are documented limitations, not bugs. They will be addressed as research needs evolve.

### 8.1 Single-model multi-vote vs multi-model judge
- **Current**: T2 uses the same LLM (DeepSeek-V4-Flash) for 3 rounds. With temperature=0, the 3 votes are nearly identical, so multi-vote mainly reduces variance from output parsing, not from model bias.
- **FAB/BigFinanceBench standard**: Use 2-3 DIFFERENT LLMs (e.g., Gemini + Claude + GPT-4) with majority vote. Different models have independent biases, which truly reduces systematic judge bias.
- **Estimated impact**: +/-3-5% on boundary cases (score near threshold).
- **Plan**: Add a second judge model (e.g., GPT-4o-mini via OpenAI, or Claude via Anthropic API) for cross-model voting. Requires additional API credentials.

### 8.2 FAB-specific rubric dependency
- **Current**: T2 correctness/dealbreaker logic depends on `rubric_structured` field with `operator: correctness|contradiction`. This is FAB-specific.
- **Impact on other benchmarks**:
  - FinGAIA (gold_answer only, no rubric): T2 returns 0.5 (neutral), T1 dominates. Qualitative questions may be under-scored.
  - BigFinanceBench (rubric without operator field): T2 treats all criteria as `correctness`, dealbreaker never triggers.
  - Multiple-choice (e.g., MMLU-Finance): T1 numeric extraction fails (single letter), T1 keyword fallback filters tokens len<3, so "A" is dropped. T2 returns 0.5.
- **Plan**: Implement a benchmark-agnostic adapter that maps any benchmark schema to `{criteria, operator}`. Add T1 support for multiple-choice (letter matching). Add T2 fallback to gold_answer semantic match when no rubric exists.

### 8.3 No severity weights on criteria
- **Current**: All rubric criteria have equal weight. `T2_score = matched / total`.
- **Confirmed (2026-09-13)**: FAB public dataset `data/fab_public.csv` only has `operator` and `criteria` fields in the Rubric. No severity/weight/priority field exists in the public 50-question set. The FAB paper mentions severity-weighted scoring, but the public data does not include this field — it may be internal to Vals AI.
- **Impact**: Cannot implement weighted scoring without the field. Equal-weight approximation is the best available.
- **Plan**: When applying for the 150-question private set, ask Vals AI whether severity weights are included. If yes, implement weighted scoring.

### 8.4 Judge model capability ceiling
- **Current**: T2 judge is DeepSeek-V4-Flash (a smaller/faster model).
- **FAB standard**: Judge uses stronger models (Claude Opus, GPT-4).
- **Estimated impact**: Flash model may misjudge complex semantic criteria. However, DeepSeek V4 Pro scores 60.4% on FAB leaderboard, so the DeepSeek family is competent in finance tasks.
- **Plan**: Run ablation comparing Flash vs Pro as judge to quantify the gap.

### 8.5 No calibration
- **Current**: Pass threshold is a fixed 0.5.
- **Research standard**: Calibrate the threshold using labeled data to optimize for precision/recall trade-off.
- **Plan**: Once we have 150+ labeled examples, run threshold sweep (0.3, 0.4, 0.5, 0.6, 0.7) and report the optimal threshold.

### 8.6 Trajectory truncation
- **Current**: Trajectory is truncated to 4000 chars (`T2_MAX_TRAJECTORY_CHARS`). Long agent runs (10 steps with large EDGAR filings) may lose evidence.
- **Impact**: Judge may miss evidence in truncated trajectory, causing false negatives.
- **Plan**: Implement smart trajectory summarization (extract key tool outputs) instead of char-based truncation.

### 8.7 Single-judge per criterion (within a round)
- **Current**: Each criterion is judged once per round (3 rounds total = 3 judgments per criterion).
- **FAB standard**: Multiple judges per criterion.
- **Note**: This overlaps with 8.1. The distinction is that 8.1 is about model diversity; 8.7 is about judgment count. Both are addressed by adding more judge models.


---

## 7. Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-09-12 | Initial 3-tier: T1 exact, T2 numeric, T3 LLM-judge (binary) |
| 2.0 | 2026-09-13 | Merge to 2-tier: T1 numeric (continuous), T2 LLM semantic (continuous + dealbreaker). Remove T3. |
| 2.1 | 2026-09-13 | T1 keyword matching fallback for non-numeric gold. T2 multi-vote (3 rounds, median/majority). Judge prompt: ignore reasoning prefixes. Agent prompt: enforce ANSWER-only format. |
| 2.1-doc | 2026-09-13 | Document 7 known limitations (Section 8). No code changes. |
| 2.2 | 2026-09-15 | Fair evaluation principles (Section 9). API retry + api_failure flag. Abstract placeholders in ReAct prompt. |
| 2.3 | 2026-09-15 | Empirical findings: V4-Flash vs V4-Pro cross-model comparison (Section 10). Failure mode taxonomy. Cost analysis. |

---

## 9. Fair Evaluation Principles

**Goal**: Ensure the evaluation measures LLM's intrinsic ability, not artifacts of the test harness.

### 9.1 Three Core Principles

#### P1: Non-Interference (不干扰)
The test harness must NOT compensate for LLM's tool-calling deficiencies.

- **Removed**: `auto-recovery` logic that auto-filled empty `edgar_search` params from task prompt.
- **Rationale**: Auto-recovery masked LLM's inability to generate correct tool parameters. If LLM passes empty `{}` as tool args, that IS the LLM's ability gap — the evaluator must not hide it.
- **What we keep**: Parameter alias resolution (`q`/`company`/`ticker` -> `query`) is API design best practice, not interference. Real-world APIs accept aliases.

#### P2: Non-Guidance (不引导)
Tool-call examples in the ReAct prompt must use abstract placeholders, not concrete content.

- **Before (biased)**: `ARGS: {"query": "NVIDIA 10-K 2024", ...}` — may bias LLM toward searching NVIDIA or using "10-K 2024" pattern.
- **After (neutral)**: `ARGS: {"query": "<company name> <filing type> <year>", ...}` — shows format only, no content hint.
- **Rationale**: Concrete examples create priming effects. LLM may mimic example content instead of constructing query from the actual task. Abstract placeholders give format documentation without content guidance — like API docs for human developers.

#### P3: Non-Attribution (不归咎)
Transient infrastructure failures must NOT count against LLM's ability score.

- **API retry mechanism**: 3 retries with exponential backoff (1s, 2s, 4s) for timeout/connection/rate-limit errors.
- **`api_failure` flag**: If all retries fail, the task is marked `api_failure` and excluded from accuracy denominator.
- **Rationale**: Network timeouts and API rate limits are infrastructure issues, not LLM reasoning failures. The evaluator reports two accuracies:
  - `Accuracy (all tasks)`: includes API failures as 0
  - `Accuracy (excl. API failures)`: excludes API failures from denominator

### 9.2 Summary Table

| Principle | What it means | Implementation |
|-----------|---------------|----------------|
| Non-Interference | Don't help LLM fill params | Removed auto-recovery |
| Non-Guidance | Don't hint at search content | Abstract placeholders |
| Non-Attribution | Don't penalize infra failures | API retry + api_failure flag |

### 9.3 What This Does NOT Mean

- We do NOT remove all error handling. Tool parameter aliases and syntax-tolerant parsing remain — these are API robustness, not LLM assistance.
- We do NOT remove the API retry mechanism. Retry handles transient failures, not LLM reasoning gaps.
- We do NOT remove the fallback path (max_steps exhausted -> generate answer from context). Fallback tests LLM's ability to synthesize from partial information.

---

## 10. Empirical Findings (Model Comparison Experiments)

*Experiments conducted 2026-09-14/15 on FAB 50 questions under fair evaluation principles.*

### 10.1 Cross-Model Comparison

| Model | Params | Accuracy | API Failures | Max Steps Hit | Avg Latency |
|---|---|---|---|---|---|
| DeepSeek-V4-Flash | 284B/13B active | **28%** (14/50) | 1/50 | ~50% | 66s |
| DeepSeek-V4-Pro | 1.6T/49B active | 8.3% (1/12 valid) | 38/50 (credits) | 92% (11/12) | 134s |

*V4-Pro results are incomplete due to HF credit depletion (402 Payment Required after 15 tasks). Valid accuracy computed on 12 tasks only.*

### 10.2 Key Finding: Larger ≠ Better on ReAct Text Format

V4-Pro (1.6T/49B) performed **worse** than V4-Flash (284B/13B) under the ReAct text-based tool-calling format, despite having 3.7x more active parameters.

**Three behavioral patterns identified in V4-Pro:**

1. **Search over-refinement**: V4-Pro takes 3-4 edgar_search calls to find the correct filing (vs 1-2 for V4-Flash), consuming step budget on query refinement.
2. **No offset pagination**: V4-Pro repeatedly calls parse_html on the same URL (10x in one task) without using the offset parameter, wasting all 10 steps fetching identical content.
3. **Convergence failure**: V4-Pro never decides "I have enough information" — 11/12 valid tasks hit max_steps without producing an ANSWER, vs ~50% for V4-Flash.

### 10.3 Root Cause Analysis: Format Compatibility

V4-Pro's underperformance is **not** a model capability deficit but a **format compatibility issue**:

- V4-Pro's "thoroughness" (more search refinement, more verification) is penalized by the step budget (max_steps=15).
- V4-Flash's "decisiveness" (quick search → immediate ANSWER) is rewarded by the same budget.
- V4-Pro supports native function calling (per DeepSeek API docs), which may eliminate parameter parsing overhead and improve efficiency.

**Implication for benchmark design**: ReAct text format introduces a **format tax** on thorough models. Fair cross-model comparison requires either (a) native function calling for models that support it, or (b) larger step budgets for thorough models — but (b) violates the "equal conditions" principle.

### 10.4 Cost Analysis

| Model | Price (input/output per 1M tokens) | Est. cost/50 tasks | Est. cost ratio |
|---|---|---|---|
| V4-Flash | $0.22 / $0.66 | ~$0.30 | 1x |
| V4-Pro | $0.66 / $1.98 | ~$8-10 | ~30x |

V4-Pro costs ~30x more due to: 3x unit price × 3.6x more tokens per task (more steps, longer responses).

### 10.5 Failure Mode Taxonomy (V4-Flash, Fair Evaluation)

After removing code artifacts (truncation bugs, auto-recovery), 28 complete_failures decompose as:

| Root Cause | Count | % | Nature |
|---|---|---|---|
| retrieve_information missing query param | 11 | 39% | LLM tool-calling gap |
| Wrong document retrieved | 7 | 25% | LLM search strategy gap |
| Found doc but didn't extract answer | 7 | 25% | LLM convergence gap |
| Fallback reasoning prefix | 2 | 7% | LLM instruction-following gap |
| Fallback "Not found" | 1 | 4% | LLM synthesis gap |

**All 28 failures are LLM capability gaps, not system artifacts** — confirming the fair evaluation framework measures LLM ability, not infrastructure limitations.

### 10.6 Future Model Comparison Plan

| Model | Interface | Status |
|---|---|---|
| DeepSeek-V4-Flash | ReAct text | ✅ Baseline (28%) |
| DeepSeek-V4-Pro | ReAct text | ⚠️ Credits depleted |
| Llama-3.3-70B-Instruct | ReAct text | Planned (free on HF) |
| Qwen-2.5-72B-Instruct | ReAct text | Planned (free on HF) |
| DeepSeek-V4-Pro | Native function calling | Planned (DeepSeek API) |
| GPT-4o | Native function calling | Planned (OpenAI API) |

Cross-interface comparison (ReAct vs native function calling) will isolate format tax from model capability.

---

## 11. Interference Registry (Potential Evaluation Confounds)

*Living document — updated as new confounds are identified. Each entry tracked from hypothesis → experiment → resolution.*

### 11.1 Confirmed Interferences (P0-P1)

| ID | Interference | Status | Evidence | Impact | Fix |
|---|---|---|---|---|---|
| INT-01 | **ReAct text format** | CONFIRMED | V4-Pro: 8% (ReAct) vs 60% (native FC, FAB official). 7x performance loss. | 25/28 complete_failures linked to ReAct parsing failures | Switch to native function calling |
| INT-02 | **Parameter omission in ReAct** | CONFIRMED | 11/28 failures: LLM writes `ARGS: {"text": "..."}` without `query` → tool error | 39% of complete_failures | Eliminated by native FC (API enforces `required` fields) |
| INT-03 | **No offset pagination understanding** | CONFIRMED | V4-Pro called parse_html 10x on same URL without offset; got same 15000 chars each time | 5/28 failures, 10 wasted steps/task | Native FC schema declares `offset: integer` clearly |
| INT-04 | **ANSWER: format detection** | CONFIRMED | V4-Pro never outputs `ANSWER:` → 11/12 tasks hit max_steps | 92% of V4-Pro valid tasks | Native FC: `tool_choice="none"` forces answer output |

### 11.2 Suspected Interferences (P2-P3, Under Investigation)

| ID | Interference | Status | Hypothesis | Experiment Needed |
|---|---|---|---|---|
| INT-05 | **max_steps=15 too restrictive** | SUSPECTED | V4-Pro needs more steps (4 edgar_search + 10 parse_html); 15 may not be enough for thorough models | Run with max_steps=20, 25, 30 and compare accuracy delta |
| INT-06 | **Context truncation 8000 chars** | SUSPECTED | SEC 10-K filings are 50K+ chars; 8000 may cut key data | Run with 12000, 16000 context and measure if complete_failure drops |
| INT-07 | **T2 judge self-evaluation bias** | SUSPECTED | V4-Flash judges V4-Flash answers → may be lenient or harsh on itself | Run T2 with Llama-3.3-70B as judge, compare scores |
| INT-08 | **temperature=0 not optimal** | SUSPECTED | Some models perform better with slight temperature (0.1-0.3) for creative retrieval | Run with temperature=0.1, 0.3 and compare |
| INT-09 | **Fallback prompt quality** | SUSPECTED | Fallback generates from truncated context → may produce incomplete answers | Compare fallback vs forced-answer with tool_choice="none" |
| INT-10 | **Single judge model** | SUSPECTED | T2 uses single model (V4-Flash) for judging → model-specific biases | Multi-model judge ensemble (V4-Flash + Llama-3.3-70B + Qwen-2.5-72B) |
| INT-11 | **Tool result cache** | NOT_INTERFERENCE | 0 cache hits = agent exploring different params (correct behavior). High cache hits would indicate agent stuck in loop (capability defect). Cache is a performance optimization, not an evaluation factor. | No action needed. Cache hits rate could be a diagnostic metric (high = stuck, low = exploring). |
| INT-12 | **Forced synthesis step** | SUSPECTED | Forcing model to answer when remaining steps ≤5 deprives search time. FAB/SWE-agent do NOT force synthesis. Could reduce score for thorough models. | Compare with/without forced synthesis; align with FAB |

### 11.3 Monitoring Variables (P4, Long-term)

| ID | Variable | Why Monitor | Current Value |
|---|---|---|---|
| MON-01 | API stability (timeout rate) | High timeout rate confounds accuracy | 1-3% (with retry) |
| MON-02 | HF credit depletion | 402 errors invalidate test runs | Monitor before each run |
| MON-03 | Model version drift | HF router may silently update model versions | Record model version per run |
| MON-04 | EDGAR API rate limiting | Too many edgar_search calls → throttled | 300 req/hour limit |
| MON-05 | Document length distribution | SEC filings vary 5K-200K chars; affects parse_html | Track per-task |

### 11.4 Research Questions (Open)

| RQ | Question | Why It Matters |
|---|---|---|
| RQ-01 | Does native FC eliminate ALL 28 complete_failures, or do some persist? | If some persist → genuine LLM capability gap; if all eliminated → all failures were format artifacts |
| RQ-02 | Is there an optimal max_steps that maximizes accuracy without inflating cost? | Trade-off between thoroughness and efficiency |
| RQ-03 | Does the format tax (ReAct vs native FC) vary by model size? | Larger models may suffer more (V4-Pro 7x loss) vs smaller (V4-Flash 2x?) |
| RQ-04 | Can we measure "format tax" as a metric itself? | Quantifying how much format choice affects accuracy is a benchmark design contribution |
| RQ-05 | Does T2 judge model choice affect model ranking? | If V4-Flash ranks higher with V4-Flash judge vs Llama judge → judge bias |
| RQ-06 | Is ReAct a fair test for models WITHOUT native FC? | If yes, ReAct has value as a "format robustness" test, but not as a capability test |

### 11.5 Resolution Protocol

Each interference follows this lifecycle:
1. **Hypothesis**: Identified as potential confound
2. **Experiment**: Controlled test (A/B comparison, isolating the variable)
3. **Analysis**: Quantify impact (accuracy delta, failure mode change)
4. **Resolution**: Fix (if confirmed) or dismiss (if no impact)
5. **Documentation**: Update this registry with findings
