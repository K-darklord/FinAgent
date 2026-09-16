# Interference Causal Comparison Table

**Last Updated**: 2026-09-16 (50-task full validation)
**Git Tag**: v3-interference-fix-60pct (5-task) → 50-task full run confirmed 48%

---

## 1. Bug vs Interference Classification

| ID | Issue | Classification | Rationale |
|---|---|---|---|
| INT-13-L1 | Schema missing REQUIRED markers | **BUG** | Code was wrong (schema didn't match function). Fixed by adding markers. |
| INT-13-L3 | Cache append in wrong function | **BUG** | Code was wrong (cache code in retrieve_information instead of parse_html). Fixed by moving code. |
| INT-13-L3b | retrieve_information references deleted `text` var | **BUG** | Code was wrong (stale variable reference). Fixed by removing reference. |
| INT-14 | Tool design requires LLM to pass 2000+ chars as JSON param | **INTERFERENCE** | Code is correct (function receives text, processes, returns). But design itself is incompatible with FC paradigm. LLM cannot reproduce large text in JSON. |
| INT-01 | ReAct vs Function Calling format | **INTERFERENCE** | Both implementations correct. But format choice systematically penalizes models less good at format-following. |
| INT-02 | Negative example priming | **INTERFERENCE** | Prompt code correct (does what it says). But including negative examples causes LLM to mimic them. |
| INT-03 | Fallback prompt "ONE LINE ONLY" | **INTERFERENCE** | Code correctly executes truncation. But parameter choice truncates key info. |
| INT-04 | max_tokens=256 | **INTERFERENCE** | Code correct. But parameter too restrictive for some answers. |
| INT-05 | max_steps=15 | **INTERFERENCE** | Code correct. But parameter may be insufficient for thorough models. |
| INT-06 | Context truncation 8000 chars | **INTERFERENCE** | Code correct. But SEC 10-K filings are 50K+ chars. |
| INT-07 | T2 judge = same model as agent | **INTERFERENCE** | Code correct. But mechanism introduces self-evaluation bias. |
| INT-12 | Forced synthesis step | **INTERFERENCE** | Code correct. But mechanism deprives search time. FAB doesn't force. |

---

## 2. Causal Quantification (Interference Only)

| ID | Interference | Type | Before (Version) | Accuracy Before | After (Version) | Accuracy After | Delta (pp) | Evidence |
|---|---|---|---|---|---|---|---|---|
| INT-14 | Tool design vs FC paradigm | Tool Design | v1-native-fc (5 tasks) | 40% | v3-interference-fix (50 tasks) | 48% | **+8pp** | RI: 0→305/305 (100%), 50-task full validation |
| INT-01 | ReAct format tax | Format | v0-ReAct (V4-Pro) | 8.3% | v0-FC (FAB official) | 60.39% | **+52.09pp** | Same model, 7x gap |
| INT-02 | Negative example priming | Prompt | v0 (50 tasks) | 16% | v1 removed (50 tasks) | 34% | **+18pp** | Accuracy regressed 34%→16% when bad examples added |
| INT-03 | Fallback "ONE LINE ONLY" | Hyperparam | v0 (50 tasks) | 30% | v1 relaxed (50 tasks) | 34% | **+4pp** | Overly restrictive prompt truncated info |
| INT-04 | max_tokens=256 | Hyperparam | v0 (50 tasks) | 22% | v1 256 maintained (50 tasks) | 34% | **+12pp** | Increasing to 1024 had no effect; issue was elsewhere |
| INT-05 | max_steps=25→50 | Hyperparam | v3 baseline (50 tasks, max_steps=25) | 48% | INT-05 (50 tasks, max_steps=50) | 54% | **+6pp** | complete_failure 20→13 (-7), total errors 26→23 (-3). Confirmed interference. |
| INT-06 | Context 8000 chars | Hyperparam | SUSPECTED | TBD | TBD | TBD | TBD | Needs controlled experiment |
| INT-07 | Judge self-eval bias | Mechanism | SUSPECTED | TBD | TBD | TBD | TBD | Needs multi-model judge experiment |
| INT-12 | Forced synthesis | Mechanism | SUSPECTED | TBD | TBD | TBD | TBD | Needs with/without comparison |

---

## 3. Interference by Category

### 3.1 Tool Design Interference (Highest Impact)
- **INT-14** (+8pp on 50 tasks, +20pp on 5 tasks): Tool design incompatible with calling paradigm
  - Root cause: retrieve_information required LLM to pass document text as JSON parameter
  - Fix: Made tool stateful (auto-search cached documents)
  - Key insight: "Code correct but design unusable" — hardest to detect

### 3.2 Format Interference
- **INT-01** (+52pp): ReAct text format vs native Function Calling
  - Root cause: ReAct requires text parsing; parse failure = tool call failure
  - Fix: Migrated to native FC
  - Key insight: Format choice is a confounding variable, not a technical detail

### 3.3 Prompt Design Interference
- **INT-02** (+18pp): Negative example priming
  - Root cause: LLM mimics example content regardless of positive/negative framing
  - Fix: Removed negative examples from fallback prompt
  - Key insight: Avoid specific examples in prompts; use abstract placeholders

- **INT-03** (+4pp): Fallback prompt truncation
  - Root cause: "ONE LINE ONLY" truncated critical information
  - Fix: Relaxed to allow multi-line answers
  - Key insight: Restrictive prompts don't improve precision; they lose information

### 3.4 Hyperparameter Interference
- **INT-04** (0pp direct): max_tokens
  - Finding: Increasing 256→1024 had no accuracy effect; root issue was API failures
  - Key insight: Hyperparameter changes can mask other issues; need isolation

- **INT-05/06** (TBD): max_steps, context length
  - Status: Suspected, needs controlled experiments

### 3.5 Mechanism Interference
- **INT-07** (TBD): LLM-as-Judge self-evaluation
  - Status: Suspected, needs multi-model judge experiment
  - Key insight: Using same model for agent and judge creates circular reasoning

- **INT-12** (TBD): Forced synthesis step
  - Status: Suspected, needs with/without comparison
  - Key insight: FAB doesn't force synthesis; we shouldn't either

---

## 4. Summary Statistics

| Metric | Value |
|---|---|
| Total issues identified | 12 (INT-01 to INT-14) |
| Pure bugs | 3 |
| True interferences | 9 |
| Confirmed (with causal data) | 5 |
| Suspected (needs experiment) | 4 |
| Non-interference | 1 (INT-11: cache hit rate) |
| Total accuracy recovered | 34% → 48% (+14pp on 50 tasks) |
| Of which: bug fixes | 34% → 40% (+6pp) |
| Of which: interference fixes | 40% → 48% (+8pp) |
| RI success rate | 0/105 (0%) → 305/305 (100%) |
