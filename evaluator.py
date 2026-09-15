"""
evaluator.py
=============
Unified evaluation module: merges scorer.py + analysis.py.

I provide two scoring paths:
  1. Legacy classify_error() for the mini benchmark (rule-based, unchanged).
  2. Rubric-coverage scoring for FAB questions (uses structured rubric criteria).

Error taxonomy (6 labels, intentionally simple — upgrade is future work):
  - retrieval_failure      : did not retrieve / retrieved wrong source
  - numeric_error          : wrong number, unit, or rounding
  - citation_missing       : missing source citation
  - tool_error             : tool call itself errored
  - qualitative_incomplete : qualitative answer missing key points
  - correct                : passed
"""
from __future__ import annotations
import csv
import json
import re
import logging
from pathlib import Path
from collections import Counter

from runner import OUTPUT_DIR
import config

logger = logging.getLogger(__name__)


# ======================================================================
# Scoring helpers (preserved verbatim from scorer.py)
# ======================================================================

def _extract_numbers(s: str) -> list[float]:
    """I extract all numbers from a string, handling thousand-separator commas."""
    cleaned = re.sub(r"(?<=\d),(?=\d{3}\b)", "", str(s))
    return [float(x) for x in re.findall(r"-?\d+\.?\d*", cleaned)]


def score_numeric(gold: str, pred: str, tol: float = 0.05) -> bool:
    """I parse numbers and allow relative tolerance. I pick the pred number closest to gold."""
    try:
        g_nums = _extract_numbers(gold)
        p_nums = _extract_numbers(pred)
        if not g_nums or not p_nums:
            return False
        g = g_nums[0]
        p = min(p_nums, key=lambda x: abs(x - g))
    except (IndexError, ValueError):
        return False
    if g == 0:
        return abs(p) < tol
    return abs(g - p) / abs(g) <= tol


def score_exact(gold: str, pred: str) -> bool:
    """I check if gold is a substring of pred (case-insensitive)."""
    return str(gold).strip().lower() in str(pred).strip().lower()


def classify_error(task: dict) -> str:
    """I classify an error based on trajectory + answer (legacy rule-based path).
    I keep the original 6-label taxonomy unchanged."""
    traj = task.get("trajectory", [])
    tool_steps = [s for s in traj if s.get("tool_name")]

    # 1) Tool itself errored
    if any("tool_error" in s.get("tool_output", "") for s in tool_steps):
        return "tool_error"

    # 2) No tool calls for retrieval/reasoning tasks
    if not tool_steps and task.get("category", "") in (
        "Quantitative Retrieval", "Qualitative Retrieval", "Numerical Reasoning"):
        return "retrieval_failure"

    # 3) Numerical reasoning -> numeric comparison
    if task.get("category") == "Numerical Reasoning":
        return "correct" if score_numeric(task.get("gold_answer", ""),
                                          task.get("final_answer", "")) else "numeric_error"

    # 4) Qualitative retrieval -> keyword coverage
    if task.get("category") == "Qualitative Retrieval":
        keywords = ["productivity", "intelligent cloud", "more personal computing"]
        hit = sum(1 for k in keywords if k.lower() in task.get("final_answer", "").lower())
        if hit >= 2:
            return "correct"
        return "qualitative_incomplete"

    # 5) Quantitative retrieval -> numeric with looser tolerance
    if task.get("category") == "Quantitative Retrieval":
        return "correct" if score_numeric(task.get("gold_answer", ""),
                                          task.get("final_answer", ""),
                                          tol=config.SCORING_QUANTITATIVE_TOL) else "numeric_error"
    return "correct"


# ======================================================================
# FAB rubric-based scoring (new path, same 6 labels)
# ======================================================================

def _rubric_coverage(row: dict, threshold: float = None) -> str:
    """I score FAB rows using their structured rubric.
    For each 'correctness' criterion I check (case-insensitive substring) if it
    appears in final_answer. I also penalize 'contradiction' criteria that match.
    I return one of the existing 6 labels, never a new one."""
    if threshold is None:
        threshold = config.SCORING_RUBRIC_COVERAGE

    metadata = row.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            metadata = {}

    structured = metadata.get("rubric_structured", [])
    if not structured:
        # Fall back to legacy classify_error path
        return classify_error(row)

    pred = str(row.get("final_answer", "")).lower().strip()
    if not pred:
        return "qualitative_incomplete"

    correctness = [c for c in structured if c.get("operator") == "correctness"]
    contradictions = [c for c in structured if c.get("operator") == "contradiction"]

    # If a forbidden contradiction string is present, the answer is wrong
    for c in contradictions:
        crit = c.get("criteria", "").strip().lower()
        if crit and crit in pred:
            return "numeric_error" if any(ch.isdigit() for ch in pred) else "qualitative_incomplete"

    # Check correctness criteria coverage
    if not correctness:
        return classify_error(row)

    hit = sum(1 for c in correctness
              if c.get("criteria", "").strip().lower() in pred)
    cov = hit / len(correctness)

    if cov >= threshold:
        return "correct"

    # Distinguish numeric vs qualitative errors
    gold = str(row.get("gold_answer", ""))
    if any(ch.isdigit() for ch in gold):
        return "numeric_error"
    return "qualitative_incomplete"



# ======================================================================
# Tier 3: LLM-as-Judge scoring (uses HF router API)
# ======================================================================


def _normalize_answer(s: str) -> str:
    """I normalize an answer string for robust comparison:
    - lowercase, strip whitespace
    - remove commas, dollar signs, percent signs, parentheses
    - remove common units (million, billion, etc.)
    - collapse multiple spaces
    """
    s = str(s).strip().lower()
    for ch in ["\$", "€", "£", "¥", ",", "(", ")", "%"]:
        s = s.replace(ch, " ")
    for unit in ["million", "billion", "trillion", "thousand", "mn", "bn", "mm"]:
        s = s.replace(unit, " ")
    s = " ".join(s.split())
    return s


def _extract_all_numbers(s: str) -> list:
    """I extract all numbers from a string, handling commas and units."""
    s = str(s).replace(",", "")
    matches = re.findall(r"-?\d+\.?\d*", s)
    nums = []
    for m in matches:
        try:
            nums.append(float(m))
        except ValueError:
            pass
    return nums


def _is_fab_row(row: dict) -> bool:
    """I check if a row is a FAB question (has rubric_structured or fab_ prefix)."""
    task_id = str(row.get("task_id", ""))
    metadata = row.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            metadata = {}
    return task_id.startswith("fab_") or bool(metadata.get("rubric_structured"))


def _rubric_coverage_normalized(row: dict) -> bool:
    """I check rubric criteria using normalized comparison. Fallback for T2 when no LLM token."""
    metadata = row.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            metadata = {}

    structured = metadata.get("rubric_structured", [])
    if not structured:
        return False

    pred = _normalize_answer(str(row.get("final_answer", "")))
    if not pred:
        return False

    correctness = [c for c in structured if c.get("operator") == "correctness"]
    if not correctness:
        return False

    hit = 0
    for c in correctness:
        crit = _normalize_answer(c.get("criteria", ""))
        if not crit:
            continue
        if crit in pred:
            hit += 1
            continue
        # Numeric fallback
        crit_nums = _extract_all_numbers(c.get("criteria", ""))
        pred_nums = _extract_all_numbers(str(row.get("final_answer", "")))
        sig_crit = [n for n in crit_nums if not (1900 < n < 2100)]
        if sig_crit and pred_nums:
            all_match = True
            for cn in sig_crit:
                found = False
                for pn in pred_nums:
                    if cn == 0:
                        if pn == 0:
                            found = True
                            break
                    elif abs(pn - cn) / max(abs(cn), 1e-10) < config.SCORING_NUMERIC_TOLERANCE:
                        found = True
                        break
                if not found:
                    all_match = False
                    break
            if all_match:
                hit += 1

    cov = hit / len(correctness) if correctness else 0
    return cov >= config.SCORING_RUBRIC_COVERAGE



_STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "been", "have", "has", "had",
    "this", "that", "with", "from", "they", "them", "their", "there", "these",
    "those", "what", "which", "who", "when", "where", "why", "how", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "only", "own", "same", "than", "too", "very", "can", "will", "just",
    "should", "now", "also", "not", "but", "however", "into", "its",
}

# ======================================================================
# Tier 1: Numeric Accuracy (rule-based, deterministic, continuous 0-1)
# ======================================================================

def _score_t1_numeric(row: dict) -> float:
    """I check whether the predicted answer contains the core numeric values
    from the gold answer, within a configurable tolerance.
    I return a continuous score in [0, 1].
    I do NOT care about reasoning text — I only look at numbers."""
    gold_raw = str(row.get("gold_answer", "")).strip()
    pred_raw = str(row.get("final_answer", "")).strip()
    if not gold_raw or not pred_raw:
        return 0.0

    # Extract numbers from gold, filter out years (1900-2100)
    gold_nums = _extract_all_numbers(gold_raw)
    sig_gold = [n for n in gold_nums if not (1900 < n < 2100)]

    if not sig_gold:
        # Text-only answer: fallback to keyword matching (continuous score)
        # I extract significant keywords from gold and count how many appear in pred
        gold_norm = _normalize_answer(gold_raw)
        pred_norm = _normalize_answer(pred_raw)

        # Exact substring match -> full score
        if gold_norm in pred_norm:
            return 1.0

        # Keyword coverage: extract significant tokens from gold (len >= 3, not stopwords)
        gold_tokens = [t for t in gold_norm.split() if len(t) >= 3 and t not in _STOPWORDS]
        if not gold_tokens:
            return 0.0

        pred_tokens_set = set(pred_norm.split())
        matched = sum(1 for t in gold_tokens if t in pred_tokens_set)
        return matched / len(gold_tokens)

    # Extract numbers from pred
    pred_nums = _extract_all_numbers(pred_raw)
    if not pred_nums:
        return 0.0

    # Count how many significant gold numbers appear in pred
    matched = 0
    for gn in sig_gold:
        for pn in pred_nums:
            if gn == 0:
                if pn == 0:
                    matched += 1
                    break
            elif abs(pn - gn) / max(abs(gn), 1e-10) < config.T1_NUMERIC_TOLERANCE:
                matched += 1
                break

    return matched / len(sig_gold)


# ======================================================================
# Tier 2: LLM Semantic Judgment (continuous 0-1, with dealbreaker)
# ======================================================================

_judge_client = None

def _get_judge_client():
    """I lazily create an OpenAI client for LLM judge via HF router."""
    global _judge_client
    if _judge_client is None:
        import os
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if not token:
            return None
        from openai import OpenAI
        _judge_client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=token,
        )
    return _judge_client


def _format_trajectory_summary(row: dict, max_chars: int = None) -> str:
    """I extract a compact summary of the agent trajectory for the LLM judge.
    I include tool names, inputs, and key observations, truncated to max_chars."""
    if max_chars is None:
        max_chars = config.T2_MAX_TRAJECTORY_CHARS

    traj = row.get("trajectory", [])
    if not traj:
        return "[no trajectory]"

    parts = []
    total_len = 0
    for step in traj:
        step_num = step.get("step", "?")
        tool = step.get("tool_name", "")
        tool_input = str(step.get("tool_input", ""))[:200]
        obs = str(step.get("observation", ""))[:300]
        thought = str(step.get("thought", ""))[:100]

        if tool:
            entry = f"Step {step_num}: Called {tool}({tool_input}) -> {obs}"
        else:
            entry = f"Step {step_num}: {thought}"
        parts.append(entry)
        total_len += len(entry)
        if total_len > max_chars:
            break

    result = "\n".join(parts)
    return result[:max_chars]


def _llm_judge_correctness_single(row: dict, criteria_list: list,
                                         question: str, pred: str,
                                         traj_summary: str) -> float:
    """I run ONE LLM judge call and return the score (0-1)."""
    client = _get_judge_client()
    if client is None:
        return 0.0

    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(criteria_list))

    prompt = (
        f"You are an expert financial evaluator.\n"
        f"Judge whether the answer satisfies EACH criterion below.\n"
        f"IMPORTANT: The answer may contain reasoning text (e.g., 'The question asks...').\n"
        f"Ignore the reasoning prefix — focus on whether the factual content satisfies each criterion.\n"
        f"Consider the agent retrieved evidence in the trajectory.\n"
        f"Reply with ONLY a comma-separated list of YES/NO (one per criterion).\n"
        f"Example: YES,NO,YES,YES,NO\n\n"
        f"Question: {question}\n\n"
        f"Agent Trajectory (retrieved evidence):\n{traj_summary}\n\n"
        f"Answer: {pred[:2000]}\n\n"
        f"Criteria:\n{numbered}\n\n"
        f"Verdicts (comma-separated YES/NO):"
    )

    try:
        resp = client.chat.completions.create(
            model=config.T2_JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=config.T2_TEMPERATURE,
        )
        verdict_text = (resp.choices[0].message.content or "").strip().upper()
        verdicts = [v.strip() for v in verdict_text.split(",")]
        hit = sum(1 for v in verdicts[:len(criteria_list)] if v.startswith("YES"))
        return hit / len(criteria_list)
    except Exception as e:
        logger.warning(f"LLM judge correctness batch failed: {e}")
        return 0.0


def _llm_judge_correctness(row: dict) -> float:
    """I use an LLM to judge correctness criteria with multi-vote (3 rounds).
    I run the judge 3 times and take the MEDIAN score to reduce variance.
    I batch ALL correctness criteria into ONE prompt per round (3 API calls total).
    I return a continuous score in [0, 1]."""
    client = _get_judge_client()
    if client is None:
        # No token: fall back to rubric coverage as approximation
        return 1.0 if _rubric_coverage_normalized(row) else 0.0

    metadata = row.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            metadata = {}

    structured = metadata.get("rubric_structured", [])
    if not structured:
        return 0.5  # No rubric: neutral, let T1 decide

    pred = str(row.get("final_answer", "")).strip()
    if not pred:
        return 0.0

    question = str(row.get("prompt", ""))
    correctness = [c for c in structured if c.get("operator") == "correctness"]

    if not correctness:
        return 0.5  # No correctness criteria: neutral

    criteria_list = [c.get("criteria", "").strip() for c in correctness
                     if c.get("criteria", "").strip()]
    if not criteria_list:
        return 0.0

    traj_summary = _format_trajectory_summary(row)

    # Multi-vote: run 3 times and take median
    n_votes = 3
    scores = []
    for _ in range(n_votes):
        s = _llm_judge_correctness_single(row, criteria_list, question, pred, traj_summary)
        scores.append(s)

    # Median: sort and pick middle
    scores.sort()
    median_score = scores[len(scores) // 2]
    return median_score


def _llm_judge_dealbreaker_single(row: dict, contra_list: list,
                                        pred: str) -> bool:
    """I run ONE dealbreaker check. Return True if any contradiction is YES."""
    client = _get_judge_client()
    if client is None:
        return False

    numbered_contra = "\n".join(f"{i+1}. {c}" for i, c in enumerate(contra_list))

    prompt = (
        f"You are an expert financial fact-checker.\n"
        f"Does the answer CONTRADICT any statement below?\n"
        f"If the answer states something opposite to a statement, reply YES for that.\n"
        f"If the answer is silent or agrees, reply NO.\n"
        f"IMPORTANT: Ignore reasoning prefixes in the answer. Focus on factual claims.\n"
        f"Reply with ONLY a comma-separated list of YES/NO.\n\n"
        f"Answer: {pred[:2000]}\n\n"
        f"Statements (all must be true):\n{numbered_contra}\n\n"
        f"Contradiction verdicts (comma-separated YES/NO):"
    )

    try:
        resp = client.chat.completions.create(
            model=config.T2_JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=config.T2_TEMPERATURE,
        )
        contra_text = (resp.choices[0].message.content or "").strip().upper()
        contra_verdicts = [v.strip() for v in contra_text.split(",")]
        return any(v.startswith("YES") for v in contra_verdicts[:len(contra_list)])
    except Exception as e:
        logger.warning(f"LLM judge dealbreaker check failed: {e}")
        return False


def _llm_judge_dealbreaker(row: dict) -> bool:
    """I check if the answer contradicts any dealbreaker statement.
    I run 3 times and use majority vote (2/3) to reduce variance.
    If YES on any contradiction criterion (in majority of rounds), dealbreaker triggered.
    I return True if a dealbreaker is triggered (score should be 0)."""
    metadata = row.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            metadata = {}

    structured = metadata.get("rubric_structured", [])
    contradictions = [c for c in structured if c.get("operator") == "contradiction"]

    if not contradictions:
        return False

    pred = str(row.get("final_answer", "")).strip()
    if not pred:
        return False

    contra_list = [c.get("criteria", "").strip() for c in contradictions
                   if c.get("criteria", "").strip()]
    if not contra_list:
        return False

    # Multi-vote: run 3 times, majority (>=2) triggers dealbreaker
    n_votes = 3
    triggered_count = 0
    for _ in range(n_votes):
        if _llm_judge_dealbreaker_single(row, contra_list, pred):
            triggered_count += 1
    return triggered_count >= 2


def _score_t2_llm_semantic(row: dict) -> tuple:
    """I run the LLM semantic judge with dealbreaker check.
    I return (score: float, dealbreaker_triggered: bool).
    If dealbreaker triggers, score = 0.0."""
    # Step 1: Check dealbreakers (contradictions)
    dealbreaker = _llm_judge_dealbreaker(row)
    if dealbreaker:
        return (0.0, True)

    # Step 2: Score correctness criteria
    score = _llm_judge_correctness(row)
    return (score, False)


# ======================================================================
# Final score aggregation
# ======================================================================

def _label_row_tiered(row: dict) -> dict:
    """I run T1 (numeric) and T2 (LLM semantic) and aggregate the final score.
    final_score = max(T1, T2), unless dealbreaker triggers (then 0).
    I return continuous scores + final pass/fail + error type."""
    t1_score = _score_t1_numeric(row)
    t2_score, dealbreaker = _score_t2_llm_semantic(row)

    # API failure override: if the agent failed due to API timeout/connection,
    # mark as api_failure and exclude from accuracy denominator
    if row.get("api_failure", False):
        return {
            "tier1_numeric": 0.0,
            "tier2_llm_semantic": 0.0,
            "dealbreaker_triggered": False,
            "final_score": 0.0,
            "is_correct": False,
            "error_type": "api_failure",
        }

    # Dealbreaker override: if T2 detected a contradiction, force 0
    if dealbreaker:
        final_score = 0.0
        error_type = "factual_contradiction"
    else:
        final_score = max(t1_score, t2_score)
        # Determine error type
        if final_score >= config.FINAL_PASS_THRESHOLD:
            error_type = "correct"
        elif t1_score == 0 and t2_score == 0:
            error_type = "complete_failure"
        elif t1_score > 0 and t1_score < config.FINAL_PASS_THRESHOLD:
            error_type = "numeric_error"
        else:
            error_type = "qualitative_incomplete"

    return {
        "tier1_numeric": t1_score,
        "tier2_llm_semantic": t2_score,
        "dealbreaker_triggered": dealbreaker,
        "final_score": final_score,
        "is_correct": final_score >= config.FINAL_PASS_THRESHOLD,
        "error_type": error_type,
    }


# ======================================================================
# Evaluator class (merged Analysis + evaluate)
# ======================================================================

class Evaluator:
    """I unify scoring and analysis into one class.
    I read trajectories.jsonl, label each row, write results.csv + error_report.json,
    then provide stats and visualization."""

    def __init__(self, output_dir=OUTPUT_DIR,
                 numeric_tol=None,
                 rubric_threshold=None):
        self.output_dir = Path(output_dir)
        self.numeric_tol = numeric_tol or config.SCORING_NUMERIC_TOL
        self.rubric_threshold = rubric_threshold or config.SCORING_RUBRIC_COVERAGE
        self.results = None   # list[dict] of scored rows

    # ---- Scoring ----
    def score(self, traj_path=None) -> dict:
        """I read trajectories.jsonl, label each row, and write results.csv + error_report.json.
        Every row is wrapped in try/except: on any failure I log a warning and label
        the row 'qualitative_incomplete' so a single bad row never aborts the whole run."""
        if traj_path is None:
            traj_path = self.output_dir / "trajectories.jsonl"
        traj_path = Path(traj_path)

        rows = []
        with traj_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        scored = []
        for r in rows:
            try:
                tiered = _label_row_tiered(r)
                r["tier1_numeric"] = round(tiered["tier1_numeric"], 4)
                r["tier2_llm_semantic"] = round(tiered["tier2_llm_semantic"], 4)
                r["dealbreaker_triggered"] = tiered["dealbreaker_triggered"]
                r["final_score"] = round(tiered["final_score"], 4)
                r["is_correct"] = tiered["is_correct"]
                r["error_type"] = tiered["error_type"]
            except Exception as e:
                logger.warning(f"Scoring failed for {r.get('task_id', '?')}: {e}")
                r["error_type"] = "qualitative_incomplete"
                r["is_correct"] = False
                r["tier1_numeric"] = 0.0
                r["tier2_llm_semantic"] = 0.0
                r["dealbreaker_triggered"] = False
                r["final_score"] = 0.0
            scored.append(r)

        self.results = scored

        # Write results.csv
        out_csv = self.output_dir / "results.csv"
        fields = ["run_id", "task_id", "category", "difficulty", "gold_answer",
                  "final_answer", "is_correct", "error_type",
                  "tier1_numeric", "tier2_llm_semantic", "dealbreaker_triggered",
                  "final_score",
                  "tool_calls", "total_latency_ms", "model_name"]
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(scored)

        # Build report
        n = len(scored)
        # Separate API failures from valid tasks
        n_api_failures = sum(1 for r in scored if r.get("error_type") == "api_failure")
        n_valid = n - n_api_failures
        correct = sum(1 for r in scored if r.get("is_correct"))
        errors = Counter(r.get("error_type", "unknown") for r in scored)
        avg_latency = (sum(r.get("total_latency_ms", 0) for r in scored) / n) if n else 0
        total_cost = sum(r.get("total_cost_usd", 0) for r in scored)

        # Tier breakdown (diagnostic: continuous scores) — exclude api_failures
        valid_scored = [r for r in scored if r.get("error_type") != "api_failure"]
        vn = len(valid_scored) if valid_scored else 1
        t1_avg = sum(r.get("tier1_numeric", 0) for r in valid_scored) / vn
        t2_avg = sum(r.get("tier2_llm_semantic", 0) for r in valid_scored) / vn
        final_avg = sum(r.get("final_score", 0) for r in valid_scored) / vn
        dealbreakers = sum(1 for r in valid_scored if r.get("dealbreaker_triggered"))

        report = {
            "n_tasks": n,
            "n_valid_tasks": n_valid,
            "n_api_failures": n_api_failures,
            "accuracy": correct / n if n else 0,
            "accuracy_excl_api_failures": correct / n_valid if n_valid else 0,
            "error_distribution": dict(errors),
            "avg_latency_ms": round(avg_latency, 1),
            "total_cost_usd": round(total_cost, 4),
            "tier_breakdown": {
                "t1_numeric_avg": round(t1_avg, 4),
                "t2_llm_semantic_avg": round(t2_avg, 4),
                "final_score_avg": round(final_avg, 4),
                "dealbreakers_triggered": dealbreakers,
            },
        }

        print("\n=== Evaluation Report (v2.0 — continuous scoring) ===")
        print(f"Accuracy: {correct}/{n} = {report['accuracy']:.2%} (threshold={config.FINAL_PASS_THRESHOLD})")
        if n_api_failures > 0:
            print(f"Accuracy (excl. {n_api_failures} API failures): {correct}/{n_valid} = {report['accuracy_excl_api_failures']:.2%}")
        print(f"Avg scores: T1(numeric)={t1_avg:.3f}  T2(LLM-semantic)={t2_avg:.3f}  Final={final_avg:.3f}")
        print(f"Dealbreakers triggered: {dealbreakers}/{n}")
        print("Error distribution:", dict(errors))
        print(f"Avg latency: {report['avg_latency_ms']} ms  |  Total cost: ${report['total_cost_usd']}")
        print(f"Detailed results: {out_csv}")

        report_path = self.output_dir / "error_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    # ---- Analysis (formerly Analysis class) ----
    def analyze(self) -> dict:
        """I load results.csv (the scored file — single source of truth),
        then run basic_stats + error_distribution + plot_accuracy_by.
        matplotlib is lazily imported so basic_stats works without it."""
        if self.results is None:
            self._load_results()

        stats = self.basic_stats()
        err_dist = self.error_distribution()
        return {"basic_stats": stats, "error_distribution": err_dist}

    def _load_results(self):
        """I load results.csv into self.results (list of dicts)."""
        results_csv = self.output_dir / "results.csv"
        if not results_csv.exists():
            print("[warn] No results.csv found. Run score() first.")
            self.results = []
            return

        with results_csv.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.results = list(reader)


    def _is_correct(self, row) -> bool:
        """I normalize is_correct from bool, string, or int to a proper bool."""
        val = row.get("is_correct", False)
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        return str(val).strip().lower() in ("true", "1", "yes")

    def basic_stats(self) -> dict:
        """I compute accuracy by category and difficulty."""
        if self.results is None:
            self._load_results()

        n = len(self.results)
        n_api = sum(1 for r in self.results if r.get("error_type") == "api_failure")
        n_valid = n - n_api
        correct = sum(1 for r in self.results if self._is_correct(r))
        accuracy = correct / n if n else 0.0
        accuracy_valid = correct / n_valid if n_valid else 0.0

        print("\n=== Basic Stats ===")
        print(f"Total tasks : {n} (excl. {n_api} API failures: {n_valid} valid)")
        print(f"Correct     : {correct}")
        print(f"Accuracy    : {accuracy:.2%} (all tasks)")
        if n_api > 0:
            print(f"Accuracy    : {accuracy_valid:.2%} (excl. API failures)")

        # By category
        cats = {}
        for r in self.results:
            cat = r.get("category", "Unknown")
            cats.setdefault(cat, {"total": 0, "correct": 0})
            cats[cat]["total"] += 1
            if self._is_correct(r):
                cats[cat]["correct"] += 1

        if cats:
            print("\n--- By Category ---")
            for cat, vals in sorted(cats.items()):
                acc = vals["correct"] / vals["total"] if vals["total"] else 0
                print(f"  {cat:<30} {vals['correct']}/{vals['total']} = {acc:.2%}")

        # By difficulty
        diffs = {}
        for r in self.results:
            d = r.get("difficulty", "Unknown")
            diffs.setdefault(d, {"total": 0, "correct": 0})
            diffs[d]["total"] += 1
            if self._is_correct(r):
                diffs[d]["correct"] += 1

        if diffs:
            print("\n--- By Difficulty ---")
            for d, vals in sorted(diffs.items()):
                acc = vals["correct"] / vals["total"] if vals["total"] else 0
                print(f"  {d:<10} {vals['correct']}/{vals['total']} = {acc:.2%}")

        return {"accuracy": accuracy, "n": n, "correct": correct,
                "by_category": cats, "by_difficulty": diffs}

    def error_distribution(self) -> dict:
        """I compute error type distribution."""
        if self.results is None:
            self._load_results()

        errors = Counter(r.get("error_type", "unknown") for r in self.results
                         if not self._is_correct(r))

        print("\n=== Error Distribution ===")
        total_wrong = sum(errors.values())
        total = len(self.results)
        print(f"Total errors: {total_wrong} / {total}")
        for etype, count in errors.most_common():
            print(f"  {etype:<25} {count}")
        return dict(errors)

    def plot_accuracy_by(self, column, save=True, show=False):
        """I plot accuracy grouped by a column. matplotlib is lazily imported."""
        if self.results is None:
            self._load_results()

        groups = {}
        for r in self.results:
            val = r.get(column, "Unknown")
            groups.setdefault(val, {"total": 0, "correct": 0})
            groups[val]["total"] += 1
            if self._is_correct(r):
                groups[val]["correct"] += 1

        if not groups:
            print(f"[skip] No data for column {column}")
            return

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = sorted(groups.keys())
        accs = [groups[l]["correct"] / groups[l]["total"] if groups[l]["total"] else 0
                for l in labels]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(labels)), accs)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(f"Accuracy by {column}")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        plt.tight_layout()

        if save:
            path = self.output_dir / f"accuracy_by_{column}.png"
            fig.savefig(path)
            print(f"saved: {path}")
        if show:
            plt.show()
        else:
            plt.close(fig)


# ======================================================================
# Backward-compatible module-level entry point
# ======================================================================

def evaluate(traj_path=None) -> dict:
    """I keep the old scorer.evaluate() CLI contract. I create an Evaluator and run score()."""
    if traj_path is None:
        traj_path = OUTPUT_DIR / "trajectories.jsonl"
    return Evaluator().score(traj_path)


if __name__ == "__main__":
    ev = Evaluator()
    ev.score()
    ev.analyze()
