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

_judge_client = None

def _get_judge_client():
    """I lazily create an OpenAI client for LLM-as-Judge via HF router."""
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

def _llm_judge_row(row: dict, judge_model: str = "deepseek-ai/DeepSeek-V4-Flash") -> str:
    """I use an LLM to judge whether the answer satisfies each rubric criterion.
    For each 'correctness' criterion I ask the judge: does the answer satisfy this?
    For each 'contradiction' criterion I ask: does the answer contradict this?
    I return one of the existing 6 labels."""
    client = _get_judge_client()
    if client is None:
        # No token -> fall back to tier 2
        return _rubric_coverage(row)

    metadata = row.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            metadata = {}

    structured = metadata.get("rubric_structured", [])
    if not structured:
        return classify_error(row)

    pred = str(row.get("final_answer", "")).strip()
    if not pred:
        return "qualitative_incomplete"

    question = str(row.get("prompt", ""))
    correctness = [c for c in structured if c.get("operator") == "correctness"]
    contradictions = [c for c in structured if c.get("operator") == "contradiction"]

    if not correctness:
        return classify_error(row)

    hit = 0
    for c in correctness:
        crit = c.get("criteria", "").strip()
        if not crit:
            continue
        try:
            prompt = (
                f"You are an expert financial evaluator. "
                f"Judge whether the given answer satisfies the criterion. "
                f"Reply ONLY with YES or NO.\n\n"
                f"Question: {question}\n\n"
                f"Answer: {pred}\n\n"
                f"Criterion: {crit}\n\n"
                f"Does the answer satisfy this criterion? YES or NO:"
            )
            resp = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4,
            )
            verdict = (resp.choices[0].message.content or "").strip().upper()
            if verdict.startswith("YES"):
                hit += 1
        except Exception as e:
            logger.warning(f"LLM judge failed for criterion '{crit[:30]}': {e}")

    # Check contradictions (if any contradiction is present, answer is wrong)
    for c in contradictions:
        crit = c.get("criteria", "").strip()
        if not crit:
            continue
        try:
            prompt = (
                f"You are an expert financial evaluator. "
                f"Does the answer contradict the given statement? "
                f"Reply ONLY with YES or NO.\n\n"
                f"Answer: {pred}\n\n"
                f"Statement: {crit}\n\n"
                f"Does the answer contradict this? YES or NO:"
            )
            resp = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4,
            )
            verdict = (resp.choices[0].message.content or "").strip().upper()
            if verdict.startswith("YES"):
                return "numeric_error" if any(ch.isdigit() for ch in pred) else "qualitative_incomplete"
        except Exception as e:
            logger.warning(f"LLM judge contradiction check failed: {e}")

    cov = hit / len(correctness) if correctness else 0
    if cov >= config.SCORING_RUBRIC_COVERAGE:
        return "correct"

    gold = str(row.get("gold_answer", ""))
    if any(ch.isdigit() for ch in gold):
        return "numeric_error"
    return "qualitative_incomplete"

def _rubric_coverage_normalized(row: dict) -> bool:
    """I check rubric criteria using normalized comparison instead of raw substring.
    This handles case, punctuation, and number format differences."""
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
        # Check normalized substring
        if crit in pred:
            hit += 1
        else:
            # Try numeric match within the criterion
            crit_nums = _extract_all_numbers(c.get("criteria", ""))
            pred_nums = _extract_all_numbers(str(row.get("final_answer", "")))
            if crit_nums and pred_nums:
                for cn in crit_nums:
                    for pn in pred_nums:
                        if cn == 0:
                            if pn == 0:
                                hit += 1
                                break
                        elif abs(pn - cn) / max(abs(cn), 1e-10) < config.SCORING_NUMERIC_TOLERANCE:
                            hit += 1
                            break
                    else:
                        continue
                    break

    cov = hit / len(correctness) if correctness else 0
    return cov >= config.SCORING_RUBRIC_COVERAGE


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


def _normalize_answer(s: str) -> str:
    """I normalize an answer string for robust comparison:
    - lowercase, strip whitespace
    - remove commas, dollar signs, percent signs, parentheses
    - remove common units (million, billion, etc.)
    - collapse multiple spaces
    """
    s = str(s).strip().lower()
    for ch in ["$", "€", "£", "¥", ",", "(", ")", "%"]:
        s = s.replace(ch, " ")
    for unit in ["million", "billion", "trillion", "thousand", "mn", "bn", "mm"]:
        s = s.replace(unit, " ")
    s = " ".join(s.split())
    return s


def _extract_all_numbers(s: str) -> list:
    """I extract all numbers from a string, handling commas and units."""
    import re
    s = str(s).replace(",", "")
    matches = re.findall(r"-?\d+\.?\d*", s)
    nums = []
    for m in matches:
        try:
            nums.append(float(m))
        except ValueError:
            pass
    return nums


def _score_tier1_exact(row: dict) -> bool:
    """Tier 1: I check if gold_answer and final_answer match after normalization.
    For numeric answers: compare values within tight tolerance (0.1%).
    For text answers: check if normalized gold is substring of normalized pred.
    This is stricter than T2/T3 but more robust than raw substring match."""
    gold_raw = str(row.get("gold_answer", "")).strip()
    pred_raw = str(row.get("final_answer", "")).strip()
    if not gold_raw or not pred_raw:
        return False

    gold_norm = _normalize_answer(gold_raw)
    pred_norm = _normalize_answer(pred_raw)

    # Try numeric comparison first
    gold_nums = _extract_all_numbers(gold_raw)
    pred_nums = _extract_all_numbers(pred_raw)

    if gold_nums and pred_nums:
        for gn in gold_nums:
            for pn in pred_nums:
                if gn == 0:
                    if pn == 0:
                        return True
                elif abs(pn - gn) / max(abs(gn), 1e-10) < 0.001:
                    return True

    # Fall back to normalized substring match
    return gold_norm in pred_norm


def _score_tier2_numeric(row: dict) -> bool:
    """Tier 2: I extract numbers from gold and pred, check if they match within tolerance.
    For FAB: use rubric substring matching with normalized comparison.
    For non-FAB: use numeric tolerance (config.SCORING_NUMERIC_TOLERANCE)."""
    if _is_fab_row(row):
        # For FAB: use rubric coverage with normalized substring match
        return _rubric_coverage_normalized(row)
    # For mini benchmark: use numeric tolerance
    return classify_error(row) == "correct"


def _score_tier3_llm_judge(row: dict) -> bool:
    """Tier 3: I use an LLM judge to evaluate semantic correctness.
    This is the most lenient -- it catches semantic matches that exact/numeric miss."""
    if _is_fab_row(row):
        return _llm_judge_row(row) == "correct"
    # For non-FAB: fall back to tier 2
    return _score_tier2_numeric(row)


def _label_row_tiered(row: dict) -> dict:
    """I run all 3 tiers and return a dict with per-tier results + final error_type.
    The final is_correct = any tier passed. The tier breakdown is diagnostic:
      - tier1 pass + tier2 pass + tier3 pass -> perfect
      - tier1 fail + tier2 fail + tier3 pass  -> format compliance issue (semantic match)
      - tier1 fail + tier2 fail + tier3 fail -> genuine knowledge gap
    """
    t1 = _score_tier1_exact(row)
    t2 = _score_tier2_numeric(row)
    t3 = _score_tier3_llm_judge(row)

    # Determine final error_type (keep existing 6 labels, no new ones)
    if t1 or t2 or t3:
        error_type = "correct"
    elif _is_fab_row(row):
        gold = str(row.get("gold_answer", ""))
        error_type = "numeric_error" if any(ch.isdigit() for ch in gold) else "qualitative_incomplete"
    else:
        error_type = classify_error(row)

    return {
        "tier1_exact": t1,
        "tier2_numeric": t2,
        "tier3_llm_judge": t3,
        "is_correct": t1 or t2 or t3,
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
                r["tier1_exact"] = tiered["tier1_exact"]
                r["tier2_numeric"] = tiered["tier2_numeric"]
                r["tier3_llm_judge"] = tiered["tier3_llm_judge"]
                r["is_correct"] = tiered["is_correct"]
                r["error_type"] = tiered["error_type"]
            except Exception as e:
                logger.warning(f"Scoring failed for {r.get('task_id', '?')}: {e}")
                r["error_type"] = "qualitative_incomplete"
                r["is_correct"] = False
                r["tier1_exact"] = False
                r["tier2_numeric"] = False
                r["tier3_llm_judge"] = False
            scored.append(r)

        self.results = scored

        # Write results.csv
        out_csv = self.output_dir / "results.csv"
        fields = ["run_id", "task_id", "category", "difficulty", "gold_answer",
                  "final_answer", "is_correct", "error_type",
                  "tier1_exact", "tier2_numeric", "tier3_llm_judge",
                  "tool_calls", "total_latency_ms", "model_name"]
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(scored)

        # Build report
        n = len(scored)
        correct = sum(1 for r in scored if r.get("is_correct"))
        errors = Counter(r.get("error_type", "unknown") for r in scored)
        avg_latency = (sum(r.get("total_latency_ms", 0) for r in scored) / n) if n else 0
        total_cost = sum(r.get("total_cost_usd", 0) for r in scored)

        # Tier breakdown (diagnostic: how many passed each tier)
        t1_pass = sum(1 for r in scored if r.get("tier1_exact"))
        t2_pass = sum(1 for r in scored if r.get("tier2_numeric"))
        t3_pass = sum(1 for r in scored if r.get("tier3_llm_judge"))

        report = {
            "n_tasks": n,
            "accuracy": correct / n if n else 0,
            "error_distribution": dict(errors),
            "avg_latency_ms": round(avg_latency, 1),
            "total_cost_usd": round(total_cost, 4),
            "tier_breakdown": {
                "tier1_exact": t1_pass,
                "tier2_numeric": t2_pass,
                "tier3_llm_judge": t3_pass,
            },
        }

        print("\n=== Evaluation Report ===")
        print(f"Accuracy: {correct}/{n} = {report['accuracy']:.2%}")
        print(f"Tier breakdown: T1(exact)={t1_pass}  T2(numeric/rubric)={t2_pass}  T3(LLM-judge)={t3_pass}")
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
        correct = sum(1 for r in self.results if self._is_correct(r))
        accuracy = correct / n if n else 0.0

        print("\n=== Basic Stats ===")
        print(f"Total tasks : {n}")
        print(f"Correct     : {correct}")
        print(f"Accuracy    : {accuracy:.2%}")

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
