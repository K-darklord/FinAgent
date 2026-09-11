"""
scorer.py
=========
读 runner 产出的 trajectories.jsonl -> 评分 + 错误归因 (error taxonomy)。

对齐 Finance Agent Benchmark 的 LLM-as-Judge + rubric 合取(AND)思路，
这里先用 **确定性规则打分**（可复现、无需额外 API），
后续可平滑替换为 LLM judge（保留同一接口）。

错误 taxonomy（你论文的核心——后续聚类/消融的对象）:
  - retrieval_failure      : 没抓到 / 抓错源
  - numeric_error          : 算错数、单位/四舍五入错
  - citation_missing       : 缺来源引用
  - tool_error             : 工具调用本身报错
  - qualitative_incomplete : 定性题要点缺失
  - correct                : 通过
"""
from __future__ import annotations
import csv
import json
import re
from pathlib import Path
from collections import Counter

from runner import OUTPUT_DIR


# ------------------------- 确定性评分器（规则版） -------------------------
def score_numeric(gold: str, pred: str, tol: float = 0.05) -> bool:
    """数值题：解析数字，允许相对容差。"""
    try:
        g = float(re.findall(r"-?\d+\.?\d*", str(gold))[0])
        p = float(re.findall(r"-?\d+\.?\d*", str(pred))[0])
    except (IndexError, ValueError):
        return False
    if g == 0:
        return abs(p) < tol
    return abs(g - p) / abs(g) <= tol


def score_exact(gold: str, pred: str) -> bool:
    return str(gold).strip().lower() in str(pred).strip().lower()


def classify_error(task: dict) -> str:
    """根据 trajectory + 答案，给一条错误标签（taxonomy 起点，后续可扩）。"""
    traj = task.get("trajectory", [])
    tool_steps = [s for s in traj if s.get("tool_name")]
    # 1) 工具本身报错
    if any("tool_error" in s.get("tool_output", "") for s in tool_steps):
        return "tool_error"
    # 2) 完全没调用工具（定量/定性题应检索）
    if not tool_steps and task["category"] in (
        "Quantitative Retrieval", "Qualitative Retrieval", "Numerical Reasoning"):
        return "retrieval_failure"
    # 3) 数值题 -> 数值比对
    if task["category"] == "Numerical Reasoning":
        return "correct" if score_numeric(task["gold_answer"], task["final_answer"]) else "numeric_error"
    # 4) 定性题 -> 简单要点覆盖
    if task["category"] == "Qualitative Retrieval":
        keywords = ["productivity", "intelligent cloud", "more personal computing"]
        hit = sum(1 for k in keywords if k.lower() in task["final_answer"].lower())
        if hit >= 2:
            return "correct"
        return "qualitative_incomplete"
    # 5) 定量检索
    if task["category"] == "Quantitative Retrieval":
        return "correct" if score_numeric(task["gold_answer"], task["final_answer"], tol=0.5) else "numeric_error"
    return "correct"


def evaluate(traj_path: Path = OUTPUT_DIR / "trajectories.jsonl") -> dict:
    rows = []
    with traj_path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    for r in rows:
        r["is_correct"] = score_numeric(r["gold_answer"], r["final_answer"], tol=0.5) \
            if r["category"] == "Numerical Reasoning" else \
            score_exact(r["gold_answer"], r["final_answer"]) or \
            score_numeric(r["gold_answer"], r["final_answer"], tol=0.5)
        r["error_type"] = classify_error(r)

    # 输出 results.csv
    out = OUTPUT_DIR / "results.csv"
    fields = ["run_id", "task_id", "category", "difficulty", "gold_answer",
              "final_answer", "is_correct", "error_type", "tool_calls", "total_latency_ms"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # 汇总统计
    n = len(rows)
    correct = sum(1 for r in rows if r["error_type"] == "correct")
    errors = Counter(r["error_type"] for r in rows)
    avg_latency = sum(r.get("total_latency_ms", 0) for r in rows) / n if n else 0
    total_cost = sum(r.get("total_cost_usd", 0) for r in rows)

    report = {
        "n_tasks": n,
        "accuracy": correct / n if n else 0,
        "error_distribution": dict(errors),
        "avg_latency_ms": round(avg_latency, 1),
        "total_cost_usd": round(total_cost, 4),
    }
    print("\n=== Evaluation Report ===")
    print(f"Accuracy: {correct}/{n} = {report['accuracy']:.2%}")
    print("Error distribution:", dict(errors))
    print(f"Avg latency: {report['avg_latency_ms']} ms  |  Total cost: ${report['total_cost_usd']}")
    print(f"Detailed results: {out}")

    (OUTPUT_DIR / "error_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    evaluate()
