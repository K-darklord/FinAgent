"""
runner.py
=========
主循环：遍历 benchmark -> 调用 agent -> 记录 trajectory -> 落盘。

输出（数据线核心产物）:
  output/trajectories.jsonl   每行一条完整 trajectory（含每步 tool 调用）
  output/run_summary.csv       每行一个 task 的聚合指标（用于 scorer / 画图）

设计要点：runner 只负责「跑 + 记录」，不做评分（评分交给 evaluator.py）。
"""
from __future__ import annotations
import csv
import json
import os
from datetime import datetime
from pathlib import Path

from benchmark import load_tasks, load_fab_questions
from agent import BaseAgent, RuleBasedFinanceAgent


OUTPUT_DIR = Path(__file__).parent / "output"


def run_evaluation(
    agent: BaseAgent,
    tasks=None,
    output_dir: Path = OUTPUT_DIR,
    run_id: str | None = None,
) -> dict:
    tasks = tasks or load_tasks()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    traj_path = output_dir / "trajectories.jsonl"
    sum_path = output_dir / "run_summary.csv"

    rows = []
    with traj_path.open("w", encoding="utf-8") as ft:
        for task in tasks:
            result = agent.solve(task)
            record = {
                "run_id": run_id,
                "task_id": task.task_id,
                "category": task.category,
                "difficulty": task.difficulty,
                "prompt": task.prompt,
                "gold_answer": task.gold_answer,
                "final_answer": result.final_answer,
                "tool_calls": result.tool_calls,
                "total_latency_ms": result.total_latency_ms,
                "total_cost_usd": result.total_cost_usd,
                "model_name": result.model_name,
                "trajectory": result.trajectory,   # 完整轨迹：归因分析的原料
                "metadata": task.metadata,
                "api_failure": result.api_failure,
            }
            ft.write(json.dumps(record, ensure_ascii=False) + "\n")

            rows.append({
                "run_id": run_id,
                "task_id": task.task_id,
                "category": task.category,
                "difficulty": task.difficulty,
                "gold_answer": task.gold_answer,
                "final_answer": result.final_answer,
                "tool_calls": result.tool_calls,
                "latency_ms": result.total_latency_ms,
                "cost_usd": result.total_cost_usd,
                "model": result.model_name,
                "metadata": json.dumps(task.metadata, ensure_ascii=False),
            })

    # 汇总 CSV
    with sum_path.open("w", newline="", encoding="utf-8") as fs:
        writer = csv.DictWriter(fs, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Run {run_id} done. {len(rows)} tasks.")
    print(f"   trajectories: {traj_path}")
    print(f"   summary:      {sum_path}")
    return {"run_id": run_id, "n": len(rows), "traj_path": str(traj_path)}


if __name__ == "__main__":
    # I support env-driven agent and benchmark selection:
    #   FINAGENT_AGENT=openai python runner.py   (use OpenAI API upper bound)
    #   FINAGENT_AGENT=fingpt python runner.py   (use FinGPT baseline)
    #   FINAGENT_BENCH=fab python runner.py      (use FAB public dataset)
    agent_name = os.getenv("FINAGENT_AGENT", "rule").lower()
    bench_name = os.getenv("FINAGENT_BENCH", "mini").lower()

    if agent_name == "fingpt":
        from agent import FinGPTAgent
        agent = FinGPTAgent()
    elif agent_name == "openai":
        from agent import OpenAIAgent
        agent = OpenAIAgent()
    elif agent_name == "hf":
        from agent import HuggingFaceAgent
        agent = HuggingFaceAgent()
    else:
        from agent import RuleBasedFinanceAgent
        agent = RuleBasedFinanceAgent()

    if bench_name == "fab":
        tasks = load_fab_questions()
    else:
        tasks = load_tasks()

    # Allow limiting number of tasks for quick testing
    num_tasks = os.environ.get("FINAGENT_NUM_TASKS")
    if num_tasks:
        tasks = tasks[:int(num_tasks)]

    print(f"[runner] agent={agent_name} bench={bench_name} tasks={len(tasks)}")
    run_evaluation(agent, tasks=tasks)
