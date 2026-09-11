"""
benchmark.py
=============
真实金融评测集的 schema 定义 + 加载器。

设计原则：数据结构 **完全对齐 Finance Agent Benchmark (arXiv:2508.00828)**，
每条样本含：question / gold_answer / reasoning_steps / rubric / evidence。
这样日后你把 TASKS 换成官方 537 题 (HuggingFace/CSV) 时，下游 runner/scorer 零改动。

本文件自带一个 "live mini benchmark"：从 **公开、无需 key** 的数据源
（公司官网 IR / SEC 公开数字 / 维基百科 infobox）抓取真实财务数字，
确保 pipeline 跑的是「真调用」，而非 mock。
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass
class Task:
    task_id: str
    category: str            # 对齐 FAB 九大类: Quantitative Retrieval / Numerical Reasoning ...
    difficulty: str          # Easy / Medium / Hard
    prompt: str              # 给 agent 的题目
    gold_answer: str         # 标准答案
    reasoning_steps: list[str] = field(default_factory=list)  # 专家分步解法
    rubric: list[str] = field(default_factory=list)           # 判分检查项
    evidence: list[str] = field(default_factory=list)        # 证据来源 URL/段落
    metadata: dict = field(default_factory=dict)             # ticker, filing_type 等

    def to_dict(self):
        return asdict(self)


# ======================================================================
# Live mini benchmark：真实可抓取、答案确定的金融问题（无需 API key）
# 注：这些数字来自各公司公开披露的年度报告，抓取后自动与 gold_answer 比对
# ======================================================================
TASKS: list[Task] = [
    Task(
        task_id="live_001",
        category="Quantitative Retrieval",
        difficulty="Easy",
        prompt=(
            "What was NVIDIA's total revenue (in USD millions) for fiscal year 2024? "
            "Use the tool `fetch_url` to retrieve the official annual report, then answer."
        ),
        gold_answer="60922",   # NVDA FY2024 10-K: Revenue = $60,922 million
        reasoning_steps=[
            "Fetch NVIDIA FY2024 (annual report / 10-K) from investor site.",
            "Locate 'Consolidated Statements of Operations' (Income Statement).",
            "Read 'Revenue' line for the fiscal year ended Jan 28, 2024.",
        ],
        rubric=[
            "answer_is_numeric_and_close_to_60922_million",
            "cites_official_source_url",
            "uses_correct_fiscal_year_ending_Jan_2024",
        ],
        evidence=["https://investor.nvidia.com/financial-information/annual-reports/default.aspx"],
        metadata={"ticker": "NVDA", "filing_type": "10-K", "fiscal_year": 2024},
    ),
    Task(
        task_id="live_002",
        category="Numerical Reasoning",
        difficulty="Medium",
        prompt=(
            "Apple Inc. FY2023 net sales were $383,285 million and FY2024 net sales were "
            "$391,035 million. Compute the year-over-year revenue growth rate in percent "
            "(round to two decimals). Use `fetch_url` if you need source confirmation."
        ),
        gold_answer="2.02",  # (391035-383285)/383285 * 100 = 2.0197...
        reasoning_steps=[
            "Identify FY2023 net sales = 383,285 and FY2024 = 391,035 (USD millions).",
            "Compute (391035 - 383285) / 383285.",
            "Multiply by 100 and round to two decimals -> 2.02%.",
        ],
        rubric=[
            "formula_is_correct",
            "result_rounded_to_two_decimals",
            "answer_within_0.05_of_2.02",
        ],
        evidence=["https://www.apple.com/investor-relations/"],
        metadata={"ticker": "AAPL", "filing_type": "10-K", "metric": "revenue_growth_pct"},
    ),
    Task(
        task_id="live_003",
        category="Qualitative Retrieval",
        difficulty="Easy",
        prompt=(
            "Briefly describe the primary business segments of Microsoft as disclosed in "
            "its latest annual report. Use `fetch_url` to retrieve the source."
        ),
        gold_answer="Microsoft operates through three segments: Productivity and Business Processes, "
                    "Intelligent Cloud, and More Personal Computing.",
        reasoning_steps=[
            "Fetch Microsoft latest 10-K / annual report.",
            "Locate 'Business' section describing operating segments.",
            "Summarize the three named segments.",
        ],
        rubric=[
            "lists_all_three_segments",
            "segment_names_accurate",
            "cites_official_source",
        ],
        evidence=["https://www.microsoft.com/investor/reports-and-filings"],
        metadata={"ticker": "MSFT", "filing_type": "10-K", "aspect": "business_segments"},
    ),
]


def load_tasks(path: str | Path | None = None) -> list[Task]:
    """加载 benchmark。若传 path（CSV/JSONL），从文件读；否则用内置 live mini set。"""
    if path is None:
        return TASKS
    p = Path(path)
    tasks: list[Task] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            tasks.append(Task(**d))
    return tasks


if __name__ == "__main__":
    for t in load_tasks():
        print(f"[{t.task_id}] {t.category:<26} {t.difficulty:<6} {t.prompt[:60]}...")
