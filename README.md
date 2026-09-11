# Finance Agent Evaluation Pipeline

最小可运行骨架：**真实 benchmark 数据 + 真实工具调用 (fetch_url) + 完整 trajectory 记录 + 评分/错误归因**。

> 设计目标：把整条数据线跑通。无需训练、无需本地模型、无需 API key（默认用本地 rule-based agent，仍会真实抓取财报 URL）。

## 目录结构

```
finance_agent_eval/
├── benchmark.py   # 真实 benchmark schema (对齐 Finance Agent Benchmark 537题格式)
├── agent.py       # 真实 agent：工具循环 + trajectory 记录 (含 OpenAI/本地vLLM 骨架)
├── runner.py      # 主循环：跑全量 -> trajectories.jsonl + run_summary.csv
├── scorer.py      # 评分 + 错误 taxonomy 归因 -> results.csv + error_report.json
├── config.py      # API key 等配置（真实 LLM 时用）
├── output/        # 产物（自动生成）
└── README.md
```

## 快速开始

```bash
cd finance_agent_eval
pip install -r requirements.txt   # 仅需 requests 等标准库基本就够
python runner.py                  # 跑 benchmark
python scorer.py                  # 评分 + 错误归因
```

产物：
- `output/trajectories.jsonl` — 每条完整轨迹（step / thought / tool / observation / latency）
- `output/run_summary.csv` — 每题聚合指标
- `output/results.csv` — 评分明细
- `output/error_report.json` — 错误分布 / accuracy / latency / cost

## 切到真实 LLM（可选）

`agent.py` 已内置 `OpenAIAgent` 骨架：

```python
from agent import OpenAIAgent
from runner import run_evaluation
run_evaluation(OpenAIAgent(model="gpt-4o-mini"))
```

填 `config.py` 的 API key 即可。本地 vLLM（OpenAI 兼容接口）同理：改 `base_url`。

## 对接正式 Finance Agent Benchmark (537 题)

数据结构已对齐 FAB schema（question / gold_answer / reasoning_steps / rubric / evidence）。
把 `benchmark.py` 的 `TASKS` 换成从官方 CSV/JSONL 加载即可，下游 runner/scorer **零改动**：

```python
tasks = load_tasks("path/to/fab_questions.jsonl")
```
