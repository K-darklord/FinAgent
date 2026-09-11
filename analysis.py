# analysis.py
# 用法：
#   conda activate finagent
#   python analysis.py
#
# 说明：
# - 不要求 run_summary.csv 一定有 correct 列。若没有，会用 gold_answer / final_answer 比对生成。
# - 答案比对默认是“规范化字符串相等”，可自定义 is_correct_fn。
# - matplotlib 采用惰性 import：只有真正画图时才 import，没装也不影响基本统计。

import json
from pathlib import Path
from collections import Counter, defaultdict


class Analysis:
    def __init__(self, output_dir="output", answer_cols=("gold_answer", "final_answer")):
        self.output_dir = Path(output_dir)
        self.answer_cols = answer_cols
        self.summary = None
        self.trajectories = None

    # ---------- 加载 ----------
    def load(self):
        summary_path = self.output_dir / "run_summary.csv"
        traj_path = self.output_dir / "trajectories.jsonl"

        import pandas as pd
        self.summary = pd.read_csv(summary_path)

        trajs = []
        with open(traj_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    trajs.append(json.loads(line))
        self.trajectories = pd.DataFrame(trajs)

        print(f"Loaded {len(self.summary)} runs, {len(self.trajectories)} trajectories")
        print("Summary columns:", self.summary.columns.tolist())
        return self

    # ---------- 答案比对（核心：不强制要求 correct 列）----------
    def ensure_correct(self, is_correct_fn=None):
        """确保 summary 里有 correct 列（bool）。

        is_correct_fn(row) -> bool ：自定义判分逻辑。
        若为 None，则用 gold_answer / final_answer 做规范化相等比对。
        """
        if self.summary is None:
            self.load()

        if "correct" in self.summary.columns:
            # 统一转成 bool
            self.summary["correct"] = self.summary["correct"].astype(bool)
            return self

        gold_col, pred_col = self.answer_cols
        if gold_col not in self.summary.columns or pred_col not in self.summary.columns:
            # 找不到答案列，无法判分，全部标为 False 并提示
            print(f"[warn] 缺少答案列 ({gold_col}, {pred_col})，无法计算 correct，默认全 False")
            self.summary["correct"] = False
            return self

        if is_correct_fn is not None:
            self.summary["correct"] = self.summary.apply(is_correct_fn, axis=1)
        else:
            self.summary["correct"] = self.summary.apply(
                lambda r: _normalize(r[gold_col]) == _normalize(r[pred_col]), axis=1
            )
        return self

    # ---------- 基本统计 ----------
    def basic_stats(self, is_correct_fn=None):
        self.ensure_correct(is_correct_fn)

        total = len(self.summary)
        correct = int(self.summary["correct"].sum())
        accuracy = correct / total if total > 0 else 0.0

        print("\n=== Basic Stats ===")
        print(f"Total tasks : {total}")
        print(f"Correct     : {correct}")
        print(f"Accuracy    : {accuracy:.2%}")

        # 按 category
        if "category" in self.summary.columns:
            print("\n--- By Category ---")
            g = self.summary.groupby("category")["correct"]
            cat = (g.sum() / g.count()).rename("accuracy")
            cat_df = cat.reset_index()
            cat_df["count"] = g.count().values
            cat_df["correct"] = g.sum().values
            print(cat_df.to_string(index=False))

        # 按 difficulty
        if "difficulty" in self.summary.columns:
            print("\n--- By Difficulty ---")
            g = self.summary.groupby("difficulty")["correct"]
            diff = ((g.sum() / g.count()).rename("accuracy")).reset_index()
            diff["count"] = g.count().values
            print(diff.to_string(index=False))

        return accuracy

    # ---------- 错误分布（为 Mistake Pool 做准备）----------
    def error_distribution(self):
        self.ensure_correct()

        errors = self.summary[self.summary["correct"] == False]

        # error_type 字段如果还没有，就先给一个占位
        if "error_type" in errors.columns:
            col = "error_type"
        else:
            col = None

        print("\n=== Error Distribution ===")
        print(f"Total errors: {len(errors)} / {len(self.summary)}")

        if col is not None:
            dist = errors[col].fillna("unknown").value_counts()
            print(dist.to_string())
            return dist.to_dict()
        else:
            # 暂无 error_type：用 task_id 列出每条错误，方便人工标注
            ids = errors["task_id"].tolist() if "task_id" in errors.columns else list(range(len(errors)))
            print("（trajectories 尚无 error_type 字段，先列出错误样本 task_id）")
            print(ids)
            return {"unknown": len(errors)}

    # ---------- 画图（可选，惰性 import matplotlib）----------
    def plot_accuracy_by(self, column, save=True, show=False):
        if self.summary is None or "correct" not in self.summary.columns:
            self.basic_stats()

        if column not in self.summary.columns:
            print(f"[skip] summary 中没有 {column} 列，无法绘图")
            return

        import matplotlib.pyplot as plt
        g = self.summary.groupby(column)["correct"]
        acc = g.sum() / g.count()

        fig, ax = plt.subplots()
        acc.plot(kind="bar", ax=ax)
        ax.set_title(f"Accuracy by {column}")
        ax.set_ylabel("Accuracy")
        plt.tight_layout()

        if save:
            path = self.output_dir / f"accuracy_by_{column}.png"
            fig.savefig(path)
            print(f"saved: {path}")
        if show:
            plt.show()
        else:
            plt.close(fig)


# ---------- 工具函数 ----------
def _normalize(x):
    """把答案转成可比对的字符串：去前后空白、小写、去掉常见格式噪声。"""
    if x is None:
        return ""
    s = str(x).strip().lower()
    # 去掉多余空格 / 千分位逗号差异
    s = s.replace(",", "").replace(" ", "")
    return s


if __name__ == "__main__":
    analysis = Analysis(output_dir="output")
    analysis.load()
    analysis.basic_stats()          # 自动用 gold_answer / final_answer 比对
    analysis.error_distribution()
    analysis.plot_accuracy_by("difficulty", save=True)   # 需要时取消注释
