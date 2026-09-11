"""
dump_project.py
把 FinAgent 项目里的关键源文件合并导出成一个 project_dump.txt，
方便整体丢给 AI（Claude Code / ChatGPT 等）理解上下文。

用法（在项目根目录下执行）：
    python dump_project.py                # 默认导出到 project_dump.txt
    python dump_project.py -o out.txt     # 指定输出文件名
    python dump_project.py --tree         # 同时在开头打印目录树
"""

import argparse
import os
from datetime import datetime

# 需要排除的目录（不会递归进入）
EXCLUDE_DIRS = {
    "__pycache__", ".git", ".idea", ".vscode",
    "venv", "env", ".venv", ".env",
    "node_modules", ".pytest_cache", ".mypy_cache",
    ".DS_Store",
}

# 需要导出的文件扩展名（小写）
INCLUDE_EXTS = {
    ".py", ".pyw",
    ".csv", ".json", ".jsonl",
    ".md", ".txt", ".rst",
    ".yaml", ".yml",
    ".cfg", ".toml", ".ini",
    ".ipynb",
    ".sh", ".zsh",
    ".env.example",  # 只导示例，不导真实 .env
}

# 显式包含的文件名（即使没有扩展名也算）
INCLUDE_FILENAMES = {
    "Dockerfile", "Makefile", "LICENSE", "requirements.txt",
    "environment.yml", ".env.example",
}


def should_include(filename: str) -> bool:
    """判断是否应该导出该文件"""
    if filename in INCLUDE_FILENAMES:
        return True
    ext = os.path.splitext(filename)[1].lower()
    if ext in INCLUDE_EXTS:
        return True
    # 排除真实 .env（含密钥），只保留 .env.example
    if filename == ".env":
        return False
    return False


def collect_files(root_dir: str):
    """遍历目录，收集所有需要导出的文件路径"""
    collected = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 原地修改 dirnames，跳过排除目录
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fname in sorted(filenames):
            if should_include(fname):
                full = os.path.join(dirpath, fname)
                collected.append(full)
    return sorted(collected)


def write_dump(files, output_path: str, root_dir: str, with_tree: bool):
    with open(output_path, "w", encoding="utf-8") as out:
        out.write("# Project Dump - FinAgent\n")
        out.write(f"# Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        out.write(f"# Root: {os.path.abspath(root_dir)}\n")
        out.write(f"# Total files: {len(files)}\n\n")

        if with_tree:
            out.write("# Directory Tree\n")
            out.write("# (use: find . -type f | sort)\n\n")

        for path in files:
            rel = os.path.relpath(path, root_dir)
            out.write(f"\n{'=' * 72}\n")
            out.write(f"===== {rel} =====\n")
            out.write(f"{'=' * 72}\n\n")
            try:
                with open(path, "r", encoding="utf-8") as fp:
                    out.write(fp.read())
            except UnicodeDecodeError:
                # 二进制/编码异常的文件跳过内容
                out.write("(binary or non-utf-8 file, skipped)\n")
            except Exception as e:
                out.write(f"(error reading: {e})\n")
            if not out.tell() or True:
                out.write("\n")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Dump FinAgent project into a single text file")
    parser.add_argument("-o", "--output", default="project_dump.txt",
                        help="输出文件名 (默认: project_dump.txt)")
    parser.add_argument("-r", "--root", default=".",
                        help="项目根目录 (默认: 当前目录)")
    parser.add_argument("--tree", action="store_true",
                        help="在开头打印目录树")
    args = parser.parse_args()

    files = collect_files(args.root)
    out_path = write_dump(files, args.output, args.root, args.tree)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"✅ 已生成: {out_path}")
    print(f"   文件数: {len(files)}")
    print(f"   大小:   {size_kb:.1f} KB")
    print(f"\n接下来：用 PyCharm 打开该文件 → 全选复制 → 粘贴给 AI 即可。")


if __name__ == "__main__":
    main()
