# config.py — API keys / model config (真实 LLM 时用)
# 推荐用环境变量，不要硬编码到代码里
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# 本地 vLLM / OpenAI 兼容服务
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# 评测全局配置
MAX_TOOL_CALLS_PER_TASK = 10
# Alias for agent.py
MAX_TOOL_CALLS = MAX_TOOL_CALLS_PER_TASK
REQUEST_TIMEOUT = 15



# ======================================================================
# FinGPT baseline agent config
# ======================================================================
FINGPT_BASE_MODEL = os.getenv("FINGPT_BASE_MODEL", "tiiuae/falcon-7b")
FINGPT_PEFT_MODEL = os.getenv("FINGPT_PEFT_MODEL", "FinGPT/fingpt-mt_falcon-7b_lora")
FINGPT_USE_8BIT = os.getenv("FINGPT_USE_8BIT", "1") == "1"   # bitsandbytes 8-bit quantization
FINGPT_DEVICE = os.getenv("FINGPT_DEVICE", "cuda")           # cpu / cuda / mps
FINGPT_MAX_NEW_TOKENS = int(os.getenv("FINGPT_MAX_NEW_TOKENS", "128"))

# ======================================================================
# HuggingFace Inference API agent (baseline via remote inference)
# ======================================================================
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_MODEL = os.getenv("HF_MODEL", "deepseek-ai/DeepSeek-V4-Flash")
HF_JUDGE_MODEL = os.getenv("HF_JUDGE_MODEL", "deepseek-ai/DeepSeek-V4-Flash")
HF_BASE_URL = "https://router.huggingface.co/v1"

# ======================================================================
# Finance Agent Benchmark (FAB) data config
# ======================================================================
FAB_PUBLIC_CSV_URL = "https://raw.githubusercontent.com/vals-ai/finance-agent/main/data/public.csv"
FAB_DATA_PATH = os.getenv("FAB_DATA_PATH", "data/fab_public.csv")

# ======================================================================
# Scoring tolerance / robustness config
# ======================================================================
SCORING_NUMERIC_TOL = float(os.getenv("SCORING_NUMERIC_TOL", "0.05"))       # relative tolerance for numeric scoring
SCORING_QUANTITATIVE_TOL = float(os.getenv("SCORING_QUANTITATIVE_TOL", "0.5"))  # looser tol for quantitative retrieval
SCORING_RUBRIC_COVERAGE = float(os.getenv("SCORING_RUBRIC_COVERAGE", "0.6"))  # threshold for rubric-based scoring
# Numeric comparison tolerance (e.g. 0.05 = 5% difference allowed)
SCORING_NUMERIC_TOLERANCE = float(os.getenv("SCORING_NUMERIC_TOLERANCE", "0.05"))
