# config.py — API keys / model config (真实 LLM 时用)
# 推荐用环境变量，不要硬编码到代码里
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# 本地 vLLM / OpenAI 兼容服务
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# 评测全局配置
MAX_TOOL_CALLS_PER_TASK = 5
REQUEST_TIMEOUT = 15
