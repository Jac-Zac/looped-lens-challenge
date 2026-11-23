"""Everything you might want to change, in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # final-evals/
DATA_DIR = ROOT / "datasets"
PROMPT_DIR = ROOT / "judge_prompts"
RUNS_DIR = ROOT / "runs"

# Question files: every *.jsonl in DATA_DIR, one JSON object per line (see data.py).

# Submission format: per question, up to MAX_LAYERS readouts of up to N_TOKENS tokens.
MAX_LAYERS = 10
N_TOKENS = 10
MAX_TOKEN_CHARS = 64                   # the model's longest token is 46 characters; text is not a token

# Judge
JUDGE_MODEL = "gpt-6-luna"
# A model named "vendor/model" is reached through OpenRouter. One with no vendor in its name
# ("gpt-6-luna") is reached at OpenAI itself, with OpenAI's key.
API_BASE = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"
OPENAI_API_BASE = "https://api.openai.com/v1"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
ENV_FILE = ROOT / ".env"               # used only if the variable is not set; one NAME=key per line
PROMPT_VERSION = "j3-v3"               # bump when a prompt file changes (invalidates the cache)
CONCURRENCY = 64
MAX_ATTEMPTS = 5
MAX_OUTPUT_TOKENS = 6000
