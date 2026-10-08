"""Shared paths and settings. Every script imports from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
for p in (DATA_RAW, DATA_PROC, RESULTS):
    p.mkdir(parents=True, exist_ok=True)

PROMPTS_CSV = ROOT / "data" / "prompts.csv"
RESPONSES = DATA_RAW / "responses.jsonl"
SEEDS = [0, 1, 2]

# Same generation settings for EVERY model (document these in the report)
MAX_TOKENS = 512
TEMPERATURE = 1.0

# family -> (provider, model id).  CHECK these IDs against each provider's
# current model list before running; names change and old ones get retired.
MODELS = {
    "gpt":    ("openai",     "gpt-4o-mini"),
    "claude": ("anthropic",  "claude-haiku-4-5-20251001"),
    "gemini": ("google",     "gemini-2.5-flash"),
}
