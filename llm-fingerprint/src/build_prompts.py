"""Build data/prompts.csv: prompt_id, category, source, prompt."""
import re
import argparse
import pandas as pd
from datasets import load_dataset
from config import PROMPTS_CSV

# drop prompts containing emails, phone numbers or URLs
PII = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|\+?\d[\d\s().-]{8,}\d|https?://\S+")
DOLLY_MAP = {
    "open_qa": "factual_qa", "general_qa": "factual_qa",
    "creative_writing": "writing", "brainstorming": "brainstorming",
    "summarization": "summarization",
}


def ok(t):
    return 10 <= len(t) <= 1000 and not PII.search(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_category", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rows = []
    for r in load_dataset("databricks/databricks-dolly-15k", split="train"):
        cat = DOLLY_MAP.get(r["category"])
        if cat:
            p = r["instruction"].strip()
            if r["context"].strip():
                p += "\n\n" + r["context"].strip()
            rows.append((cat, "dolly", p))
    for r in load_dataset("openai/gsm8k", "main", split="train"):
        rows.append(("math", "gsm8k", r["question"].strip()))
    for r in load_dataset("google-research-datasets/mbpp", "full", split="train"):
        rows.append(("coding", "mbpp", r["text"].strip()))

    df = pd.DataFrame(rows, columns=["category", "source", "prompt"])
    df = df[df.prompt.map(ok)].drop_duplicates("prompt")
    parts = [g.sample(min(a.per_category, len(g)), random_state=a.seed)
             for _, g in df.groupby("category")]
    df = pd.concat(parts).sample(frac=1, random_state=a.seed).reset_index(drop=True)
    df.insert(0, "prompt_id", [f"p{i:04d}" for i in range(len(df))])
    df.to_csv(PROMPTS_CSV, index=False)
    print(df.category.value_counts())
    print(f"Saved {len(df)} prompts to {PROMPTS_CSV}")


if __name__ == "__main__":
    main()
