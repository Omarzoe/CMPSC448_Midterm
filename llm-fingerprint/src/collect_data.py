"""Query every LLM with every prompt; append to data/raw/responses.jsonl.
Safe to stop and re-run: finished (prompt_id, model) pairs are skipped.

Test first:  python src/collect_data.py --limit 400
"""
import os
import json
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from config import ROOT, PROMPTS_CSV, RESPONSES, MODELS, MAX_TOKENS, TEMPERATURE

load_dotenv(ROOT / ".env")
_clients = {}
_lock = threading.Lock()


def make_client(provider):
    if provider == "openai":
        from openai import OpenAI
        return OpenAI()
    if provider == "openrouter":
        from openai import OpenAI
        return OpenAI(base_url="https://openrouter.ai/api/v1",
                      api_key=os.environ["OPENROUTER_API_KEY"])
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic()
    if provider == "google":
        from google import genai
        return genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    raise ValueError(provider)


def generate(provider, model, prompt):
    """Return (text, finish_reason). No system prompt, same settings for all."""
    c = _clients[provider]
    if provider in ("openai", "openrouter"):
        kw = {"max_completion_tokens": MAX_TOKENS} if provider == "openai" else {"max_tokens": MAX_TOKENS}
        r = c.chat.completions.create(model=model, temperature=TEMPERATURE,
                                      messages=[{"role": "user", "content": prompt}], **kw)
        ch = r.choices[0]
        return ch.message.content or "", str(ch.finish_reason)
    if provider == "anthropic":
        r = c.messages.create(model=model, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                              messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in r.content if b.type == "text"), str(r.stop_reason)
    if provider == "google":
        from google.genai import types
        # NOTE: "thinking" models can spend max_output_tokens on hidden reasoning;
        # if Gemini responses come back empty/truncated, disable thinking in this config.
        r = c.models.generate_content(
            model=model, contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=MAX_TOKENS, temperature=TEMPERATURE))
        fr = r.candidates[0].finish_reason if r.candidates else "none"
        return r.text or "", str(fr)


def with_retry(fn, tries=6):
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(min(60, 2 ** i))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only first N prompts (testing)")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    prompts = pd.read_csv(PROMPTS_CSV)
    if a.limit:
        prompts = prompts.head(a.limit)

    for fam in a.models:
        prov = MODELS[fam][0]
        if prov not in _clients:
            _clients[prov] = make_client(prov)

    done = set()
    if RESPONSES.exists():
        with open(RESPONSES, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                done.add((d["prompt_id"], d["model_family"]))

    tasks = [(r, fam) for _, r in prompts.iterrows() for fam in a.models
             if (r.prompt_id, fam) not in done]
    print(f"{len(tasks)} calls to make ({len(done)} already done)")

    def work(r, fam):
        prov, model = MODELS[fam]
        text, fin = with_retry(lambda: generate(prov, model, r.prompt))
        rec = dict(model_family=fam, model_version=model, provider=prov,
                   prompt_id=r.prompt_id, category=r.category, prompt=r.prompt,
                   response=text, finish_reason=fin,
                   timestamp=datetime.now(timezone.utc).isoformat(),
                   temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        with _lock:
            with open(RESPONSES, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(work, r, fam): (r.prompt_id, fam) for r, fam in tasks}
        for fu in tqdm(as_completed(futs), total=len(futs)):
            try:
                fu.result()
            except Exception as e:
                print("FAILED", futs[fu], repr(e)[:200])  # re-run the script to retry


if __name__ == "__main__":
    main()
