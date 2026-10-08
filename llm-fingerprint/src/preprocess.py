"""Clean responses.jsonl -> data/processed/dataset.csv (+ mismatched variant) + splits."""
import json
import numpy as np
import pandas as pd
from config import RESPONSES, DATA_PROC, SEEDS

REFUSAL = r"^(?:i'm sorry|i am sorry|i apologize|i can't|i cannot|sorry,|i'm unable|i am unable)"


def make_splits(df, name):
    """Split by prompt_id so all answers to one prompt land in the same split."""
    ids = df.prompt_id.unique()
    for seed in SEEDS:
        perm = np.random.RandomState(seed).permutation(ids)
        n = len(perm)
        n_test, n_val = int(0.15 * n), int(0.15 * n)
        split = {"test": perm[:n_test].tolist(),
                 "val": perm[n_test:n_test + n_val].tolist(),
                 "train": perm[n_test + n_val:].tolist()}
        with open(DATA_PROC / f"{name}_split_seed{seed}.json", "w") as f:
            json.dump(split, f)


def make_mismatch(df, seed=123):
    """Each prompt answered by ONE model, with category skewed per model.
    Input-only classification should now beat chance (dataset artifact)."""
    rng = np.random.RandomState(seed)
    models, cats = sorted(df.model_family.unique()), sorted(df.category.unique())
    keep = []
    for _, g in df.groupby("prompt_id"):
        fav = models[cats.index(g.category.iloc[0]) % len(models)]
        p = np.array([0.6 if m == fav else 0.4 / (len(models) - 1) for m in models])
        keep.append(g[g.model_family == rng.choice(models, p=p)])
    return pd.concat(keep).reset_index(drop=True)


def main():
    df = pd.read_json(RESPONSES, lines=True)
    n0 = len(df)
    df["response"] = df.response.fillna("").astype(str)
    df = df[df.response.str.strip().str.len() > 0]

    # keep only prompts answered by ALL models -> balanced, identical prompts
    n_models = df.model_family.nunique()
    full = df.groupby("prompt_id").model_family.nunique()
    df = df[df.prompt_id.isin(full[full == n_models].index)]
    df = df.drop_duplicates(["prompt_id", "model_family"]).reset_index(drop=True)

    fr = df.finish_reason.str.lower()
    df["truncated"] = fr.str.contains("length") | fr.str.contains("max_tokens")
    df["refusal"] = df.response.str.strip().str.lower().str[:60].str.contains(REFUSAL, regex=True)
    df["n_words"] = df.response.str.split().str.len()
    df.insert(0, "row_id", range(len(df)))

    print(f"kept {len(df)} of {n0} rows, {df.prompt_id.nunique()} prompts, {n_models} models")
    print(df.groupby("model_family").agg(n=("row_id", "count"),
                                         mean_words=("n_words", "mean"),
                                         median_words=("n_words", "median"),
                                         truncated=("truncated", "mean"),
                                         refusal=("refusal", "mean")).round(3))
    print(pd.crosstab(df.category, df.model_family))

    df.to_csv(DATA_PROC / "dataset.csv", index=False)
    make_splits(df, "dataset")

    mm = make_mismatch(df)
    mm.to_csv(DATA_PROC / "dataset_mismatch.csv", index=False)
    make_splits(mm, "dataset_mismatch")
    print("mismatch variant:\n", mm.model_family.value_counts())


if __name__ == "__main__":
    main()
