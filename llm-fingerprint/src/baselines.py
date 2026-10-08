"""Majority-class and TF-IDF (word + char n-grams) + logistic regression baselines.
Run:  python src/baselines.py            (or --dataset dataset_mismatch)
"""
import json
import argparse
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from config import RESULTS, SEEDS
from data import load_split
from metrics import score


def texts(df, mode):
    if mode == "output":
        return df.response.tolist()
    if mode == "input":
        return df.prompt.tolist()
    return (df.prompt + " [SEP] " + df.response).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    a = ap.parse_args()

    for seed in SEEDS:
        sp = load_split(a.dataset, seed)
        labels = sorted(sp["train"].model_family.unique())
        l2i = {l: i for i, l in enumerate(labels)}
        y = {k: np.array([l2i[m] for m in d.model_family]) for k, d in sp.items()}

        for mode in ["output", "input", "both"]:
            vec = FeatureUnion([
                ("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, lowercase=False,
                                      sublinear_tf=True, token_pattern=r"\w+|[^\w\s]")),
                ("c", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=3,
                                      lowercase=False, sublinear_tf=True, max_features=200000)),
            ])
            Xtr = vec.fit_transform(texts(sp["train"], mode))
            Xte = vec.transform(texts(sp["test"], mode))
            for name, clf in [("tfidf_lr", LogisticRegression(C=10, max_iter=1000)),
                              ("majority", DummyClassifier(strategy="most_frequent"))]:
                clf.fit(Xtr, y["train"])
                pred = clf.predict(Xte)
                res = score(y["test"], pred, labels)
                res.update(dataset=a.dataset, model=name, mode=mode, seed=seed, labels=labels,
                           history=[], y_true=y["test"].tolist(), y_pred=pred.tolist(),
                           test_row_ids=sp["test"].row_id.tolist())
                with open(RESULTS / f"{a.dataset}_{name}_{mode}_seed{seed}.json", "w") as f:
                    json.dump(res, f)
                print(f"seed{seed} {name:9s} {mode:6s} acc={res['acc']:.3f} f1={res['macro_f1']:.3f}")

                if name == "tfidf_lr" and mode == "output" and seed == SEEDS[0]:
                    names = vec.get_feature_names_out()
                    print("  top features per model (check for trivial cues!):")
                    for i, l in enumerate(labels):
                        top = np.argsort(clf.coef_[i])[-10:][::-1]
                        print(f"   {l}: {[names[j] for j in top]}")


if __name__ == "__main__":
    main()
