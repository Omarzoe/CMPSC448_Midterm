"""Aggregate results: tables (mean ± std), confusion matrices, curves, significance tests.
Run:  python src/evaluate.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binomtest, ttest_rel

from config import RESULTS


def main():
    runs = [json.load(open(f)) for f in sorted(RESULTS.glob("*_seed*.json"))]
    df = pd.DataFrame([{k: r[k] for k in ("dataset", "model", "mode", "seed", "acc", "macro_f1")}
                       for r in runs])

    # ---- summary table
    g = df.groupby(["dataset", "model", "mode"]).agg(
        acc_mean=("acc", "mean"), acc_std=("acc", "std"),
        f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"), n_seeds=("seed", "count")).round(4)
    g.to_csv(RESULTS / "summary.csv")
    print(g.to_string())

    # ---- confusion matrices summed over seeds
    groups = {}
    for r in runs:
        groups.setdefault((r["dataset"], r["model"], r["mode"]), []).append(r)
    for (ds, m, mode), rs in groups.items():
        cm = np.sum([r["confusion"] for r in rs], axis=0).astype(float)
        cm = cm / cm.sum(1, keepdims=True).clip(min=1)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                    xticklabels=rs[0]["labels"], yticklabels=rs[0]["labels"])
        plt.xlabel("predicted"); plt.ylabel("true"); plt.title(f"{ds} | {m} | {mode}")
        plt.tight_layout()
        plt.savefig(RESULTS / f"cm_{ds}_{m}_{mode}.png", dpi=150)
        plt.close()

    # ---- training curves (seed 0)
    for r in runs:
        if r["seed"] == 0 and r["history"]:
            h = pd.DataFrame(r["history"])
            fig, ax = plt.subplots(1, 2, figsize=(9, 3))
            ax[0].plot(h.epoch, h.train_loss); ax[0].set_title("train loss"); ax[0].set_xlabel("epoch")
            ax[1].plot(h.epoch, h.val_f1); ax[1].set_title("val macro-F1"); ax[1].set_xlabel("epoch")
            fig.suptitle(f"{r['dataset']} | {r['model']} | {r['mode']}")
            plt.tight_layout()
            plt.savefig(RESULTS / f"curve_{r['dataset']}_{r['model']}_{r['mode']}.png", dpi=150)
            plt.close()

    # ---- RQ2 significance: output-only vs input+output
    print("\nRQ2: output-only vs input+output")
    for ds in df.dataset.unique():
        for m in df.model.unique():
            a = {r["seed"]: r for r in groups.get((ds, m, "output"), [])}
            b = {r["seed"]: r for r in groups.get((ds, m, "both"), [])}
            seeds = sorted(set(a) & set(b))
            if not seeds:
                continue
            ps = []
            for s in seeds:
                ca = np.array(a[s]["y_true"]) == np.array(a[s]["y_pred"])
                cb = np.array(b[s]["y_true"]) == np.array(b[s]["y_pred"])
                only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
                ps.append(binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0)
            accs_a = [a[s]["acc"] for s in seeds]; accs_b = [b[s]["acc"] for s in seeds]
            t = ttest_rel(accs_b, accs_a).pvalue if len(seeds) > 1 else float("nan")
            print(f"{ds} | {m}: output={np.mean(accs_a):.3f} both={np.mean(accs_b):.3f} "
                  f"| McNemar p per seed={np.round(ps, 3).tolist()} | paired-t p={t:.3f}")


if __name__ == "__main__":
    main()
