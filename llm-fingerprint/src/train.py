"""Train one CNN/LSTM run.
Example:  python src/train.py --model cnn --mode output --seed 0
Debug:    python src/train.py --model cnn --mode output --limit 200   (should overfit)
"""
import json
import copy
import random
import argparse
import numpy as np
import torch
import torch.nn as nn

from config import RESULTS
from data import prepare
from models import TextCNN, BiLSTM
from metrics import score


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps += model(x.to(device)).argmax(1).cpu().tolist()
        ys += y.tolist()
    return np.array(ys), np.array(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "lstm"], required=True)
    ap.add_argument("--mode", choices=["output", "input", "both"], default="output")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--emb", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="debug: train on first N rows only")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loaders, vocab_size, labels, sp = prepare(a.dataset, a.seed, a.mode, a.max_len, a.batch, a.limit)
    cls = TextCNN if a.model == "cnn" else BiLSTM
    model = cls(vocab_size, len(labels), emb=a.emb).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    loss_fn = nn.CrossEntropyLoss()
    print(f"{a.model}/{a.mode}/seed{a.seed} on {device}: vocab={vocab_size}, "
          f"train={len(sp['train'])}, val={len(sp['val'])}, test={len(sp['test'])}, classes={labels}")

    best_f1, best_state, bad, history = -1, None, 0, []
    for ep in range(1, a.epochs + 1):
        model.train()
        tot, correct, n = 0.0, 0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(y); n += len(y)
            correct += (logits.argmax(1) == y).sum().item()
        yv, pv = predict(model, loaders["val"], device)
        v = score(yv, pv, labels)
        history.append({"epoch": ep, "train_loss": tot / n, "train_acc": correct / n,
                        "val_acc": v["acc"], "val_f1": v["macro_f1"]})
        print(f"ep{ep:02d} loss={tot/n:.4f} train_acc={correct/n:.3f} "
              f"val_acc={v['acc']:.3f} val_f1={v['macro_f1']:.3f}")
        if v["macro_f1"] > best_f1:
            best_f1, best_state, bad = v["macro_f1"], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop")
                break

    model.load_state_dict(best_state)
    yt, pt = predict(model, loaders["test"], device)   # test is evaluated ONCE
    res = score(yt, pt, labels)
    res.update(dataset=a.dataset, model=a.model, mode=a.mode, seed=a.seed, labels=labels,
               best_val_f1=best_f1, history=history, hparams=vars(a),
               y_true=yt.tolist(), y_pred=pt.tolist(),
               test_row_ids=sp["test"].row_id.tolist())
    out = RESULTS / f"{a.dataset}_{a.model}_{a.mode}_seed{a.seed}.json"
    if not a.limit:
        with open(out, "w") as f:
            json.dump(res, f)
    print(f"TEST acc={res['acc']:.4f} macro_f1={res['macro_f1']:.4f}  -> {out.name}")


if __name__ == "__main__":
    main()
