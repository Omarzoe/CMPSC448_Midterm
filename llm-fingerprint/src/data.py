"""Tokenizer, vocabulary, and PyTorch loaders shared by CNN and LSTM."""
import re
import json
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from config import DATA_PROC

# keeps punctuation, markdown symbols and newlines as tokens (formatting is a fingerprint)
TOK = re.compile(r"\n|\w+|[^\w\s]")
PAD, UNK, SEP = "<pad>", "<unk>", "<sep>"
PROMPT_CAP = 64  # in "both" mode, prompt tokens are capped so the response still fits


def tokenize(t):
    return TOK.findall(str(t))


def row_tokens(prompt, response, mode, max_len):
    if mode == "output":
        return tokenize(response)[:max_len]
    if mode == "input":
        return tokenize(prompt)[:max_len]
    return (tokenize(prompt)[:PROMPT_CAP] + [SEP] + tokenize(response))[:max_len]


def build_vocab(token_lists, min_freq=2):
    cnt = Counter(t for toks in token_lists for t in toks)
    itos = [PAD, UNK, SEP] + [t for t, c in cnt.most_common() if c >= min_freq and t != SEP]
    return {t: i for i, t in enumerate(itos)}


def encode(tokens, vocab, max_len):
    ids = [vocab.get(t, 1) for t in tokens[:max_len]]
    return ids + [0] * (max_len - len(ids))


def load_split(dataset, seed):
    df = pd.read_csv(DATA_PROC / f"{dataset}.csv")
    with open(DATA_PROC / f"{dataset}_split_seed{seed}.json") as f:
        split = json.load(f)
    return {k: df[df.prompt_id.isin(set(v))].reset_index(drop=True) for k, v in split.items()}


def prepare(dataset, seed, mode, max_len=256, batch=64, limit=0):
    sp = load_split(dataset, seed)
    labels = sorted(set().union(*[set(d.model_family) for d in sp.values()]))
    l2i = {l: i for i, l in enumerate(labels)}
    if limit:
        sp["train"] = sp["train"].head(limit)

    toks = {k: [row_tokens(p, r, mode, max_len) for p, r in zip(d.prompt, d.response)]
            for k, d in sp.items()}
    vocab = build_vocab(toks["train"])  # TRAIN ONLY

    loaders = {}
    for k, d in sp.items():
        x = torch.tensor([encode(t, vocab, max_len) for t in toks[k]])
        y = torch.tensor([l2i[m] for m in d.model_family])
        loaders[k] = DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=(k == "train"))
    return loaders, len(vocab), labels, sp
