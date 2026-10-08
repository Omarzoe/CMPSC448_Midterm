import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TextCNN(nn.Module):
    """Kim-style CNN: embedding -> parallel conv1d -> global max pool -> dropout -> linear."""
    def __init__(self, vocab, n_cls, emb=128, n_filters=100, kernel_sizes=(2, 3, 4, 5), drop=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb, n_filters, k) for k in kernel_sizes])
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), n_cls)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)                      # B, emb, L
        h = [F.relu(c(e)).max(dim=2).values for c in self.convs]
        return self.fc(self.drop(torch.cat(h, dim=1)))


class BiLSTM(nn.Module):
    """Embedding -> BiLSTM (packed, ignores padding) -> masked max-pool -> dropout -> linear."""
    def __init__(self, vocab, n_cls, emb=128, hidden=128, layers=1, drop=0.4):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                            dropout=drop if layers > 1 else 0.0)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(2 * hidden, n_cls)

    def forward(self, x):
        mask = (x != 0)
        lengths = mask.sum(1).clamp(min=1).cpu()
        packed = pack_padded_sequence(self.emb(x), lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        out = out.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values
        return self.fc(self.drop(out))
