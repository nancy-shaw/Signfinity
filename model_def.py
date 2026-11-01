"""BiLSTM + Attention model, copied as is from the training notebook."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMAttention(nn.Module):
    def __init__(self, input_size=126, hidden_size=64, num_layers=1,
                 num_classes=129, dropout=0.65):
        super().__init__()

        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        self.attention = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.bilstm(x)
        attn_scores = self.attention(lstm_out)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        context = self.layer_norm(context)
        context = self.dropout(context)
        out = self.classifier(context)
        return out