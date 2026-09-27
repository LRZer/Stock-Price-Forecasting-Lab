"""Modern PyTorch versions of the original model families.

The same 18 architectures can predict a next-day close (price task) or produce
a next-day up logit (direction task). Seq2seq and VAE names describe compact
one-step teaching implementations, not exact archived TensorFlow reproductions.
"""

from __future__ import annotations

import torch
from torch import nn


CLASSICAL_MODELS = ("naive", "ridge", "hist-gbdt", "stacked")

_RECURRENT = {
    "lstm": ("lstm", False, False),
    "bidirectional-lstm": ("lstm", True, False),
    "lstm-2path": ("lstm", False, True),
    "gru": ("gru", False, False),
    "bidirectional-gru": ("gru", True, False),
    "gru-2path": ("gru", False, True),
    "vanilla": ("rnn", False, False),
    "bidirectional-vanilla": ("rnn", True, False),
    "vanilla-2path": ("rnn", False, True),
}
_SEQ2SEQ = {
    "lstm-seq2seq": ("lstm", False, False),
    "bidirectional-lstm-seq2seq": ("lstm", True, False),
    "lstm-seq2seq-vae": ("lstm", False, True),
    "gru-seq2seq": ("gru", False, False),
    "bidirectional-gru-seq2seq": ("gru", True, False),
    "gru-seq2seq-vae": ("gru", False, True),
}
NEURAL_MODELS = tuple(_RECURRENT) + tuple(_SEQ2SEQ) + (
    "attention-is-all-you-need",
    "cnn-seq2seq",
    "dilated-cnn-seq2seq",
)
# The original 18 price families remain unchanged. These compact additions are
# direction-only candidates; they are not claims of superior stock accuracy.
DIRECTION_EXTRA_MODELS = ("tcn-residual", "gru-attention")
DIRECTION_NEURAL_MODELS = NEURAL_MODELS + DIRECTION_EXTRA_MODELS
ALL_MODELS = CLASSICAL_MODELS + NEURAL_MODELS


def _rnn_class(cell: str) -> type[nn.Module]:
    return {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[cell]


def _final_state(state: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
                 bidirectional: bool) -> torch.Tensor:
    hidden = state[0] if isinstance(state, tuple) else state
    if bidirectional:
        return torch.cat((hidden[-2], hidden[-1]), dim=-1)
    return hidden[-1]


class RecurrentForecaster(nn.Module):
    def __init__(self, cell: str, hidden_size: int, bidirectional: bool, two_path: bool,
                 input_size: int = 1, task: str = "price"):
        super().__init__()
        rnn_type = _rnn_class(cell)
        self.forward_rnn = rnn_type(input_size, hidden_size, batch_first=True,
                                    bidirectional=bidirectional)
        self.change_rnn = rnn_type(1, hidden_size, batch_first=True) if two_path else None
        self.bidirectional = bidirectional
        self.task = task
        features = hidden_size * (2 if bidirectional or two_path else 1)
        self.head = nn.Sequential(nn.LayerNorm(features), nn.Linear(features, hidden_size),
                                  nn.GELU(), nn.Linear(hidden_size, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, state = self.forward_rnn(x)
        features = _final_state(state, self.bidirectional)
        if self.change_rnn is not None:
            first_channel = x[:, :, :1]
            changes = torch.cat((torch.zeros_like(first_channel[:, :1]),
                                 first_channel[:, 1:] - first_channel[:, :-1]), dim=1)
            _, change_state = self.change_rnn(changes)
            features = torch.cat((features, _final_state(change_state, False)), dim=-1)
        self.aux_loss = x.new_zeros(())
        output = self.head(features).squeeze(-1)
        return x[:, -1, 0] + output if self.task == "price" else output


class Seq2SeqForecaster(nn.Module):
    def __init__(self, cell: str, hidden_size: int, bidirectional: bool, variational: bool,
                 input_size: int = 1, task: str = "price"):
        super().__init__()
        self.encoder = _rnn_class(cell)(input_size, hidden_size, batch_first=True,
                                        bidirectional=bidirectional)
        self.bidirectional = bidirectional
        self.variational = variational
        self.cell = cell
        self.task = task
        context_size = hidden_size * (2 if bidirectional else 1)
        if variational:
            self.mean = nn.Linear(context_size, hidden_size)
            self.log_variance = nn.Linear(context_size, hidden_size)
            context_size = hidden_size
        self.initial_hidden = nn.Linear(context_size, hidden_size)
        self.decoder = (nn.LSTMCell(input_size, hidden_size) if cell == "lstm"
                        else nn.GRUCell(input_size, hidden_size))
        self.head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, state = self.encoder(x)
        context = _final_state(state, self.bidirectional)
        if self.variational:
            mean = self.mean(context)
            log_variance = self.log_variance(context).clamp(-10, 10)
            if self.training:
                context = mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)
            else:
                context = mean
            self.aux_loss = -0.5 * (1 + log_variance - mean.square() - log_variance.exp()).mean()
        else:
            self.aux_loss = x.new_zeros(())
        hidden = torch.tanh(self.initial_hidden(context))
        token = x[:, -1, :]
        if self.cell == "lstm":
            hidden, _ = self.decoder(token, (hidden, torch.zeros_like(hidden)))
        else:
            hidden = self.decoder(token, hidden)
        output = self.head(hidden).squeeze(-1)
        return x[:, -1, 0] + output if self.task == "price" else output


class AttentionForecaster(nn.Module):
    def __init__(self, lookback: int, hidden_size: int, input_size: int = 1,
                 task: str = "price"):
        super().__init__()
        attention_heads = 4
        if hidden_size % attention_heads:
            raise ValueError("hidden_size must be divisible by 4 for attention")
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.task = task
        self.position = nn.Parameter(torch.zeros(1, lookback, hidden_size))
        layer = nn.TransformerEncoderLayer(hidden_size, attention_heads,
                                           dim_feedforward=hidden_size * 2,
                                           dropout=0.1, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.input_projection(x) + self.position)
        self.aux_loss = x.new_zeros(())
        output = self.head(encoded[:, -1]).squeeze(-1)
        return x[:, -1, 0] + output if self.task == "price" else output


class CNNForecaster(nn.Module):
    def __init__(self, hidden_size: int, dilated: bool, input_size: int = 1,
                 task: str = "price"):
        super().__init__()
        dilation_1, dilation_2 = (2, 4) if dilated else (1, 1)
        self.encoder = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 3, padding=dilation_1, dilation=dilation_1),
            nn.GELU(),
            nn.Conv1d(hidden_size, hidden_size, 3, padding=dilation_2, dilation=dilation_2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.task = task
        self.aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x.transpose(1, 2)).squeeze(-1)
        self.aux_loss = x.new_zeros(())
        output = self.head(features).squeeze(-1)
        return x[:, -1, 0] + output if self.task == "price" else output


class _CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.dilation = dilation
        self.first = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.second = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Left padding keeps the block causal within the already observed window.
        hidden = self.activation(self.first(nn.functional.pad(x, (2 * self.dilation, 0))))
        hidden = self.second(nn.functional.pad(hidden, (2 * self.dilation, 0)))
        return self.activation(x + hidden)


class ResidualTCNClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_projection = nn.Conv1d(input_size, hidden_size, kernel_size=1)
        self.blocks = nn.Sequential(*(_CausalResidualBlock(hidden_size, dilation)
                                      for dilation in (1, 2, 4)))
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.blocks(self.input_projection(x.transpose(1, 2)))
        self.aux_loss = x.new_zeros(())
        return self.head(hidden[:, :, -1]).squeeze(-1)


class AttentionGRUClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.encoder = nn.GRU(input_size, hidden_size, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        self.head = nn.Sequential(nn.LayerNorm(2 * hidden_size),
                                  nn.Linear(2 * hidden_size, hidden_size), nn.GELU(),
                                  nn.Linear(hidden_size, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x)
        weights = torch.softmax(self.attention(encoded).squeeze(-1), dim=1)
        context = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        features = torch.cat((encoded[:, -1], context), dim=-1)
        self.aux_loss = x.new_zeros(())
        return self.head(features).squeeze(-1)


def build_model(name: str, *, lookback: int, hidden_size: int = 32,
                input_size: int = 1, task: str = "price") -> nn.Module:
    if hidden_size < 4:
        raise ValueError("hidden_size must be at least 4")
    if input_size < 1 or task not in ("price", "direction"):
        raise ValueError("input_size must be positive and task must be price or direction")
    if name in DIRECTION_EXTRA_MODELS:
        if task != "direction":
            raise ValueError(f"{name} supports the direction task only")
        return (ResidualTCNClassifier(input_size, hidden_size) if name == "tcn-residual"
                else AttentionGRUClassifier(input_size, hidden_size))
    if name in _RECURRENT:
        cell, bidirectional, two_path = _RECURRENT[name]
        return RecurrentForecaster(cell, hidden_size, bidirectional, two_path,
                                   input_size=input_size, task=task)
    if name in _SEQ2SEQ:
        cell, bidirectional, variational = _SEQ2SEQ[name]
        return Seq2SeqForecaster(cell, hidden_size, bidirectional, variational,
                                 input_size=input_size, task=task)
    if name == "attention-is-all-you-need":
        return AttentionForecaster(lookback, hidden_size, input_size=input_size, task=task)
    if name in ("cnn-seq2seq", "dilated-cnn-seq2seq"):
        return CNNForecaster(hidden_size, name.startswith("dilated"), input_size=input_size,
                             task=task)
    raise ValueError(f"Unknown neural model: {name}")
