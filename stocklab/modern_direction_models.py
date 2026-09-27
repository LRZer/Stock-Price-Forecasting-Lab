"""Compact, direction-only adaptations of eight time-series model ideas.

These teaching implementations borrow the defining data flow of the cited
forecasting papers. They are not reproductions of those papers' architectures
or published forecasting results.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


MODERN_DIRECTION_MODELS = (
    "dlinear-direction",
    "tsmixer-direction",
    "patchtst-direction",
    "itransformer-direction",
    "nhits-direction",
    "tide-direction",
    "timesnet-direction",
    "mamba-style-direction",
)


def _zero_head(layer: nn.Linear) -> None:
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)


class DLinearDirection(nn.Module):
    """Moving-average decomposition, separate temporal linears, binary head."""

    def __init__(self, lookback: int, input_size: int):
        super().__init__()
        self.trend = nn.Linear(lookback, 1)
        self.seasonal = nn.Linear(lookback, 1)
        self.head = nn.Linear(input_size, 1)
        _zero_head(self.head)
        self.aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.transpose(1, 2)
        trend = F.avg_pool1d(F.pad(channels, (1, 1), mode="replicate"), 3, stride=1)
        features = (self.trend(trend) + self.seasonal(channels - trend)).squeeze(-1)
        self.aux_loss = x.new_zeros(())
        return self.head(features).squeeze(-1)


class _MixerBlock(nn.Module):
    def __init__(self, lookback: int, input_size: int, hidden_size: int):
        super().__init__()
        self.time_norm = nn.LayerNorm(lookback)
        self.time_mlp = nn.Sequential(nn.Linear(lookback, lookback * 2), nn.GELU(),
                                      nn.Linear(lookback * 2, lookback))
        self.feature_norm = nn.LayerNorm(input_size)
        self.feature_mlp = nn.Sequential(nn.Linear(input_size, hidden_size), nn.GELU(),
                                         nn.Linear(hidden_size, input_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        by_channel = x.transpose(1, 2)
        x = x + self.time_mlp(self.time_norm(by_channel)).transpose(1, 2)
        return x + self.feature_mlp(self.feature_norm(x))


class TSMixerDirection(nn.Module):
    """Residual time and feature mixing with an observed-window readout."""

    def __init__(self, lookback: int, input_size: int, hidden_size: int):
        super().__init__()
        self.blocks = nn.Sequential(*(_MixerBlock(lookback, input_size, hidden_size)
                                      for _ in range(2)))
        self.head = nn.Sequential(nn.LayerNorm(lookback * input_size),
                                  nn.Linear(lookback * input_size, 1))
        _zero_head(self.head[-1])
        self.aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = self.blocks(x)
        self.aux_loss = x.new_zeros(())
        return self.head(mixed.flatten(1)).squeeze(-1)


class PatchTSTDirection(nn.Module):
    """Shared, channel-independent attention over observed temporal patches."""

    def __init__(self, lookback: int, input_size: int, hidden_size: int):
        super().__init__()
        patch_size = min(4, lookback)
        step = max(1, patch_size // 2)
        right_pad = (step - (lookback - patch_size) % step) % step
        patch_count = 1 + (lookback + right_pad - patch_size) // step
        self.patch_size = patch_size
        self.step = step
        self.right_pad = right_pad
        self.embed = nn.Linear(patch_size, hidden_size)
        self.position = nn.Parameter(torch.zeros(1, patch_count, hidden_size))
        layer = nn.TransformerEncoderLayer(hidden_size, 4, hidden_size * 2,
                                           dropout=0.1, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Sequential(nn.LayerNorm(input_size * hidden_size),
                                  nn.Linear(input_size * hidden_size, 1))
        _zero_head(self.head[-1])
        self.aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, channels = x.shape
        channels_first = x.transpose(1, 2)
        if self.right_pad:
            channels_first = F.pad(channels_first, (0, self.right_pad), mode="replicate")
        patches = channels_first.unfold(-1, self.patch_size, self.step)
        tokens = self.embed(patches).reshape(batch * channels, -1, self.position.shape[-1])
        encoded = self.encoder(tokens + self.position)
        features = encoded.mean(dim=1).reshape(batch, -1)
        self.aux_loss = x.new_zeros(())
        return self.head(features).squeeze(-1)


class ITransformerDirection(nn.Module):
    """One token per feature; self-attention mixes feature histories."""

    def __init__(self, lookback: int, input_size: int, hidden_size: int):
        super().__init__()
        self.embed = nn.Linear(lookback, hidden_size)
        layer = nn.TransformerEncoderLayer(hidden_size, 4, hidden_size * 2,
                                           dropout=0.1, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Sequential(nn.LayerNorm(input_size * hidden_size),
                                  nn.Linear(input_size * hidden_size, 1))
        _zero_head(self.head[-1])
        self.aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(self.embed(x.transpose(1, 2))).flatten(1)
        self.aux_loss = x.new_zeros(())
        return self.head(features).squeeze(-1)


class _NHITSBlock(nn.Module):
    def __init__(self, lookback: int, input_size: int, hidden_size: int, pool: int):
        super().__init__()
        self.pool = pool
        pooled_length = (lookback + pool - 1) // pool
        self.encoder = nn.Sequential(nn.Linear(pooled_length * input_size, hidden_size),
                                     nn.GELU(), nn.Linear(hidden_size, hidden_size), nn.GELU())
        self.backcast = nn.Linear(hidden_size, pooled_length * input_size)
        self.logit = nn.Linear(hidden_size, 1)
        _zero_head(self.logit)
        self.pooled_length = pooled_length
        self.input_size = input_size

    def forward(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # The newest observation is included when the length is not divisible.
        channels = residual.transpose(1, 2)
        pad = (-channels.shape[-1]) % self.pool
        if pad:
            channels = F.pad(channels, (0, pad), mode="replicate")
        pooled = F.avg_pool1d(channels, self.pool, stride=self.pool)
        hidden = self.encoder(pooled.flatten(1))
        coarse = self.backcast(hidden).reshape(-1, self.input_size, self.pooled_length)
        backcast = F.interpolate(coarse, size=residual.shape[1], mode="linear",
                                 align_corners=False).transpose(1, 2)
        return residual - backcast, self.logit(hidden).squeeze(-1)


class NHITSDirection(nn.Module):
    """Multi-rate pooling and residual backcasts for a one-step binary logit."""

    def __init__(self, lookback: int, input_size: int, hidden_size: int):
        super().__init__()
        self.blocks = nn.ModuleList(_NHITSBlock(lookback, input_size, hidden_size, pool)
                                    for pool in (4, 2, 1))
        self.aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        output = x.new_zeros(x.shape[0])
        for block in self.blocks:
            residual, component = block(residual)
            output = output + component
        self.aux_loss = x.new_zeros(())
        return output


class _DenseResidual(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                 nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class TiDEDirection(nn.Module):
    """Dense feature projection, encoder/decoder blocks and a linear skip."""

    def __init__(self, lookback: int, input_size: int, hidden_size: int):
        super().__init__()
        feature_width = max(4, hidden_size // 4)
        self.feature_projection = nn.Sequential(nn.Linear(input_size, feature_width),
                                                nn.GELU())
        self.encoder = nn.Sequential(
            nn.Linear(lookback * (input_size + feature_width), hidden_size),
            nn.GELU(), _DenseResidual(hidden_size))
        self.decoder = nn.Sequential(_DenseResidual(hidden_size),
                                     nn.Linear(hidden_size, lookback * feature_width),
                                     nn.GELU())
        self.head = nn.Linear(lookback * feature_width, 1)
        self.skip = nn.Linear(lookback * input_size, 1)
        _zero_head(self.head)
        _zero_head(self.skip)
        self.aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = x.flatten(1)
        projected = self.feature_projection(x)
        encoded = self.encoder(torch.cat((x, projected), dim=-1).flatten(1))
        self.aux_loss = x.new_zeros(())
        return (self.head(self.decoder(encoded)) + self.skip(raw)).squeeze(-1)


class TimesNetDirection(nn.Module):
    """Small periodic 2D convolutions with per-example spectral weighting."""

    def __init__(self, lookback: int, input_size: int, hidden_size: int):
        super().__init__()
        self.periods = tuple(p for p in (2, 4, 5, 10) if p <= lookback) or (1,)
        self.conv = nn.Sequential(nn.Conv2d(input_size, hidden_size, 3, padding=1),
                                  nn.GELU(), nn.Conv2d(hidden_size, hidden_size, 3,
                                                        padding=1), nn.GELU())
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))
        _zero_head(self.head[-1])
        self.aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        spectrum = torch.fft.rfft(x.transpose(1, 2), dim=-1).abs().mean(dim=1)
        frequencies = [min(round(length / period), spectrum.shape[-1] - 1)
                       for period in self.periods]
        weights = torch.softmax(spectrum[:, frequencies], dim=-1)
        features = []
        for period in self.periods:
            pad = (-length) % period
            values = x.transpose(1, 2)
            if pad:
                values = F.pad(values, (0, pad), mode="replicate")
            grid = values.reshape(batch, channels, (length + pad) // period, period)
            features.append(self.conv(grid).mean(dim=(2, 3)))
        combined = (torch.stack(features, dim=1) * weights.unsqueeze(-1)).sum(dim=1)
        self.aux_loss = x.new_zeros(())
        return self.head(combined).squeeze(-1)


class MambaStyleDirection(nn.Module):
    """Pure-PyTorch selective diagonal state space; no official fused scan."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_projection = nn.Linear(input_size, 2 * hidden_size)
        self.depthwise = nn.Conv1d(hidden_size, hidden_size, 3,
                                   groups=hidden_size)
        self.ssm_parameters = nn.Linear(hidden_size, 3 * hidden_size)
        self.decay_log = nn.Parameter(torch.zeros(hidden_size))
        self.direct = nn.Parameter(torch.ones(hidden_size))
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))
        _zero_head(self.head[-1])
        self.aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.input_projection(x).chunk(2, dim=-1)
        value = F.silu(self.depthwise(F.pad(value.transpose(1, 2), (2, 0)))
                       .transpose(1, 2))
        delta_raw, input_b, output_c = self.ssm_parameters(value).chunk(3, dim=-1)
        delta = F.softplus(delta_raw)
        decay_rate = self.decay_log.exp()
        state = torch.zeros_like(value[:, 0])
        output = state
        for step in range(x.shape[1]):
            decay = torch.exp(-delta[:, step] * decay_rate)
            state = decay * state + delta[:, step] * input_b[:, step] * value[:, step]
            output = (output_c[:, step] * state + self.direct * value[:, step]) * \
                     torch.sigmoid(gate[:, step])
        self.aux_loss = x.new_zeros(())
        return self.head(output).squeeze(-1)


def build_modern_direction_model(name: str, *, lookback: int, input_size: int,
                                 hidden_size: int = 32) -> nn.Module:
    if lookback < 1 or input_size < 1 or hidden_size < 4 or hidden_size % 4:
        raise ValueError("lookback/input_size must be positive; hidden_size must be divisible by 4")
    if name == "dlinear-direction":
        return DLinearDirection(lookback, input_size)
    if name == "tsmixer-direction":
        return TSMixerDirection(lookback, input_size, hidden_size)
    if name == "patchtst-direction":
        return PatchTSTDirection(lookback, input_size, hidden_size)
    if name == "itransformer-direction":
        return ITransformerDirection(lookback, input_size, hidden_size)
    if name == "nhits-direction":
        return NHITSDirection(lookback, input_size, hidden_size)
    if name == "tide-direction":
        return TiDEDirection(lookback, input_size, hidden_size)
    if name == "timesnet-direction":
        return TimesNetDirection(lookback, input_size, hidden_size)
    if name == "mamba-style-direction":
        return MambaStyleDirection(input_size, hidden_size)
    raise ValueError(f"Unknown modern direction model: {name}")
