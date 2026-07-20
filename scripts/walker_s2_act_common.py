"""Shared model and checkpoint contract for state-based Walker S2 ACT."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


FEATURE_CONTRACT = "walker_s2_object_relative_state_act_v1"
ENV_OBS_DIM = 82
ACTION_DIM = 7
ARM_ACTION_DIM = 6
GRIP_ACTION_INDEX = 6


def normalize_observation(
    observation: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    clip: float,
) -> torch.Tensor:
    normalized = (observation - mean) / std
    return torch.clamp(normalized, -clip, clip) if clip > 0.0 else normalized


class WalkerS2StateACT(nn.Module):
    """Predict a future object-relative action chunk from recent state history."""

    def __init__(
        self,
        input_dim: int = ENV_OBS_DIM,
        action_dim: int = ACTION_DIM,
        history_len: int = 8,
        chunk_len: int = 20,
        d_model: int = 128,
        nhead: int = 4,
        encoder_layers: int = 3,
        decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if history_len < 1 or chunk_len < 1:
            raise ValueError("history_len and chunk_len must be positive.")
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}.")

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.history_len = history_len
        self.chunk_len = chunk_len
        self.d_model = d_model
        self.nhead = nhead
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.observation_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.observation_position = nn.Parameter(torch.zeros(1, history_len, d_model))
        self.action_queries = nn.Parameter(torch.zeros(1, chunk_len, d_model))
        nn.init.normal_(self.observation_position, std=0.02)
        nn.init.normal_(self.action_queries, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers, norm=nn.LayerNorm(d_model))
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, observation_history: torch.Tensor) -> torch.Tensor:
        if observation_history.ndim != 3:
            raise ValueError(
                f"Expected observation history [batch, history, features], got {tuple(observation_history.shape)}."
            )
        if observation_history.shape[1:] != (self.history_len, self.input_dim):
            raise ValueError(
                f"Expected history shape [batch, {self.history_len}, {self.input_dim}], "
                f"got {tuple(observation_history.shape)}."
            )

        memory = self.observation_projection(observation_history) + self.observation_position
        memory = self.encoder(memory)
        queries = self.action_queries.expand(observation_history.shape[0], -1, -1)
        decoded = self.decoder(queries, memory)
        raw = self.action_head(decoded)
        arm = torch.tanh(raw[..., :ARM_ACTION_DIM])
        grip = torch.sigmoid(raw[..., GRIP_ACTION_INDEX : GRIP_ACTION_INDEX + 1])
        return torch.cat((arm, grip), dim=-1)

    def export_config(self) -> dict[str, int | float]:
        return {
            "input_dim": self.input_dim,
            "action_dim": self.action_dim,
            "history_len": self.history_len,
            "chunk_len": self.chunk_len,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
        }

    @classmethod
    def from_config(cls, config: dict) -> "WalkerS2StateACT":
        return cls(
            input_dim=int(config["input_dim"]),
            action_dim=int(config["action_dim"]),
            history_len=int(config["history_len"]),
            chunk_len=int(config["chunk_len"]),
            d_model=int(config["d_model"]),
            nhead=int(config["nhead"]),
            encoder_layers=int(config["encoder_layers"]),
            decoder_layers=int(config["decoder_layers"]),
            dim_feedforward=int(config["dim_feedforward"]),
            dropout=float(config.get("dropout", 0.0)),
        )


def load_walker_s2_act(
    checkpoint_path: Path,
    device: str | torch.device,
) -> tuple[WalkerS2StateACT, torch.Tensor, torch.Tensor, dict]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"ACT checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("feature_contract") != FEATURE_CONTRACT:
        raise ValueError(
            f"Checkpoint contract is {checkpoint.get('feature_contract')!r}; expected {FEATURE_CONTRACT!r}."
        )
    if int(checkpoint.get("environment_obs_dim", -1)) != ENV_OBS_DIM:
        raise ValueError(
            f"Checkpoint observation dimension is {checkpoint.get('environment_obs_dim')}; expected {ENV_OBS_DIM}."
        )
    if int(checkpoint.get("action_dim", -1)) != ACTION_DIM:
        raise ValueError(f"Checkpoint action dimension is {checkpoint.get('action_dim')}; expected {ACTION_DIM}.")

    model = WalkerS2StateACT.from_config(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    mean = checkpoint["obs_mean"].to(device=device, dtype=torch.float32)
    std = checkpoint["obs_std"].to(device=device, dtype=torch.float32)
    return model, mean, std, checkpoint
