"""Model configuration dataclass and tensor dimension conventions.

This module defines ``ModelConfig``, the single source of truth for all
transformer hyperparameters. Every other module in ``kamui.model`` imports
its hyperparameters from an instance of this class — never from magic
numbers scattered in code.
"""

from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Union


@dataclass
class ModelConfig:
    """Configuration class for the KAMUI transformer model.

    Holds all hyperparameters, performs validation, computes helper dimensions,
    and estimates the parameter count under explicit architectural assumptions.

    Attributes:
        n_layers: Number of transformer blocks.
        d_model: Residual stream / embedding dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward network hidden layer dimension.
        vocab_size: Size of the token vocabulary.
        context_length: Maximum sequence length.
        positional_encoding: Type of positional encoding ('learned' or 'sinusoidal').
        dropout: Dropout rate (must be in [0, 1]).
    """
    n_layers: int = 6
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1024
    vocab_size: int = 8192
    context_length: int = 256
    positional_encoding: str = "learned"
    dropout: float = 0.0

    def __post_init__(self) -> None:
        """Enforces tensor dimension and hyperparameter sanity checks.

        Raises:
            ValueError: If any validation rule is violated.
        """
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be > 0, got {self.n_layers}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {self.d_model}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be > 0, got {self.n_heads}")
        if self.d_ff <= 0:
            raise ValueError(f"d_ff must be > 0, got {self.d_ff}")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {self.vocab_size}")
        if self.context_length <= 0:
            raise ValueError(f"context_length must be > 0, got {self.context_length}")
        
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads}) "
                f"to yield an integer d_head."
            )

        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")

        if self.positional_encoding not in ("learned", "sinusoidal"):
            raise ValueError(
                f"positional_encoding must be 'learned' or 'sinusoidal', got '{self.positional_encoding}'"
            )

    @property
    def d_head(self) -> int:
        """Dimension of each attention head.

        Calculated as d_model // n_heads.
        """
        return self.d_model // self.n_heads

    @property
    def attention_parameters(self) -> int:
        """Estimates total attention parameters across all layers.

        Assumptions:
            - Multi-head causal self-attention is applied at each of the `n_layers`.
            - In each layer, the input (residual stream of shape d_model) is projected
              to Query, Key, and Value spaces. Each of these 3 projections is a linear
              layer of shape (d_model, d_model) with a bias vector of shape (d_model).
            - The output of the combined heads is projected back to the residual stream
              using an output projection of shape (d_model, d_model) with a bias of shape (d_model).
            - Formula: n_layers * 4 * d_model * (d_model + 1)
        """
        return self.n_layers * 4 * self.d_model * (self.d_model + 1)

    @property
    def feedforward_parameters(self) -> int:
        """Estimates total feed-forward network parameters across all layers.

        Assumptions:
            - A position-wise MLP is applied at each of the `n_layers`.
            - The MLP consists of two linear layers:
              1. An expansion layer: weight matrix of shape (d_model, d_ff) and bias of shape (d_ff).
              2. A projection layer: weight matrix of shape (d_ff, d_model) and bias of shape (d_model).
            - Formula: n_layers * (2 * d_model * d_ff + d_model + d_ff)
        """
        return self.n_layers * (2 * self.d_model * self.d_ff + self.d_model + self.d_ff)

    @property
    def embedding_parameters(self) -> int:
        """Estimates total embedding parameters.

        Assumptions:
            - Token embedding matrix of shape (vocab_size, d_model).
            - If `positional_encoding` is "learned", a learned positional embedding
              matrix of shape (context_length, d_model) is present.
            - If `positional_encoding` is "sinusoidal", positional encodings are fixed
              and add 0 trainable parameters.
            - Formula: (vocab_size * d_model) + (context_length * d_model if learned else 0)
        """
        pos_params = self.context_length * self.d_model if self.positional_encoding == "learned" else 0
        return (self.vocab_size * self.d_model) + pos_params

    @property
    def estimated_total_parameters(self) -> int:
        """Estimates the total trainable parameter count of the model.

        Assumptions:
            - Includes attention, feedforward, and embedding parameters.
            - Includes normalization layer parameters:
              - 2 LayerNorms per block (attention input, FFN input). Each LayerNorm has
                learnable scale (gamma) and bias (beta) parameters of shape (d_model,).
              - 1 final LayerNorm before unembedding, also with scale and bias parameters of shape (d_model,).
              - LayerNorm parameters formula: n_layers * 4 * d_model + 2 * d_model.
            - Includes unembedding layer parameters:
                - The weight matrix is tied to the token embedding matrix transpose, so it
                  adds 0 trainable parameters.
                - The unembedding linear layer has NO learnable bias (bias=False), matching
                  standard GPT-2 weight-tying conventions.
        """
        ln_params = (self.n_layers * 4 * self.d_model) + (2 * self.d_model)
        return (
            self.embedding_parameters
            + self.attention_parameters
            + self.feedforward_parameters
            + ln_params
        )

    def __repr__(self) -> str:
        """Returns a helpful, formatted string representation of the model configuration."""
        return (
            f"ModelConfig(\n"
            f"  n_layers={self.n_layers},\n"
            f"  d_model={self.d_model},\n"
            f"  n_heads={self.n_heads} (d_head={self.d_head}),\n"
            f"  d_ff={self.d_ff},\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  context_length={self.context_length},\n"
            f"  positional_encoding='{self.positional_encoding}',\n"
            f"  dropout={self.dropout},\n"
            f"  estimated_total_parameters={self.estimated_total_parameters:,}\n"
            f")"
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ModelConfig":
        """Loads a ModelConfig from a YAML file.

        The YAML file must contain a 'model' section.

        Args:
            path: Path to the YAML file.

        Returns:
            An instance of ModelConfig.
        """
        import yaml
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse YAML file at {path}: {e}") from e

        if not isinstance(data, dict):
            raise ValueError(f"YAML file at {path} must contain a dictionary mapping.")

        if "model" not in data:
            raise ValueError("YAML configuration must contain a 'model' section.")

        model_data = data["model"]
        if not isinstance(model_data, dict):
            raise ValueError("Model configuration section in YAML must be a dictionary.")

        # Check for unknown configuration keys
        dataclass_fields = {field.name for field in fields(cls)}
        for key in model_data:
            if key not in dataclass_fields:
                raise ValueError(f"Unknown configuration key in 'model' section: '{key}'")

        return cls(**model_data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Serializes the ModelConfig instance to a YAML file.

        If the file already exists, preserves other sections (e.g., 'training',
        'data', 'logging') and only updates the 'model' block.

        Args:
            path: Path where the YAML file should be written.
        """
        import yaml
        
        path_obj = Path(path)
        existing_data = {}
        if path_obj.exists() and path_obj.is_file():
            try:
                with open(path_obj, "r", encoding="utf-8") as f:
                    content = yaml.safe_load(f)
                    if isinstance(content, dict):
                        existing_data = content
            except Exception:
                pass

        existing_data["model"] = asdict(self)
        
        with open(path_obj, "w", encoding="utf-8") as f:
            yaml.safe_dump(existing_data, f, default_flow_style=False, sort_keys=False)
