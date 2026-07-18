"""Unit tests for ModelConfig."""

from pathlib import Path

import pytest

from kamui.model.config import ModelConfig


def test_valid_config_creation() -> None:
    """Verifies that ModelConfig can be instantiated with valid parameters and defaults."""
    config = ModelConfig()
    assert config.n_layers == 6
    assert config.d_model == 256
    assert config.n_heads == 8
    assert config.d_ff == 1024
    assert config.vocab_size == 8192
    assert config.context_length == 256
    assert config.positional_encoding == "learned"
    assert config.dropout == 0.0


def test_invalid_layer_count() -> None:
    """Verifies that invalid layer count values raise ValueError."""
    with pytest.raises(ValueError, match="n_layers must be > 0"):
        ModelConfig(n_layers=0)
    with pytest.raises(ValueError, match="n_layers must be > 0"):
        ModelConfig(n_layers=-5)


def test_invalid_d_model() -> None:
    """Verifies that invalid d_model values raise ValueError."""
    with pytest.raises(ValueError, match="d_model must be > 0"):
        ModelConfig(d_model=0)
    with pytest.raises(ValueError, match="d_model must be > 0"):
        ModelConfig(d_model=-128)


def test_invalid_n_heads() -> None:
    """Verifies that invalid n_heads values raise ValueError."""
    with pytest.raises(ValueError, match="n_heads must be > 0"):
        ModelConfig(n_heads=0)
    with pytest.raises(ValueError, match="n_heads must be > 0"):
        ModelConfig(n_heads=-4)


def test_invalid_d_ff() -> None:
    """Verifies that invalid d_ff values raise ValueError."""
    with pytest.raises(ValueError, match="d_ff must be > 0"):
        ModelConfig(d_ff=0)
    with pytest.raises(ValueError, match="d_ff must be > 0"):
        ModelConfig(d_ff=-512)


def test_invalid_vocab_size() -> None:
    """Verifies that invalid vocab_size values raise ValueError."""
    with pytest.raises(ValueError, match="vocab_size must be > 0"):
        ModelConfig(vocab_size=0)
    with pytest.raises(ValueError, match="vocab_size must be > 0"):
        ModelConfig(vocab_size=-100)


def test_invalid_context_length() -> None:
    """Verifies that invalid context_length values raise ValueError."""
    with pytest.raises(ValueError, match="context_length must be > 0"):
        ModelConfig(context_length=0)
    with pytest.raises(ValueError, match="context_length must be > 0"):
        ModelConfig(context_length=-16)


def test_invalid_head_dimensions() -> None:
    """Verifies that d_model not divisible by n_heads raises ValueError."""
    with pytest.raises(ValueError, match="must be divisible by n_heads"):
        ModelConfig(d_model=128, n_heads=5)


def test_valid_context_length_non_power_of_two() -> None:
    """Verifies that non-power-of-two context lengths are now valid."""
    config = ModelConfig(context_length=300)
    assert config.context_length == 300
    config_another = ModelConfig(context_length=15)
    assert config_another.context_length == 15


def test_invalid_dropout() -> None:
    """Verifies that dropout values outside [0, 1] raise ValueError."""
    with pytest.raises(ValueError, match="dropout must be in"):
        ModelConfig(dropout=-0.1)
    with pytest.raises(ValueError, match="dropout must be in"):
        ModelConfig(dropout=1.1)


def test_invalid_positional_encoding() -> None:
    """Verifies that invalid positional encoding strings raise ValueError."""
    with pytest.raises(ValueError, match="positional_encoding must be"):
        ModelConfig(positional_encoding="rotary")


def test_computed_properties() -> None:
    """Verifies all computed properties return correct values."""
    config = ModelConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        d_ff=256,
        vocab_size=1000,
        context_length=32,
        positional_encoding="learned",
        dropout=0.1,
    )
    assert config.d_head == 16

    # 2 layers * 4 projections/layer * 64 * 65 = 33280
    assert config.attention_parameters == 2 * 4 * 64 * 65

    # 2 layers * (2 * 64 * 256 + 64 + 256) = 2 * (32768 + 320) = 66176
    assert config.feedforward_parameters == 2 * (2 * 64 * 256 + 64 + 256)

    # vocab (1000 * 64) + context (32 * 64) = 64000 + 2048 = 66048
    assert config.embedding_parameters == (1000 * 64) + (32 * 64)

    # layer norms: 2 * 4 * 64 + 2 * 64 = 512 + 128 = 640
    # unembed bias is 0 (bias is False)
    # total: 66048 + 33280 + 66176 + 640 = 166144
    assert config.estimated_total_parameters == 166144


def test_sinusoidal_embedding_parameters() -> None:
    """Verifies that sinusoidal positional encoding returns 0 parameters for position."""
    config = ModelConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        d_ff=256,
        vocab_size=1000,
        context_length=32,
        positional_encoding="sinusoidal",
        dropout=0.1,
    )
    assert config.embedding_parameters == 1000 * 64


def test_repr() -> None:
    """Verifies that __repr__ returns a formatted string containing key fields."""
    config = ModelConfig()
    rep = repr(config)
    assert "ModelConfig" in rep
    assert "n_layers=6" in rep
    assert "d_model=256" in rep
    assert "estimated_total_parameters=" in rep


def test_yaml_roundtrip(tmp_path: Path) -> None:
    """Verifies that ModelConfig can be saved to and loaded from YAML correctly."""
    config = ModelConfig(
        n_layers=4,
        d_model=128,
        n_heads=4,
        d_ff=512,
        vocab_size=4096,
        context_length=128,
        positional_encoding="sinusoidal",
        dropout=0.2,
    )

    # Save using Path object
    yaml_path_obj = tmp_path / "config.yaml"
    config.to_yaml(yaml_path_obj)

    # Load using Path object
    loaded_config_obj = ModelConfig.from_yaml(yaml_path_obj)
    assert config == loaded_config_obj

    # Save using string path
    yaml_path_str = str(tmp_path / "config_str.yaml")
    config.to_yaml(yaml_path_str)

    # Load using string path
    loaded_config_str = ModelConfig.from_yaml(yaml_path_str)
    assert config == loaded_config_str


def test_yaml_from_existing_configs() -> None:
    """Verifies loading ModelConfig from existing project configurations."""
    project_root = Path(__file__).parents[2]
    config_paths = [
        project_root / "configs" / "nano.yaml",
        project_root / "configs" / "small.yaml",
        project_root / "configs" / "medium.yaml",
        project_root / "configs" / "gpt2_compatible.yaml",
    ]
    for path in config_paths:
        assert path.exists()
        config = ModelConfig.from_yaml(path)
        assert isinstance(config, ModelConfig)


def test_yaml_validation_errors(tmp_path: Path) -> None:
    """Verifies that invalid YAML structures raise appropriate ValueErrors."""
    import yaml

    # 1. YAML is not a dictionary mapping
    invalid_path = tmp_path / "invalid_list.yaml"
    with open(invalid_path, "w", encoding="utf-8") as f:
        yaml.safe_dump([1, 2, 3], f)

    with pytest.raises(ValueError, match="must contain a dictionary mapping"):
        ModelConfig.from_yaml(invalid_path)

    # 2. 'model' key exists but is not a dictionary mapping
    invalid_model_path = tmp_path / "invalid_model.yaml"
    with open(invalid_model_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"model": "not-a-dict"}, f)

    with pytest.raises(
        ValueError, match="Model configuration section in YAML must be a dictionary"
    ):
        ModelConfig.from_yaml(invalid_model_path)


def test_yaml_unknown_keys(tmp_path: Path) -> None:
    """Verifies that unknown configuration keys in the model section raise ValueError."""
    import yaml

    yaml_path = tmp_path / "unknown_keys.yaml"
    data = {"model": {"n_layers": 4, "d_model": 128, "unknown_key": "some-value"}}
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)

    with pytest.raises(ValueError, match="Unknown configuration key"):
        ModelConfig.from_yaml(yaml_path)


def test_yaml_malformed(tmp_path: Path) -> None:
    """Verifies that malformed YAML files raise a ValueError."""
    yaml_path = tmp_path / "malformed.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("model:\n  n_layers: 4\n  d_model: [invalid_list")

    with pytest.raises(ValueError, match="Failed to parse YAML"):
        ModelConfig.from_yaml(yaml_path)


def test_yaml_preserve_non_model_sections(tmp_path: Path) -> None:
    """Verifies that to_yaml does not destroy non-model sections in existing files."""
    import yaml

    yaml_path = tmp_path / "unified.yaml"
    data = {
        "model": {
            "n_layers": 4,
            "d_model": 128,
            "n_heads": 4,
            "d_ff": 512,
            "vocab_size": 4096,
            "context_length": 128,
            "positional_encoding": "learned",
            "dropout": 0.0,
        },
        "training": {"batch_size": 16, "learning_rate": 3e-4},
        "data": {"dataset": "tinystories"},
        "logging": {"output_dir": "checkpoints/nano"},
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)

    # Load, modify, and save
    config = ModelConfig.from_yaml(yaml_path)
    config.n_layers = 10
    config.to_yaml(yaml_path)

    # Read back raw data to check preservation
    with open(yaml_path, encoding="utf-8") as f:
        saved_data = yaml.safe_load(f)

    assert saved_data["model"]["n_layers"] == 10
    assert saved_data["model"]["d_model"] == 128
    assert saved_data["training"]["batch_size"] == 16
    assert saved_data["training"]["learning_rate"] == 3e-4
    assert saved_data["data"]["dataset"] == "tinystories"
    assert saved_data["logging"]["output_dir"] == "checkpoints/nano"


def test_yaml_invalid_model_block_structure(tmp_path: Path) -> None:
    """Verifies that a non-dictionary 'model' section raises a ValueError."""
    import yaml

    yaml_path = tmp_path / "invalid_model_block.yaml"
    data = {"model": [1, 2, 3]}
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)

    with pytest.raises(
        ValueError, match="Model configuration section in YAML must be a dictionary"
    ):
        ModelConfig.from_yaml(yaml_path)


def test_yaml_missing_model_section(tmp_path: Path) -> None:
    """Verifies that a missing 'model' section raises a ValueError."""
    import yaml

    yaml_path = tmp_path / "missing_model.yaml"
    data = {"training": {"batch_size": 16}}
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)

    with pytest.raises(ValueError, match="YAML configuration must contain a 'model' section"):
        ModelConfig.from_yaml(yaml_path)


@pytest.mark.parametrize(
    "config_name",
    ["nano.yaml", "small.yaml", "medium.yaml", "gpt2_compatible.yaml"],
)
def test_shipped_configs_are_valid(config_name: str) -> None:
    """Every YAML config shipped in configs/ must load into a valid ModelConfig."""
    config_path = Path(__file__).parents[2] / "configs" / config_name
    config = ModelConfig.from_yaml(config_path)
    assert config.d_model % config.n_heads == 0
    assert config.vocab_size > 0


def test_to_yaml_with_existing_malformed_file(tmp_path: Path) -> None:
    """Verifies that to_yaml behaves correctly when the existing file is malformed."""
    config = ModelConfig()
    yaml_path = tmp_path / "malformed_existing.yaml"

    # Create a malformed file
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("[invalid_yaml")

    # to_yaml should handle the exception and successfully write the config
    config.to_yaml(yaml_path)

    # Read back to verify
    loaded = ModelConfig.from_yaml(yaml_path)
    assert loaded == config
