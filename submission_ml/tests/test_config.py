from pathlib import Path

from src.utils.config import load_config


def test_load_config_extends_baseline():
    config = load_config(Path("configs/resnet18.yaml"))
    assert config["model"]["name"] == "resnet18"
    assert config["training"]["batch_size"] == 32
    assert "bpd" in config["classes"]


def test_nitro_config_targets_cuda():
    config = load_config(Path("configs/nitro4060.yaml"))
    assert config["training"]["device"] == "cuda"
    assert config["training"]["mixed_precision"] is True
    assert config["cache"]["enabled"] is True
