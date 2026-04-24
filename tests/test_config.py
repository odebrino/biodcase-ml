from pathlib import Path

from src.utils.config import load_config


def test_load_config_extends_baseline():
    config = load_config(Path("legacy/cnn/configs/resnet18.yaml"))
    assert config["model"]["name"] == "resnet18"
    assert config["training"]["batch_size"] == 32
    assert "bpd" in config["classes"]


def test_nitro_config_targets_cuda():
    config = load_config(Path("legacy/cnn/configs/nitro4060.yaml"))
    assert config["training"]["device"] == "cuda"
    assert config["training"]["mixed_precision"] is True
    assert config["cache"]["enabled"] is True


def test_aplose_512_98_preset_expands_audio_settings():
    config = load_config(Path("configs/aplose_512_98.yaml"))

    assert config["audio"]["preset"] == "aplose_512_98"
    assert config["audio"]["n_fft"] == 512
    assert config["audio"]["win_length"] == 256
    assert config["audio"]["hop_length"] == 5
    assert config["audio"]["overlap_percent"] == 98
    assert config["audio"]["margin_seconds"] == 1.0


def test_aplose_256_90_preset_expands_audio_settings():
    config = load_config(Path("configs/aplose_256_90.yaml"))

    assert config["audio"]["preset"] == "aplose_256_90"
    assert config["audio"]["n_fft"] == 256
    assert config["audio"]["win_length"] == 256
    assert config["audio"]["hop_length"] == 26
    assert config["audio"]["overlap_percent"] == 90


def test_knn_configs_do_not_inherit_cnn_model_or_training_sections():
    for path in [Path("configs/knn_submission.yaml"), Path("configs/knn_search.yaml")]:
        config = load_config(path)
        assert "model" not in config
        assert "training" not in config
        assert config["submission"]["feature_set"]
        assert config["audio"]["preset"] == "aplose_512_98"
