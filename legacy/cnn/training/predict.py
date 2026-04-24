import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from legacy.cnn.models.resnet import create_model
from src.data.spectrogram import event_tensor_from_waveform, read_waveform
from src.utils.config import load_config


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Install the PyTorch stack described in the README.") from exc
    return torch


def load_checkpoint(checkpoint_path: Path, config: dict, device):
    torch = require_torch()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_config = checkpoint.get("args") or config
    class_names = checkpoint.get("class_names", saved_config["classes"])
    model = create_model(saved_config["model"]["name"], len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names, saved_config


def predict_tensor(model, class_names: list[str], tensor, device) -> dict:
    torch = require_torch()
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor.unsqueeze(0).to(device)), dim=1).squeeze(0).cpu()
    pred_idx = int(probabilities.argmax().item())
    return {
        "predicted_label": class_names[pred_idx],
        "predicted_idx": pred_idx,
        "probabilities": {
            class_name: float(probabilities[idx].item()) for idx, class_name in enumerate(class_names)
        },
    }


def image_tensor(image_path: Path, img_size: int):
    torch = require_torch()
    image = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def predict_image(checkpoint_path: Path, config: dict, image_path: Path) -> dict:
    torch = require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, saved_config = load_checkpoint(checkpoint_path, config, device)
    tensor = image_tensor(image_path, int(saved_config["model"]["img_size"]))
    result = predict_tensor(model, class_names, tensor, device)
    result["image_path"] = str(image_path)
    return result


def predict_audio(
    checkpoint_path: Path,
    config: dict,
    audio_path: Path,
    start_seconds: float,
    end_seconds: float,
    low_frequency: float,
    high_frequency: float,
) -> dict:
    torch = require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, saved_config = load_checkpoint(checkpoint_path, config, device)
    waveform, sample_rate = read_waveform(audio_path)
    row = {
        "audio_path": str(audio_path),
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "clip_start_seconds": start_seconds,
        "clip_end_seconds": end_seconds,
        "low_frequency": low_frequency,
        "high_frequency": high_frequency,
    }
    tensor = event_tensor_from_waveform(
        waveform,
        sample_rate,
        row,
        saved_config["audio"],
        int(saved_config["model"]["img_size"]),
    )
    result = predict_tensor(model, class_names, tensor, device)
    result.update(
        {
            "audio_path": str(audio_path),
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "low_frequency": low_frequency,
            "high_frequency": high_frequency,
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a processed image or one event from a WAV file.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="legacy/cnn/configs/nitro4060.yaml")
    parser.add_argument("--image", default=None)
    parser.add_argument("--audio", default=None)
    parser.add_argument("--start-seconds", type=float, default=None)
    parser.add_argument("--end-seconds", type=float, default=None)
    parser.add_argument("--low-frequency", type=float, default=0.0)
    parser.add_argument("--high-frequency", type=float, default=125.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.image:
        result = predict_image(Path(args.checkpoint), config, Path(args.image))
    elif args.audio and args.start_seconds is not None and args.end_seconds is not None:
        result = predict_audio(
            Path(args.checkpoint),
            config,
            Path(args.audio),
            args.start_seconds,
            args.end_seconds,
            args.low_frequency,
            args.high_frequency,
        )
    else:
        raise SystemExit("Use either --image or --audio with --start-seconds and --end-seconds.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
