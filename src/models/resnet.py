
def create_model(name: str, num_classes: int, pretrained: bool = False):
    try:
        import torch.nn as nn
        from torchvision import models
    except ImportError as exc:
        raise SystemExit(
            "Model creation requires torch and torchvision. Install the base dependencies with "
            "`pip install -r requirements.txt`, then install the PyTorch stack with "
            "`pip install -r requirements-cu124.txt`."
        ) from exc

    if name != "resnet18":
        raise ValueError(f"Unsupported model: {name}")

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
