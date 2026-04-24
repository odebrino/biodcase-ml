
def require_torch():
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise SystemExit("Install the PyTorch stack described in the README.") from exc
    return torch, functional


class FocalLoss:
    def __init__(self, gamma: float = 2.0, weight=None) -> None:
        self.gamma = gamma
        self.weight = weight

    def __call__(self, logits, targets):
        torch, functional = require_torch()
        ce = functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.weight is not None:
            # Class weights scale the final per-sample loss; they do not change pt.
            loss = loss * self.weight.to(logits.device)[targets]
        return loss.mean()


def make_loss(loss_cfg: dict, class_weights=None):
    torch, _ = require_torch()
    name = loss_cfg.get("name", "cross_entropy")
    if name == "cross_entropy":
        return torch.nn.CrossEntropyLoss(weight=class_weights)
    if name == "focal":
        return FocalLoss(gamma=float(loss_cfg.get("gamma", 2.0)), weight=class_weights)
    raise ValueError(f"Unknown loss: {name}")
