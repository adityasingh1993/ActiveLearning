import torch
import torch.nn as nn
import copy
from typing import Tuple


def enable_dropout(model: nn.Module) -> None:
    """Enable only dropout layers while keeping normalization layers in eval mode (H-1 fix)."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


class EMATeacher(nn.Module):
    """Exponential Moving Average teacher model with stochastic MC Dropout support."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.shadow = copy.deepcopy(model)
        for param in self.shadow.parameters():
            param.detach_()

    @torch.no_grad()
    def update(self, model: nn.Module, decay: float = 0.999):
        for shadow_param, model_param in zip(self.shadow.parameters(), model.parameters()):
            shadow_param.data = decay * shadow_param.data + (1.0 - decay) * model_param.data

    def forward(self, *args, **kwargs):
        self.shadow.eval()
        with torch.no_grad():
            return self.shadow(*args, **kwargs)

    def forward_mc_dropout(self, x: torch.Tensor, num_passes: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform Monte Carlo dropout passes through shadow model (H-1 fix).

        Returns:
            mean_probs: Average predicted probabilities [B, C, D, H, W]
            uncertainty_map: Epistemic uncertainty map (variance of probabilities) [B, 1, D, H, W]
        """
        self.shadow.eval()
        enable_dropout(self.shadow)

        mc_preds = []
        with torch.no_grad():
            for _ in range(num_passes):
                out = self.shadow(x)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                elif out.ndim == 6:
                    out = out[:, 0]

                probs = torch.sigmoid(out) if out.shape[1] == 1 else torch.softmax(out, dim=1)
                mc_preds.append(probs)

        # Stack passes: [T, B, C, D, H, W]
        mc_preds_stacked = torch.stack(mc_preds, dim=0)

        # Mean predicted probability across passes
        mean_probs = mc_preds_stacked.mean(dim=0)

        # Variance across passes as epistemic uncertainty map
        uncertainty_map = mc_preds_stacked.var(dim=0).mean(dim=1, keepdim=True)

        return mean_probs, uncertainty_map

    def state_dict(self, *args, **kwargs):
        return self.shadow.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return self.shadow.load_state_dict(state_dict, *args, **kwargs)
