import torch
import torch.nn as nn
import copy

class EMATeacher(nn.Module):
    """Exponential Moving Average teacher model."""
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
            
    def state_dict(self, *args, **kwargs):
        return self.shadow.state_dict(*args, **kwargs)
        
    def load_state_dict(self, state_dict, *args, **kwargs):
        return self.shadow.load_state_dict(state_dict, *args, **kwargs)
