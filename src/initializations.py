import torch
from typing import Callable

TIME_STEP_FLOOR = 0.001
TIME_STEP_MAX   = 0.1
INV_SOFTPLUS_EPSILON = 1e-8

def linear_decay(t_max, t_min, L, idx):
    """
    Linear decay from t_min to t_max across layers
    """
    frac = (11 - idx) / (L - 1) if L > 1 else 0.0
    t = t_max + (t_min - t_max) * frac 
    return t

def linear_decay_reverse(t_max, t_min, L, idx):
    """
    Linear decay from t_max to t_min across layers
    """
    frac = idx / (L - 1) if L > 1 else 0.0
    t = t_max + (t_min - t_max) * frac 
    return t

def uniform(t_max, t_min):
    """
    Uniform distribution between t_min and t_max
    """
    dist = torch.distributions.Uniform(t_min, t_max)
    t = dist.sample()
    return t


def inverse_softplus(y: torch.Tensor, eps=INV_SOFTPLUS_EPSILON) -> torch.Tensor:
    """
    Inverse of softplus: log(exp(y) - 1), stabilized.
    """
    y_safe = torch.clamp(y, min=eps)
    return torch.log(torch.clamp(torch.exp(y_safe) - 1.0, min=eps))


def initialize_dt_bias(
    model: torch.nn.Module,
    t_max: float = TIME_STEP_MAX,
    t_min: float = TIME_STEP_FLOOR,
    init_fn_type: str = 'uniform',
):
    """
    Reinitialize each layer's dt_bias so that init_fn is used to compute the dt_bias values for each layer

    Args:
        model: your nn.Module with model.backbone.layers[i].mixer.dt_bias
        t_max: starting value (for layer 0)
        t_min: ending value (for last layer)
        init_fn: function to compute the dt_bias values for each layer
    """
    if init_fn_type == 'uniform':
        init_fn = uniform
    elif init_fn_type == 'linear_decay':
        init_fn = linear_decay
    elif init_fn_type == 'linear_decay_reverse':
        init_fn = linear_decay_reverse
    infos = []
    device = None
    if not (hasattr(model, 'backbone') and hasattr(model.backbone, 'layers')):
        raise RuntimeError("Expected model.backbone.layers to exist")
    for idx, layer in enumerate(model.backbone.layers):
        mixer = getattr(layer, 'mixer', None)
        if mixer is None: continue
        dtb = getattr(mixer, 'dt_bias', None)
        if not isinstance(dtb, torch.nn.Parameter): continue
        if device is None:
            device = dtb.device
        elif dtb.device != device:
            raise RuntimeError(f"Mixed devices for dt_bias; saw {dtb.device} vs {device}")
        infos.append((dtb, idx, dtb.numel()))

    L = len(infos)
    if L == 0:
        raise RuntimeError("No dt_bias parameters found!")

    with torch.no_grad():
        for param, idx, num_heads in infos:
            dt_bias_value = init_fn(t_max, t_min, L, idx)
            targets = torch.full((num_heads,), dt_bias_value, device=device)
            raw = inverse_softplus(targets)
            param.data.copy_(raw.view(param.shape).to(param.dtype))
