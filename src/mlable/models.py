import torch.nn

# GENERIC ######################################################################

def freeze(model: torch.nn.Module) -> None:
    """Disable gradients for all the parameters of a given model."""
    for __p in model.parameters():
        __p.requires_grad_(False)
