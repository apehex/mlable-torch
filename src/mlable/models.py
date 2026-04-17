import gc

import torch.cuda
import torch.nn

# GENERIC ######################################################################

def freeze(model: torch.nn.Module) -> None:
    """Disable gradients for all the parameters of a given model."""
    for __p in model.parameters():
        __p.requires_grad_(False)

# MEMORY #######################################################################

def free_memory(
    model: object=None
) -> None:
    # drop references
    if model is not None:
        del model
    # run garbage collection
    gc.collect()
    # free CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
