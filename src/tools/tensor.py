import torch
from torch import Tensor

def unpad_by_mask(padded_tensor: Tensor, padding_value: int):
    """
    Remove padding values from each sequence in the batch
    """
    mask = padded_tensor != padding_value
    return torch.stack([seq[mask[i]] for i, seq in enumerate(padded_tensor)])