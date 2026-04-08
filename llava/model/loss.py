from typing import List, Union

import torch
from torch.nn.functional import cross_entropy

from llava.constants import IGNORE_INDEX

__all__ = ["soft_cross_entropy"]


def reweight_cross_entropy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    sample_weight: torch.FloatTensor,
    ignore_index: int = IGNORE_INDEX
) -> torch.Tensor:
    # Remove last token from outputs and first token from targets
    outputs = outputs[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    device, dtype = outputs.device, outputs.dtype
    sample_weight = sample_weight.to(device=device, dtype=dtype)
    # Remove outputs and targets with ignore_index
    tok_loss = 0
    total_wtokens = 0
    try:
        # import ipdb; ipdb.set_trace()
        for i in range(len(outputs)):
            indices = targets[i] != ignore_index
            Mi = int(indices.sum())
            tok_loss_i =cross_entropy(outputs[i][indices], targets[i][indices], reduction="none")  # [Mi]
            tok_loss += tok_loss_i.sum() * sample_weight[i]
            total_wtokens += Mi * sample_weight[i] # each token is equal, but if stop label is shorter, it will less important
            # total_wsamples += sample_weight[i]  # each sample is equal
    except:
        # import ipdb; ipdb.set_trace()
        print('outputs num = ',len(outputs), 'target num = ',len(targets) ,'sample_weight num = ',len(sample_weight))

    return tok_loss /  total_wtokens.clamp_min(1.0) # return tok_loss / total_wsamples.clamp_min(1.0)


def soft_cross_entropy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    soft_tokens: Union[torch.Tensor, List[int]],
    std: float = 1,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    # Remove last token from outputs and first token from targets
    outputs = outputs[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()

    # Flatten outputs and targets
    targets = targets.view(-1)
    outputs = outputs.view(targets.size(0), -1)

    # Remove outputs and targets with ignore_index
    indices = targets != ignore_index
    outputs = outputs[indices]
    targets = targets[indices]

    # Convert soft token IDs to tensor
    if isinstance(soft_tokens, list):
        soft_tokens = torch.tensor(soft_tokens).to(targets)

    # Calculate loss for non-soft tokens
    indices = torch.isin(targets, soft_tokens, invert=True)
    loss = cross_entropy(outputs[indices], targets[indices], reduction="sum")

    # Calculate loss for soft tokens
    indices = torch.isin(targets, soft_tokens)
    targets_indices = torch.zeros_like(outputs[indices])
    for k, target in enumerate(targets[indices]):
        dist = torch.exp(-((target - soft_tokens) ** 2) / (2 * std**2))
        targets_indices[k][soft_tokens] = dist / dist.sum()
    loss += cross_entropy(outputs[indices], targets_indices, reduction="sum")

    # Return average loss
    return loss / targets.size(0)
