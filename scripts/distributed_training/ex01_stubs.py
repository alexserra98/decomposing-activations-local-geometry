"""
Exercise 1 stubs — fill in the four functions below.

Run the tests with:
    torchrun --standalone --nnodes=1 --nproc_per_node=2 test_ex01.py
"""
import torch
import torch.distributed as dist


def share_from_rank0(value: float, rank: int, device: torch.device) -> torch.Tensor:
    """Create a 1-element tensor on rank 0 with `value`, then broadcast to all ranks.

    Args:
        value:  The float to share (only meaningful on rank 0).
        rank:   This process's rank.
        device: The CUDA device for this rank.

    Returns:
        A 1-element tensor on `device` holding `value` on every rank.
    """
    raise NotImplementedError


def distributed_sum(local_value: float, rank: int, device: torch.device) -> float:
    """Return the sum of all ranks' local_value.

    Both ranks must return the same result.

    Args:
        local_value: This rank's contribution to the sum.
        rank:        This process's rank.
        device:      The CUDA device for this rank.

    Returns:
        The sum of local_value across all ranks, as a Python float.
    """
    raise NotImplementedError


def distributed_max(local_value: float, rank: int, device: torch.device) -> float:
    """Return the maximum of all ranks' local_value.

    Both ranks must return the same result.

    Args:
        local_value: This rank's value.
        rank:        This process's rank.
        device:      The CUDA device for this rank.

    Returns:
        The maximum of local_value across all ranks, as a Python float.
    """
    raise NotImplementedError


def gather_all(
    local_tensor: torch.Tensor,
    world_size: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Collect local_tensor from every rank so all ranks have the full list.

    Args:
        local_tensor: This rank's tensor. All ranks must pass a tensor of the same shape.
        world_size:   Total number of ranks.
        device:       The CUDA device for this rank.

    Returns:
        A list of `world_size` tensors, where result[r] is rank r's local_tensor.
        Every rank returns the same list.
    """
    raise NotImplementedError
