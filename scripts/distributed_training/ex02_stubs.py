"""
Exercise 2 stubs — fill in the three functions below.

Run the tests with:
    torchrun --standalone --nnodes=1 --nproc_per_node=2 test_ex02.py
"""
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader


def make_loader(rank: int, world_size: int, batch_size: int) -> DataLoader:
    """Return a DataLoader for this rank's partition of the dataset.

    The full dataset has 100 samples: x ~ N(0, 1) of shape (4,), y in {0, 1}.
    Split it evenly across ranks (rank r gets indices [r*50, (r+1)*50)).
    Use the same underlying TensorDataset on every rank.

    Args:
        rank:       This process's rank.
        world_size: Total number of ranks (2 in these exercises).
        batch_size: Batch size for the DataLoader.

    Returns:
        A DataLoader that yields (x, y) batches from this rank's slice.

    Hint: torch.utils.data.Subset(dataset, indices) gives you a subset of a dataset.
    """
    raise NotImplementedError


def train_one_step_ddp(
    model: nn.Module,
    loader: DataLoader,
    rank: int,
    device: torch.device,
) -> float:
    """Wrap model in DDP, take one optimizer step on the first batch, return the loss.

    Steps:
        1. Wrap model in DistributedDataParallel.
        2. Create an SGD optimizer (lr=0.01).
        3. Get one batch from loader.
        4. Forward pass → F.cross_entropy loss.
        5. loss.backward()
        6. optimizer.step()
        7. Return float(loss.item()).

    Args:
        model:  A 2-layer MLP, already on `device`. DO NOT recreate it.
        loader: DataLoader from make_loader.
        rank:   This process's rank.
        device: The CUDA device for this rank.

    Returns:
        The cross-entropy loss before the parameter update, as a Python float.

    Hint: DistributedDataParallel(model, device_ids=[rank])
    """
    raise NotImplementedError


def params_are_equal_across_ranks(
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
) -> bool:
    """Return True if all ranks hold identical parameters.

    Check the first parameter tensor: all_gather it from all ranks and
    compare each rank's copy to rank 0's copy.

    Args:
        model:      The model after training (DDP wrapper already removed by caller).
        rank:       This process's rank.
        world_size: Total number of ranks.
        device:     The CUDA device for this rank.

    Returns:
        True if all ranks have the same first parameter; False otherwise.

    Hint: list(model.parameters())[0] gives the first parameter tensor.
          Use dist.all_gather to collect copies from all ranks.
    """
    raise NotImplementedError
