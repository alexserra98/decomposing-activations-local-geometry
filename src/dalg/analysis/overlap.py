from src.annotations import Array, _N_JOBS
from src.error import MetricComputationError, DataRetrievalError
import logging
from dadapy.data import Data
import tqdm
import numpy as np
from joblib import Parallel, delayed
from functools import partial
import logging
from jaxtyping import Float, Int, Bool
from typing import Tuple, List, Dict
import torch

def compute_knn_euclidian(X, X_new, k):
    """
    Compute the k-nearest neighbors distances and indices using euclidian distance with PyTorch.

    Parameters:
    - X (torch.Tensor): Reference tensor of shape (N, D)
    - X_new (torch.Tensor): Query tensor of shape (M, D)
    - k (int): Number of nearest neighbors to find

    Returns:
    - distances_k (torch.Tensor): Tensor of shape (M, k) containing cosine distances to the k-nearest neighbors
    - indices_k (torch.Tensor): Tensor of shape (M, k) containing indices of the k-nearest neighbors
    """
    X = X.float()
    X_new = X_new.float()
    
    # Compute euclidian distance
    euclidian_dist = torch.cdist(X_new, X, p=2)  # Shape: (M, N)

    # Find the k smallest distances and their indices
    distances_k, indices_k = torch.topk(euclidian_dist, k=k, dim=1, largest=False)

    return distances_k.cpu().float().numpy(), indices_k.cpu().numpy()

def return_data_overlap(
        self,
        input_i: Float[Array, "num_layers num_instances d_model"] |
            Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
        input_j:  Float[Array, "num_layers num_instances d_model"] |
            Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
        k: Int
    ) -> Float[Array, "num_layers"]:
    """
    Process a single layer
    Inputs:
        layer: Int
        input_i: Float[Array, "num_layers, num_instances, model_dim"]
        input_j: Float[Array, "num_layers, num_instances, model_dim"]
        k: Int
            the number of neighbours considered for the overlap
    Returns:
        Array
    """

    if isinstance(input_i, tuple):
        mat_dist_i, mat_coord_i = input_i
        data = Data(distances=(mat_dist_i, mat_coord_i), maxk=k)
        mat_dist_j, mat_coord_j = input_j
        overlap = data.return_data_overlap(distances=(mat_dist_j,
                                                        mat_coord_j), k=k)
        return overlap
    elif isinstance(input_i, np.ndarray):
        data = Data(coordinates=input_i, maxk=k)
        overlap = data.return_data_overlap(input_j, k=k)
        return overlap


def return_label_overlap(
        self, 
        tensors: Float[Array, "num_layers num_instances d_model"] |
        Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
        labels: Float[Int, "num_instances"],
        k: Int,
) -> Float[Array, "num_layers"]:
    """
    Inputs:
        tensors: Float[Array, "num_layers num_instances d_model"] |
        Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
            It can either receive the hidden states or the distance matrices
        labels: Float[Int, "num_instances"]
        k: Int
            the number of neighbours considered for the overlap
    Returns:
        Float[Array, "num_layers"]
    """
    tensors = tensors[layer]
    try:
        # do clustering
        if isinstance(tensors, tuple):
            mat_dist, mat_coord = tensors
            data = Data(distances=(mat_dist, mat_coord), maxk=k)
            overlap = data.return_label_overlap(labels, k=k)
            return overlap
        elif isinstance(tensors, np.ndarray):
            # do clustering
            data = Data(coordinates=tensors, maxk=k)
            overlap = data.return_label_overlap(labels, k=k)
            return overlap
    except Exception as e:
        raise MetricComputationError(f"Error raised by Dadapy: {e}")


