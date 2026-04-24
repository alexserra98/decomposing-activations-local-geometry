from __future__ import annotations

from typing import List, Tuple, Optional, Union
from collections import Counter

import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer, utils

from dalg.data.concept_dataset import ConceptDataset, SupervisedConceptDataset


# Maps mode names to hook-string factories (layer -> cache key).
_HOOK_FN = {
    "mlp":          lambda L: f"blocks.{L}.mlp.hook_post",
    "mlp_out":      lambda L: f"blocks.{L}.hook_mlp_out",
    "residual":     lambda L: utils.get_act_name("resid_post", L),
    "residual_pre": lambda L: utils.get_act_name("resid_pre", L),
    "attn_out":     lambda L: f"blocks.{L}.hook_attn_out",
}


class ActivationGenerator:
    """Extracts token-level activations from a HookedTransformer model."""

    def __init__(
        self,
        model_name: str,
        model_device: str = "cpu",
        data_device: str = "cpu",
        mode: str = "residual",
    ):
        """
        Args:
            model_name: TransformerLens model name (e.g. "gpt2-small").
            model_device: Device for the language model forward passes.
            data_device: Device for the output activation tensors.
            mode: Which activation stream to extract. Supported values:
                  "mlp", "mlp_out", "residual", "residual_pre", "attn_out".
        """
        if mode not in _HOOK_FN:
            raise ValueError(f"Unknown mode '{mode}'. Choose from {sorted(_HOOK_FN)}")
        self.model = HookedTransformer.from_pretrained(model_name, device=model_device)
        self.data_device = data_device
        self.mode = mode

    def _hook_str(self, layer: int) -> str:
        """Return the TransformerLens cache key for the chosen stream at a given layer."""
        return _HOOK_FN[self.mode](layer)

    def _tokenize_dataset(self, dataset, batch_size: int) -> List[torch.Tensor]:
        """Tokenize every batch in the dataset; returns a list of (B, T) token tensors."""
        return [
            self.model.to_tokens(batch["prompt"])
            for batch in dataset.get_batches(batch_size=batch_size)
        ]

    def build_vocab_frequency(self, dataset, batch_size: int = 5) -> Counter:
        """
        Count how often each token appears across the dataset (padding excluded).

        Args:
            dataset: Dataset with a get_batches method.
            batch_size: Number of prompts per tokenization call.

        Returns:
            Counter mapping token_id -> total occurrences across the dataset.
        """
        pad_id = self.model.tokenizer.pad_token_id
        counter: Counter = Counter()
        for input_ids in tqdm(self._tokenize_dataset(dataset, batch_size), desc="Building vocab frequency"):
            for tok in input_ids.flatten().tolist():
                if tok != pad_id:
                    counter[tok] += 1
        return counter

    @torch.no_grad()
    def generate_activations(
        self,
        dataset: Union[ConceptDataset, SupervisedConceptDataset],
        layers: List[int],
        batch_size: int = 5,
        stack: bool = False,
    ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], torch.Tensor]:
        """
        Extract activations for every non-padding, non-BOS token in the dataset.

        Args:
            dataset: Dataset yielding batches with at least a "prompt" key.
            layers: Layer indices from which to extract activations.
            batch_size: Number of prompts per forward pass.
            stack: If False (default), return a list with one tensor per layer,
                   each of shape (num_tokens, d_model). If True, concatenate all
                   layers along the feature dimension and return a single tensor
                   of shape (num_tokens, len(layers) * d_model).

        Returns:
            A tuple (activations, freq) where:
              - activations: list of (num_tokens, d_model) tensors (one per layer),
                             or a single (num_tokens, len(layers)*d_model) tensor
                             when stack=True.
              - freq: (num_tokens,) int tensor; each entry is the global frequency
                      of that token in the dataset (useful for inverse-frequency
                      weighted sampling during initialization).
        """
        vocab_freq = self.build_vocab_frequency(dataset, batch_size=batch_size)
        pad_id = self.model.tokenizer.pad_token_id
        bos_id = self.model.tokenizer.bos_token_id

        per_layer: List[List[torch.Tensor]] = [[] for _ in layers]
        all_token_ids: List[torch.Tensor] = []

        for input_ids in tqdm(self._tokenize_dataset(dataset, batch_size), desc="Generating activations"):
            input_ids = input_ids.to(self.data_device)
            _, cache = self.model.run_with_cache(input_ids)

            mask = (input_ids != pad_id) & (input_ids != bos_id)
            all_token_ids.append(input_ids[mask].cpu())

            for idx, layer in enumerate(layers):
                acts = cache[self._hook_str(layer)].detach().to(self.data_device)
                per_layer[idx].append(acts[mask].cpu())

            del cache
            torch.cuda.empty_cache()

        self.model.reset_hooks()
        layer_tensors = [torch.cat(parts, dim=0) for parts in per_layer]
        token_ids_all = torch.cat(all_token_ids, dim=0)
        freq = torch.tensor([vocab_freq[t.item()] for t in token_ids_all])

        if stack:
            return torch.cat(layer_tensors, dim=1), freq
        return layer_tensors, freq

    @torch.no_grad()
    def generate_period_activations(
        self,
        dataset: Union[ConceptDataset, SupervisedConceptDataset],
        layers: List[int],
        batch_size: int = 5,
    ) -> List[torch.Tensor]:
        """
        Extract activations at sentence-ending period tokens (".").

        Args:
            dataset: Dataset yielding batches with at least a "prompt" key.
            layers: Layer indices from which to extract activations.
            batch_size: Number of prompts per forward pass.

        Returns:
            List of tensors (one per layer), each of shape (num_period_tokens, d_model).
        """
        period_id = self.model.tokenizer.encode(".")[0]
        per_layer: List[List[torch.Tensor]] = [[] for _ in layers]

        for input_ids in tqdm(self._tokenize_dataset(dataset, batch_size), desc="Generating period activations"):
            input_ids = input_ids.to(self.data_device)
            _, cache = self.model.run_with_cache(input_ids)
            mask = input_ids == period_id

            for idx, layer in enumerate(layers):
                acts = cache[self._hook_str(layer)].detach().to(self.data_device)
                per_layer[idx].append(acts[mask].cpu())

            del cache
            torch.cuda.empty_cache()

        return [torch.cat(parts, dim=0) for parts in per_layer]


def extract_token_ids(
    dataset,
    act_generator: ActivationGenerator,
    batch_size: int = 5,
) -> Tuple[torch.Tensor, List[int], Optional[List]]:
    """
    Tokenize the dataset and return token IDs, sample IDs, and per-token labels —
    without running the language model forward pass.

    Each token at position j in sample i receives sample index i (0-based). Padding
    and BOS tokens are excluded.

    Args:
        dataset: Dataset with a get_batches method returning dicts with at least
                 "prompt". If batches also contain a "label" key (e.g.
                 SupervisedConceptDataset), per-token labels are returned.
        act_generator: Provides the tokenizer and data_device.
        batch_size: Number of prompts per tokenization call.

    Returns:
        token_ids:  (num_tokens,) LongTensor of token IDs for every non-pad/BOS position.
        sample_ids: List[int] of length num_tokens; the 0-based dataset index per token.
        labels:     Per-token label list (each sample's label repeated for all its tokens)
                    if the dataset provides "label" batches; otherwise None.
    """
    model = act_generator.model
    pad_id = model.tokenizer.pad_token_id
    bos_id = model.tokenizer.bos_token_id

    all_token_ids: List[torch.Tensor] = []
    sample_ids: List[int] = []
    labels: Optional[List] = None
    sample_idx = 0

    for batch in tqdm(dataset.get_batches(batch_size=batch_size), desc="Extracting token IDs"):
        input_ids = model.to_tokens(batch["prompt"], padding_side="left").to(act_generator.data_device)
        mask = (input_ids != pad_id) & (input_ids != bos_id)
        counts = mask.sum(dim=1).tolist()

        batch_labels = batch.get("label")
        if batch_labels is not None:
            if labels is None:
                labels = []
            for n, lab in zip(counts, batch_labels):
                n = int(n)
                labels.extend([lab] * n)
                sample_ids.extend([sample_idx] * n)
                sample_idx += 1
        else:
            for n in counts:
                sample_ids.extend([sample_idx] * int(n))
                sample_idx += 1

        all_token_ids.append(input_ids[mask].cpu())

    return torch.cat(all_token_ids, dim=0), sample_ids, labels
