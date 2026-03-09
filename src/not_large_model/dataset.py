"""Dataset: all mod² pairs (a, b), result = (a + b) % mod."""

import torch
from torch import Tensor

from not_large_model.vocab import Vocab


def make_dataset(vocab: Vocab) -> tuple[Tensor, Tensor]:
    """Return (inputs, targets) tensors of shape (mod², 4).

    inputs:  token indices for [a, b, =, result]
    targets: next-token prediction targets
             position 0: predict b given a
             position 1: predict = given a, b
             position 2: predict result given a, b, =
             position 3: -100 (ignored)
    """
    mod = vocab.mod
    seqs = []
    for a in range(mod):
        for b in range(mod):
            result = (a + b) % mod
            seqs.append(vocab.encode([str(a), str(b), "=", str(result)]))

    inputs = torch.tensor(seqs, dtype=torch.long)       # (mod², 4)
    targets = torch.full_like(inputs, fill_value=-100)
    targets[:, 2] = inputs[:, 3]                        # only predict result at pos 2
    return inputs, targets


def split_dataset(
    vocab: Vocab,
    train_frac: float = 0.5,
    seed: int = 42,
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    """Split all mod² pairs into train and test sets.

    Returns ((train_inputs, train_targets), (test_inputs, test_targets)).
    """
    inputs, targets = make_dataset(vocab)
    n = len(inputs)
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng)
    n_train = int(n * train_frac)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    return (inputs[train_idx], targets[train_idx]), (inputs[test_idx], targets[test_idx])
