"""Evaluates all mod² combinations, prints accuracy table, and plots accuracy vs epochs."""

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from not_large_model.dataset import make_dataset
from not_large_model.model import TinyTransformer
from not_large_model.train import History
from not_large_model.vocab import Vocab


def evaluate(model: TinyTransformer, vocab: Vocab) -> float:
    """Print next-token prediction accuracy on all mod² pairs."""
    device = next(model.parameters()).device
    inputs, _ = make_dataset(vocab)
    inputs = inputs.to(device)
    n = len(inputs)

    model.eval()
    with torch.no_grad():
        logits = model(inputs)              # (n, 4, vocab_size)
        preds = logits.argmax(dim=-1)       # (n, 4)

    correct = 0
    print(f"\n{'Input':>12}  {'Pred':>4}  {'Expected':>8}  OK?")
    print("-" * 38)
    for i in range(n):
        seq = inputs[i].tolist()
        a_tok, b_tok, _eq, _res = vocab.decode(seq)
        pred_result = vocab.decode([preds[i, 2].item()])[0]
        expected = vocab.decode([seq[3]])[0]
        ok = pred_result == expected
        correct += int(ok)
        print(f"{a_tok} {b_tok} = {pred_result:>4}  (expected {expected})  {'✓' if ok else '✗'}")

    accuracy = correct / n
    print(f"\nAccuracy: {correct}/{n} = {accuracy:.0%}")
    return accuracy


def plot_accuracy(history: History, mod: int, path: str = "assets/accuracy.png") -> None:
    """Plot train vs test accuracy over epochs and save to *path*."""
    epochs = [h[0] for h in history]
    train_accs = [h[1] * 100 for h in history]
    test_accs = [h[2] * 100 for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_accs, label="memorize (train)", linewidth=2)
    ax.plot(epochs, test_accs, label="generalize (test)", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Accuracy vs Epochs — modular addition on Z{mod}")
    ax.set_ylim(-5, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nPlot saved to {path}")
