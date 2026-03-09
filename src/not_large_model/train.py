"""Training loop with AdamW, prints loss every 100 epochs."""

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from not_large_model.model import TinyTransformer

History = list[tuple[int, float, float]]  # (epoch, train_acc, test_acc)


def _accuracy(model: TinyTransformer, inputs: Tensor, device: torch.device) -> float:
    """Fraction of sequences where the result token (position 2) is predicted correctly."""
    model.eval()
    with torch.no_grad():
        logits = model(inputs.to(device))       # (N, 4, vocab_size)
        preds = logits.argmax(dim=-1)           # (N, 4)
    correct = (preds[:, 2] == inputs[:, 3].to(device)).sum().item()
    model.train()
    return correct / len(inputs)


def train(
    train_inputs: Tensor,
    train_targets: Tensor,
    test_inputs: Tensor,
    test_targets: Tensor,
    vocab_size: int = 6,
    d_model: int = 128,
    epochs: int = 5000,
    lr: float = 3e-4,
    weight_decay: float = 1.0,
    batch_size: int | None = None,
    until_grok: bool = False,
    eval_every: int | None = None,
) -> tuple[TinyTransformer, History]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyTransformer(vocab_size=vocab_size, d_model=d_model).to(device)
    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    effective_batch = batch_size if batch_size is not None else len(train_inputs)
    loader = DataLoader(
        TensorDataset(train_inputs, train_targets),
        batch_size=effective_batch,
        shuffle=True,
    )

    # Auto-scale eval frequency: ~200 checkpoints over the full run, min every 50 epochs.
    if eval_every is None:
        eval_every = max(50, epochs // 200)

    history: History = []
    loss = torch.tensor(0.0)

    bar = tqdm(total=epochs, unit="epoch", dynamic_ncols=True)
    model.train()
    for epoch in range(1, epochs + 1):
        for batch_inputs, batch_targets in loader:
            logits = model(batch_inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), batch_targets.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        bar.update(1)

        if epoch % eval_every == 0:
            train_acc = _accuracy(model, train_inputs, device)
            test_acc = _accuracy(model, test_inputs, device)
            history.append((epoch, train_acc, test_acc))
            bar.set_postfix(loss=f"{loss.item():.4f}", train=f"{train_acc:.0%}", test=f"{test_acc:.0%}")

            if until_grok and test_acc == 1.0:
                print(f"\nGrokked at epoch {epoch}!")
                break
    else:
        if until_grok:
            print(f"\nReached epoch limit ({epochs}) without fully grokking.")

    bar.close()
    return model, history
