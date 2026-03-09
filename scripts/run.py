"""Entrypoint: train then evaluate."""

import argparse

from not_large_model.vocab import Vocab
from not_large_model.dataset import split_dataset
from not_large_model.train import train
from not_large_model.eval import evaluate, plot_accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny transformer on modular addition.")
    parser.add_argument("--mod", type=int, default=5, help="Modulus for the addition group, e.g. 5 gives Z₅ with 25 pairs, 11 gives Z₁₁ with 121 pairs (default: 5)")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs, or max cap when --until-grok is set (default: 5000)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--weight-decay", type=float, default=1.0, help="AdamW weight decay (default: 1.0)")
    parser.add_argument("--train-frac", type=float, default=0.5, help="Fraction of pairs used for training (default: 0.5)")
    parser.add_argument("--until-grok", action="store_true", help="Keep training until test accuracy reaches 100%% (uses --epochs as a safety cap)")
    parser.add_argument("--save", type=str, default="model.pt", help="Path to save the trained model (default: model.pt)")
    args = parser.parse_args()

    vocab = Vocab(args.mod)
    (train_inputs, train_targets), (test_inputs, test_targets) = split_dataset(vocab, train_frac=args.train_frac)
    n_train, n_test = len(train_inputs), len(test_inputs)
    total = vocab.mod ** 2
    print(f"Z{args.mod}: {total} total pairs — {n_train} train / {n_test} test\n")

    print("=== Training ===")
    model, history = train(
        train_inputs, train_targets, test_inputs, test_targets,
        vocab_size=vocab.vocab_size,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        until_grok=args.until_grok,
    )

    print(f"\n=== Evaluation (all {total} pairs) ===")
    evaluate(model, vocab)

    model.save(args.save, mod=args.mod)
    print(f"Model saved to {args.save}")

    plot_accuracy(history, mod=args.mod)


if __name__ == "__main__":
    main()
