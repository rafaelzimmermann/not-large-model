"""Load a saved model and predict (a + b) % mod."""

import argparse
import torch
from not_large_model.model import TinyTransformer
from not_large_model.vocab import Vocab


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict (a + b) % mod using a saved model.")
    parser.add_argument("a", type=int)
    parser.add_argument("b", type=int)
    parser.add_argument("--model", type=str, default="model.pt", help="Path to saved model (default: model.pt)")
    args = parser.parse_args()

    model, mod = TinyTransformer.load(args.model)
    vocab = Vocab(mod)

    if not (0 <= args.a < mod and 0 <= args.b < mod):
        parser.error(f"Both numbers must be in range [0, {mod - 1}] for this model (Z{mod})")

    model.eval()
    tokens = vocab.encode([str(args.a), str(args.b), "=", str(0)])  # placeholder result
    x = torch.tensor([tokens])
    with torch.no_grad():
        logits = model(x)
    pred = logits[0, 2].argmax().item()
    expected = (args.a + args.b) % mod
    print(f"{args.a} + {args.b} = {vocab.decode([pred])[0]}  (expected {expected}, {'✓' if pred == expected else '✗'})")


if __name__ == "__main__":
    main()
