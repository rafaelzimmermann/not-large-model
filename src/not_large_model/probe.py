"""Fourier probe of token embeddings.

After grokking, token embeddings encode each number k as a point on a
circle: some embedding dimensions follow cos(2πkf/mod) and others
sin(2πkf/mod) for a small set of frequencies f. This script makes that
visible by computing the Fourier power spectrum of the embeddings and
plotting the dominant wave patterns.
"""

import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from not_large_model.model import TinyTransformer
from not_large_model.vocab import Vocab


def main() -> None:
    parser = argparse.ArgumentParser(description="Fourier probe of token embeddings.")
    parser.add_argument("--model", type=str, default="model.pt", help="Path to saved model (default: model.pt)")
    parser.add_argument("--out", type=str, default="assets/probe.png", help="Output image path (default: assets/probe.png)")
    args = parser.parse_args()

    model, mod = TinyTransformer.load(args.model)
    vocab = Vocab(mod)
    model.eval()

    with torch.no_grad():
        emb = model.tok_emb.weight[:mod].cpu().numpy()  # (mod, d_model) — digits only, no "="

    tokens = np.arange(mod)

    # --- Fourier power spectrum ---
    # FFT across the token dimension for every embedding dimension,
    # then sum squared magnitudes over all embedding dims.
    fft = np.fft.rfft(emb, axis=0)               # (mod//2+1, d_model)
    power = (np.abs(fft) ** 2).sum(axis=1)        # (mod//2+1,)
    freqs = np.arange(len(power))

    # --- Wave patterns for the top-2 frequencies ---
    top2_freqs = np.argsort(power[1:])[-2:][::-1] + 1  # skip DC (freq=0)

    # For each top frequency, find the embedding dim most aligned with
    # cos and sin at that frequency (highest absolute FFT coefficient).
    wave_plots = []
    for f in top2_freqs:
        cos_ref = np.cos(2 * np.pi * f * tokens / mod)
        sin_ref = np.sin(2 * np.pi * f * tokens / mod)
        cos_scores = np.abs(emb.T @ cos_ref)
        sin_scores = np.abs(emb.T @ sin_ref)
        cos_dim = cos_scores.argmax()
        sin_dim = sin_scores.argmax()
        wave_plots.append((f, cos_ref, sin_ref, emb[:, cos_dim], emb[:, sin_dim], cos_dim, sin_dim))

    # --- Embedding heatmap (top-16 dims by variance) ---
    top_dims = np.argsort(emb.var(axis=0))[-16:][::-1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: Fourier power spectrum
    ax = axes[0]
    bars = ax.bar(freqs, power, color=["tab:orange" if f in top2_freqs else "tab:blue" for f in freqs])
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Power")
    ax.set_title(f"Fourier power of token embeddings (Z{mod})")
    ax.set_xticks(freqs)

    # Panel 2: wave patterns for top-2 frequencies
    ax = axes[1]
    colors = ["tab:orange", "tab:green"]
    for i, (f, cos_ref, sin_ref, cos_emb, sin_emb, cos_dim, sin_dim) in enumerate(wave_plots):
        # Normalise to [-1, 1] for comparison
        def norm(v): return v / (np.abs(v).max() + 1e-8)
        c = colors[i]
        ax.plot(tokens, norm(cos_ref), "--", color=c, alpha=0.4, label=f"cos(2π·{f}·k/{mod})")
        ax.plot(tokens, norm(cos_emb), "o-", color=c, label=f"emb dim {cos_dim} (f={f})")
    ax.set_xlabel("Token k")
    ax.set_ylabel("Value (normalised)")
    ax.set_title("Dominant Fourier modes in embeddings")
    ax.set_xticks(tokens)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 3: embedding heatmap
    ax = axes[2]
    im = ax.imshow(emb[:, top_dims].T, aspect="auto", cmap="RdBu", interpolation="nearest")
    ax.set_xlabel("Token k")
    ax.set_ylabel("Embedding dim (top 16 by variance)")
    ax.set_title("Token embedding values\n(sinusoidal rows = Fourier features)")
    ax.set_xticks(tokens)
    fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"Fourier probe — Z{mod} after grokking", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
