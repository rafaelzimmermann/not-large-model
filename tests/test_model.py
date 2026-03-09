import torch
from not_large_model.model import TinyTransformer


def test_output_shape():
    model = TinyTransformer(vocab_size=6, d_model=32)
    x = torch.randint(0, 6, (4, 4))  # batch=4, seq_len=4
    logits = model(x)
    assert logits.shape == (4, 4, 6)


def test_ffn_default_is_4x_dmodel():
    model = TinyTransformer(vocab_size=6, d_model=32)
    layer = model.transformer.layers[0]
    assert layer.linear1.out_features == 128  # 4 * 32


def test_no_dropout():
    model = TinyTransformer(vocab_size=6, d_model=32)
    layer = model.transformer.layers[0]
    assert layer.dropout.p == 0.0


def test_causal_mask_prevents_future_leakage():
    """Position 0 logits must not change when tokens at positions 1-3 change."""
    model = TinyTransformer(vocab_size=6, d_model=32)
    model.eval()
    x1 = torch.tensor([[0, 1, 5, 2]])
    x2 = torch.tensor([[0, 3, 5, 4]])  # different tokens at positions 1-3
    with torch.no_grad():
        logits1 = model(x1)
        logits2 = model(x2)
    assert torch.allclose(logits1[:, 0, :], logits2[:, 0, :])
