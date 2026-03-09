import pytest
from not_large_model.vocab import Vocab


def test_vocab_size():
    assert Vocab(5).vocab_size == 6   # 0-4 + "="
    assert Vocab(11).vocab_size == 12  # 0-10 + "="


def test_encode_decode_roundtrip():
    vocab = Vocab(5)
    tokens = ["0", "3", "=", "3"]
    assert vocab.decode(vocab.encode(tokens)) == tokens


def test_eq_token_is_last():
    vocab = Vocab(5)
    assert vocab.eq_idx == 5


def test_unknown_token_raises():
    vocab = Vocab(5)
    with pytest.raises(KeyError):
        vocab.encode(["5"])  # out of range for Z5
