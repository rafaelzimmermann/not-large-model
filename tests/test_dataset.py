import torch
from not_large_model.vocab import Vocab
from not_large_model.dataset import make_dataset, split_dataset


def test_make_dataset_size():
    vocab = Vocab(5)
    inputs, targets = make_dataset(vocab)
    assert inputs.shape == (25, 4)
    assert targets.shape == (25, 4)


def test_make_dataset_correctness():
    vocab = Vocab(5)
    inputs, targets = make_dataset(vocab)
    for i in range(len(inputs)):
        a, b, eq, result = inputs[i].tolist()
        assert eq == vocab.eq_idx
        assert result == (a + b) % 5


def test_targets_only_at_position_2():
    vocab = Vocab(5)
    _, targets = make_dataset(vocab)
    assert (targets[:, 0] == -100).all()
    assert (targets[:, 1] == -100).all()
    assert (targets[:, 2] != -100).all()
    assert (targets[:, 3] == -100).all()


def test_split_dataset_sizes():
    vocab = Vocab(11)
    (train_in, train_tgt), (test_in, test_tgt) = split_dataset(vocab, train_frac=0.5)
    assert len(train_in) + len(test_in) == 121
    assert len(train_in) == len(train_tgt)
    assert len(test_in) == len(test_tgt)


def test_split_dataset_no_overlap():
    vocab = Vocab(11)
    (train_in, _), (test_in, _) = split_dataset(vocab, train_frac=0.5)
    train_set = {tuple(r.tolist()) for r in train_in}
    test_set = {tuple(r.tolist()) for r in test_in}
    assert train_set.isdisjoint(test_set)


def test_split_dataset_reproducible():
    vocab = Vocab(5)
    split1 = split_dataset(vocab, seed=42)
    split2 = split_dataset(vocab, seed=42)
    assert torch.equal(split1[0][0], split2[0][0])
