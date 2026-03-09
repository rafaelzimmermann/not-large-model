"""Vocabulary for modular addition: tokens are digit strings plus "="."""

SEQ_LEN = 4  # [a, b, =, result] — fixed regardless of modulus


class Vocab:
    """Token set for a given modulus. Tokens: ["0", ..., "mod-1", "="]."""

    def __init__(self, mod: int) -> None:
        self.mod = mod
        self.tokens = [str(i) for i in range(mod)] + ["="]
        self.vocab_size = len(self.tokens)
        self._tok2idx = {t: i for i, t in enumerate(self.tokens)}
        self._idx2tok = {i: t for i, t in enumerate(self.tokens)}
        self.eq_idx = self._tok2idx["="]

    def encode(self, tokens: list[str]) -> list[int]:
        return [self._tok2idx[t] for t in tokens]

    def decode(self, indices: list[int]) -> list[str]:
        return [self._idx2tok[i] for i in indices]
