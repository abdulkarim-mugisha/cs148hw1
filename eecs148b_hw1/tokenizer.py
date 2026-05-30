from __future__ import annotations

import os
import re
from collections import Counter
from typing import Iterable, Iterator

import regex

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# ---------------------------------------------------------------------------
# BPE Training
# ---------------------------------------------------------------------------

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer on the text at input_path.

    Returns:
        vocab:  dict[int, bytes] mapping token ID → token bytes
        merges: ordered list of (bytes, bytes) merge pairs
    """
    with open(input_path, encoding="utf-8") as f:
        corpus = f.read()

    # Initial vocabulary: 256 single-byte entries
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []

    # Add special tokens to vocabulary before training
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    # Split on special tokens so no merges cross document boundaries
    if special_tokens:
        special_pat = "|".join(re.escape(tok) for tok in special_tokens)
        chunks = re.split(special_pat, corpus)
    else:
        chunks = [corpus]

    # Pre-tokenize each chunk and count unique symbol sequences
    pretokens: dict[tuple[bytes, ...], int] = {}
    for chunk in chunks:
        for match in regex.finditer(PAT, chunk):
            word_bytes = match.group(0).encode("utf-8")
            symbols = tuple(bytes([b]) for b in word_bytes)
            pretokens[symbols] = pretokens.get(symbols, 0) + 1

    # Build pair counts and an inverted index: pair → set of pretokens containing it.
    # The index lets us update only the affected pretokens on each merge step instead
    # of scanning the entire pretoken table.
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    pair_index: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}

    for symbols, count in pretokens.items():
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + count
            if pair not in pair_index:
                pair_index[pair] = set()
            pair_index[pair].add(symbols)

    while len(vocab) < vocab_size and pair_counts:
        # Most frequent pair; ties broken lexicographically (max over the pair tuple)
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merges.append(best)
        merged = best[0] + best[1]
        vocab[len(vocab)] = merged

        # Process only the pretokens that actually contain the best pair
        affected = list(pair_index.get(best, ()))
        pair_index[best] = set()  # consumed — clear the index entry

        for old_sym in affected:
            count = pretokens.get(old_sym)
            if count is None:
                continue  # already removed via a collision in a prior iteration

            # Build the new symbol sequence after applying the merge
            new_sym_list: list[bytes] = []
            i = 0
            while i < len(old_sym):
                if (
                    i + 1 < len(old_sym)
                    and old_sym[i] == best[0]
                    and old_sym[i + 1] == best[1]
                ):
                    new_sym_list.append(merged)
                    i += 2
                else:
                    new_sym_list.append(old_sym[i])
                    i += 1
            new_sym = tuple(new_sym_list)

            # Remove every pair that old_sym contributed to pair_counts / pair_index
            for j in range(len(old_sym) - 1):
                p = (old_sym[j], old_sym[j + 1])
                cur = pair_counts.get(p, 0) - count
                if cur > 0:
                    pair_counts[p] = cur
                else:
                    pair_counts.pop(p, None)
                if p in pair_index:
                    pair_index[p].discard(old_sym)

            # Add every pair that new_sym contributes
            for j in range(len(new_sym) - 1):
                p = (new_sym[j], new_sym[j + 1])
                pair_counts[p] = pair_counts.get(p, 0) + count
                if p not in pair_index:
                    pair_index[p] = set()
                pair_index[p].add(new_sym)

            # Update pretoken table (handle the rare collision case where new_sym exists)
            del pretokens[old_sym]
            pretokens[new_sym] = pretokens.get(new_sym, 0) + count

    return vocab, merges


# ---------------------------------------------------------------------------
# Tokenizer class
# ---------------------------------------------------------------------------

class Tokenizer:
    """Byte-level BPE tokenizer that encodes/decodes text using a trained vocabulary."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab: dict[int, bytes] = dict(vocab)
        self.merges = merges
        self.special_tokens: list[str] = list(special_tokens) if special_tokens else []

        # Reverse map: bytes → int (built from vocab)
        self._bytes_to_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}

        # Add special tokens to vocab if they aren't already present
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self._bytes_to_id:
                new_id = max(self.vocab.keys()) + 1
                self.vocab[new_id] = token_bytes
                self._bytes_to_id[token_bytes] = new_id

        # Merge rank lookup: (bytes, bytes) → rank (lower = higher priority)
        self._merge_ranks: dict[tuple[bytes, bytes], int] = {
            merge: rank for rank, merge in enumerate(merges)
        }

        # Compiled pattern for splitting on special tokens (longest match first)
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            escaped = "|".join(re.escape(tok) for tok in sorted_specials)
            # Capturing group so re.split keeps the matched separators
            self._special_split_pattern = f"({escaped})"
        else:
            self._special_split_pattern = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """Construct a Tokenizer from serialized vocab and merges files.

        Vocab file: JSON mapping string token ID → list of byte values.
        Merges file: one merge per line as two space-separated latin-1 encoded tokens.
        """
        import json

        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab = json.load(f)

        vocab: dict[int, bytes] = {}
        for k, v in raw_vocab.items():
            if isinstance(v, list):
                vocab[int(k)] = bytes(v)
            elif isinstance(v, str):
                vocab[int(k)] = v.encode("latin-1")
            else:
                vocab[int(k)] = bytes(v)

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split(" ")
                if len(parts) == 2:
                    merges.append(
                        (parts[0].encode("latin-1"), parts[1].encode("latin-1"))
                    )

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode text into a list of token IDs."""
        if not text:
            return []

        ids: list[int] = []

        if self._special_split_pattern is not None:
            # re.split with a capturing group returns:
            #   [non-special, special, non-special, special, ..., non-special]
            # Parts at odd indices are matched special tokens.
            parts = re.split(self._special_split_pattern, text)
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    # Special token — map directly to its ID
                    ids.append(self._bytes_to_id[part.encode("utf-8")])
                else:
                    ids.extend(self._encode_chunk(part))
        else:
            ids.extend(self._encode_chunk(text))

        return ids

    def _encode_chunk(self, text: str) -> list[int]:
        """Encode a plain-text chunk (containing no special tokens) into IDs."""
        if not text:
            return []

        ids: list[int] = []
        for match in regex.finditer(PAT, text):
            word_bytes = match.group(0).encode("utf-8")
            symbols: list[bytes] = [bytes([b]) for b in word_bytes]
            symbols = self._apply_merges(symbols)
            for s in symbols:
                ids.append(self._bytes_to_id[s])
        return ids

    def _apply_merges(self, symbols: list[bytes]) -> list[bytes]:
        """Greedily apply BPE merges (lowest rank first) until no more apply."""
        while len(symbols) > 1:
            # Find the pair with the lowest merge rank (highest priority)
            best_rank: int | None = None
            best_idx = -1
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self._merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i

            if best_idx == -1:
                break  # No applicable merge found

            merged = symbols[best_idx] + symbols[best_idx + 1]
            symbols = symbols[:best_idx] + [merged] + symbols[best_idx + 2:]

        return symbols

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode strings from an iterable, yielding token IDs one at a time.

        This avoids materialising the full token sequence in memory, making it
        suitable for large inputs (e.g., a file handle iterated line by line).
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs back into a Unicode string."""
        token_bytes = b"".join(self.vocab[i] for i in ids)
        return token_bytes.decode("utf-8", errors="replace")
