import os
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import BinaryIO

import regex as re

PAT_RE = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

type Word = str


class PreTokenizerBase(ABC):
    @property
    @abstractmethod
    def spec_tokens(self) -> list[str]:
        pass

    @abstractmethod
    def pretokenize(self, text: str) -> Iterable[Word]:
        pass

    @abstractmethod
    def pretokenize_file(self, file: BinaryIO) -> Iterable[Word]:
        pass


class PreTokenizer(PreTokenizerBase):
    def __init__(
        self, spec_tokens: list[str], split_spec_token: bytes = b"<|endoftext|>", num_processes: int = 1
    ) -> None:
        self.sorted_spec_tokens: list[str] = sorted(spec_tokens, key=lambda token: len(token), reverse=True)
        self.num_processes = num_processes
        assert isinstance(split_spec_token, bytes), "Must represent special token as a bytestring"
        self.split_spec_token = split_spec_token

    def _find_chunk_boundaries(
        self,
        file: BinaryIO,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """

        # Get total file size in bytes,
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        desired_num_chunks = self.num_processes
        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(self.split_spec_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    def _split_text_by_spec_tokens_rec(self, text: str, spec_token_i: int) -> list[str]:

        if spec_token_i == len(self.sorted_spec_tokens):
            return [text]

        spec_token_to_split = self.sorted_spec_tokens[spec_token_i]
        out_texts: list[str] = []

        spec_token_start = text.find(spec_token_to_split, 0, len(text))

        if spec_token_start == -1:
            out_texts.append(text)
        else:
            i = 0
            while spec_token_start != -1:
                if spec_token_start > i:  # avoid empty substrings
                    out_texts.append(text[i:spec_token_start])
                i = spec_token_start + len(spec_token_to_split)
                spec_token_start = text.find(
                    spec_token_to_split,
                    spec_token_start + len(spec_token_to_split),
                    len(text),
                )
            if i != len(text):
                out_texts.append(text[i:])

        res: list[str] = []
        for out_text in out_texts:
            res.extend(self._split_text_by_spec_tokens_rec(out_text, spec_token_i + 1))

        return res

    def _split_text_by_spec_tokens(self, text: str) -> list[Word]:
        return self._split_text_by_spec_tokens_rec(text, 0)

    def _pretokenize(self, text: str) -> Iterable[Word]:
        for part in self._split_text_by_spec_tokens(text):
            for match in PAT_RE.finditer(part):
                yield match.group()

    def _pretokenize_file(self, file: BinaryIO) -> Iterable[Word]:
        boundaries = self._find_chunk_boundaries(file)
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            file.seek(start)
            chunk = file.read(end - start).decode("utf-8", errors="ignore")
            yield from self._pretokenize(chunk)

    def pretokenize(self, text: str) -> Iterable[Word]:
        return self._pretokenize(text)

    def pretokenize_file(self, file: BinaryIO) -> Iterable[str]:
        return self._pretokenize_file(file)

    @property
    def spec_tokens(self) -> list[str]:
        return self.sorted_spec_tokens
