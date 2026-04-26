import os
from typing import BinaryIO, Iterable
import regex as re

PAT_RE = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes,
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

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
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _split_text_by_spec_tokens_rec(
    text: str, sorted_spec_tokens: list[str], spec_token_i: int
):

    if spec_token_i == len(sorted_spec_tokens):
        return [text]

    spec_token_to_split = sorted_spec_tokens[spec_token_i]
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
        res.extend(
            _split_text_by_spec_tokens_rec(
                out_text, sorted_spec_tokens, spec_token_i + 1
            )
        )

    return res


def _split_text_by_spec_tokens(text: str, spec_tokens: list):
    sorted_spec_tokens = list(sorted(spec_tokens, key=len, reverse=True))
    return _split_text_by_spec_tokens_rec(text, sorted_spec_tokens, 0)


def _pretokenize_chunk(chunk: str, spec_tokens: list[str]) -> Iterable[str]:
    for part in _split_text_by_spec_tokens(chunk, spec_tokens):
        for match in PAT_RE.finditer(part):
            yield match.group()


def pretokenize_text(
    f: BinaryIO, spec_tokens: list[str], num_processes: int
) -> Iterable[str]:
    boundaries = _find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)

        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        for word in _pretokenize_chunk(chunk, spec_tokens):
            yield word
