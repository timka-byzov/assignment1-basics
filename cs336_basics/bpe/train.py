from collections import defaultdict
from cs336_basics.bpe.common import (
    FreqTable,
    UTF_8_BYTES,
    ByteDictionary,
    BytePair,
    ByteWord,
)


def _build_initial_dict(spec_tokens: list[bytes]) -> ByteDictionary:
    spec_tokens_count = len(spec_tokens)

    initial_dict = {i: spec_tokens[i] for i in range(spec_tokens_count)}
    initial_dict.update({spec_tokens_count + i: bytes([i]) for i in range(UTF_8_BYTES)})

    return initial_dict


def _merge_word_tokens(word_tokens: ByteWord, merge_pair: BytePair) -> list[bytes]:

    new_word_tokens = []
    i = 0
    while i < len(word_tokens) - 1:
        if word_tokens[i] == merge_pair[0] and word_tokens[i + 1] == merge_pair[1]:
            new_word_tokens.append(merge_pair[0] + merge_pair[1])
            i += 1
        else:
            new_word_tokens.append(word_tokens[i])
        i += 1

    if i == len(word_tokens) - 1:
        new_word_tokens.append(word_tokens[-1])

    return new_word_tokens


def _merge_iteration(
    pretokenized_text: list[ByteWord],
) -> tuple[bool, list[ByteWord], BytePair]:

    iteration_frequencies: FreqTable = defaultdict(int)
    most_freq_pair: BytePair | None = None
    for word_tokens in pretokenized_text:
        for i in range(len(word_tokens) - 1):
            curr_pair = word_tokens[i], word_tokens[i + 1]

            iteration_frequencies[curr_pair] += 1

            if (
                most_freq_pair is None
                or iteration_frequencies[curr_pair]
                > iteration_frequencies[most_freq_pair]
                or iteration_frequencies[curr_pair]
                == iteration_frequencies[most_freq_pair]
                and curr_pair > most_freq_pair  # compare tuples of bytes
            ):
                most_freq_pair = curr_pair

    if most_freq_pair is None:  # full words joined or all pairs has spec tokens
        return False, [], (bytes(), bytes())

    new_pretokenized_text = []
    for word_tokens in pretokenized_text:
        new_pretokenized_text.append(_merge_word_tokens(word_tokens, most_freq_pair))

    return True, new_pretokenized_text, (most_freq_pair[0], most_freq_pair[1])


def train(
    pretokenized_text: list[ByteWord],
    spec_tokens: list[bytes],
    vocab_size: int,
    num_threads=4,
) -> tuple[ByteDictionary, list[BytePair]]:

    assert vocab_size >= len(spec_tokens) + UTF_8_BYTES

    byte_dict = _build_initial_dict(spec_tokens)
    local_text = pretokenized_text
    merges: list[tuple[bytes, bytes]] = []

    while len(byte_dict) < vocab_size:  # TODO: or smth else
        has_merged, text, new_token_pair = _merge_iteration(local_text)
        if not has_merged:
            break

        else:
            local_text = text
            merges.append(new_token_pair)
            byte_dict[len(byte_dict)] = new_token_pair[0] + new_token_pair[1]

    return byte_dict, merges
