from collections import defaultdict
from cs336_basics.bpe.common import (
    FreqTable,
    UTF_8_BYTES,
    ByteVocab,
    BytePair,
    ByteWord,
)


class MostFreqPair:
    def __init__(self) -> None:
        self.value: BytePair | None = None
        self.count = 0

    def try_update(self, new_pair: BytePair, new_count: int) -> bool:

        if (
            self.value is None
            or new_count > self.count
            or new_count == self.count
            and new_pair > self.value  # compare tuples of bytes
        ):
            self.value = new_pair
            self.count = new_count
            return True

        return False


class TokenNode:
    def __init__(self, token: bytes) -> None:
        self.token: bytes = token
        self.prev: TokenNode | None = None
        self.next: TokenNode | None = None


class TrainWord:
    def __init__(self, word: ByteWord) -> None:
        self.head = TokenNode(word[0])
        prev_node = self.head
        for i in range(1, len(word)):
            new_node = TokenNode(word[i])
            prev_node.next = new_node
            new_node.prev = prev_node

            prev_node = new_node

    def merge(self, most_freq_pair: BytePair, frequencis: FreqTable) -> None:
        curr_node = self.head

        while curr_node:
            next_node = curr_node.next
            if next_node:
                if (curr_node.token, next_node.token) == most_freq_pair:
                    if curr_node.prev:
                        key = (curr_node.prev.token, curr_node.token)
                        frequencis[key] -= 1
                        if frequencis[key] == 0:
                            del frequencis[key]
                    if next_node.next:
                        key = (next_node.token, next_node.next.token)
                        frequencis[key] -= 1
                        if frequencis[key] == 0:
                            del frequencis[key]

                    new_node = TokenNode(curr_node.token + next_node.token)
                    if curr_node.prev:
                        new_node.prev = curr_node.prev
                        curr_node.prev.next = new_node
                        frequencis[(new_node.prev.token, new_node.token)] += 1
                    else:
                        self.head = new_node

                    if next_node.next:
                        new_node.next = next_node.next
                        next_node.next.prev = new_node
                        frequencis[(new_node.token, new_node.next.token)] += 1

                    next_node = next_node.next

            curr_node = next_node


def _build_initial_dict(spec_tokens: list[bytes]) -> ByteVocab:
    spec_tokens_count = len(spec_tokens)

    initial_dict = {i: spec_tokens[i] for i in range(spec_tokens_count)}
    initial_dict.update({spec_tokens_count + i: bytes([i]) for i in range(UTF_8_BYTES)})

    return initial_dict


def _build_init_frequencis(
    pretokenized_text: list[ByteWord],
) -> tuple[FreqTable, MostFreqPair]:
    most_freq_pair = MostFreqPair()
    frequencis = defaultdict(int)
    for word_tokens in pretokenized_text:
        for i in range(len(word_tokens) - 1):
            curr_pair = word_tokens[i], word_tokens[i + 1]

            frequencis[curr_pair] += 1
            most_freq_pair.try_update(curr_pair, frequencis[curr_pair])

    return frequencis, most_freq_pair


def _merge_iteration(
    frequencis: FreqTable, most_freq_pair: BytePair, text: list[TrainWord]
) -> MostFreqPair:

    del frequencis[most_freq_pair]
    new_most_freq_pair = MostFreqPair()

    for word in text:
        word.merge(most_freq_pair, frequencis)

    for pair, freq in frequencis.items():
        new_most_freq_pair.try_update(pair, freq)

    return new_most_freq_pair


def train(
    pretokenized_text: list[ByteWord],
    spec_tokens: list[bytes],
    vocab_size: int,
    num_threads=4,
) -> tuple[ByteVocab, list[BytePair]]:

    assert vocab_size >= len(spec_tokens) + UTF_8_BYTES

    train_text = [TrainWord(word) for word in pretokenized_text]

    byte_vocab = _build_initial_dict(spec_tokens)

    frequencis, most_freq_pair = _build_init_frequencis(pretokenized_text)
    merges: list[BytePair] = []

    while len(byte_vocab) < vocab_size and most_freq_pair.value:  # TODO: or smth else

        merges.append(most_freq_pair.value)
        byte_vocab[len(byte_vocab)] = most_freq_pair.value[0] + most_freq_pair.value[1]

        new_most_freq_pair = _merge_iteration(
            frequencis, most_freq_pair.value, train_text
        )
        most_freq_pair = new_most_freq_pair

    return byte_vocab, merges
