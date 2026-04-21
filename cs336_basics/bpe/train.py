from collections import defaultdict
from dataclasses import dataclass
from cs336_basics.bpe.common import (
    FrequencyTable,
    UTF_8_BYTES,
    ByteVocab,
    BytePair,
    ByteWord,
    reverse_bytes,
)
import heapq


@dataclass(order=True)
class FrequencyHeapItem:
    count: int
    pair: BytePair


class FrequencyHeap:
    def __init__(self) -> None:
        self.heap: list[FrequencyHeapItem] = []
        heapq.heapify(self.heap)

    def push(self, item: FrequencyHeapItem) -> None:
        reverse_count = -item.count
        reverse_bytes_0 = reverse_bytes(item.pair[0])
        reverse_bytes_1 = reverse_bytes(item.pair[1])

        heapq.heappush(
            self.heap,
            FrequencyHeapItem(reverse_count, (reverse_bytes_0, reverse_bytes_1)),
        )

    def pop(self) -> FrequencyHeapItem:
        item = heapq.heappop(self.heap)
        count = -item.count
        bytes_0 = reverse_bytes(item.pair[0])
        bytes_1 = reverse_bytes(item.pair[1])

        return FrequencyHeapItem(count, (bytes_0, bytes_1))

    def empty(self) -> bool:
        return len(self.heap) == 0


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

    def merge(
        self,
        most_freq_pair: BytePair,
        frequencis: FrequencyTable,
        freq_heap: FrequencyHeap,
    ) -> None:
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

                        key = (new_node.prev.token, new_node.token)
                        frequencis[key] += 1
                        freq_heap.push(FrequencyHeapItem(frequencis[key], key))

                    else:
                        self.head = new_node

                    if next_node.next:
                        new_node.next = next_node.next
                        next_node.next.prev = new_node

                        key = (new_node.token, new_node.next.token)
                        frequencis[key] += 1
                        freq_heap.push(FrequencyHeapItem(frequencis[key], key))

                    next_node = next_node.next

            curr_node = next_node


def _build_initial_dict(spec_tokens: list[bytes]) -> ByteVocab:
    spec_tokens_count = len(spec_tokens)

    initial_dict = {i: spec_tokens[i] for i in range(spec_tokens_count)}
    initial_dict.update({spec_tokens_count + i: bytes([i]) for i in range(UTF_8_BYTES)})

    return initial_dict


def _build_init_frequencis(
    pretokenized_text: list[ByteWord],
) -> tuple[FrequencyTable, FrequencyHeap]:
    frequencis = defaultdict(int)
    for word_tokens in pretokenized_text:
        for i in range(len(word_tokens) - 1):
            curr_pair = word_tokens[i], word_tokens[i + 1]

            frequencis[curr_pair] += 1

    freq_heap = FrequencyHeap()
    for pair, freq in frequencis.items():
        freq_heap.push(FrequencyHeapItem(freq, pair))
    return frequencis, freq_heap


def _merge_iteration(
    frequencis: FrequencyTable,
    freq_heap: FrequencyHeap,
    most_freq_pair: BytePair,
    text: list[TrainWord],
) -> BytePair | None:

    del frequencis[most_freq_pair]

    for word in text:
        word.merge(most_freq_pair, frequencis, freq_heap)

    if freq_heap.empty():
        return None

    item = freq_heap.pop()
    while (
        item.pair not in frequencis or item.count != frequencis[item.pair]
    ) and not freq_heap.empty():
        item = freq_heap.pop()

    if item.pair in frequencis and item.count == frequencis[item.pair]:
        return item.pair


def train(
    pretokenized_text: list[ByteWord],
    spec_tokens: list[bytes],
    vocab_size: int,
    num_threads=4,
) -> tuple[ByteVocab, list[BytePair]]:

    assert vocab_size >= len(spec_tokens) + UTF_8_BYTES

    train_text = [TrainWord(word) for word in pretokenized_text]

    byte_vocab = _build_initial_dict(spec_tokens)

    frequencis, freq_heap = _build_init_frequencis(pretokenized_text)
    most_freq_pair = freq_heap.pop().pair if not freq_heap.empty() else None
    merges: list[BytePair] = []

    while len(byte_vocab) < vocab_size and most_freq_pair:  # TODO: or smth else

        merges.append(most_freq_pair)
        byte_vocab[len(byte_vocab)] = most_freq_pair[0] + most_freq_pair[1]

        new_most_freq_pair = _merge_iteration(
            frequencis, freq_heap, most_freq_pair, train_text
        )
        most_freq_pair = new_most_freq_pair

    return byte_vocab, merges
