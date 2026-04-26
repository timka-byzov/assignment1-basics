from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable
from cs336_basics.bpe.common import (
    FrequencyTable,
    UTF_8_BYTES,
    ByteVocab,
    BytePair,
    ByteWord,
)

import heapq


@dataclass(order=False)
class ReversedBytes:
    value: bytes

    def __lt__(self, other: "ReversedBytes") -> bool:
        return self.value > other.value

    def __le__(self, other: "ReversedBytes") -> bool:
        return self.value >= other.value

    def __gt__(self, other: "ReversedBytes") -> bool:
        return self.value < other.value

    def __ge__(self, other: "ReversedBytes") -> bool:
        return self.value <= other.value


@dataclass(order=True)
class FrequencyHeapItem:
    count: int  # negative
    pair: tuple[ReversedBytes, ReversedBytes]


@dataclass(frozen=True)  # иммутабельный, хэшируемый
class FrequencyCounterItem:
    count: int
    value: BytePair


class FrequencyCounter:
    def __init__(self, freq_table: FrequencyTable | None = None) -> None:
        self.heap: list[FrequencyHeapItem] = []
        self.freq_table = freq_table if freq_table is not None else defaultdict(int)

    def _empty(self) -> bool:
        return len(self.heap) == 0

    def _to_heap_item(self, item: FrequencyCounterItem) -> FrequencyHeapItem:
        return FrequencyHeapItem(
            -item.count,
            (ReversedBytes(item.value[0]), ReversedBytes(item.value[1])),
        )

    def _pop_heap(self) -> FrequencyCounterItem | None:
        if self._empty():
            return None

        item = heapq.heappop(self.heap)

        return FrequencyCounterItem(
            -item.count, (item.pair[0].value, item.pair[1].value)
        )

    def _push_heap(self, item: FrequencyCounterItem) -> None:
        heapq.heappush(self.heap, self._to_heap_item(item))

    def add(self, value: BytePair):
        self.freq_table[value] += 1
        self._push_heap(FrequencyCounterItem(self.freq_table[value], value))

    def sub(self, value: BytePair):
        self.freq_table[value] -= 1
        self._push_heap(FrequencyCounterItem(self.freq_table[value], value))

    def pop_most_freq_item(self) -> FrequencyCounterItem | None:
        item = self._pop_heap()
        while item is not None and (
            item.count != self.freq_table[item.value] or item.count == 0
        ):
            item = self._pop_heap()
        return item


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

    def merge(self, counter: FrequencyCounter, most_freq_pair: BytePair) -> None:
        curr_node = self.head
        while curr_node:
            next_node = curr_node.next
            if next_node:
                if (curr_node.token, next_node.token) == most_freq_pair:
                    # remove joined pair from statistic
                    counter.sub((curr_node.token, next_node.token))
                    if curr_node.prev:
                        counter.sub((curr_node.prev.token, curr_node.token))

                    if next_node.next:
                        counter.sub((next_node.token, next_node.next.token))

                    # add joined pair and
                    new_node = TokenNode(curr_node.token + next_node.token)
                    if curr_node.prev:
                        new_node.prev = curr_node.prev
                        curr_node.prev.next = new_node
                        counter.add((new_node.prev.token, new_node.token))
                    else:
                        self.head = new_node

                    if next_node.next:
                        new_node.next = next_node.next
                        next_node.next.prev = new_node
                        counter.add((new_node.token, new_node.next.token))

                    next_node = next_node.next

            curr_node = next_node


def _build_initial_dict(spec_tokens: list[bytes]) -> ByteVocab:
    spec_tokens_count = len(spec_tokens)

    initial_dict = {i: spec_tokens[i] for i in range(spec_tokens_count)}
    initial_dict.update({spec_tokens_count + i: bytes([i]) for i in range(UTF_8_BYTES)})

    return initial_dict


def _build_init_counter(
    pretokenized_text: Iterable[ByteWord],
) -> FrequencyCounter:
    counter = FrequencyCounter()
    for word_tokens in pretokenized_text:
        for i in range(len(word_tokens) - 1):
            curr_pair = word_tokens[i], word_tokens[i + 1]
            counter.add(curr_pair)
    return counter


def _merge_iteration(
    counter: FrequencyCounter, most_freq_pair: BytePair, chunk: list[TrainWord]
) -> None:

    for word in chunk:
        word.merge(counter, most_freq_pair)


def train(
    pretokenized_text: list[ByteWord],
    spec_tokens: list[bytes],
    vocab_size: int,
    num_threads=4,
) -> tuple[ByteVocab, list[BytePair]]:

    assert vocab_size >= len(spec_tokens) + UTF_8_BYTES

    train_text = [TrainWord(word) for word in pretokenized_text]

    byte_vocab = _build_initial_dict(spec_tokens)

    counter = _build_init_counter(pretokenized_text)
    merges: list[BytePair] = []

    step = 0

    most_freq_pair = counter.pop_most_freq_item()

    while len(byte_vocab) < vocab_size and most_freq_pair is not None:

        # if step == 9:
        #     print(f"winner: {most_freq_pair.value}, count: {most_freq_pair.count}")
        #     print(f"freq (b'e', b'r'): {counter.freq_table.get((b'e', b'r'), 0)}")
        #     print(f"freq (b' ', b's'): {counter.freq_table.get((b' ', b's'), 0)}")
        #     break

        merges.append(most_freq_pair.value)
        byte_vocab[len(byte_vocab)] = most_freq_pair.value[0] + most_freq_pair.value[1]

        _merge_iteration(counter, most_freq_pair.value, train_text)

        most_freq_pair = counter.pop_most_freq_item()

        step += 1

    return byte_vocab, merges
