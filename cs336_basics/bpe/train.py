import heapq
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from cs336_basics.bpe.common import (
    UTF_8_BYTES,
    BytePair,
    ByteVocab,
    ByteWord,
    FrequencyTable,
)


@dataclass(order=False)
class _ReversedBytes:
    value: bytes

    def __lt__(self, other: "_ReversedBytes") -> bool:
        return self.value > other.value

    def __le__(self, other: "_ReversedBytes") -> bool:
        return self.value >= other.value

    def __gt__(self, other: "_ReversedBytes") -> bool:
        return self.value < other.value

    def __ge__(self, other: "_ReversedBytes") -> bool:
        return self.value <= other.value


@dataclass(order=True)
class _FrequencyHeapItem:
    count: int  # negative
    pair: tuple[_ReversedBytes, _ReversedBytes]


@dataclass(frozen=True)  # иммутабельный, хэшируемый
class FrequencyCounterItem:
    count: int
    value: BytePair


class FrequencyCounter:
    def __init__(self, freq_table: FrequencyTable | None = None) -> None:
        self.heap: list[_FrequencyHeapItem] = []
        self.freq_table = freq_table if freq_table is not None else defaultdict(int)

    def _empty(self) -> bool:
        return len(self.heap) == 0

    def _to_heap_item(self, item: FrequencyCounterItem) -> _FrequencyHeapItem:
        return _FrequencyHeapItem(
            -item.count,
            (_ReversedBytes(item.value[0]), _ReversedBytes(item.value[1])),
        )

    def _pop_heap(self) -> FrequencyCounterItem | None:
        if self._empty():
            return None

        item = heapq.heappop(self.heap)

        return FrequencyCounterItem(-item.count, (item.pair[0].value, item.pair[1].value))

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
        while item is not None and (item.count != self.freq_table[item.value] or item.count == 0):
            item = self._pop_heap()
        return item


class TokenOccurrence:
    def __init__(
        self,
        token: bytes,
        prev_same: "TokenOccurrence | None" = None,
        next_same: "TokenOccurrence | None" = None,
        left: "TokenOccurrence | None" = None,
        right: "TokenOccurrence | None" = None,
    ) -> None:
        self.prev_same = prev_same
        self.next_same = next_same
        self.token = token
        self.left = left
        self.right = right


class IndexRecord:
    def __init__(self, head: TokenOccurrence, tail: TokenOccurrence) -> None:
        self.head = head
        self.tail = tail


class Index:
    def __init__(self, pretokenized_text: Iterable[ByteWord]) -> None:
        self._counter = FrequencyCounter()
        self._index: dict[bytes, IndexRecord | None] = {}
        self._build_index(pretokenized_text)  # possible optimization - combine heap and index building

    def _build_index(
        self,
        pretokenized_text: Iterable[ByteWord],
    ) -> None:

        for word in pretokenized_text:
            left_occurrence: TokenOccurrence | None = None
            for token in word:
                occurrence = self._add(token, left=left_occurrence, right=None)
                if left_occurrence:
                    left_occurrence.right = occurrence
                left_occurrence = occurrence

    def _remove(self, node: TokenOccurrence) -> None:
        record = self._index[node.token]
        assert record is not None

        if not (node.prev_same or node.next_same):
            self._index[node.token] = None
            return

        # delete head
        if node.prev_same is None:
            assert node.next_same is not None
            record.head = node.next_same

        # delete tail
        if node.next_same is None:
            assert node.prev_same is not None
            record.tail = node.prev_same

        if node.next_same:
            node.next_same.prev_same = node.prev_same
        if node.prev_same:
            node.prev_same.next_same = node.next_same

    # добавляем в хвост, соблюдаем порядок
    def _add(self, token: bytes, left: TokenOccurrence | None, right: TokenOccurrence | None) -> TokenOccurrence:

        if left:
            self._counter.add((left.token, token))
        if right:
            self._counter.add((token, right.token))

        record = self._index.get(token)
        if record is None:
            occ = TokenOccurrence(token, left=left, right=right)
            self._index[token] = IndexRecord(head=occ, tail=occ)
            return occ
        else:
            curr_tail = record.tail
            new_tail = TokenOccurrence(token, prev_same=curr_tail, left=left, right=right)
            curr_tail.next_same = new_tail
            record.tail = new_tail
            return new_tail

    def _join_pair(self, left: TokenOccurrence, right: TokenOccurrence) -> None:

        left_neighbour = left.left
        right_neighbour = right.right

        self._counter.sub((left.token, right.token))
        if left_neighbour:
            self._counter.sub((left_neighbour.token, left.token))
        if right_neighbour:
            self._counter.sub((right.token, right_neighbour.token))

        new_occ = self._add(token=left.token + right.token, left=left.left, right=right.right)
        if left_neighbour:
            left_neighbour.right = new_occ
        if right_neighbour:
            right_neighbour.left = new_occ

        self._remove(left)
        self._remove(right)

    def join_most_freq_pair(self) -> BytePair | None:
        pair = self._counter.pop_most_freq_item()
        if not pair:
            return None
        prev_token, next_token = pair.value

        record = self._index[prev_token]
        assert record is not None
        curr_node: TokenOccurrence | None = record.head
        while curr_node is not None:
            next_node = curr_node.next_same
            right_node = curr_node.right
            if right_node is not None and right_node.token == next_token:
                if next_node is right_node:
                    next_node = right_node.next_same
                self._join_pair(curr_node, right_node)

            curr_node = next_node

        return prev_token, next_token


class Trainer:
    def __init__(self, pretokenized_text: Iterable[ByteWord], spec_tokens: list[bytes], vocab_size: int) -> None:

        assert vocab_size >= len(spec_tokens) + UTF_8_BYTES

        self.vocab_size = vocab_size
        self.spec_tokens = spec_tokens
        self.pretokenized_text = pretokenized_text

    def _build_initial_dict(self, spec_tokens: list[bytes]) -> ByteVocab:
        spec_tokens_count = len(spec_tokens)
        initial_dict = {i: spec_tokens[i] for i in range(spec_tokens_count)}
        initial_dict.update({spec_tokens_count + i: bytes([i]) for i in range(UTF_8_BYTES)})

        return initial_dict

    def train(self) -> tuple[list[BytePair], ByteVocab]:
        index = Index(self.pretokenized_text)
        merges: list[BytePair] = []
        byte_vocab = self._build_initial_dict(self.spec_tokens)
        idx = len(byte_vocab)

        while idx < self.vocab_size and (most_freq_pair := index.join_most_freq_pair()) is not None:
            merges.append(most_freq_pair)
            byte_vocab[idx] = most_freq_pair[0] + most_freq_pair[1]
            idx += 1

        return merges, byte_vocab
