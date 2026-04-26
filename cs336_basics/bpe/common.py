type ByteWord = tuple[bytes, ...]
type BytePair = tuple[bytes, bytes]
type FrequencyTable = dict[BytePair, int]
type ByteVocab = dict[int, bytes]

from typing import Iterable

UTF_8_BYTES = 256


def convert_text_to_bytewords(text: Iterable[str]) -> Iterable[ByteWord]:
    for word in text:
        yield tuple(map(lambda char: bytes(char, encoding="utf-8"), word))


def reverse_bytes(text: bytes) -> bytes:
    return bytes([(UTF_8_BYTES - 1) - text[i] for i in range(len(text))])
