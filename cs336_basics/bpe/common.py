from collections.abc import Iterable

type ByteWord = tuple[bytes, ...]
type BytePair = tuple[bytes, bytes]
type FrequencyTable = dict[BytePair, int]
type ByteVocab = dict[int, bytes]


UTF_8_BYTES = 256


def convert_text_to_bytewords(text: Iterable[str]) -> Iterable[ByteWord]:
    for word in text:
        yield tuple(bytes([byte]) for byte in word.encode("utf-8"))


def reverse_bytes(text: bytes) -> bytes:
    return bytes([(UTF_8_BYTES - 1) - text[i] for i in range(len(text))])
