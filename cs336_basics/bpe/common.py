type ByteWord = tuple[bytes, ...]
type BytePair = tuple[bytes, bytes]
type FreqTable = dict[BytePair, int]
type ByteVocab = dict[int, bytes]

UTF_8_BYTES = 256


def convert_text_to_bytewords(text: list[str]) -> list[ByteWord]:
    return [
        tuple(map(lambda char: bytes(char, encoding="utf-8"), word)) for word in text
    ]
