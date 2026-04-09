from pathlib import Path
from cs336_basics.bpe.pretokenization import pretokenize_text
from cs336_basics.bpe.train import train
from cs336_basics.bpe.common import convert_text_to_bytewords, ByteVocab


class BPE:
    def __init__(self) -> None:
        self.byte_dict: ByteVocab | None = None

    def fit(
        self, file_path: str, spec_tokens: list[str], vocab_size: int, num_processes=4
    ):

        with open(file_path, "rb") as f:
            pretokenized_text = pretokenize_text(f, spec_tokens, num_processes)
            byte_dict, merges = train(
                convert_text_to_bytewords(pretokenized_text),
                [bytes(token, encoding="utf-8") for token in spec_tokens],
                vocab_size,
            )

            self.byte_dict = byte_dict

        return byte_dict, merges

    def transform(self):
        pass


if __name__ == "__main__":
    print("Start")
    bpe = BPE()
    bpe.fit(
        "/home/byzov-timofey/projects/stanford/assignment1-basics/tests/fixtures/corpus.en",
        ["<|endoftext|>"],
        500,
    )
    print("Done")
