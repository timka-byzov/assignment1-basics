from cs336_basics.bpe.common import BytePair, ByteVocab, convert_text_to_bytewords
from cs336_basics.bpe.pretokenization import PreTokenizerBase
from cs336_basics.bpe.train import train


class BPE:
    def __init__(self, pretokenizer: PreTokenizerBase) -> None:
        self.pretokenizer = pretokenizer
        self.byte_dict: ByteVocab | None = None

    def fit(self, dataset_path: str, vocab_size: int) -> list[BytePair]:
        with open(dataset_path, "rb") as f:
            pretokenized_text = self.pretokenizer.pretokenize_file(f)
            byte_dict, merges = train(
                list(convert_text_to_bytewords(pretokenized_text)),
                [bytes(token, encoding="utf-8") for token in self.pretokenizer.spec_tokens],
                vocab_size,
            )
        self.byte_dict = byte_dict
        return merges

    def transform(self):
        pass
