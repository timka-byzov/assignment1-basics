from cs336_basics.bpe.common import BytePair, ByteVocab, convert_text_to_bytewords
from cs336_basics.bpe.pretokenization import PreTokenizerBase
from cs336_basics.bpe.train import Trainer


class BPE:
    def __init__(self, pretokenizer: PreTokenizerBase) -> None:
        self.pretokenizer = pretokenizer
        self.byte_vocab: ByteVocab | None = None

    def fit(self, dataset_path: str, vocab_size: int) -> list[BytePair]:
        with open(dataset_path, "rb") as f:
            pretokenized_text = self.pretokenizer.pretokenize_file(f)
            trainer = Trainer(
                list(convert_text_to_bytewords(pretokenized_text)),
                [bytes(token, encoding="utf-8") for token in self.pretokenizer.spec_tokens],
                vocab_size,
            )
            merges, self.byte_vocab = trainer.train()

        return merges

    def transform(self):
        pass
