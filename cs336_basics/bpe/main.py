from cs336_basics.bpe.pretokenization import PreTokenizer
from cs336_basics.bpe.tokenizer import BPE

if __name__ == "__main__":
    print("Start")
    pretokenizer = PreTokenizer(spec_tokens=[])
    bpe = BPE(pretokenizer)
    bpe.fit("/home/byzov-timofey/projects/stanford/assignment1-basics/tests/fixtures/corpus.en", 1024)
    print("Done")
