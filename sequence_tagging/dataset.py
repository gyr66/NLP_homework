import torch
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator


class NERDataset(Dataset):
    def __init__(self, corpus_file_path, label_file_path, max_length, vocab=None):
        with open(corpus_file_path, "r", encoding="utf-8") as f:
            corpus = f.readlines()
        with open(label_file_path, "r", encoding="utf-8") as f:
            label = f.readlines()
        corpus = [line.strip().split() for line in corpus]
        label = [line.strip().split() for line in label]
        assert len(corpus) == len(label), "corpus and label must have same length"

        if vocab is None:
            self.vocab = build_vocab_from_iterator(
                corpus, specials=["<unk>", "<pad>"], special_first=True
            )
        else:
            self.vocab = vocab
        self.vocab.set_default_index(self.vocab["<unk>"])

        self.tag2id = {
            "O": 0,
            "B-PER": 1,
            "I-PER": 2,
            "B-ORG": 3,
            "I-ORG": 4,
            "B-LOC": 5,
            "I-LOC": 6,
            "PAD": 7,
        }

        self.corpus = []
        self.label = []
        self.valid_len = []
        for corpus_line, label_line in zip(corpus, label):
            assert len(corpus_line) == len(
                label_line
            ), "corpus and label must have same length"
            self.valid_len.append(len(corpus_line))
            self.corpus.append(self.vocab(corpus_line))
            self.label.append([self.tag2id[tag] for tag in label_line])
            if len(self.corpus[-1]) > max_length:
                self.corpus[-1] = self.corpus[-1][:max_length]
                self.label[-1] = self.label[-1][:max_length]
                self.valid_len[-1] = max_length
            else:
                self.corpus[-1] = self.corpus[-1] + [self.vocab.get_stoi()["<pad>"]] * (
                    max_length - len(self.corpus[-1])
                )
                self.label[-1] = self.label[-1] + [self.tag2id["PAD"]] * (
                    max_length - len(self.label[-1])
                )

        self.corpus = torch.Tensor(self.corpus).long()
        self.label = torch.Tensor(self.label).long()
        self.valid_len = torch.Tensor(self.valid_len).long()

    def __getitem__(self, item):
        return self.corpus[item], self.label[item], self.valid_len[item]

    def __len__(self):
        return len(self.label)
