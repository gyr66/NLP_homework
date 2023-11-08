import torch
import torch.nn as nn
from torchcrf import CRF


class NERModel(nn.Module):
    def __init__(
        self, vocab_size, tag_size, embedding_dim, hidden_dim, num_layers, dropout=0.1
    ):
        self.vocab_size = vocab_size
        self.tag_size = tag_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, tag_size)
        self.crf = CRF(tag_size, batch_first=True)

    def load_pretrained_embedding(self, vocab, pretrained_embedding):
        for idx, word in enumerate(vocab.get_itos()):
            if word in pretrained_embedding:
                self.embedding.weight.data[idx] = torch.from_numpy(
                    pretrained_embedding[word]
                )

    def forward(self, input_ids, valid_lens, labels=None):
        embeddings = self.embedding(input_ids)
        outputs, _ = self.lstm(embeddings)
        outputs = self.dropout(outputs)
        logits = self.classifier(outputs)
        seq_len = input_ids.size(1)
        mask = (
            torch.arange(seq_len)[None, :].to(valid_lens.device) < valid_lens[:, None]
        )
        tag_ids = self.crf.decode(logits, mask=mask)
        loss = None
        if labels is not None:
            loss = -self.crf(logits, labels, mask=mask, reduction="mean")
        return tag_ids, loss
