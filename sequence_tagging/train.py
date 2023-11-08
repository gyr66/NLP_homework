import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)
import pickle
import os
import argparse

from config import Config
from dataset import NERDataset
from model import NERModel

DEVICE = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")


def validate_epoch(model, dataloader):
    def flat_list(nested_list):
        return [e for inner_list in nested_list for e in inner_list]

    preds = []
    true_labels = []
    model.eval()
    for input_ids, labels, valid_lens in dataloader:
        input_ids, labels, valid_lens = (
            input_ids.to(DEVICE),
            labels.to(DEVICE),
            valid_lens.to(DEVICE),
        )
        with torch.no_grad():
            tag_ids, _ = model(input_ids, valid_lens)
        labels = [
            label[:valid_len]
            for label, valid_len in zip(labels.tolist(), valid_lens.tolist())
        ]
        preds.extend(flat_list(tag_ids))
        true_labels.extend(flat_list(labels))
    precision = precision_score(true_labels, preds, average="macro")
    recall = recall_score(true_labels, preds, average="macro")
    f1 = f1_score(true_labels, preds, average="macro")
    return precision, recall, f1


def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0.0
    for input_ids, labels, valid_lens in dataloader:
        optimizer.zero_grad()
        input_ids, labels, valid_lens = (
            input_ids.to(DEVICE),
            labels.to(DEVICE),
            valid_lens.to(DEVICE),
        )
        _, loss = model(input_ids, valid_lens, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(dataloader.dataset)


def main():
    config = Config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=config.max_length)
    parser.add_argument("--embedding_dim", type=int, default=config.embedding_dim)
    parser.add_argument("--hidden_dim", type=int, default=config.hidden_dim)
    parser.add_argument("--num_layers", type=int, default=config.num_layers)
    parser.add_argument("--dropout", type=float, default=config.dropout)
    parser.add_argument("--epoch", type=int, default=config.epoch)
    parser.add_argument("--learning_rate", type=float, default=config.learning_rate)
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--save_model", type=str, default=config.save_model)
    parser.add_argument("--dry_run", action="store_true")
    config = parser.parse_args()

    train_dataset = NERDataset(
        "data/train_corpus.txt", "data/train_label.txt", config.max_length
    )
    vocab = train_dataset.vocab
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    tag2id = train_dataset.tag2id
    test_dataset = NERDataset(
        "data/test_corpus.txt", "data/test_label.txt", config.max_length, vocab
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, num_workers=16
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.batch_size, num_workers=16
    )
    if os.path.exists(config.save_model):
        nerlstm = torch.load(config.save_model)
    else:
        nerlstm = NERModel(
            len(vocab),
            len(tag2id),
            config.embedding_dim,
            config.hidden_dim,
            config.num_layers,
            config.dropout,
        ).to(DEVICE)
        with open("word2vec.pkl", "rb") as f:
            word2vec = pickle.load(f)
        nerlstm.load_pretrained_embedding(vocab, word2vec)

    if not config.dry_run:
        optimizer = Adam(nerlstm.parameters(), config.learning_rate)

        best_f1 = 0.0
        for epoch in range(config.epoch):
            train_running_loss = train_epoch(nerlstm, train_dataloader, optimizer)
            print("Epoch: {} | Loss: {}".format(epoch + 1, train_running_loss))
            prec, rec, f1 = validate_epoch(nerlstm, test_dataloader)
            print("Precision: {} | Recall: {} | F1: {}".format(prec, rec, f1))
            if f1 > best_f1:
                torch.save(nerlstm, config.save_model)
                best_f1 = f1
    else:
        prec, rec, f1 = validate_epoch(nerlstm, test_dataloader)
        print("Precision: {} | Recall: {} | F1: {}".format(prec, rec, f1))


if __name__ == "__main__":
    main()
