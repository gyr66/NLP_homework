import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torchvision.models as models
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall

import os
import argparse
from tqdm import tqdm

from model import LeNet, VGG16, BiRNN, DNN
from config import Config


def train_epoch(model, criterion, optimizer, dataloader, clip, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(dataloader, desc="Training: ", leave=False):
        if isinstance(model, BiRNN):
            # 原始的图像形状为: [N, 3, 224, 224]
            # 需要将其变为: [N, 224, 3*224]
            batch_size = inputs.size(0)
            inputs = inputs.permute(0, 2, 1, 3)
            inputs = inputs.reshape(batch_size, 224, -1)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() * 100 / len(dataloader.dataset)

    return epoch_loss, epoch_acc


def val_epoch(model, criterion, dataloader, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    precision_metric = MulticlassPrecision(average=None, num_classes=2).to(device)
    recall_metric = MulticlassRecall(average=None, num_classes=2).to(device)

    for inputs, labels in tqdm(dataloader, desc="Evaluating: ", leave=False):
        if isinstance(model, BiRNN):
            # 原始的图像形状为: [N, 3, 224, 224]
            # 需要将其变为: [N, 224, 3*224]
            batch_size = inputs.size(0)
            inputs = inputs.permute(0, 2, 1, 3)
            inputs = inputs.reshape(batch_size, 224, -1)
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, labels)

        precision_metric.update(preds, labels)
        recall_metric.update(preds, labels)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() * 100 / len(dataloader.dataset)
    precision = precision_metric.compute()
    recall = recall_metric.compute()

    return epoch_loss, epoch_acc, precision, recall


if __name__ == "__main__":
    config = Config()

    parser = argparse.ArgumentParser(description="Image Classification.")
    parser.add_argument(
        "--data_path",
        type=str,
        default=config.data_path,
        help="location of the image",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["lenet", "vgg", "resnet", "rnn", "dnn"],
        default=config.model,
        help="choice of the model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=config.batch_size, help="batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=config.lr, help="initial learning rate"
    )
    parser.add_argument(
        "--clip", type=float, default=config.clip, help="gradient clipping"
    )
    parser.add_argument(
        "--epochs", type=int, default=config.epochs, help="upper epoch limit"
    )
    parser.add_argument(
        "--saved",
        action="store_true",
        help="whether continue training using the model saved before",
        default=config.saved,
    )
    parser.add_argument(
        "--only_test",
        action="store_true",
        help="whether only test model",
        default=config.only_test,
    )
    parser.add_argument(
        "--multi_gpu",
        action="store_true",
        help="whether use multi gpu to train",
        default=config.multi_gpu,
    )
    parser.add_argument(
        "--gpus", nargs="+", type=int, help="gpu id list to use", default=config.gpus
    )

    config = parser.parse_args()

    print(f"Using model {config.model}")

    DEVICE = torch.device(
        f"cuda:{config.gpus[0]}" if torch.cuda.is_available() else "cpu"
    )

    image_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    train_dataset = datasets.ImageFolder(
        root=os.path.join(config.data_path, "train"),
        transform=image_transforms["train"],
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=16
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(config.data_path, "val"), transform=image_transforms["val"]
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=16
    )

    if config.saved:
        model = torch.load(f"checkpoint/{config.model}.pt").to(DEVICE)
        print(f"Load {config.model} from checkpoint...")
    else:
        if config.model == "lenet":
            model = LeNet()
        elif config.model == "vgg":
            model = VGG16()
        elif config.model == "resnet":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 2)
        elif config.model == "rnn":
            model = BiRNN()
        elif config.model == "dnn":
            model = DNN()

    model = model.to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {num_params:,} trainable parameters")
    if config.multi_gpu:
        model = nn.DataParallel(model, device_ids=config.gpus)

    criterion = nn.CrossEntropyLoss()

    if config.only_test:
        val_loss, val_acc, precision, recall = val_epoch(
            model, criterion, val_loader, DEVICE
        )
        print(f"{config.model} Valid Accuracy: {val_acc:.2f}")
        for i, class_name in enumerate(["Cat", "Dog"]):
            print(f"{class_name}\t{precision[i]:.2f}\t{recall[i]:.2f}")
        exit(0)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=2
    )

    best_valid_acc = 0.0
    for epoch in range(config.epochs):
        train_loss, train_acc = train_epoch(
            model, criterion, optimizer, train_loader, config.clip, DEVICE
        )
        val_loss, val_acc, _, _ = val_epoch(model, criterion, val_loader, DEVICE)
        lr_scheduler.step(val_loss)
        if val_acc > best_valid_acc:
            best_valid_acc = val_acc
            torch.save(model, f"checkpoint/{config.model}.pt")
        print(f"Epoch: {epoch}")
        print(f"Train Loss: {train_loss:.3f}\tTrain Accuracy: {train_acc:.2f}%")
        print(f"Valid Loss: {val_loss:.3f}\tValid Accuracy: {val_acc:.2f}%")
    print(f"Best valid accuracy is {best_valid_acc}")
