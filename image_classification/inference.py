import torch
from torchvision import transforms
from PIL import Image
import argparse


def load_image(image_path, transform):
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0)
    return img


def inference(model, image_path, device):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    model = model.to(device)
    model.eval()
    img_tensor = load_image(image_path, transform)
    img_tensor = img_tensor.to(device)
    outputs = model(img_tensor)
    class_id = torch.argmax(outputs.data, 1)
    return class_id.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Classification.")
    parser.add_argument("image_path", type=str)
    parser.add_argument(
        "--model",
        type=str,
        choices=["lenet", "vgg", "resnet", "rnn", "dnn"],
        default="resnet",
        help="choice of the model",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.model == "lenet":
        model = torch.load("checkpoint/lenet.pt")
    elif args.model == "vgg":
        model = torch.load("checkpoint/vgg.pt")
    elif args.model == "resnet":
        model = torch.load("checkpoint/resnet.pt")
    elif args.model == "rnn":
        model = torch.load("checkpoint/rnn.pt")
    elif args.model == "dnn":
        model = torch.load("checkpoint/dnn.pt")
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    class_id = inference(model, args.image_path, device)
    if class_id == 0:
        print("Cat")
    elif class_id == 1:
        print("Dog")
