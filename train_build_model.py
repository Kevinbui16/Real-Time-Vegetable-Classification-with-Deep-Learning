import torch
import torch.nn as nn
from dataset import VegetableDataset
from model import CNN
from torchvision.transforms import Compose, Resize, ToTensor, RandomAffine, ColorJitter, Normalize
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import matplotlib.pyplot as plt
import argparse
import warnings

warnings.filterwarnings("ignore")


def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="Wistia")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)


def get_args():
    parser = argparse.ArgumentParser("Test Argument")
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--image_size", "-i", type=int, default=224)
    parser.add_argument("--num_epochs", "-e", type=int, default=100)
    parser.add_argument("--learning_rate", "-r", type=int, default=1e-2)
    parser.add_argument("--momentum", "-m", type=int, default=0.9)
    parser.add_argument("--data_path", "-d", type=str, default="Vegetable_Images")
    parser.add_argument("--log_path", "-l", type=str, default="tensorboard/vegetables_build_model")
    parser.add_argument("--checkpoint_path", "-c", type=str, default="trained_models/vegetables_build_model")
    parser.add_argument("--pretrained_checkpoint_path", "-p", type=str, default=None)
    args = parser.parse_args()
    return args


def train(args):
    device = torch.device("mps")
    train_transform = Compose([
        ToTensor(),
        RandomAffine(
            degrees=(-5, 5),
            translate=(0.15, 0.15),
            scale=(0.85, 1.1),
            shear=10
        ),
        ColorJitter(
            brightness=0.125,
            contrast=0.5,
            saturation=0.5,
            hue=0.05
        ),
        Resize((args.image_size, args.image_size)),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    ])
    val_transform = Compose([
        ToTensor(),
        Resize((args.image_size, args.image_size)),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    train_dataset = VegetableDataset(root=args.data_path, is_train=True, transform=train_transform)
    train_param = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": 8,
        "drop_last": True

    }
    train_dataloader = DataLoader(dataset=train_dataset, **train_param)
    num_iter_per_epoch = len(train_dataloader)

    val_dataset = VegetableDataset(root=args.data_path, is_train=False, transform=val_transform)
    val_param = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": 8,
        "drop_last": False,

    }
    val_dataloader = DataLoader(dataset=val_dataset, **val_param)

    model = CNN(num_classes=len(train_dataset.categories)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    if args.pretrained_checkpoint_path is not None:
        checkpoint = torch.load(args.pretrained_checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
    else:
        best_acc = -1
        start_epoch = 0

    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)

    if not os.path.isdir(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    writer = SummaryWriter(args.log_path)

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        trained_losses = []
        progress_bar = tqdm(train_dataloader, colour="cyan")
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)
            loss = criterion(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            trained_losses.append(loss.item())
            average_loss = np.mean(trained_losses)

            progress_bar.set_description(
                "Epoch: {}/{}. Train Loss: {}".format(epoch + 1, args.num_epochs, round(average_loss, 2)))
            writer.add_scalar("Train/loss", average_loss, epoch * num_iter_per_epoch + iter)

        all_predictions = []
        all_labels = []
        all_losses = []
        model.eval()
        with torch.no_grad():
            progress_bar = tqdm(val_dataloader, colour="yellow")
            for iter, (images, labels) in enumerate(progress_bar):
                images = images.to(device)
                labels = labels.to(device)

                predictions = model(images)
                loss = criterion(predictions, labels)
                predictions = torch.argmax(predictions, dim=1)

                all_predictions.extend(predictions.tolist())
                all_labels.extend(labels.tolist())
                all_losses.append(loss.item())

            acc = accuracy_score(all_labels, all_predictions)
            conf_matrix = confusion_matrix(all_labels, all_predictions)
            loss = np.mean(all_losses)
            progress_bar.set_description("Accuracy score: {}. Loss score: {}".format(round(acc, 2), round(loss, 2)))
            plot_confusion_matrix(writer, conf_matrix, train_dataset.categories, epoch)
            writer.add_scalar("Val/accuracy", acc, epoch)
            writer.add_scalar("Val/loss", loss, epoch)

            checkpoint = {
                "epoch": epoch + 1,
                "best_acc": best_acc,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_path, "last.pt"))
            if acc > best_acc:
                best_acc = acc
                torch.save(checkpoint, os.path.join(args.checkpoint_path, "best.pt"))


if __name__ == '__main__':
    args = get_args()
    train(args)