import os
import random
import numpy as np
import json
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from model import UNetViT

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('mps') if torch.backends.mps.is_available()
          else torch.device('cpu'))
print(f"Using device: {device}")


class ColorizationDataset(Dataset):
    def __init__(self, gray_dir, rgb_dir):
        self.gray_dir = gray_dir
        self.rgb_dir = rgb_dir
        self.filenames = [
            f for f in os.listdir(gray_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        self.gray_transform = transforms.ToTensor()
        self.rgb_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        gray_path = os.path.join(self.gray_dir, filename)
        rgb_path = os.path.join(self.rgb_dir, filename)

        gray_img = Image.open(gray_path).convert("L")
        rgb_img = Image.open(rgb_path).convert("RGB")

        gray_tensor = self.gray_transform(gray_img)
        rgb_tensor = self.rgb_transform(rgb_img)

        return gray_tensor, rgb_tensor


def train_one_epoch(model, dataloader, loss_fn, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Training]", leave=False)
    for gray, rgb in progress_bar:
        gray, rgb = gray.to(device), rgb.to(device)
        optimizer.zero_grad()
        output = model(gray)
        loss = loss_fn(output, rgb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch}/{total_epochs} Training Loss: {avg_loss:.4f}")
    return avg_loss


def validate(model, dataloader, loss_fn, device, epoch, total_epochs):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Validation]", leave=False)
        for gray, rgb in progress_bar:
            gray, rgb = gray.to(device), rgb.to(device)
            output = model(gray)
            loss = loss_fn(output, rgb)
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch}/{total_epochs} Validation Loss: {avg_loss:.4f}")
    return avg_loss


def get_checkpoint_filename(epoch, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"model_e{epoch}_{timestamp}.pt"
    return os.path.join(save_dir, filename)


def train_model_pipeline():
    with open("hyperparameters.json", "r") as f:
        hparams = json.load(f)

    epochs = hparams.get("epochs", 10)
    lr = hparams.get("learning_rate", 1e-4)
    batch_size = hparams.get("batch_size", 16)
    vit_embed_dim = hparams.get("vit_embed_dim", 512)
    vit_heads = hparams.get("vit_heads", 8)

    gray_train_dir = "./data_subset/gray/train"
    rgb_train_dir = "./data_subset/rgb/train"
    gray_val_dir = "./data_subset/gray/val"
    rgb_val_dir = "./data_subset/rgb/val"

    train_dataset = ColorizationDataset(gray_train_dir, rgb_train_dir)
    val_dataset = ColorizationDataset(gray_val_dir, rgb_val_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = UNetViT(vit_embed_dim=vit_embed_dim, vit_heads=vit_heads).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch, epochs)
        validate(model, val_loader, loss_fn, device, epoch, epochs)
        checkpoint_path = get_checkpoint_filename(epoch)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    train_model_pipeline()