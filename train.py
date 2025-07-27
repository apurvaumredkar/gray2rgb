import os
import random
import numpy as np
import json
import time
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch import autocast, GradScaler

import kornia

from perceptual_loss import VGGPerceptualLoss
from model import UNetViT

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    device = torch.device('cuda')
    device_type = 'cuda'
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    device_type = 'mps'
else:
    device = torch.device('cpu')
    device_type = 'cpu'

print(f"Using device: {device}")


class ColorizationDataset(Dataset):
    def __init__(self, rgb_dir):
        self.rgb_dir = rgb_dir
        self.filenames = [
            f for f in os.listdir(rgb_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        rgb_path = os.path.join(self.rgb_dir, filename)
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        L = gray_img.astype(np.float32) / 255.0 * 100.0
        L = (L / 50.0) - 1.0
        L = L[None, :, :]
        L = torch.from_numpy(L).float()
        rgb_img = torch.from_numpy(
            (rgb_img / 255.0).astype(np.float32)).permute(2, 0, 1)
        return L, rgb_img


def lab_to_rgb_torch(L, ab):
    lab = torch.cat([(L + 1) * 50.0, ab * 128.0], dim=1)
    rgb = kornia.color.lab_to_rgb(lab)
    return rgb.clamp(0, 1)


def train_one_epoch(model, dataloader, perc_loss_fn, optimizer, device, device_type, epoch, total_epochs, scaler):
    model.train()
    total_loss = 0
    progress_bar = tqdm(
        dataloader, desc=f"Epoch {epoch}/{total_epochs} [Training]", leave=False)
    for L, gt_rgb in progress_bar:
        L, gt_rgb = L.to(device), gt_rgb.to(device)
        optimizer.zero_grad()
        with autocast(device_type=device_type):
            pred_ab = model(L)
            pred_rgb = lab_to_rgb_torch(L, pred_ab).to(device)
            loss = perc_loss_fn(pred_rgb, gt_rgb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch}/{total_epochs} Training Loss: {avg_loss:.4f}")
    return avg_loss


def validate(model, dataloader, perc_loss_fn, device, device_type, epoch, total_epochs):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        progress_bar = tqdm(
            dataloader, desc=f"Epoch {epoch}/{total_epochs} [Validation]", leave=False)
        for L, gt_rgb in progress_bar:
            L, gt_rgb = L.to(device), gt_rgb.to(device)
            with autocast(device_type=device_type):
                pred_ab = model(L)
                pred_rgb = lab_to_rgb_torch(L, pred_ab).to(device)
                loss = perc_loss_fn(pred_rgb, gt_rgb)
            total_loss += loss.item()
            progress_bar.set_postfix(
                loss=loss.item())
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
    lr = hparams.get("learning_rate", 1e-3)
    batch_size = hparams.get("batch_size", 8)
    vit_embed_dim = hparams.get("vit_embed_dim", 128)
    vit_heads = hparams.get("vit_heads", 4)

    rgb_train_dir = "./data_subset/train"
    rgb_val_dir = "./data_subset/val"

    train_dataset = ColorizationDataset(rgb_train_dir)
    val_dataset = ColorizationDataset(rgb_val_dir)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = UNetViT(in_channels=1, out_channels=2,
                    vit_embed_dim=vit_embed_dim, vit_heads=vit_heads).to(device)
    perc_loss_fn = VGGPerceptualLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()

    epoch_logs = []

    for epoch in range(1, epochs + 1):
        train_start = time.time()
        train_loss = train_one_epoch(model, train_loader, perc_loss_fn, optimizer, device,
                        device_type, epoch, epochs, scaler)
        train_time = time.time() - train_start
        
        checkpoint_path = get_checkpoint_filename(epoch)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        val_start = time.time()
        val_loss = validate(model, val_loader, perc_loss_fn, device,
                 device_type, epoch, epochs)
        val_time = time.time() - val_start

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_time": train_time,
            "val_time": val_time
        })

        with open("training_metrics.json", "w") as f:
            json.dump({"hyperparameters": hparams, "epochs": epoch_logs}, f, indent=2)
    
    print("Saved all training logs to training_metrics.json")


if __name__ == "__main__":
    train_model_pipeline()