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
import torchvision.transforms as T
from PIL import Image
from model import ColorizationNet
import kornia.color as kc
import lpips

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import kornia.color as kc


class ColorizationDataset(Dataset):
    def __init__(self, rgb_dir, is_train=True):
        self.rgb_dir = rgb_dir
        self.filenames = [
            f for f in os.listdir(rgb_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.is_train = is_train

        # Define augmentations
        if is_train:
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(15),
                T.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Add tiny translations
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                T.ToTensor(),
                T.RandomErasing(
                    p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0
                )
            ])
        else:
            self.transforms = T.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        rgb_path = os.path.join(self.rgb_dir, filename)

        img = Image.open(rgb_path).convert("RGB")
        img = self.transforms(img)  
        img = img.unsqueeze(0)    # pyright: ignore[reportAttributeAccessIssue]
        lab = kc.rgb_to_lab(img)[0]  
        L = lab[0:1] / 100.0
        ab = (lab[1:3] + 128.0) / 127.5 - 1.0

        return L, ab


def lab_to_rgb_torch(L, ab):
    lab = torch.cat([(L * 100.0), (ab + 1.0) * 127.5 - 128.0], dim=1)
    rgb = kc.lab_to_rgb(lab).clamp(0, 1)
    return rgb


def get_checkpoint_filename(epoch, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"model_e{epoch}_{timestamp}.pt"
    return os.path.join(save_dir, filename)


def train_one_epoch(
    model,
    dataloader,
    l1_loss_fn,
    perc_loss_fn,
    l1_weight,
    perc_weight,
    vibrance_weight,
    vibrance_threshold,
    optimizer,
    device,
    epoch,
    total_epochs,
    autocast_ctx,
    scaler,
):
    model.train()
    total_loss = 0
    progress_bar = tqdm(
        dataloader, desc=f"Epoch {epoch}/{total_epochs} [Training]", leave=False
    )
    for L, ab in progress_bar:
        L, ab = L.to(device), ab.to(device)
        optimizer.zero_grad()
        with autocast_ctx:
            pred_ab = model(L)
            # L1 loss
            loss_l1 = l1_loss_fn(pred_ab, ab)

            # Vibrance loss: penalize dull ab regions
            saturation = torch.sqrt(pred_ab[:, 0] ** 2 + pred_ab[:, 1] ** 2 + 1e-6)
            vibrance_loss = torch.mean((vibrance_threshold - saturation).clamp(min=0.0))

            # Perceptual loss
            pred_rgb = lab_to_rgb_torch(L, pred_ab)
            gt_rgb = lab_to_rgb_torch(L, ab)
            loss_perc = perc_loss_fn(pred_rgb * 2 - 1, gt_rgb * 2 - 1).mean()
            
            # Total loss
            loss = l1_weight * loss_l1 + perc_weight * loss_perc + vibrance_weight * vibrance_loss

        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch}/{total_epochs} Training Loss: {avg_loss:.4f}")
    return avg_loss


def validate(
    model,
    dataloader,
    l1_loss_fn,
    perc_loss_fn,
    l1_weight,
    perc_weight,
    device,
    epoch,
    total_epochs,
    autocast_ctx,
):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        progress_bar = tqdm(
            dataloader, desc=f"Epoch {epoch}/{total_epochs} [Validation]", leave=False
        )
        for L, ab in progress_bar:
            L, ab = L.to(device), ab.to(device)
            with autocast_ctx:
                pred_ab = model(L)
                loss_l1 = l1_loss_fn(pred_ab, ab)
                pred_rgb = lab_to_rgb_torch(L, pred_ab)
                gt_rgb = lab_to_rgb_torch(L, ab)
                loss_perc = perc_loss_fn(pred_rgb * 2 - 1, gt_rgb * 2 - 1).mean()
                loss = l1_weight * loss_l1 + perc_weight * loss_perc
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch}/{total_epochs} Validation Loss: {avg_loss:.4f}")
    return avg_loss


def train_model_pipeline():
    with open("hyperparameters.json", "r") as f:
        hparams = json.load(f)

    vit_embed_dim = hparams.get("vit_embed_dim", 128)
    vit_heads = hparams.get("vit_heads", 4)
    epochs = hparams.get("epochs", 10)
    batch_size = hparams.get("batch_size", 8)
    lr = hparams.get("learning_rate", 1e-3)
    wd = hparams.get("weight_decay", 1e-4)
    n_workers = int(hparams.get("num_workers", 4))
    dropout = hparams.get("dropout", 0.2)
    l1_weight = hparams.get("l1_weight", 1.0)
    perc_weight = hparams.get("perc_weight", 0.1)
    vibrance_weight = hparams.get("vibrance_weight", 0.1)
    vibrance_threshold = hparams.get("vibrance_threshold", 0.3)

    rgb_train_dir = "./data_subset/train"
    rgb_val_dir = "./data_subset/val"

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_type = "mps"
    else:
        device = torch.device("cpu")
        device_type = "cpu"

    print(f"Using device: {device}")
    autocast_ctx = torch.autocast(device_type=device_type)
    scaler = torch.GradScaler(device=device_type)

    model = ColorizationNet(
        in_channels=1,
        out_channels=2,
        vit_embed_dim=vit_embed_dim,
        vit_heads=vit_heads,
        dropout=dropout,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5, min_lr=1e-6)

    train_dataset = ColorizationDataset(rgb_train_dir, is_train=True)
    val_dataset = ColorizationDataset(rgb_val_dir, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    )

    l1_loss_fn = nn.L1Loss()
    perc_loss_fn = lpips.LPIPS(net='alex').to(device)

    epoch_logs = []

    torch.backends.cudnn.benchmark = True

    for epoch in range(1, epochs + 1):
        train_start = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            l1_loss_fn,
            perc_loss_fn,
            l1_weight,
            perc_weight,
            vibrance_weight,
            vibrance_threshold,
            optimizer,
            device,
            epoch,
            epochs,
            autocast_ctx,
            scaler,
        )
        train_time = time.time() - train_start

        checkpoint_path = get_checkpoint_filename(epoch)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        val_start = time.time()
        val_loss = validate(
            model,
            val_loader,
            l1_loss_fn,
            perc_loss_fn,
            l1_weight,
            perc_weight,
            device,
            epoch,
            epochs,
            autocast_ctx,
        )
        val_time = time.time() - val_start

        scheduler.step(val_loss)

        if device.type == "cuda":
            torch.cuda.empty_cache()

        epoch_logs.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_time": train_time,
                "val_time": val_time,
            }
        )

        with open("training_metrics.json", "w") as f:
            json.dump({"hyperparameters": hparams, "epochs": epoch_logs}, f, indent=2)

    print("Training logs saved to training_metrics.json")


if __name__ == "__main__":
    train_model_pipeline()
