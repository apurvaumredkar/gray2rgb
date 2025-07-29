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
import warnings

warnings.filterwarnings("ignore")


class ColorizationDataset(Dataset):
    def __init__(self, rgb_dir, resolution=256):
        self.rgb_dir = rgb_dir
        self.resolution = resolution
        self.filenames = [
            f
            for f in os.listdir(rgb_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        self.transform = T.Compose([T.RandomHorizontalFlip(p=0.5), T.ToTensor()])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        rgb_path = os.path.join(self.rgb_dir, filename)

        img = Image.open(rgb_path).convert("RGB")
        img = self.transform(img)

        img = img.unsqueeze(0) # pyright: ignore[reportAttributeAccessIssue]
        lab = kc.rgb_to_lab(img)[0]

        L = lab[0:1] / 100.0
        ab = lab[1:3] / 110.0

        return L, ab


def lab_to_rgb_torch(L, ab):
    lab = torch.cat([L * 100.0, ab * 110.0], dim=1)
    rgb = kc.lab_to_rgb(lab).clamp(0, 1)
    return rgb


class ColorLoss(nn.Module):
    def __init__(
        self, device, l1_weight=1.0, perceptual_weight=0.3, confidence_weight=0.1
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.confidence_weight = confidence_weight

        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = lpips.LPIPS(net="alex").to(device)

    def forward(self, pred_colors, pred_confidence, target_colors, L_channel):

        l1_loss = self.l1_loss(pred_colors, target_colors)

        confidence_loss = torch.mean((1.0 - pred_confidence) ** 2)

        if self.perceptual_weight > 0:
            pred_rgb = lab_to_rgb_torch(L_channel, pred_colors)
            target_rgb = lab_to_rgb_torch(L_channel, target_colors)

            perc_loss = self.perceptual_loss(
                pred_rgb * 2 - 1, target_rgb * 2 - 1
            ).mean()
        else:
            perc_loss = torch.tensor(0.0, device=pred_colors.device)

        weighted_l1 = torch.mean(
            self.l1_loss(pred_colors, target_colors) * (pred_confidence + 0.1)
        )

        total_loss = (
            self.l1_weight * l1_loss
            + self.perceptual_weight * perc_loss
            + self.confidence_weight * confidence_loss
            + 0.5 * weighted_l1
        )

        return total_loss, {
            "l1_loss": l1_loss.item(),
            "perceptual_loss": (
                perc_loss.item() if isinstance(perc_loss, torch.Tensor) else 0.0
            ),
            "confidence_loss": confidence_loss.item(),
            "weighted_l1": weighted_l1.item(),
        }


def train_one_epoch(
    model, dataloader, criterion, optimizer, device, scaler, epoch, total_epochs
):
    model.train()
    total_loss = 0
    loss_components = {
        "l1_loss": 0,
        "perceptual_loss": 0,
        "confidence_loss": 0,
        "weighted_l1": 0,
    }

    progress_bar = tqdm(
        dataloader, desc=f"Epoch {epoch}/{total_epochs} [Training]", leave=False
    )

    for batch_idx, (L, ab) in enumerate(progress_bar):
        L, ab = L.to(device), ab.to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            pred_ab, pred_confidence = model(L)
            loss, components = criterion(pred_ab, pred_confidence, ab, L)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] = loss_components[key] + components[key]

        progress_bar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "l1": f'{components["l1_loss"]:.4f}',
                "perc": f'{components["perceptual_loss"]:.4f}',
            }
        )

    avg_loss = total_loss / len(dataloader)
    for key in loss_components:
        loss_components[key] = loss_components[key] / len(dataloader) # pyright: ignore[reportArgumentType]

    return avg_loss, loss_components


def validate(model, dataloader, criterion, device, epoch, total_epochs):
    model.eval()
    total_loss = 0
    loss_components = {
        "l1_loss": 0,
        "perceptual_loss": 0,
        "confidence_loss": 0,
        "weighted_l1": 0,
    }

    with torch.no_grad():
        progress_bar = tqdm(
            dataloader, desc=f"Epoch {epoch}/{total_epochs} [Validation]", leave=False
        )

        for L, ab in progress_bar:
            L, ab = L.to(device), ab.to(device)

            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                pred_ab, pred_confidence = model(L)
                loss, components = criterion(pred_ab, pred_confidence, ab, L)

            total_loss += loss.item()
            for key in loss_components:
                loss_components[key] += components[key]

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    for key in loss_components:
        loss_components[key] = loss_components[key] / len(dataloader) # pyright: ignore[reportArgumentType]

    return avg_loss, loss_components


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(model, optimizer, filepath, device):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    return epoch, loss


def main():

    try:
        with open("hyperparameters.json", "r") as f:
            hparams = json.load(f)
    except FileNotFoundError:
        print("hyperparameters.json not found, using defaults")
        hparams = {}

    config = {
        "resolution": hparams.get("resolution", 256),
        "batch_size": hparams.get("batch_size", 8),
        "learning_rate": hparams.get("learning_rate", 1e-4),
        "weight_decay": hparams.get("weight_decay", 1e-5),
        "epochs": hparams.get("epochs", 100),
        "num_workers": hparams.get("num_workers", 4),
        "dropout": hparams.get("dropout", 0.15),
        "vit_embed_dim": hparams.get("vit_embed_dim", 256),
        "vit_heads": hparams.get("vit_heads", 8),
        "num_vit_layers": hparams.get("num_vit_layers", 2),
        "l1_weight": hparams.get("l1_weight", 1.0),
        "perceptual_weight": hparams.get("perceptual_weight", 0.3),
        "confidence_weight": hparams.get("confidence_weight", 0.1),
        "train_dir": hparams.get("train_dir", "./data_subset/train"),
        "val_dir": hparams.get("val_dir", "./data_subset/val"),
        "checkpoint_dir": hparams.get("checkpoint_dir", "./checkpoints"),
        "resume_from": hparams.get("resume_from", None),
    }

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    model = ColorizationNet(
        in_channels=1,
        out_channels=2,
        dropout=config["dropout"],
        vit_embed_dim=config["vit_embed_dim"],
        vit_heads=config["vit_heads"],
        num_vit_layers=config["num_vit_layers"],
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    criterion = ColorLoss(
        device=device,
        l1_weight=config["l1_weight"],
        perceptual_weight=config["perceptual_weight"],
        confidence_weight=config["confidence_weight"],
    )

    train_dataset = ColorizationDataset(config["train_dir"], config["resolution"])
    val_dataset = ColorizationDataset(config["val_dir"], config["resolution"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=True if config["num_workers"] > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=True if config["num_workers"] > 0 else False,
    )

    scaler = torch.GradScaler()

    start_epoch = 1
    if config.get("resume_from") and os.path.exists(config["resume_from"]):
        print(f"Resuming training from {config['resume_from']}")
        start_epoch, _ = load_checkpoint(
            model, optimizer, config["resume_from"], device
        )
        start_epoch += 1
    else:
        print("Starting training from scratch")

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_components": [],
        "val_components": [],
    }

    best_val_loss = float("inf")

    print(f"Starting training for {config['epochs']} epochs")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(start_epoch, config["epochs"] + 1):

        train_loss, train_components = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            epoch,
            config["epochs"],
        )

        val_loss, val_components = validate(
            model, val_loader, criterion, device, epoch, config["epochs"]
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_components"].append(train_components)
        history["val_components"].append(val_components)

        print(f"Epoch {epoch}/{config['epochs']}:")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        checkpoint_path = os.path.join(
            config["checkpoint_dir"], f"checkpoint_epoch_{epoch:03d}.pt"
        )
        save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(config["checkpoint_dir"], "best_model.pt")
            save_checkpoint(model, optimizer, epoch, val_loss, best_path)
            print(f"  New best model saved! Val Loss: {val_loss:.4f}")

        with open(
            os.path.join(config["checkpoint_dir"], "training_history.json"), "w"
        ) as f:
            json.dump({"config": config, "history": history}, f, indent=2)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
