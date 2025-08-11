import os
import random
import numpy as np
import json
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image
from model import ViTUNetColorizer, PatchDiscriminator
import kornia.color as kc
from skimage import color
import argparse
from datetime import datetime

from evaluate import mae_metric, psnr_metric, ssim_metric

import warnings

warnings.filterwarnings("ignore")


class PerceptualLoss(nn.Module):
    def __init__(self, loss_type="l2", resize=True):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.eval()
        self.feature_layers = [4, 9, 16, 23, 30]
        self.vgg = nn.ModuleList([vgg[i] for i in range(max(self.feature_layers) + 1)])  # type: ignore

        for param in self.vgg.parameters():
            param.requires_grad = False

        self.loss_type = loss_type
        self.resize = resize

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _preprocess(self, img):
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)
        if self.resize:
            img = F.interpolate(
                img, size=(224, 224), mode="bicubic", align_corners=False
            )
        img = (img - self.mean) / self.std
        return img

    def forward(self, input, target):
        input = self._preprocess(input)
        target = self._preprocess(target)
        loss = 0.0
        x, y = input, target
        for i, layer in enumerate(self.vgg):
            x, y = layer(x), layer(y)
            if i in self.feature_layers:
                if self.loss_type == "l1":
                    loss += F.l1_loss(x, y)
                else:
                    loss += F.mse_loss(x, y)
        return loss / len(self.feature_layers)


class WeightedL1Loss(nn.Module):
    def __init__(
        self, weights_tensor=None, bin_edges_a=None, bin_edges_b=None, num_bins=64
    ):
        super().__init__()
        self.num_bins = num_bins
        if weights_tensor is not None:
            self.register_buffer("weights", weights_tensor)
            self.bin_edges_a = bin_edges_a
            self.bin_edges_b = bin_edges_b
        else:
            self.weights = None

    def set_weights(self, weights_tensor, bin_edges_a, bin_edges_b):
        self.register_buffer("weights", weights_tensor)
        self.bin_edges_a = bin_edges_a
        self.bin_edges_b = bin_edges_b

    def get_pixel_weights(self, rgb_images):
        B, C, H, W = rgb_images.shape
        device = rgb_images.device
        if self.weights is None:
            return torch.ones(B, H, W, device=device)

        rgb_np = rgb_images.permute(0, 2, 3, 1).cpu().numpy()
        lab_batch = color.rgb2lab(rgb_np)
        a_channel = lab_batch[:, :, :, 1]
        b_channel = lab_batch[:, :, :, 2]

        a_bins = np.digitize(a_channel, self.bin_edges_a) - 1  # type: ignore
        b_bins = np.digitize(b_channel, self.bin_edges_b) - 1  # type: ignore
        a_bins = np.clip(a_bins, 0, self.num_bins - 1)
        b_bins = np.clip(b_bins, 0, self.num_bins - 1)

        pixel_weights_np = self.weights.cpu().numpy()[a_bins, b_bins]

        return torch.from_numpy(pixel_weights_np).to(device)

    def forward(self, pred_ab, target_ab, target_rgb):
        raw_l1_metric = torch.abs(pred_ab - target_ab).mean()
        l1_loss_map = torch.abs(pred_ab - target_ab).mean(dim=1)
        pixel_weights = self.get_pixel_weights(target_rgb)
        weighted_l1_loss = (l1_loss_map * pixel_weights).mean()

        return weighted_l1_loss, raw_l1_metric


class CombinedLoss(nn.Module):
    def __init__(
        self,
        lambda_l1,
        lambda_perceptual,
        lambda_adv,
        weights_tensor,
        bin_edges_a,
        bin_edges_b,
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_adv = lambda_adv
        self.weighted_l1 = WeightedL1Loss(weights_tensor, bin_edges_a, bin_edges_b)
        self.perceptual = PerceptualLoss(loss_type="l2", resize=True)

    def generator_loss(self, pred_ab, target_ab, pred_rgb, target_rgb, disc_pred_fake):
        weighted_l1_component, raw_l1_metric = self.weighted_l1(
            pred_ab, target_ab, target_rgb
        )
        perceptual_loss = self.perceptual(pred_rgb, target_rgb)
        adv_loss = torch.mean((disc_pred_fake - 1) ** 2)

        total_loss = (
            self.lambda_l1 * weighted_l1_component
            + self.lambda_perceptual * perceptual_loss
            + self.lambda_adv * adv_loss
        )

        return total_loss, {
            "l1": raw_l1_metric.item(),
            "perceptual": perceptual_loss.item(),
            "adversarial": adv_loss.item(),
        }

    def discriminator_loss(self, disc_pred_real, disc_pred_fake):
        real_loss = torch.mean((disc_pred_real - 1) ** 2)
        fake_loss = torch.mean(disc_pred_fake**2)
        total_loss = 0.5 * (real_loss + fake_loss)

        return total_loss, {"real": real_loss.item(), "fake": fake_loss.item()}


class ColorizationDataset(Dataset):
    def __init__(self, rgb_dir):
        self.rgb_dir = rgb_dir
        self.filenames = [
            f
            for f in os.listdir(rgb_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        rgb_path = os.path.join(self.rgb_dir, filename)

        img = Image.open(rgb_path).convert("RGB")
        img = T.ToTensor()(img)
        img = img.unsqueeze(0)
        lab = kc.rgb_to_lab(img)[0]

        L = lab[0:1] / 100.0
        ab = lab[1:3] / 110.0

        return L, ab


def lab_to_rgb_torch(L, ab):
    lab = torch.cat([L * 100.0, ab * 110.0], dim=1)
    rgb = kc.lab_to_rgb(lab).clamp(0, 1)
    return rgb


def train_one_epoch(
    generator,
    discriminator,
    dataloader,
    criterion,
    optimizer_G,
    optimizer_D,
    device,
    scaler,
    epoch,
    total_epochs,
):
    generator.train()
    discriminator.train()
    generator.set_epoch(epoch)

    total_g_loss = 0.0
    total_d_loss = 0.0
    loss_components = {"l1_loss": 0.0, "perceptual_loss": 0.0, "adversarial_loss": 0.0}

    progress_bar = tqdm(
        dataloader, desc=f"Epoch {epoch}/{total_epochs} [Training]", leave=False
    )

    for _, (L, ab) in enumerate(progress_bar):
        L, ab = L.to(device), ab.to(device)

        optimizer_D.zero_grad()
        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            fake_ab = generator(L)

            pred_real = discriminator(L, ab)
            pred_fake = discriminator(L, fake_ab.detach())

            d_loss, _ = criterion.discriminator_loss(pred_real, pred_fake)

        scaler.scale(d_loss).backward()
        scaler.step(optimizer_D)
        scaler.update()

        optimizer_G.zero_grad()
        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            fake_ab_gen = generator(L)
            pred_fake_gen = discriminator(L, fake_ab_gen)

            fake_rgb = lab_to_rgb_torch(L, fake_ab_gen)
            real_rgb = lab_to_rgb_torch(L, ab)

            g_loss, g_metrics = criterion.generator_loss(
                pred_ab=fake_ab_gen,
                target_ab=ab,
                pred_rgb=fake_rgb,
                target_rgb=real_rgb,
                disc_pred_fake=pred_fake_gen,
            )

        scaler.scale(g_loss).backward()
        scaler.step(optimizer_G)
        scaler.update()

        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()
        for key in ["l1", "perceptual", "adversarial"]:
            if key in g_metrics:
                loss_components[key + "_loss"] += g_metrics[key]

        progress_bar.set_postfix(
            {
                "G_loss": f"{g_loss.item():.4f}",
                "D_loss": f"{d_loss.item():.4f}",
                "l1": f'{g_metrics["l1"]:.4f}',
            }
        )

    avg_g_loss = total_g_loss / len(dataloader)
    avg_d_loss = total_d_loss / len(dataloader)

    for key in loss_components:
        loss_components[key] = float(loss_components[key] / len(dataloader))

    return avg_g_loss, avg_d_loss, loss_components


def validate(
    generator, discriminator, dataloader, criterion, device, epoch, total_epochs
):
    generator.eval()
    discriminator.eval()
    total_g_loss = 0.0
    total_d_loss = 0.0

    total_mae = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0

    with torch.no_grad():
        progress_bar = tqdm(
            dataloader, desc=f"Epoch {epoch}/{total_epochs} [Validation]", leave=False
        )

        for L, ab in progress_bar:
            L, ab = L.to(device), ab.to(device)

            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                fake_ab = generator(L)
                pred_real = discriminator(L, ab)
                pred_fake = discriminator(L, fake_ab)

                fake_rgb = lab_to_rgb_torch(L, fake_ab)
                real_rgb = lab_to_rgb_torch(L, ab)

                g_loss, _ = criterion.generator_loss(
                    pred_ab=fake_ab,
                    target_ab=ab,
                    pred_rgb=fake_rgb,
                    target_rgb=real_rgb,
                    disc_pred_fake=pred_fake,
                )
                d_loss, _ = criterion.discriminator_loss(pred_real, pred_fake)

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

            batch_mae = 0.0
            batch_psnr = 0.0
            batch_ssim = 0.0

            for i in range(L.size(0)):
                pred_rgb_np = fake_rgb[i].cpu()
                target_rgb_np = real_rgb[i].cpu()

                batch_mae += float(mae_metric(pred_rgb_np, target_rgb_np))
                batch_psnr += float(psnr_metric(pred_rgb_np, target_rgb_np))
                batch_ssim += float(ssim_metric(pred_rgb_np, target_rgb_np))  # type: ignore

            batch_size = L.size(0)
            total_mae += batch_mae
            total_psnr += batch_psnr
            total_ssim += batch_ssim
            num_samples += batch_size

            progress_bar.set_postfix(
                {
                    "G_loss": f"{g_loss.item():.4f}",
                    "D_loss": f"{d_loss.item():.4f}", 
                    "mae": f"{float(batch_mae/batch_size):.4f}",
                    "psnr": f"{float(batch_psnr/batch_size):.2f}",
                    "ssim": f"{float(batch_ssim/batch_size):.2f}",
                }
            )

    avg_g_loss = total_g_loss / len(dataloader)
    avg_d_loss = total_d_loss / len(dataloader)

    avg_metrics = {
        "mae": float(total_mae / num_samples),
        "psnr": float(total_psnr / num_samples),
        "ssim": float(total_ssim / num_samples),
    }

    return avg_g_loss, avg_d_loss, avg_metrics


def save_checkpoint(
    generator, discriminator, optimizer_G, optimizer_D, epoch, loss, filepath
):
    checkpoint = {
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_G_state_dict": optimizer_G.state_dict(),
        "optimizer_D_state_dict": optimizer_D.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint_for_resume(
    generator, discriminator, optimizer_G, optimizer_D, filepath, device
):
    print(f"Resuming training from checkpoint: {filepath}")
    checkpoint = torch.load(filepath, map_location=device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
    optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
    return checkpoint["epoch"]


def main():
    parser = argparse.ArgumentParser(description="Train a colorization GAN.")
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint .pt file to resume training from.",
    )
    parser.add_argument(
        "--resume_history",
        type=str,
        default=None,
        help="Path to the training_history.json file to continue logging.",
    )
    args = parser.parse_args()

    try:
        with open("hyperparameters.json", "r") as f:
            hparams = json.load(f)
    except FileNotFoundError:
        print("hyperparameters.json not found, using defaults")
        hparams = {}

    config = {
        "resolution": hparams.get("resolution", 256),
        "batch_size": hparams.get("batch_size", 32),
        "learning_rate": hparams.get("learning_rate", 1e-4),
        "learning_rate_vit": hparams.get("learning_rate_vit", 1e-5),
        "weight_decay": hparams.get("weight_decay", 1e-4),
        "epochs": hparams.get("epochs", 50),
        "num_workers": hparams.get("num_workers", 4),
        "prefetch_factor": hparams.get("prefetch_factor", 2),
        "freeze_vit_epochs": hparams.get("freeze_vit_epochs", 10),
        "lambda_l1": hparams.get("lambda_l1", 100.0),
        "lambda_perceptual": hparams.get("lambda_perceptual", 10.0),
        "lambda_adv": hparams.get("lambda_adv", 1.0),
        "train_dir": hparams.get("train_dir", "./data_subset/train"),
        "val_dir": hparams.get("val_dir", "./data_subset/val"),
        "checkpoint_dir": hparams.get("checkpoint_dir", "./checkpoints"),
    }

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    try:
        lab_weights_data = np.load("lab_weights.npz")
        weights = torch.from_numpy(lab_weights_data["weights"]).float()
        bin_edges_a = lab_weights_data["a_edges"]
        bin_edges_b = lab_weights_data["b_edges"]
    except FileNotFoundError:
        print("Error: 'lab_weights.npz' not found.")
        print("Please run 'generate_data_subset.py' first.")
        return

    weights = weights.to(device)

    generator = ViTUNetColorizer(
        vit_model_name="vit_tiny_patch16_224",
        freeze_vit_epochs=config["freeze_vit_epochs"],
    ).to(device)
    discriminator = PatchDiscriminator(in_channels=3).to(device)

    param_groups = generator.get_param_groups(
        lr_decoder=config["learning_rate"], lr_vit=config["learning_rate_vit"]
    )
    optimizer_G = AdamW(param_groups, weight_decay=config["weight_decay"])
    optimizer_D = AdamW(
        discriminator.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    
    start_epoch = 1
    history = None

    if args.resume_checkpoint:
        last_epoch = load_checkpoint_for_resume(
            generator, discriminator, optimizer_G, optimizer_D, args.resume_checkpoint, device
        )
        start_epoch = last_epoch + 1

    if args.resume_history:
        with open(args.resume_history, "r") as f:
            loaded_data = json.load(f)
        
        if args.resume_checkpoint:
            last_epoch = start_epoch - 1
            num_history_epochs = len(loaded_data["history"]["train_g_loss"])
            if num_history_epochs < last_epoch:
                raise ValueError(
                    f"History file has only {num_history_epochs} epochs, but checkpoint is from epoch {last_epoch}."
                )
        history = loaded_data["history"]
        print(f"Loaded history with {len(history['train_g_loss'])} epochs.")


    if history is None:
        print("Starting training from scratch.")
        history = {
            "train_g_loss": [], "train_d_loss": [],
            "val_g_loss": [], "val_d_loss": [],
            "val_metrics": [], "epoch_duration_seconds": [],
            "lr_g_decoder": [], "lr_g_vit": [], "lr_d": [],
        }

    
    remaining_epochs = config["epochs"] - (start_epoch - 1)
    if remaining_epochs <= 0:
        print("Training is already complete according to the checkpoint epoch.")
        return
        
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=remaining_epochs, eta_min=1e-5)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=remaining_epochs, eta_min=1e-5)
    
    train_dataset = ColorizationDataset(config["train_dir"])
    val_dataset = ColorizationDataset(config["val_dir"])

    criterion = CombinedLoss(
        lambda_l1=config["lambda_l1"],
        lambda_perceptual=config["lambda_perceptual"],
        lambda_adv=config["lambda_adv"],
        weights_tensor=weights,
        bin_edges_a=bin_edges_a,
        bin_edges_b=bin_edges_b,
    ).to(device)

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"], pin_memory=True, persistent_workers=True,
        prefetch_factor=config["prefetch_factor"]
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"], pin_memory=True, persistent_workers=True,
        prefetch_factor=config["prefetch_factor"]
    )
    scaler = torch.GradScaler()

    print(f"Starting training from epoch {start_epoch} for {remaining_epochs} epochs.")
    for epoch in range(start_epoch, config["epochs"] + 1):
        start_time = time.time()
        
        train_g_loss, train_d_loss, _ = train_one_epoch(
            generator, discriminator, train_loader, criterion,
            optimizer_G, optimizer_D, device, scaler, epoch, config["epochs"],
        )
        
        val_g_loss, val_d_loss, val_metrics = validate(
            generator, discriminator, val_loader, criterion,
            device, epoch, config["epochs"],
        )
        
        end_time = time.time()
        epoch_duration = end_time - start_time

        history["train_g_loss"].append(train_g_loss)
        history["train_d_loss"].append(train_d_loss)
        history["val_g_loss"].append(val_g_loss)
        history["val_d_loss"].append(val_d_loss)
        history["val_metrics"].append(val_metrics)
        history["epoch_duration_seconds"].append(epoch_duration)
        
        lrs_g = scheduler_G.get_last_lr()
        history["lr_g_decoder"].append(lrs_g[0])
        history["lr_g_vit"].append(lrs_g[1])
        history["lr_d"].append(scheduler_D.get_last_lr()[0])

        print(f"\nEpoch {epoch}/{config['epochs']} completed in {epoch_duration:.2f}s")
        print(f"  Train -> G_Loss: {train_g_loss:.4f} | D_Loss: {train_d_loss:.4f}")
        print(f"  Val   -> G_Loss: {val_g_loss:.4f} | D_Loss: {val_d_loss:.4f}")
        print(f"  Val Metrics -> MAE: {val_metrics['mae']:.4f} | PSNR: {val_metrics['psnr']:.2f} | SSIM: {val_metrics['ssim']:.3f}")
        print(f"  Learning Rates -> G_Decoder: {lrs_g[0]:.2e}, G_ViT: {lrs_g[1]:.2e}, D: {scheduler_D.get_last_lr()[0]:.2e}")

        scheduler_G.step()
        scheduler_D.step()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(config["checkpoint_dir"], f"checkpoint_epoch_{epoch:03d}_{timestamp}.pt")
        save_checkpoint(
                generator, discriminator, optimizer_G, optimizer_D,
                epoch, val_g_loss, checkpoint_path
            )
        print(f"  Saved checkpoint to {checkpoint_path}")

        with open(os.path.join("training_history.json"), "w") as f:
            json.dump({"config": config, "history": history}, f, indent=2)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nTraining completed! Final metrics saved in training_history.json")


if __name__ == "__main__":
    main()