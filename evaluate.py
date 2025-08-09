import os
import re
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torchvision.transforms as T
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from model import ViTUNetColorizer
from PIL import Image
import kornia.color as kc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def lab_to_rgb_torch(L, ab):
    lab = torch.cat([L * 100.0, ab * 110.0], dim=0).unsqueeze(0)
    rgb = kc.lab_to_rgb(lab).squeeze(0).clamp(0, 1)
    return rgb


def save_prediction(pred_tensor, save_path):
    save_image(pred_tensor.clamp(0, 1), save_path)


def find_best_checkpoint(ckpt_dir="checkpoints"):
    pattern = re.compile(r"checkpoint_epoch_(\d+)_(\d{8}_\d{6})\.pt$")
    candidates = []

    print(f"Searching for checkpoints in '{ckpt_dir}'...")
    for fname in os.listdir(ckpt_dir):
        if pattern.match(fname):
            candidates.append(fname)

    if not candidates:
        raise FileNotFoundError(
            f"No valid timestamped checkpoints found in '{ckpt_dir}'. "
            "Please ensure checkpoints are named like 'checkpoint_epoch_XXX_YYYYMMDD_HHMMSS.pt'."
        )

    candidates.sort()

    latest_checkpoint = candidates[-1]
    print(f"No checkpoint specified. Found latest checkpoint: {latest_checkpoint}")
    return os.path.join(ckpt_dir, latest_checkpoint)


def load_checkpoint(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "generator_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["generator_state_dict"])
        epoch = checkpoint.get("epoch", 0)
        loss = checkpoint.get("loss", 0)
        print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded checkpoint (state_dict only)")


def load_hparams(path="hyperparameters.json"):
    defaults = {
        "batch_size": 16,
        "resolution": 256,
        "num_workers": 4,
    }

    try:
        with open(path, "r") as f:
            hparams = json.load(f)
    except FileNotFoundError:
        print(f"Warning: {path} not found, using defaults")
        hparams = {}

    for key, value in defaults.items():
        hparams.setdefault(key, value)

    return hparams


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
        return L, ab, filename


def mae_metric(pred, target):
    return torch.mean(torch.abs(pred - target)).item()


def psnr_metric(pred, target):
    pred_np = pred.cpu().numpy().transpose(1, 2, 0)
    target_np = target.cpu().numpy().transpose(1, 2, 0)
    return compare_psnr(target_np, pred_np, data_range=1.0)


def ssim_metric(pred, target):
    pred_np = pred.cpu().numpy().transpose(1, 2, 0)
    target_np = target.cpu().numpy().transpose(1, 2, 0)
    win_size = min(7, pred_np.shape[0], pred_np.shape[1])
    if win_size % 2 == 0:
        win_size -= 1

    return compare_ssim(
        target_np, pred_np, channel_axis=2, data_range=1.0, win_size=win_size
    )


def calculate_color_metrics(pred_ab, target_ab):
    mae_ab = mae_metric(pred_ab, target_ab)
    pred_sat = torch.sqrt(pred_ab[0] ** 2 + pred_ab[1] ** 2)
    target_sat = torch.sqrt(target_ab[0] ** 2 + target_ab[1] ** 2)
    saturation_diff = torch.mean(torch.abs(pred_sat - target_sat)).item()
    return {"mae_ab": mae_ab, "saturation_diff": saturation_diff}


def evaluate_model(checkpoint_path=None, test_dir=None, save_predictions=True):
    hparams = load_hparams("hyperparameters.json")
    if test_dir is None:
        test_dir = "./data_subset/test"
    save_dir = "./predictions"
    metrics_json = "eval_metrics.json"

    if save_predictions:
        os.makedirs(save_dir, exist_ok=True)

    if checkpoint_path is None:
        checkpoint_path = find_best_checkpoint("checkpoints")
    print(f"Using checkpoint: {checkpoint_path}")

    test_dataset = ColorizationDataset(test_dir)
    test_loader = DataLoader(
        test_dataset,
        batch_size=hparams["batch_size"],
        num_workers=hparams["num_workers"],
        shuffle=False,
        pin_memory=True,
    )
    print(f"Found {len(test_dataset)} test images in '{test_dir}'")

    model = ViTUNetColorizer(vit_model_name="vit_tiny_patch16_224").to(device)
    load_checkpoint(checkpoint_path, model, device)
    model.eval()

    metrics = {
        "RGB_MAE": [],
        "RGB_PSNR": [],
        "RGB_SSIM": [],
        "AB_MAE": [],
        "Saturation_Diff": [],
    }

    print("Starting evaluation...")
    with torch.no_grad():
        for L, ab_gt, fnames in tqdm(test_loader, desc="Evaluating"):
            L, ab_gt = L.to(device), ab_gt.to(device)
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                pred_ab = model(L)

            for i in range(L.size(0)):
                pred_ab_i = pred_ab[i].cpu()
                input_L_i = L[i].cpu()
                ab_gt_i = ab_gt[i].cpu()
                fname = fnames[i]
                pred_rgb = lab_to_rgb_torch(input_L_i, pred_ab_i)
                gt_rgb = lab_to_rgb_torch(input_L_i, ab_gt_i)

                if save_predictions:
                    save_prediction(pred_rgb, os.path.join(save_dir, fname))

                metrics["RGB_MAE"].append(mae_metric(pred_rgb, gt_rgb))
                metrics["RGB_PSNR"].append(psnr_metric(pred_rgb, gt_rgb))
                metrics["RGB_SSIM"].append(ssim_metric(pred_rgb, gt_rgb))
                color_metrics = calculate_color_metrics(pred_ab_i, ab_gt_i)
                metrics["AB_MAE"].append(color_metrics["mae_ab"])
                metrics["Saturation_Diff"].append(color_metrics["saturation_diff"])

    avg_metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
    avg_metrics["Total_Images"] = len(test_dataset)
    avg_metrics["Checkpoint"] = os.path.basename(checkpoint_path) # type: ignore

    with open(metrics_json, "w") as f:
        json.dump(avg_metrics, f, indent=4)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total Images: {avg_metrics['Total_Images']}")
    print(f"Checkpoint: {avg_metrics['Checkpoint']}")
    print("-" * 50)
    print("RGB Metrics:")
    print(f"  MAE:  {avg_metrics['RGB_MAE']:.4f}")
    print(f"  PSNR: {avg_metrics['RGB_PSNR']:.2f} dB")
    print(f"  SSIM: {avg_metrics['RGB_SSIM']:.4f}")
    print("-" * 50)
    print("Color Metrics (LAB Space):")
    print(f"  AB MAE:           {avg_metrics['AB_MAE']:.4f}")
    print(f"  Saturation Diff:  {avg_metrics['Saturation_Diff']:.4f}")
    print("=" * 50)

    if save_predictions:
        print(f"Predictions saved to: {save_dir}")
    print(f"Metrics saved to: {metrics_json}")

    return avg_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate colorization model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint. If omitted, uses latest checkpoint.",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=None,
        help="Path to test images directory. Default: ./data_subset/test",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Don't save prediction images (faster evaluation)",
    )
    args = parser.parse_args()
    evaluate_model(
        checkpoint_path=args.checkpoint,
        test_dir=args.test_dir,
        save_predictions=not args.no_save,
    )
