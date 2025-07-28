import os
import re
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from model import ColorizationNet
from PIL import Image
import kornia.color as kc

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
)
device_type = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

autocast_ctx = torch.autocast(device_type=device_type)


def lab_to_rgb_torch(L, ab):
    lab = torch.cat([(L * 100.0), (ab + 1.0) * 127.5 - 128.0], dim=0).unsqueeze(0)
    rgb = kc.lab_to_rgb(lab).squeeze(0).clamp(0, 1)
    return rgb


def save_prediction(pred_tensor, save_path):
    save_image(pred_tensor.clamp(0, 1), save_path)


def find_latest_checkpoint(ckpt_dir="checkpoints"):
    pattern = re.compile(r"model_e(\d+)_(\d{8}_\d{6})\.pt$")
    candidates = []
    for fname in os.listdir(ckpt_dir):
        m = pattern.match(fname)
        if m:
            epoch = int(m.group(1))
            ts_key = int(m.group(2).replace("_", ""))
            candidates.append((ts_key, epoch, fname))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints matching pattern in {ckpt_dir}")
    candidates.sort(reverse=True)
    return os.path.join(ckpt_dir, candidates[0][2])


def load_hparams(path="hyperparameters.json"):
    with open(path, "r") as f:
        return json.load(f)


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
        img = np.array(img).astype(np.float32) / 255.0

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        lab = kc.rgb_to_lab(img_tensor)[0]
        L = lab[0:1] / 100.0
        ab = (lab[1:3] + 128.0) / 127.5 - 1.0
        return L, ab, filename


def mae_metric(pred, target):
    return torch.mean(torch.abs(pred - target)).item()


def psnr_metric(pred, target):
    p = pred.cpu().numpy().transpose(1, 2, 0)
    t = target.cpu().numpy().transpose(1, 2, 0)
    return compare_psnr(t, p, data_range=1.0)


def ssim_metric(pred, target):
    p = pred.cpu().numpy().transpose(1, 2, 0)
    t = target.cpu().numpy().transpose(1, 2, 0)
    return compare_ssim(t, p, channel_axis=2, data_range=1.0, win_size=7)


def evaluate_model(checkpoint_path=None):
    hparams = load_hparams("hyperparameters.json")
    rgb_test_dir = "./data_subset/test"
    save_dir = "./predictions"
    metrics_json = "eval_metrics.json"

    batch_size = hparams.get("batch_size", 8)
    vit_embed_dim = hparams.get("vit_embed_dim", 128)
    vit_heads = hparams.get("vit_heads", 4)
    dropout = hparams.get("dropout", 0.1)
    n_workers = hparams.get("num_workers", 4)

    os.makedirs(save_dir, exist_ok=True)

    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint("checkpoints")
    print(f"Using checkpoint: {checkpoint_path}")

    test_dataset = ColorizationDataset(rgb_test_dir)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=False
    )

    model = ColorizationNet(
        in_channels=1,
        out_channels=2,
        vit_embed_dim=vit_embed_dim,
        vit_heads=vit_heads,
        dropout=dropout,
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    metrics = {"MAE": [], "PSNR": [], "SSIM": []}

    with torch.no_grad():
        for L, ab, fnames in tqdm(test_loader, desc="Evaluating", leave=True):
            L, ab = L.to(device), ab.to(device)
            with autocast_ctx:
                pred_ab = model(L)
                for i in range(L.size(0)):
                    pred_ab_i = pred_ab[i].cpu()
                    input_L = L[i].cpu()
                    fname = fnames[i]
                    pred_rgb = lab_to_rgb_torch(input_L, pred_ab_i)
                    save_prediction(pred_rgb, os.path.join(save_dir, fname))
                    pred_rgb_tensor = pred_rgb.clamp(0, 1).float()

                    gt_path = os.path.join(rgb_test_dir, fname)
                    gt_img = Image.open(gt_path).convert("RGB")
                    gt_rgb = np.array(gt_img).astype(np.float32) / 255.0
                    gt_rgb = np.transpose(gt_rgb, (2, 0, 1))
                    gt_rgb_tensor = torch.from_numpy(gt_rgb).float()

                    if pred_rgb_tensor.shape != gt_rgb_tensor.shape:
                        H, W = gt_rgb_tensor.shape[1:]
                        pred_rgb_tensor = torch.nn.functional.interpolate(
                            pred_rgb_tensor.unsqueeze(0),
                            size=(H, W),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)

                    metrics["MAE"].append(mae_metric(pred_rgb_tensor, gt_rgb_tensor))
                    metrics["PSNR"].append(psnr_metric(pred_rgb_tensor, gt_rgb_tensor))
                    metrics["SSIM"].append(ssim_metric(pred_rgb_tensor, gt_rgb_tensor))

    avg_metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
    with open(metrics_json, "w") as f:
        json.dump(avg_metrics, f, indent=4)

    print("Evaluation complete. Metrics saved to", metrics_json)
    print(avg_metrics)
    return avg_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate colorization model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pt). If omitted, latest in ./checkpoints is used.",
    )
    args = parser.parse_args()
    evaluate_model(checkpoint_path=args.checkpoint)
