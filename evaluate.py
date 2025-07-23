import os
import re
import json
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from model import UNetViT

"""
evaluate.py

Evaluates a UNetViT colorization model on the test split.

- If --checkpoint is NOT provided, the newest checkpoint in ./checkpoints
  (by timestamp in filename model_e<epoch>_<YYYYMMDD_HHMMSS>.pt) is used.
- vit_embed_dim and vit_heads are read from hyperparameters.json unless
  explicitly supplied via CLI.

Outputs:
- predictions/ : generated RGB images
- eval_results.json : averaged MAE, PSNR, SSIM metrics
"""

device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))


class ColorizationDataset(Dataset):
    def __init__(self, gray_dir, rgb_dir):
        self.gray_dir = gray_dir
        self.rgb_dir = rgb_dir
        self.filenames = [f for f in os.listdir(gray_dir)
                          if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        self.gray_tf = transforms.ToTensor()
        self.rgb_tf = transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        g_path = os.path.join(self.gray_dir, fname)
        r_path = os.path.join(self.rgb_dir, fname)

        gray = Image.open(g_path).convert("L")
        rgb = Image.open(r_path).convert("RGB")

        return self.gray_tf(gray), self.rgb_tf(rgb), fname


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


def evaluate_model(checkpoint_path=None,
                   gray_test_dir="./data_subset/gray/test",
                   rgb_test_dir="./data_subset/rgb/test",
                   save_dir="./predictions",
                   metrics_json="eval_results.json",
                   batch_size=8,
                   vit_embed_dim=None,
                   vit_heads=None,
                   hparam_path="hyperparameters.json"):
    os.makedirs(save_dir, exist_ok=True)

    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint("checkpoints")
    print(f"Using checkpoint: {checkpoint_path}")

    if vit_embed_dim is None or vit_heads is None:
        hparams = load_hparams(hparam_path)
        vit_embed_dim = vit_embed_dim or hparams.get("vit_embed_dim", 512)
        vit_heads = vit_heads or hparams.get("vit_heads", 8)

    test_dataset = ColorizationDataset(gray_test_dir, rgb_test_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = UNetViT(vit_embed_dim=vit_embed_dim, vit_heads=vit_heads).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_idict = model.load_state_dict(state)   # type: ignore

    metrics = {"MAE": [], "PSNR": [], "SSIM": []}

    model.eval()
    with torch.no_grad():
        for gray, rgb, fnames in tqdm(test_loader, desc="Evaluating", leave=True):
            gray, rgb = gray.to(device), rgb.to(device)
            preds = model(gray)
            for pred, tgt, fname in zip(preds, rgb, fnames):
                metrics["MAE"].append(mae_metric(pred, tgt))
                metrics["PSNR"].append(psnr_metric(pred, tgt))
                metrics["SSIM"].append(ssim_metric(pred, tgt))
                save_prediction(pred, os.path.join(save_dir, fname))

    avg_metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
    with open(metrics_json, "w") as f:
        json.dump(avg_metrics, f, indent=4)

    print("Evaluation complete. Metrics saved to", metrics_json)
    return avg_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate UNetViT colorization model.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pt). If omitted, latest in ./checkpoints is used.")
    parser.add_argument("--batch_size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--gray_dir", type=str, default="./data_subset/gray/test")
    parser.add_argument("--rgb_dir", type=str, default="./data_subset/rgb/test")
    parser.add_argument("--pred_dir", type=str, default="./predictions")
    parser.add_argument("--metrics_out", type=str, default="eval_results.json")
    parser.add_argument("--hparams", type=str, default="hyperparameters.json",
                        help="Path to hyperparameters JSON.")
    parser.add_argument("--vit_embed_dim", type=int, default=None,
                        help="Override vit_embed_dim (else taken from hparams).")
    parser.add_argument("--vit_heads", type=int, default=None,
                        help="Override vit_heads (else taken from hparams).")
    args = parser.parse_args()

    evaluate_model(checkpoint_path=args.checkpoint,
                   gray_test_dir=args.gray_dir,
                   rgb_test_dir=args.rgb_dir,
                   save_dir=args.pred_dir,
                   metrics_json=args.metrics_out,
                   batch_size=args.batch_size,
                   vit_embed_dim=args.vit_embed_dim,
                   vit_heads=args.vit_heads,
                   hparam_path=args.hparams)
