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
from model import UNetViT
import cv2

device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))

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

        
        lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB).astype(np.float32)
        ab = (lab[:,:,1:3] - 128.0) / 128.0
        ab = ab.transpose(2, 0, 1)

        L = torch.from_numpy(L).float()
        ab = torch.from_numpy(ab).float()
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

def lab_to_rgb_torch(L, ab):
    L = (L + 1.0) * 50.0
    ab = ab * 128.0
    lab = torch.cat([L, ab], dim=0).permute(1,2,0).cpu().numpy()  
    lab = np.clip(lab, [0, -128, -128], [100, 127, 127])  
    rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    rgb = np.clip(rgb, 0, 1)
    rgb = torch.from_numpy(rgb).permute(2,0,1).float()
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

def evaluate_model(checkpoint_path=None,
                   rgb_test_dir="./data_subset/test",
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

    test_dataset = ColorizationDataset(rgb_test_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = UNetViT(in_channels=1, out_channels=2, vit_embed_dim=vit_embed_dim, vit_heads=vit_heads).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)

    metrics = {"MAE": [], "PSNR": [], "SSIM": []}

    model.eval()
    with torch.no_grad():
        for L, ab, fnames in tqdm(test_loader, desc="Evaluating", leave=True):
            L = L.to(device)
            preds_ab = model(L)
            for i in range(L.size(0)):
                pred_ab = preds_ab[i].cpu()
                input_L = L[i].cpu()
                fname = fnames[i]

                
                pred_rgb = lab_to_rgb_torch(input_L, pred_ab)
                save_prediction(pred_rgb, os.path.join(save_dir, fname))

                pred_rgb_tensor = pred_rgb.clamp(0, 1).float()

                
                gt_path = os.path.join(rgb_test_dir, fname)
                gt_rgb = cv2.imread(gt_path)
                gt_rgb = cv2.cvtColor(gt_rgb, cv2.COLOR_BGR2RGB)
                gt_rgb = np.clip(gt_rgb / 255.0, 0, 1)
                gt_rgb = np.transpose(gt_rgb, (2, 0, 1))
                gt_rgb_tensor = torch.from_numpy(gt_rgb).float()

                if pred_rgb_tensor.shape != gt_rgb_tensor.shape:
                    H, W = gt_rgb_tensor.shape[1:]
                    pred_rgb_tensor = torch.nn.functional.interpolate(
                        pred_rgb_tensor.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
                    ).squeeze(0)

                metrics["MAE"].append(mae_metric(pred_rgb_tensor, gt_rgb_tensor))
                metrics["PSNR"].append(psnr_metric(pred_rgb_tensor, gt_rgb_tensor))
                metrics["SSIM"].append(ssim_metric(pred_rgb_tensor, gt_rgb_tensor))

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
    parser.add_argument("--rgb_dir", type=str, default="./data_subset/test")
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
                   rgb_test_dir=args.rgb_dir,
                   save_dir=args.pred_dir,
                   metrics_json=args.metrics_out,
                   batch_size=args.batch_size,
                   vit_embed_dim=args.vit_embed_dim,
                   vit_heads=args.vit_heads,
                   hparam_path=args.hparams)