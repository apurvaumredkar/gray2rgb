import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torchvision.utils import save_image
from tqdm import tqdm
from model import UNetViT


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

        return gray_tensor, rgb_tensor, filename


def mae_metric(pred, target):
    return torch.mean(torch.abs(pred - target)).item()


def psnr_metric(pred, target):
    pred_np = pred.cpu().numpy().transpose(1, 2, 0)
    target_np = target.cpu().numpy().transpose(1, 2, 0)
    return compare_psnr(target_np, pred_np, data_range=1.0)


def ssim_metric(pred, target):
    pred_np = pred.cpu().numpy().transpose(1, 2, 0)
    target_np = target.cpu().numpy().transpose(1, 2, 0)
    ssim = compare_ssim(
        target_np,
        pred_np,
        channel_axis=2,
        data_range=1.0,
        win_size=7
    )
    return ssim


def save_prediction(pred_tensor, save_path):
    save_image(pred_tensor.clamp(0, 1), save_path)


def evaluate(model, test_loader, device, save_dir="./predictions", metrics_json="metrics_results.json"):
    os.makedirs(save_dir, exist_ok=True)

    model.eval()

    metrics = {
        "MAE": [],
        "PSNR": [],
        "SSIM": []
    }

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating", leave=True)
        for gray, rgb, filenames in progress_bar:
            gray, rgb = gray.to(device), rgb.to(device)
            preds = model(gray)

            for pred, target, fname in zip(preds, rgb, filenames):
                metrics["MAE"].append(mae_metric(pred, target))
                metrics["PSNR"].append(psnr_metric(pred, target))
                metrics["SSIM"].append(ssim_metric(pred, target))

                save_path = os.path.join(save_dir, fname)
                save_prediction(pred, save_path)

    avg_metrics = {k: float(np.mean(v)) for k, v in metrics.items()}

    with open(metrics_json, "w") as f:
        json.dump(avg_metrics, f, indent=4)

    return avg_metrics


def main():
    gray_test_dir = "./data_subset/gray/test"
    rgb_test_dir = "./data_subset/rgb/test"

    test_dataset = ColorizationDataset(gray_test_dir, rgb_test_dir)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = UNetViT().to(device)
    checkpoint_path = "checkpoints/model_e10_20250722_040306.pt"  # Update as needed
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    metrics = evaluate(model, test_loader, device)
    print("Evaluation complete. Metrics saved to metrics_results.json")


if __name__ == "__main__":
    main()
