import os
import math
import cv2
import base64
import torch
import numpy as np
import gradio as gr
from PIL import Image
import tempfile

try:
    import kornia.color as kc
except Exception:
    kc = None

from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from model import ViTUNetColorizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKPT = "checkpoints/checkpoint_epoch_017_20250810_193435.pt"
model = None
if os.path.exists(CKPT):
    print(f"Loading model from: {CKPT}")
    model = ViTUNetColorizer(vit_model_name="vit_tiny_patch16_224").to(device)
    state = torch.load(CKPT, map_location=device)
    sd = state.get("generator_state_dict", state)
    model.load_state_dict(sd)
    model.eval()
else:
    print(f"Warning: Checkpoint not found at {CKPT}. The app will not be able to colorize images.")

def to_L(rgb_np: np.ndarray):
    if kc is None:
        gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
        L = gray / 100.0
        return torch.from_numpy(L).unsqueeze(0).unsqueeze(0).float().to(device)
    t = torch.from_numpy(rgb_np.astype(np.float32)/255.).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        return kc.rgb_to_lab(t)[:,0:1]/100.0

def lab_to_rgb(L, ab):
    if kc is None:
        lab = torch.cat([L*100.0, torch.clamp(ab, -1, 1)*110.0], dim=1)[0].permute(1,2,0).cpu().numpy()
        lab = np.clip(lab, [0,-128,-128], [100,127,127]).astype(np.float32)
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return (np.clip(rgb,0,1)*255).astype(np.uint8)
    lab = torch.cat([L*100.0, torch.clamp(ab, -1, 1)*110.0], dim=1)
    with torch.no_grad():
        rgb = kc.lab_to_rgb(lab)
    return (torch.clamp(rgb,0,1)[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)

def pad_to_multiple(img_np, m=16):
    h,w = img_np.shape[:2]
    ph, pw = math.ceil(h/m)*m, math.ceil(w/m)*m
    return cv2.copyMakeBorder(img_np,0,ph-h,0,pw-w,cv2.BORDER_CONSTANT,value=(0,0,0)), (h,w)

def compute_metrics(pred, gt):
    p = pred.astype(np.float32)/255.; g = gt.astype(np.float32)/255.
    mae  = float(np.mean(np.abs(p-g)))
    psnr = float(psnr_metric(g, p, data_range=1.0))
    try:
        ssim = float(ssim_metric(g, p, channel_axis=2, data_range=1.0, win_size=7))
    except TypeError:
        ssim = float(ssim_metric(g, p, multichannel=True, data_range=1.0, win_size=7))
    return round(mae,4), round(psnr,2), round(ssim,4)

def to_grayscale(image):
    if image is None:
        return None
    return image.convert("L").convert("RGB")

def infer(image: Image.Image, want_metrics: bool):
    if image is None:
        return None, None, None, None, None
    if model is None:
        return None, None, None, None, "<div>Checkpoint not found.</div>"

    pil = image.convert("RGB")
    rgb = np.array(pil)
    
    proc, (oh, ow) = pad_to_multiple(rgb, 16); back = (ow, oh)

    L = to_L(proc)
    with torch.no_grad():
        ab = model(L)
    out = lab_to_rgb(L, ab)

    out = out[:back[1], :back[0]]

    mae = psnr = ssim = None
    if want_metrics:
        mae, psnr, ssim = compute_metrics(out, np.array(pil))

    gray_pil = pil.convert("L").convert("RGB")
    _, bo = cv2.imencode(".jpg", cv2.cvtColor(np.array(gray_pil), cv2.COLOR_RGB2BGR))
    _, bc = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    so = "data:image/jpeg;base64," + base64.b64encode(bo).decode()
    sc = "data:image/jpeg;base64," + base64.b64encode(bc).decode()
 
    compare_html = f"""
    <div style="margin:auto; border-radius:14px; overflow:hidden;">
        <img-comparison-slider>
            <img slot="first" src="{so}" />
            <img slot="second" src="{sc}" />
        </img-comparison-slider>
    </div>
    """

    return out, mae, psnr, ssim, compare_html

def save_for_download(image_array):
    """Saves a NumPy array to a temporary file and returns the path."""
    if image_array is not None:
        pil_img = Image.fromarray(image_array)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            pil_img.save(temp_file.name)
            return temp_file.name
    return None

def make_theme():
    try:
        from gradio.themes.utils import colors, fonts, sizes
        return gr.themes.Soft(
            primary_hue=colors.indigo,
            neutral_hue=colors.gray,
            font=fonts.GoogleFont("Inter"),
        ).set(radius_size=sizes.radius_lg, spacing_size=sizes.spacing_md)
    except Exception:
        return gr.themes.Soft()

THEME = make_theme()

PLACEHOLDER_HTML = """
<div style='display:flex; justify-content:center; align-items:center; height:480px; border: 2px dashed #4B5563; border-radius:12px; color:#4B5563; font-family:sans-serif;'>
    <span>Result will be shown here</span>
</div>
"""

HEAD = """
<script type="module" src="https://unpkg.com/img-comparison-slider@8/dist/index.js"></script>
<link rel="stylesheet" href="https://unpkg.com/img-comparison-slider@8/dist/themes/default.css" />
"""

with gr.Blocks(theme=THEME, title="Image Colorizer", head=HEAD) as demo:
    gr.Markdown("# 🎨 Image Colorizer\nWorks best on natural scenes. Learn more about the dataset we trained on [here.](http://places.csail.mit.edu/)")
    
    result_state = gr.State()

    with gr.Row():
        with gr.Column(scale=5):
            img_in = gr.Image(
                label="Upload image",
                type="pil",
                image_mode="RGB",
                height=320,
                sources=["upload", "webcam", "clipboard"]
            )
            img_in.upload(fn=to_grayscale, inputs=img_in, outputs=img_in)
            show_m = gr.Checkbox(label="Show metrics", value=True)
            with gr.Row():
                run = gr.Button("Colorize")
                clr = gr.Button("Clear")
                download_btn = gr.DownloadButton("Download Result", visible=False)

            examples = gr.Examples(
                examples=[os.path.join("examples", f) for f in os.listdir("examples")] if os.path.exists("examples") else [],
                inputs=img_in,
                examples_per_page=8,
                label=None
            )

        with gr.Column(scale=7):
            out_html = gr.HTML(label="Result", value=PLACEHOLDER_HTML)
            with gr.Row():
                mae_box  = gr.Number(label="MAE",       interactive=False, precision=4)
                psnr_box = gr.Number(label="PSNR (dB)", interactive=False, precision=2)
                ssim_box = gr.Number(label="SSIM",      interactive=False, precision=4)

    def _go(image, want_metrics):
        out_image, mae, psnr, ssim, cmp_html = infer(image, want_metrics)
        if not want_metrics:
            mae = psnr = ssim = None
        
        download_button_update = gr.update(visible=True) if out_image is not None else gr.update(visible=False)
        
        return out_image, cmp_html, mae, psnr, ssim, download_button_update

    run.click(
        _go,
        inputs=[img_in, show_m],
        outputs=[result_state, out_html, mae_box, psnr_box, ssim_box, download_btn]
    )

    def _clear():
        return None, None, PLACEHOLDER_HTML, None, None, None, gr.update(visible=False)
    
    clr.click(
        _clear,
        inputs=None,
        outputs=[img_in, result_state, out_html, mae_box, psnr_box, ssim_box, download_btn]
    )
    
    download_btn.click(
        save_for_download,
        inputs=[result_state],
        outputs=[download_btn]
    )

if __name__ == "__main__":
    try:
        demo.launch(show_api=False)
    except TypeError:
        demo.launch()