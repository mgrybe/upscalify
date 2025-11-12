import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from .metrics import calculate_psnr, calculate_ssim
import os

# ---------- helpers ----------

def _tensor_to_uint8_hwc(t: torch.Tensor) -> np.ndarray:
    """(C,H,W) float in [0,1] -> (H,W,C) uint8 on CPU."""
    return (t.detach().clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy())

# ---------- metrics (eval) ----------

@torch.inference_mode()
#@torch._dynamo.disable()  # don't graph-capture this routine
def evaluate_sr(model, eval_loader, device, autocast_ctx, scale_factor, test_y_channel=False):
    """
    Returns (mean_psnr, mean_ssim) over eval_loader.
    Assumes calculate_psnr / calculate_ssim are available:
        calculate_psnr(img1, img2, crop_border, img_format='HWC', test_y_channel=False)
    """
    m = model
    was_training = m.training
    m.eval()

    prev_bench = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    psnrs, ssims = [], []
    crop = int(scale_factor)

    try:
        with autocast_ctx:
            for lq, hq in eval_loader:
                lq = lq.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                hq = hq.to(device, non_blocking=True).to(memory_format=torch.channels_last)

                #print(f'lq.shape={lq.shape}, hq.shape={hq.shape}')
                sr = m(lq)  # (B,C,H,W), values ~[0,1]

                # take first in batch
                sr_np = _tensor_to_uint8_hwc(sr[0])
                hq_np = _tensor_to_uint8_hwc(hq[0])

                psnrs.append(calculate_psnr(sr_np, hq_np, crop, 'HWC', test_y_channel))
                ssims.append(calculate_ssim(sr_np, hq_np, crop, 'HWC', test_y_channel))
    finally:
        torch.backends.cudnn.benchmark = prev_bench
        if was_training:
            m.train()

    return float(np.mean(psnrs)), float(np.mean(ssims))

@torch.inference_mode()
def evaluate_sr_v2(
        model,
        eval_loader,
        device,
        autocast_ctx,
        scale_factor: int,
        test_y_channel: bool = False,
        collect_n: int = 0,     # how many (LR, SR, HR) samples to return for TB
):
    """
    Returns (mean_psnr, mean_ssim, samples)
      - mean_psnr, mean_ssim: floats over eval_loader
      - samples: list of (lr, sr, hr) tensors in CHW, float32, [0,1], on CPU
                 ready for SummaryWriter.add_image(..., dataformats="CHW")

    Assumes:
      - calculate_psnr(img1, img2, crop_border, img_format='HWC', test_y_channel=False)
      - calculate_ssim(img1, img2, crop_border, img_format='HWC', test_y_channel=False)
      - _tensor_to_uint8_hwc: (C,H,W) -> (H,W,C) uint8 in [0,255]
    """
    m = model
    was_training = m.training
    m.eval()

    prev_bench = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False

    psnrs, ssims = [], []
    crop = int(scale_factor)
    samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def to_chw01_cpu(x: torch.Tensor) -> torch.Tensor:
        # (C,H,W) float in [0,1], on CPU, contiguous
        x = x.detach().float()
        # If your pipeline is already [0,1], this is a clamp; if 0..255, adapt here if needed.
        x = x.clamp(0.0, 1.0).to("cpu").contiguous()
        return x

    try:
        with autocast_ctx:
            for lq, hq in eval_loader:
                # move to device; keep memory_format if you like
                lq = lq.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                hq = hq.to(device, non_blocking=True).to(memory_format=torch.channels_last)

                sr = m(lq)  # (B,C,H,W), values ~[0,1]

                # metrics on the first item in the batch
                sr_np = _tensor_to_uint8_hwc(sr[0])
                hq_np = _tensor_to_uint8_hwc(hq[0])
                psnrs.append(calculate_psnr(sr_np, hq_np, crop, 'HWC', test_y_channel))
                ssims.append(calculate_ssim(sr_np, hq_np, crop, 'HWC', test_y_channel))

                # optionally collect a few samples for TensorBoard
                if collect_n and len(samples) < collect_n:
                    b = min(sr.size(0), collect_n - len(samples))
                    for i in range(b):
                        # Return CHW float [0,1] on CPU
                        lr_i = to_chw01_cpu(lq[i].movedim(-1, -1))  # no-op, lq is (C,H,W) already
                        sr_i = to_chw01_cpu(sr[i])
                        hr_i = to_chw01_cpu(hq[i])
                        samples.append((lr_i, sr_i, hr_i))
    finally:
        torch.backends.cudnn.benchmark = prev_bench
        if was_training:
            m.train()

    return float(np.mean(psnrs)), float(np.mean(ssims)), samples

@torch.inference_mode()
#@torch._dynamo.disable()  # don't graph-capture this routine
def evaluate_sr_v3(model, eval_loader, device, autocast_ctx, scale_factor, test_y_channel=False, save_dir=None):
    """
    Returns (mean_psnr, mean_ssim) over eval_loader.
    Assumes calculate_psnr / calculate_ssim are available:
        calculate_psnr(img1, img2, crop_border, img_format='HWC', test_y_channel=False)
    """
    m = model
    was_training = m.training
    m.eval()

    prev_bench = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    psnrs, ssims = [], []
    crop = int(scale_factor)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    try:
        with autocast_ctx:
            i = 0
            for lq, hq in eval_loader:
                i = i + 1
                lq = lq.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                hq = hq.to(device, non_blocking=True).to(memory_format=torch.channels_last)

                #print(f'lq.shape={lq.shape}, hq.shape={hq.shape}')
                sr = m(lq)  # (B,C,H,W), values ~[0,1]

                # take first in batch
                sr_np = _tensor_to_uint8_hwc(sr[0])
                hq_np = _tensor_to_uint8_hwc(hq[0])

                psnr = calculate_psnr(sr_np, hq_np, crop, 'HWC', test_y_channel)
                psnrs.append(psnr)
                ssim = calculate_ssim(sr_np, hq_np, crop, 'HWC', test_y_channel)
                ssims.append(ssim)

                if save_dir is not None:
                    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
                    # Low-resolution image
                    axs[0].imshow(torch.permute(lq[0], (1, 2, 0)).cpu().numpy())
                    axs[0].set_title('Low-Resolution')
                    axs[0].axis('off')
                    # Super-resolved image (remove batch dimension)
                    axs[1].imshow(sr_np)
                    axs[1].set_title(f'SR (PSNR: {psnr:.4f}, SSIM: {ssim:.4f})')
                    axs[1].axis('off')
                    # High-resolution image
                    axs[2].imshow(hq_np)
                    axs[2].set_title('High-Resolution')
                    axs[2].axis('off')
                    plt.tight_layout()
                    save_path = os.path.join(save_dir, f'{i + 1}.png')
                    plt.savefig(save_path, dpi=150)
                    plt.close(fig)

    finally:
        torch.backends.cudnn.benchmark = prev_bench
        if was_training:
            m.train()

    return float(np.mean(psnrs)), float(np.mean(ssims))


# ---------- visualization ----------

def _show_triplet(lq_img: torch.Tensor, sr_img: torch.Tensor, hq_img: torch.Tensor,
                  sf: int = 4, title: str | None = None):
    """
    lq_img: (C,Hl,Wl), sr_img/hq_img: (C,Hh,Wh)
    Upscales LQ by nearest for side-by-side comparison.
    """
    lq_up = F.interpolate(lq_img.unsqueeze(0), scale_factor=sf, mode='nearest').squeeze(0)

    lq_np = _tensor_to_uint8_hwc(lq_up)
    sr_np = _tensor_to_uint8_hwc(sr_img)
    hq_np = _tensor_to_uint8_hwc(hq_img)

    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1); plt.imshow(lq_np); plt.title(f"LQ ×{sf} (NN)"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(sr_np);  plt.title("SR (model)");   plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(hq_np);  plt.title("HQ (target)");  plt.axis('off')
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()

@torch.inference_mode()
#@torch._dynamo.disable()
def visualise_sr(model, eval_loader, device, scale_factor, autocast_ctx, num_images=10, test_y_channel=False, compute_metrics=True):
    """
    Shows up to `num_images` triplets from eval_loader.
    If `compute_metrics` is True, computes PSNR/SSIM per-sample for titles.
    Returns a list of dicts with optional metrics for each shown image.
    """
    m = model
    was_training = m.training
    m.eval()

    prev_bench = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False

    results = []
    shown = 0
    crop = int(scale_factor)

    try:
        with autocast_ctx:
            for lq, hq in eval_loader:
                lq = lq.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                hq = hq.to(device, non_blocking=True).to(memory_format=torch.channels_last)

                sr = m(lq)

                for b in range(lq.size(0)):
                    title = None
                    metrics = {}
                    if compute_metrics:
                        sr_np = _tensor_to_uint8_hwc(sr[b])
                        hq_np = _tensor_to_uint8_hwc(hq[b])
                        psnr = calculate_psnr(sr_np, hq_np, crop, 'HWC', test_y_channel)
                        ssim = calculate_ssim(sr_np, hq_np, crop, 'HWC', test_y_channel)
                        metrics = {"psnr": float(psnr), "ssim": float(ssim)}
                        title = f"PSNR: {psnr:.3f} dB | SSIM: {ssim:.4f}"

                    _show_triplet(lq[b], sr[b], hq[b], sf=scale_factor, title=title)
                    results.append(metrics if metrics else {})
                    shown += 1
                    if shown >= num_images:
                        return results
    finally:
        torch.backends.cudnn.benchmark = prev_bench
        if was_training:
            m.train()

    return results