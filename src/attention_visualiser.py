import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math

# ------------- load DINOv2 (ViT-S/14) -------------
backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
backbone.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone.to(device)

# ------------- preprocessing -------------
def preprocess(pil_img, size=518):
    tfm = T.Compose([
        T.Resize(size, antialias=True),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    return tfm(pil_img)

# NOTE: use raw string or forward slashes on Windows to avoid escape issues
img = Image.open(r"C:\Users\Owner\Pictures\c39db240-42ae-11f0-88e1-3f1b11853089.jpg").convert("RGB")
x = preprocess(img).unsqueeze(0).to(device)

# ------------ helper: keep only patch tokens and infer H,W ------------
def extract_patch_tokens(tokens: torch.Tensor):
    """
    tokens: (B, T, C) possibly including CLS/register tokens.
    Returns: patch_tokens (B, N, C), HW where N = HW*HW.
    """
    B, T, C = tokens.shape

    # try to find a square <= T whose remainder matches common special token counts
    # (0: no special tokens, 1: only CLS, 4: only registers, 5: CLS+registers)
    for remainder in (0, 1, 4, 5):
        N = T - remainder
        hw = int(math.isqrt(N))
        if hw * hw == N and N > 0:
            # keep the last N tokens (assume specials are at the front)
            return tokens[:, T-N:, :], hw

    # fallback: crop to the largest perfect square
    N = int(math.isqrt(T)) ** 2
    hw = int(math.isqrt(N))
    return tokens[:, :N, :], hw

# ------------- get intermediate tokens -------------
with torch.no_grad():
    # Many DINOv2 builds return patch-only tokens here (no CLS).
    tokens_list = backbone.get_intermediate_layers(x, n=1)  # list of (B, T, C)
    tokens = tokens_list[0]  # (B, T, C)

patch_tokens, HW = extract_patch_tokens(tokens)  # (B, HW*HW, C)

# ------------- fold to (C, H, W) -------------
B, N, C = patch_tokens.shape
feat_hw = patch_tokens[0].transpose(0, 1).reshape(C, HW, HW)  # (C, H, W)

# ------------- (optional) upsample to patch grid scaled to image/14 -------------
feat_up = torch.nn.functional.interpolate(
    feat_hw.unsqueeze(0),
    size=(img.size[1] // 14, img.size[0] // 14),
    mode='bilinear', align_corners=False
)[0]

# ------------- visualize channels -------------
def grid_show_feature_maps(feat: torch.Tensor, n=16, title="DINOv2 tokens"):
    C, H, W = feat.shape
    n = min(n, C)
    sel = torch.linspace(0, C-1, steps=n).long()
    tiles = []
    for c in sel:
        m = feat[c]
        m = (m - m.min()) / (m.max() - m.min() + 1e-6)
        tiles.append(m.cpu().numpy())

    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(2.5*cols, 2.5*rows))
    for i, m in enumerate(tiles, 1):
        plt.subplot(rows, cols, i)
        plt.imshow(m, cmap="gray")
        plt.axis("off")
    plt.suptitle(title + f"  (C={C}, H={H}, W={W})")
    plt.tight_layout()
    plt.show()

grid_show_feature_maps(feat_hw, n=16, title=f"DINOv2 ViT-S/14 patch-token maps (HW={HW})")
