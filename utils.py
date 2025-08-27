# SharpnessPro/utils.py
# Helpers communs pour les filtres de netteté.
import math
import torch
import torch.nn.functional as F

EPS = 1e-6

# ---------- Conversions / clamps ----------
def ensure_tensor_bchw(img: torch.Tensor) -> torch.Tensor:
    # ComfyUI: images en [B,H,W,C] float32 [0..1]
    if not isinstance(img, torch.Tensor):
        raise TypeError("Expected torch.Tensor image")
    if img.ndim != 4:
        raise ValueError("Expected image with shape [B,H,W,C]")
    if img.shape[-1] == 3:  # B,H,W,C -> B,C,H,W
        img = img.permute(0, 3, 1, 2).contiguous()
    return img

def to_bhwc(img_bchw: torch.Tensor) -> torch.Tensor:
    # B,C,H,W -> B,H,W,C
    return img_bchw.permute(0, 2, 3, 1).contiguous()

def clamp01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)

# ---------- Flou Gaussien (propre & rapide) ----------
def get_sigma_from_radius(radius: float) -> float:
    # Heuristique "douce" proche Photoshop
    return max(0.15, radius * 0.5 + 0.25)

def gaussian_kernel1d(sigma: float, truncate: float = 3.0,
                      device=None, dtype=None) -> torch.Tensor:
    if sigma <= 0:
        k = torch.tensor([1.0], device=device, dtype=dtype or torch.float32)
        return k / k.sum()
    radius = int(truncate * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype or torch.float32)
    kernel = torch.exp(-(x ** 2) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel

def gaussian_blur(img_bchw: torch.Tensor, radius: float) -> torch.Tensor:
    if radius <= 0:
        return img_bchw
    sigma = get_sigma_from_radius(radius)
    device, dtype = img_bchw.device, img_bchw.dtype
    k1d = gaussian_kernel1d(sigma, device=device, dtype=dtype)

    kH = k1d.view(1, 1, -1, 1)
    kW = k1d.view(1, 1, 1, -1)
    C = img_bchw.shape[1]

    pad_h = (kH.shape[2] - 1) // 2
    pad_w = (kW.shape[3] - 1) // 2

    x = F.pad(img_bchw, (0, 0, pad_h, pad_h), mode='reflect')
    x = F.conv2d(x, kH.expand(C, 1, -1, 1), groups=C)
    x = F.pad(x, (pad_w, pad_w, 0, 0), mode='reflect')
    x = F.conv2d(x, kW.expand(C, 1, 1, -1), groups=C)
    return x

# ---------- Luma / recomposition ----------
def rgb_to_luma(rgb_bchw: torch.Tensor) -> torch.Tensor:
    # BT.709
    r = rgb_bchw[:, 0:1]
    g = rgb_bchw[:, 1:2]
    b = rgb_bchw[:, 2:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def luma_to_rgb(luma_bchw: torch.Tensor, rgb_base_bchw: torch.Tensor) -> torch.Tensor:
    # Remplacement de luma en conservant la chroma via un scale simple
    baseL = rgb_to_luma(rgb_base_bchw)
    scale = (luma_bchw + EPS) / (baseL + EPS)
    return clamp01(rgb_base_bchw * scale)

# ---------- Blend modes simplifiés ----------
def blend_overlay(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    res = torch.where(base <= 0.5, 2 * base * blend, 1 - 2 * (1 - base) * (1 - blend))
    return clamp01(res)

def blend_soft_light(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    # Approx W3C
    res = (1 - 2 * blend) * base * base + 2 * blend * base
    return clamp01(res)

def apply_opacity(base: torch.Tensor, effect: torch.Tensor, opacity: float) -> torch.Tensor:
    return clamp01(base * (1 - opacity) + effect * opacity)

# ---------- Détection de contours ----------
def sobel_edges(img_bchw: torch.Tensor) -> torch.Tensor:
    # Magnitude Sobel sur luma, normalisée [0..1]
    gray = rgb_to_luma(img_bchw)
    kx = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]], dtype=img_bchw.dtype, device=img_bchw.device
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]], dtype=img_bchw.dtype, device=img_bchw.device
    ).view(1, 1, 3, 3)
    gx = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode='reflect'), kx)
    gy = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode='reflect'), ky)
    mag = torch.sqrt(gx * gx + gy * gy)
    mag = mag / (mag.amax(dim=[2, 3], keepdim=True) + EPS)
    return mag  # B,1,H,W

# ---------- Filtre guidé (approx) edge-aware ----------
def guided_smooth(img_bchw: torch.Tensor, radius: float, eps: float = 1e-4) -> torch.Tensor:
    """
    Smoothing edge-aware très léger (approx guided filter) appliqué sur la luma.
    Patch important : imposer un noyau IMPAIR pour conserver exactement H, W.
    """
    if radius <= 0:
        return rgb_to_luma(img_bchw)

    k = int(max(1, round(radius)))
    if k % 2 == 0:
        k += 1  # noyau impair indispensable

    device, dtype = img_bchw.device, img_bchw.dtype
    kernel = torch.ones(1, 1, k, k, device=device, dtype=dtype) / (k * k)
    pad = (k // 2, k // 2, k // 2, k // 2)  # L,R,T,B

    L = rgb_to_luma(img_bchw)

    mean_L  = F.conv2d(F.pad(L,   pad, mode="reflect"), kernel)
    mean_L2 = F.conv2d(F.pad(L*L, pad, mode="reflect"), kernel)
    var_L   = mean_L2 - mean_L * mean_L

    a = var_L / (var_L + eps)
    b = (1 - a) * mean_L

    mean_a = F.conv2d(F.pad(a, pad, mode="reflect"), kernel)
    mean_b = F.conv2d(F.pad(b, pad, mode="reflect"), kernel)

    return mean_a * L + mean_b  # B,1,H,W

# ---------- Masque de tons moyens (Clarity) ----------
def midtone_mask(img_bchw: torch.Tensor, softness: float = 0.35) -> torch.Tensor:
    """
    Masque ~1 autour des midtones (luma≈0.5), 0 vers ombres/hautes lumières.
    softness ∈ (0, 0.5] contrôle la largeur du plateau.
    """
    L = rgb_to_luma(img_bchw)  # B,1,H,W
    s = max(1e-6, min(0.5, float(softness)))
    m = torch.cos((L - 0.5) * math.pi / (2 * s)).clamp(-1, 1)
    m = (m + 1) / 2.0
    m = torch.where((L > 0.5 + s) | (L < 0.5 - s), torch.zeros_like(m), m)
    return m  # B,1,H,W
