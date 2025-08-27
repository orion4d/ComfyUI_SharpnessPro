# SharpnessPro/UnsharpMaskSharpen.py
import torch
from .utils import ensure_tensor_bchw, to_bhwc, gaussian_blur, clamp01, rgb_to_luma, luma_to_rgb

class UnsharpMaskNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "radius": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.2, "step": 0.005}),
                "luma_only": (["Yes","No"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "SharpnessPro/Sharpen"

    def apply(self, image, radius, amount, threshold, luma_only):
        x = ensure_tensor_bchw(image).to(torch.float32)
        low = gaussian_blur(x, radius)
        detail = x - low  # high freq

        if threshold > 0:
            # Suppress tiny differences
            mask = (detail.abs() > threshold).float()
            detail = detail * mask

        if luma_only == "Yes":
            L = rgb_to_luma(x)
            L_low = rgb_to_luma(low)
            dL = L - L_low
            if threshold > 0:
                m = (dL.abs() > threshold).float()
                dL = dL * m
            L_out = (L + amount * dL).clamp(0.0, 1.0)
            out = x * 0  # placeholder shape
            out = luma_to_rgb(L_out, x)
        else:
            out = (x + amount * detail).clamp(0.0, 1.0)

        return (to_bhwc(clamp01(out)),)

NODE_CLASS_MAPPINGS = {
    "UnsharpMaskSharpen": UnsharpMaskNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UnsharpMaskSharpen": "SharpnessPro · Masque flou (USM)"
}
