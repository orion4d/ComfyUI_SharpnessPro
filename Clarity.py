# SharpnessPro/Clarity.py
import torch
from .utils import ensure_tensor_bchw, to_bhwc, clamp01, gaussian_blur, rgb_to_luma, luma_to_rgb, midtone_mask

class ClarityNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "radius": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 200.0, "step": 1.0}),
                "strength": ("FLOAT", {"default": 0.35, "min": -1.0, "max": 1.0, "step": 0.01}),
                "midtone_softness": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 0.5, "step": 0.01}),
                "luma_only": (["Yes","No"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "SharpnessPro/Tone"

    def apply(self, image, radius, strength, midtone_softness, luma_only):
        x = ensure_tensor_bchw(image).to(torch.float32)
        # Local contrast via unsharp on a big radius, constrained to midtones
        low = gaussian_blur(x, radius)
        detail = x - low

        mask = midtone_mask(x, softness=midtone_softness)  # B,1,H,W
        if luma_only == "Yes":
            L = rgb_to_luma(x)
            L_low = gaussian_blur(x, radius=radius)
            L_low = rgb_to_luma(L_low)
            dL = (L - L_low) * mask
            L_out = (L + strength * dL).clamp(0,1)
            # Recombine
            eps = 1e-6
            baseL = L + eps
            scale = (L_out + eps) / baseL
            out = clamp01(x * scale)
        else:
            out = clamp01(x + strength * (detail * mask))

        return (to_bhwc(out),)

NODE_CLASS_MAPPINGS = {"Clarity": ClarityNode}
NODE_DISPLAY_NAME_MAPPINGS = {"Clarity": "SharpnessPro · Clarté"}
