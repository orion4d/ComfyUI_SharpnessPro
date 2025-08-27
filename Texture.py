# SharpnessPro/Texture.py
import torch
from .utils import ensure_tensor_bchw, to_bhwc, clamp01, gaussian_blur, rgb_to_luma

class TextureNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "radius": ("FLOAT", {"default": 2.0, "min": 0.3, "max": 10.0, "step": 0.1}),
                "strength": ("FLOAT", {"default": 0.4, "min": -1.0, "max": 1.0, "step": 0.01}),
                "luma_only": (["Yes","No"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "SharpnessPro/Tone"

    def apply(self, image, radius, strength, luma_only):
        x = ensure_tensor_bchw(image).to(torch.float32)
        # Micro-contrast = high frequency boost with small radius, symmetric, gentle
        low = gaussian_blur(x, radius=radius)
        hf = x - low  # high frequencies

        if luma_only == "Yes":
            L = rgb_to_luma(x)
            L_low = rgb_to_luma(low)
            dL = L - L_low
            L_out = (L + strength * dL).clamp(0,1)
            eps = 1e-6
            scale = (L_out + eps) / (L + eps)
            out = clamp01(x * scale)
        else:
            out = clamp01(x + strength * hf)

        return (to_bhwc(out),)

NODE_CLASS_MAPPINGS = {"Texture": TextureNode}
NODE_DISPLAY_NAME_MAPPINGS = {"Texture": "SharpnessPro · Texture"}
