# SharpnessPro/SmartSharpen.py
import torch
from .utils import (ensure_tensor_bchw, to_bhwc, clamp01, gaussian_blur, 
                    guided_smooth, rgb_to_luma)

class SmartSharpenNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "radius": ("FLOAT", {"default": 1.8, "min": 0.0, "max": 50.0, "step": 0.1}),
                "amount": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 5.0, "step": 0.05}),
                "reduce_noise": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fade_shadows": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fade_highlights": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "SharpnessPro/Sharpen"

    def apply(self, image, radius, amount, reduce_noise, fade_shadows, fade_highlights):
        x = ensure_tensor_bchw(image).to(torch.float32)

        # Edge-aware "blur" estimation on luma:
        L = rgb_to_luma(x)
        smooth_L = guided_smooth(x, radius=max(1.0, radius*2), eps=1e-3)
        # Detail mostly on luma to avoid color halos:
        dL = L - smooth_L

        # Optional noise reduction: shrink very small details
        if reduce_noise > 0:
            shrink = (dL.abs() / (dL.abs() + 0.03)).pow(1.0 - reduce_noise)  # compress micro noise
            dL = dL * shrink

        # Shadow/Highlight fade masks:
        shadows = (1.0 - L).clamp(0,1)
        highlights = L.clamp(0,1)
        dL = dL * (1.0 - fade_shadows*shadows) * (1.0 - fade_highlights*highlights)

        L_out = (L + amount * dL).clamp(0.0, 1.0)

        # Recombine to RGB preserving chroma
        eps = 1e-6
        baseL = L + eps
        scale = (L_out + eps) / baseL
        out = clamp01(x * scale)

        return (to_bhwc(out),)

NODE_CLASS_MAPPINGS = {
    "SmartSharpen": SmartSharpenNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartSharpen": "SharpnessPro · Netteté optimisée"
}
