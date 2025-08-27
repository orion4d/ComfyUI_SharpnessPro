# SharpnessPro/HighPassSharpen.py
import torch
from .utils import ensure_tensor_bchw, to_bhwc, gaussian_blur, clamp01, rgb_to_luma, luma_to_rgb, blend_overlay, blend_soft_light, apply_opacity

class HighPassSharpenNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "radius": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "blend_mode": (["Overlay","SoftLight"],),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "work_on_luma": (["Yes","No"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "SharpnessPro/Sharpen"

    def apply(self, image, radius, amount, blend_mode, opacity, work_on_luma):
        x = ensure_tensor_bchw(image).to(torch.float32)
        base = x

        if work_on_luma == "Yes":
            # high-pass on luminance (more PS-like)
            L = rgb_to_luma(base)
            low = gaussian_blur(base, radius=radius)
            L_low = rgb_to_luma(low)
            detail = (L - L_low)  # [-?, ?]
            hp_layer = (0.5 + amount * detail).clamp(0.0, 1.0).repeat(1,3,1,1)
            # Blend on RGB but create from luma details
            if blend_mode == "Overlay":
                blended = blend_overlay(base, hp_layer)
            else:
                blended = blend_soft_light(base, hp_layer)
            out = apply_opacity(base, blended, opacity)
        else:
            low = gaussian_blur(base, radius=radius)
            detail = (base - low)
            hp_layer = (0.5 + amount * detail).clamp(0.0, 1.0)
            if blend_mode == "Overlay":
                blended = blend_overlay(base, hp_layer)
            else:
                blended = blend_soft_light(base, hp_layer)
            out = apply_opacity(base, blended, opacity)

        return (to_bhwc(clamp01(out)),)

NODE_CLASS_MAPPINGS = {
    "HighPassSharpen": HighPassSharpenNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HighPassSharpen": "SharpnessPro · Passe-haut"
}
