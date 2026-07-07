import logging

import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.utils
import folder_paths


logger = logging.getLogger(__name__)


class _Krea2LoraAsModelPatch:
    """
    Minimal MODEL_PATCH backend to carry Krea2 Control LoRA weights.
    Control execution stays in controlnet patcher side.
    """

    def __init__(self, state_dict):
        self.state_dict = state_dict


class Krea2ControlNetLoraLoader:
    """
    Krea2 ControlNet LoRA file loader (MODEL_PATCH output only).
    Follows existing controlnet model loader style: select file and output model_patch.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": (folder_paths.get_filename_list("controlnet"),),
            }
        }

    RETURN_TYPES = ("MODEL_PATCH",)
    FUNCTION = "load_model_patch"
    CATEGORY = "advanced/loaders/krea2"
    DESCRIPTION = "Load a Krea2 controlnet LoRA file and output MODEL_PATCH."

    def _is_krea2_like_lora(self, state_dict: dict) -> bool:
        if not state_dict:
            return False

        keys = list(state_dict.keys())
        has_lora_pair = any(".lora_up." in k or ".lora_A." in k or ".lora.down." in k for k in keys)
        if not has_lora_pair:
            return False

        krea2_tokens = (
            ".blocks.",
            ".txtfusion.",
            ".first.",
            ".last.",
            ".tmlp.",
            ".txtmlp.",
            ".tproj.",
            ".pe_embedder.",
        )
        return any(any(token in k for token in krea2_tokens) for k in keys)

    def load_model_patch(self, name):
        lora_file = folder_paths.get_full_path_or_raise("controlnet", name)
        logger.info(f"[Krea2ControlNetLoraLoader] Loading controlnet LoRA: {lora_file}")
        lora_state_dict = comfy.utils.load_torch_file(lora_file, safe_load=True)
        if not isinstance(lora_state_dict, dict) or len(lora_state_dict) == 0:
            raise ValueError(f"Invalid or empty state dict: {lora_file}")

        if not self._is_krea2_like_lora(lora_state_dict):
            raise ValueError(
                "This file does not look like a Krea2 LoRA (missing expected Krea2/LoRA key patterns). "
                f"Refusing to apply: {lora_file}"
            )

        model = _Krea2LoraAsModelPatch(lora_state_dict)
        model_patcher = comfy.model_patcher.CoreModelPatcher(
            model,
            load_device=comfy.model_management.get_torch_device(),
            offload_device=comfy.model_management.unet_offload_device(),
            inplace_update=True,
        )
        logger.info("[Krea2ControlNetLoraLoader] Loaded successfully")
        return (model_patcher,)

