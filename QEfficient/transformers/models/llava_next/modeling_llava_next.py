# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


import numpy as np
import torch
import torch.nn as nn
from transformers.models.llava_next.modeling_llava_next import (
    LlavaNextForConditionalGeneration,
    # get_anyres_image_grid_shape,
)

from QEfficient.utils import constants
from QEfficient.utils._utils import IOInfo
from QEfficient.utils.logging_utils import logger


# def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
#     """
#     Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

#     Args:
#         image_size (`tuple`):
#             The size of the input image in the format (width, height).
#         grid_pinpoints (`List`):
#             A list containing possible resolutions. Each item in the list should be a tuple or list
#             of the form `(height, width)`.
#         patch_size (`int`):
#             The size of each image patch.

#     Returns:
#         tuple: The shape of the image patch grid in the format (width, height).
#     """
#     if not isinstance(grid_pinpoints, list):
#         raise TypeError("grid_pinpoints should be a list of tuples or lists")

#     # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
#     if not isinstance(image_size, (list, tuple)):
#         if not isinstance(image_size, (torch.Tensor, np.ndarray)):
#             raise TypeError(
#                 f"image_size invalid type: {type(image_size)} not valid, should be either list, tuple, np.ndarray or tensor"
#             )
#         image_size = image_size.tolist()

#     height, width = select_best_resolution(image_size, grid_pinpoints)
#     return height // patch_size, width // patch_size

# def select_best_resolution(original_size: tuple, possible_resolutions: list) -> tuple:
#     """
#     Selects the best resolution from a list of possible resolutions based on the original size.

#     This is done by calculating the effective and wasted resolution for each possible resolution.

#     The best fit resolution is the one that maximizes the effective resolution and minimizes the wasted resolution.

#     Args:
#         original_size (tuple):
#             The original size of the image in the format (height, width).
#         possible_resolutions (list):
#             A list of possible resolutions in the format [(height1, width1), (height2, width2), ...].

#     Returns:
#         tuple: The best fit resolution in the format (height, width).
#     """
#     original_height, original_width = original_size
#     best_fit = None
#     max_effective_resolution = 0
#     min_wasted_resolution = float("inf")

#     for height, width in possible_resolutions:
#         scale = min(width / original_width, height / original_height)
#         downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
#         effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
#         wasted_resolution = (width * height) - effective_resolution

#         if effective_resolution > max_effective_resolution or (
#             effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
#         ):
#             max_effective_resolution = effective_resolution
#             min_wasted_resolution = wasted_resolution
#             best_fit = (height, width)

#     return best_fit

from typing import Union, List, Tuple
import torch

def select_best_resolution(
    original_size: Tuple[int, int],
    possible_resolutions: List[Tuple[int, int]],
):
    """
    Returns two 0-D int64 tensors (best_h, best_w).
    This version avoids data-dependent tensor indexing and Python-side branching.
    """
    if not isinstance(original_size, (tuple, list)) or len(original_size) != 2:
        raise ValueError("original_size must be (H, W)")
    if not isinstance(possible_resolutions, list) or len(possible_resolutions) == 0:
        raise ValueError("possible_resolutions must be a non-empty list of (H, W)")

    # [N, 2] possible resolutions as int64 tensor
    pr = torch.as_tensor(possible_resolutions, dtype=torch.int64)  # on CPU is fine

    # Compute fit score (vectorized)
    oh = torch.tensor(original_size[0], dtype=torch.float32)
    ow = torch.tensor(original_size[1], dtype=torch.float32)

    heights = pr[:, 0].to(torch.float32)  # [N]
    widths  = pr[:, 1].to(torch.float32)  # [N]

    scale = torch.minimum(widths / ow, heights / oh)          # [N]
    down_h = torch.floor(scale * oh).to(torch.int64)          # [N]
    down_w = torch.floor(scale * ow).to(torch.int64)          # [N]
    effective = down_h * down_w                               # [N]
    wasted = (pr[:, 0] * pr[:, 1]) - effective                # [N]

    score = effective * 10_000_000 - wasted                   # [N]
    idx = torch.argmax(score).to(torch.int64)                 # 0-D

    # One-hot selection to avoid data-dependent indexing
    one_hot = torch.nn.functional.one_hot(idx, num_classes=pr.shape[0]).to(pr.dtype)  # [N]
    best_h = (pr[:, 0] * one_hot).sum()   # 0-D int64 tensor
    best_w = (pr[:, 1] * one_hot).sum()   # 0-D int64 tensor
    return best_h, best_w

def _to_i64_scalar(x):
    if isinstance(x, torch.Tensor):
        return x.to(torch.int64).reshape(())
    return torch.tensor(int(x), dtype=torch.int64)

def get_anyres_image_grid_shape(
    image_size: Union[Tuple[int, int], torch.Tensor, list],
    grid_pinpoints: List[Tuple[int, int]],
    divisor,  # keep your existing third arg meaning intact
):
    """
    Export-friendly: avoids data-dependent Python indexing/branching.
    Returns Python ints in eager; during export, use size-like checks before int().
    """
    # Normalize original_size to 1-D int64 tensor [2]
    if isinstance(image_size, torch.Tensor):
        if image_size.numel() != 2:
            raise ValueError("image_size tensor must have 2 elements (H, W)")
        orig_hw = image_size.to(torch.int64).flatten()
    else:
        if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
            raise ValueError("image_size must be (H, W)")
        orig_hw = torch.tensor([int(image_size[0]), int(image_size[1])], dtype=torch.int64)

    if not isinstance(grid_pinpoints, list) or len(grid_pinpoints) == 0:
        raise ValueError("grid_pinpoints must be a non-empty list of (H, W)")

    # Resolve best (H, W) as 0-D tensors
    if len(grid_pinpoints) == 1:
        best_h = torch.tensor(grid_pinpoints[0][0], dtype=torch.int64)
        best_w = torch.tensor(grid_pinpoints[0][1], dtype=torch.int64)
    else:
        # Uses the tensor-only selection above
        best_h, best_w = select_best_resolution((int(orig_hw[0]), int(orig_hw[1])), grid_pinpoints)

    div_t = _to_i64_scalar(divisor)

    # Compute grid counts as tensors
    num_h = torch.div(best_h, div_t, rounding_mode="trunc")  # 0-D int64 tensor
    num_w = torch.div(best_w, div_t, rounding_mode="trunc")  # 0-D int64 tensor

    # Mark as size-like if the API exists (PyTorch 2.5+)
    try:
        num_h = torch._constrain_as_size(num_h)  # annotate as size-like
        num_w = torch._constrain_as_size(num_w)
    except Exception:
        pass

    # Best effort: prove positive
    try:
        torch._check(num_h > 0)
        torch._check(num_w > 0)
    except Exception:
        pass

    # Convert to Python ints last. In export, _constrain_as_size helps Dynamo treat them as SymInts.
    try:
        return int(num_h), int(num_w)   # SymInt-friendly path in export builds
    except Exception:
        # Eager fallback
        return int(num_h.item()), int(num_w.item())
    
class QEffLlavaNextEncoderWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.vision_model = self.model.vision_tower
    
    
    def forward(self, pixel_values, image_sizes):
        print("here in forward")
        if pixel_values.dim() == constants.GRANITEVISION_PIXEL_VALUE_DIM:
            pixel_values_new = pixel_values.squeeze(0)

        image_feature = self.model.vision_tower(pixel_values_new, output_hidden_states=True)
        if isinstance(self.model.config.vision_feature_layer, int):
            selected_image_feature = image_feature.hidden_states[self.model.config.vision_feature_layer]
        else:
            hs_pool = [image_feature.hidden_states[layer_idx] for layer_idx in self.model.config.vision_feature_layer]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        vision_feature_select_strategy = self.model.config.vision_feature_select_strategy
        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(f"Unexpected select feature strategy: {self.model.config.vision_feature_select_strategy}")
        image_features = self.model.multi_modal_projector(selected_image_feature)
        image_features = torch.split(image_features, [image_features.shape[0]], dim=0)
        new_image_features = []

        # Image feature
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = (
                    self.model.config.vision_config.image_size // self.model.config.vision_config.patch_size
                )

                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.model.config.image_grid_pinpoints,
                    self.model.config.vision_config.image_size,
                )

                if (
                    np.prod(image_feature.shape) % (num_patch_height * num_patch_width * height * width) != 0
                    and vision_feature_select_strategy == "default"
                ):
                    logger.warning_once(
                        "Image feature shape does not line up with the provided patch size. "
                        "You may be using the `default` vision_feature_select_strategy with a"
                        " visual encoder that does not have CLS."
                    )

                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)

                if not isinstance(image_sizes[image_idx], (list, tuple)):
                    if not isinstance(image_sizes[image_idx], (torch.Tensor, np.ndarray)):
                        raise TypeError(
                            f"image_size invalid type: {type(image_sizes[image_idx])} not valid, should be either list, tuple, np.ndarray or tensor"
                        )
                original_size = image_sizes[image_idx].tolist()
                original_height, original_width = original_size
                current_height, current_width = image_feature.shape[1:]

                if torch.is_tensor(current_height):
                    current_height = current_height.item()
                    current_width = current_width.item()

                scale_factor = current_width / original_width
                new_height = int(round(original_height * scale_factor, 7))
                padding = (current_height - new_height) // 2
                image_feature = image_feature[:, padding : current_height - padding, :]
                if self.model.model.image_newline is not None:
                    image_feature = torch.cat(
                        (
                            image_feature,
                            self.model.model.image_newline[:, None, None]
                            .expand(*image_feature.shape[:-1], 1)
                            .to(image_feature.device, image_feature.dtype),
                        ),
                        dim=-1,
                    )
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
            else:
                image_feature = image_feature[0]
                if self.model.model.image_newline is not None:
                    image_feature = torch.cat(
                        (image_feature, self.model.model.image_newline[None].to(image_feature)), dim=0
                    )
            new_image_features.append(image_feature)
        image_features = torch.cat(new_image_features, dim=0)
        return image_features.unsqueeze(0)


class QEffLlavaNextDecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = self.model.config
        self.language_model = self.model.language_model
        self.lm_head = self.model.lm_head

    def forward(self, input_ids, vision_embeds, position_ids, image_idx, past_key_values):
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        image_features = vision_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        mask = input_ids == self.config.image_token_index
        indices1 = mask.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(mask.shape[0]).view(-1, 1)
        image_features_expanded = image_features[indices0, indices1]
        image_inputs_embeds = torch.where(mask.unsqueeze(-1), image_features_expanded, inputs_embeds)
        # *where to skip image encoder for decode*
        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, image_inputs_embeds)
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        image_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs[0][torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        return logits, vision_embeds, image_idx, outputs.past_key_values


class QEffLlavaNextForConditionalGeneration(LlavaNextForConditionalGeneration):
    def get_qeff_vision_encoder(self):
        return QEffLlavaNextEncoderWrapper(self)

    def get_qeff_language_decoder(self):
        return QEffLlavaNextDecoderWrapper(self)

    def get_dummy_inputs(self, kv_offload: bool = False, **kwargs):
        num_layers = self.config.text_config.num_hidden_layers
        num_key_value_heads = self.config.text_config.num_key_value_heads
        head_dim = self.config.text_config.hidden_size // self.config.text_config.num_attention_heads
        if vis_cfg := getattr(self.config, "vision_config", None):
            img_size = getattr(vis_cfg, "image_size", constants.GRANITEVISION_IMG_SIZE)
        else:
            img_size = constants.GRANITEVISION_IMG_SIZE
        if img_size != constants.GRANITEVISION_IMG_SIZE and kv_offload:
            raise NotImplementedError("Image Size other than 384 is not supported for LlavaNext models yet.")
        vision_size = constants.GRANITEVISION_FEATURE_SIZE
        vision_inputs = {
            "pixel_values": torch.zeros(
                (
                    constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
                    constants.GRANITEVISION_NUM_PATCHES,
                    constants.GRANITEVISION_NUM_CHANNELS,
                    constants.GRANITEVISION_IMG_SIZE,
                    constants.GRANITEVISION_IMG_SIZE,
                ),
                dtype=torch.float32,
            ),
            "image_sizes": torch.tensor(
                [[constants.GRANITEVISION_IMG_SIZE_HEIGHT, constants.GRANITEVISION_IMG_SIZE_WIDTH]], dtype=torch.int64
            ),
        }
        lang_inputs = {
            "input_ids": torch.ones(
                (constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, constants.GRANITEVISION_SEQ_LEN), dtype=torch.int64
            ),
            "attention_mask": torch.ones(
                (constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, constants.GRANITEVISION_SEQ_LEN), dtype=torch.int64
            ),
            "vision_embeds": torch.ones(
                (
                    constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
                    vision_size,
                    self.language_model.config.hidden_size,
                ),
                dtype=torch.float32,
            ),
            "image_idx": torch.zeros((1, 1), dtype=torch.int64),
        }
        lang_inputs["position_ids"] = lang_inputs.pop("attention_mask").cumsum(1)
        lang_inputs["past_key_values"] = []
        for i in range(num_layers):
            lang_inputs["past_key_values"].append(
                (
                    torch.zeros(
                        constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
                        num_key_value_heads,
                        constants.GRANITEVISION_CTX_LEN,
                        head_dim,
                    ),
                    torch.zeros(
                        constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
                        num_key_value_heads,
                        constants.GRANITEVISION_CTX_LEN,
                        head_dim,
                    ),
                )
            )
        lang_inputs["position_ids"] = torch.full(lang_inputs["position_ids"].shape, constants.GRANITEVISION_CTX_LEN - 1)
        inputs = {}
        if kv_offload:
            inputs["vision"] = vision_inputs
            inputs["lang"] = lang_inputs
        else:
            lang_inputs.pop("vision_embeds")
            inputs = {**vision_inputs, **lang_inputs}
        return inputs

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        img_size: int,
        kv_offload: bool = False,
        **compiler_options,
    ):
        max_num_images = compiler_options.pop("max_num_images", 1)
        num_patches = compiler_options.pop("num_patches", None)
        image_size_height = compiler_options.pop("image_size_height", None)
        image_size_width = compiler_options.pop("image_size_width", None)

        if num_patches is None:
            num_patches = constants.GRANITEVISION_NUM_PATCHES
        if image_size_height is None:
            image_size_height = constants.GRANITEVISION_IMG_SIZE_HEIGHT
        if image_size_width is None:
            image_size_width = constants.GRANITEVISION_IMG_SIZE_WIDTH

        if num_patches != constants.GRANITEVISION_NUM_PATCHES:
            logger.warning("Image Num Patches should be set to 10")
            num_patches = constants.GRANITEVISION_NUM_PATCHES

        if image_size_height != constants.GRANITEVISION_IMG_SIZE_HEIGHT:
            logger.warning(
                "Image Size Height Should be fixed to 1109. Please Reshape the image to (w x h) (1610 x 1109)"
            )
            image_size_height = constants.GRANITEVISION_IMG_SIZE_HEIGHT

        if image_size_width != constants.GRANITEVISION_IMG_SIZE_WIDTH:
            logger.warning(
                "Image Size Width Should be fixed to 1610. Please Reshape the image to (w x h) (1610 x 1109)"
            )
            image_size_width = constants.GRANITEVISION_IMG_SIZE_WIDTH

        prefill_seq_len = prefill_seq_len if prefill_seq_len else constants.GRANITEVISION_SEQ_LEN
        ctx_len = ctx_len if ctx_len else constants.GRANITEVISION_CTX_LEN
        if not kv_offload:
            raise NotImplementedError("We currently support on Dual QPC for this model please set kv_offload to True")
        if img_size is None and hasattr(self.config.vision_config, "image_size"):
            img_size = getattr(self.config.vision_config, "image_size")
        elif img_size is None:
            img_size = constants.GRANITEVISION_IMG_SIZE
            logger.warning("Setting img_size to be 384, as it was neither passed nor found in vision_config")
        if img_size != constants.GRANITEVISION_IMG_SIZE and kv_offload:
            logger.warning("Image Size other than 384 is not supported for LlavaNext models yet.")
        vision_size = constants.GRANITEVISION_FEATURE_SIZE
        vision = [
            {
                "batch_size": batch_size,
                "image_size_height": image_size_height,
                "image_size_width": image_size_width,
                "num_patches": num_patches,
                "max_num_images": max_num_images,
                "img_size": img_size,
            }
        ]
        lang = [
            {
                "batch_size": batch_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
                "image_size_height": image_size_height,
                "image_size_width": image_size_width,
                "num_patches": num_patches,
                "max_num_images": max_num_images,
                "img_size": img_size,
                "vision_size": vision_size,
            },
            {
                "batch_size": batch_size,
                "seq_len": "1",
                "ctx_len": ctx_len,
                "image_size_height": image_size_height,
                "image_size_width": image_size_width,
                "num_patches": num_patches,
                "max_num_images": max_num_images,
                "img_size": img_size,
                "vision_size": vision_size,
            },
        ]
        specializations = {}
        if kv_offload:
            specializations["vision"] = vision
            specializations["lang"] = lang
            return specializations, compiler_options
        else:
            return lang, compiler_options

    def get_onnx_dynamic_axes(self, kv_offload: bool = False):
        # Define dynamic axes
        num_layers = self.config.text_config.num_hidden_layers
        vision_dynamic_axes = {
            "pixel_values": {0: "batch_size", 1: "num_patches", 3: "img_size", 4: "img_size"},
            "image_sizes": {0: "image_size_height", 1: "image_size_width"},
        }
        lang_dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
            "vision_embeds": {0: "batch_size", 1: "vision_size"},
        }
        for i in range(num_layers):
            lang_dynamic_axes[f"past_key.{i}"] = {0: "batch_size", 2: "ctx_len"}
            lang_dynamic_axes[f"past_value.{i}"] = {0: "batch_size", 2: "ctx_len"}
        dynamic_axes = {}
        if kv_offload:
            dynamic_axes["vision"] = vision_dynamic_axes
            dynamic_axes["lang"] = lang_dynamic_axes
        else:
            dynamic_axes = {**vision_dynamic_axes, **lang_dynamic_axes}
        return dynamic_axes

    def get_output_names(self, kv_offload: bool = False):
        vision_output_names = ["vision_embeds"]
        lang_output_names = ["logits"]
        for i in range(self.language_model.config.num_hidden_layers):
            for kv in ["key", "value"]:
                lang_output_names.append(f"past_{kv}.{i}_RetainedState")

        output_names = {}
        if kv_offload:
            lang_output_names.insert(1, "vision_embeds_RetainedState")
            lang_output_names.insert(2, "image_idx_output")
            output_names["vision"] = vision_output_names
            output_names["lang"] = lang_output_names
        else:
            lang_output_names.insert(1, "pixel_values_RetainedState")
            lang_output_names.insert(2, "image_idx_output")
            return lang_output_names
        return output_names

    def get_inputs_info(self):
        return [
            IOInfo(name="input_ids", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="attention_mask", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="pixel_values", datatype=torch.float32, shape=("batch_size", 10, 3, "img_size", "img_size")),
            IOInfo(name="image_sizes", datatype=torch.int64, shape=(1109, 1610)),
        ]

