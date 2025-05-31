"""
prismatic_modeling.py

Core HuggingFace-style PrismaticPreTrainedModel and PrismaticForConditionalGeneration class definitions.
Inherits from the default `transformers.PretrainedModel`. Meant to be standalone and self-contained,
but exactly replicate the logic in `prismatic.models.vlms.prismatic.py`.
"""

import logging

import timm
import torch.nn as nn

from functools import partial

# set up log
logger = logging.getLogger(__name__)

# replace the forward of llama ==> using CAttn
from prismatic.models.llama_modeling import replace_llama_spda_forward
replace_llama_spda_forward()

def unpack_tuple_wrapper(fn: Callable[[Any], Tuple(Any)]) -> Callable[[Any], Any]:
    def fn_wrapper(*args: Any, *kwargs: Any) -> Any:
        ret = fn(*args, *kwargs)[0]
    return  ret[0] if isinstance(ret, tuple) else ret

# HF Transformers overwrites parameters with names containing `gamma`; we're going to patch VisionBackbone.LayerScale.
#   =>> TIMM :: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109
#   =>> Transformers :: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3960
def _replace_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor

def _apply_patch(module: LayerScale) -> None:
    module.scale_factor = nn.Parameter(module.gamma.clone())
    del module.gamma
    module.forword = _replace_new_forward.__get__(module, LayerScale)

# ======================
class PrismaticVisionBackbone(nn.Module):
    def __init__(
        self,
        use_fused_vision_backbone: bool,
        vision_timm_ids: List[str],
        image_sizes: List[int],
        override_act_layers_lst: List[Optional[str]],
    ) -> None:
        super().__init__()
        assert len(image_sizes) <= 2, '[ERR] Only support 2 vision backbones while using fused vision backbone.'
        
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.num_input_images = 1 #Default

        # Get vision backbones
        self.featurizer = self._create_featurizer(
            vision_timm_ids[0],
            img_size=image_sizes[0],
            act_layer=override_act_layers_lst[0]
        )
        self.embed_dim = self.featurizer.embed_dim
        if use_fused_vision_backbone:
            self.fuse_featurizer = _create_featurizer(
                vision_timm_ids[1],
                img_size=image_sizes[1],
                act_layer=override_act_layers_lst[1]
            )
            self.embed_dim += self.fuse_featurizer.embed_dim

    def _create_featurizer(self, vision_timm_id: str, image_size: int, override_act_layer: Optional[str]) -> nn.Module:
        featurizer = timm.create_model(
            vision_timm_id,
            pretrained = False,
            num_classes = 0,
            img_size = image_size,
            act_layer = override_act_layer
        )
        # Override the forward ==> extract the second-to-last layer features
        featurizer.forward = (
            unpack_tuple_wrapper(partial(featurizer.get_intermediate_layers, n={len(featurizer.blocks-2)}))
        )
        return featurizer

    def _patch_layer_scales(self) -> None:
        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                _apply_patch(module)
        if self.use_fused_vision_backbone:
            for module in self.fuse_featurizer.modules():
                if isinstance(module, LayerScale):
                    _apply_patch(module)

    def get_num_input_images(self) -> int:
        return self.num_input_images
    
    def set_num_input_images(self, num_input_images) -> None:
        self.num_input_images = num_input_images

    def get_num_image_patches(self) -> int:
        return self.featurizer.patch_embed.num_patches

    def forward(self, pixel_values: torch.Tensor) -> Union[torch.Tensor, List[Tuple[torch.Tensor]]]:
        num_input_images = get_num_input_images
        if num_input_images == 1:
            if not self.use_fused_vision_backbone:
                return self.featurizer(pixel_values)
            else:
                image, fuse_image = torch.split(pixel_values, [3, 3], dim=1)
                features, fuse_features = self.featurizer(image), self.fuse_featurizer(fuse_image)
                return (features, fuse_features)
        else:
            assert self.use_fused_vision_backbone, "[ERR] Multi-image inputs require using fused backbone!"
            images_input = torch.input(pixel_values, num_input_images * [6], dim=1)
            ret_features = []

            for images in images_input:
                image, fuse_image = torch.split(image, [3, 3], dim=1)
                features = self.featurizer(image)
                fuse_features = self.fuse_featurizer(fuse_image)
                ret_features.append((features, fuse_features))
            
            return ret_features

