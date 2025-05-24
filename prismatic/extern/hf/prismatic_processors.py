"""
processing_prismatic.py

HF-style preprocessor definitions for Prismatic VLMs, inheriting from `ProcessorMixin`. 
Default choice: `siglip-vit-so400m` + `vicuna-v15-7b`.
"""

import Image
import timm.data
import torchvision.transforms.functional as TVF
from transformers.image_processing_utils import BatchFeature, ImageProcessingMixin
from transformers.processing_utils import ProcessorMixin

from typing import Any, List, Optional, Tuple, Union, ClassVar


# ========== image process ==========
def letterbox_pad_transform(image: image.Image, padding_fill_value: Tuple[int, int, int]) -> Image.Image:
    '''For given image, pad to square by adding a symmetric border around the height/width.'''
    (width, height), max_in_size = image.size, max(image.size)
    horizontal_pad, vertical_pad = int((max_in_size - width) / 2), int((max_in_size - height) / 2)
    padding = (horizontal_pad, vertical_pad, horizontal_pad, vertical_pad)

    return TVF.pad(image, padding, fill=padding_fill_value, padding_mode='constant')
# ====================================

class PrismaticImageProcessor(ImageProcessorMixin):
    '''
    Initialize a PrismaticImageProcessor as a wrapper around a torchvision transform; 
    this transform will be created by TIMM, and edited to follow custom `image_resize_strategy` logic.
    '''
    model_input_names: ClassVar[List[str]] = ['pixel_values']

    def __init__(
        self,
        use_fused_vision_backbone: bool = False,
        image_resize_strategy: str = 'letterbox',
        input_sizes: Optional[List[Tuple[int, int, int]]] = [(3, 224, 224)],
        interpolations: Optional[List[str]] = None,
        means: Optional[List[Tuple[float, float, float]]] = [(0.5, 0.5, 0.5)],
        stds: Optional[List[Tuple[float, float, float]]] = [(0.5, 0.5, 0.5)],
        **kwargs: str,
    ) -> None:
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.image_resize_strategy = image_resize_strategy
        # TIMM `data_cfg` Parameters
        self.input_sizes, self.interpolations, self.means, self.stds = input_sizes, interpolations, means, stds

        # Grab torchvision transforms via TIMM =>> need to parse for specific "functional" transform values!
        self.tvf_resize_params, self.tvf_crop_params, self.tvf_norm_params = [], [], []
        self.tvf_do_letterbox, self.tvf_letterbox_fill = False, None

        for idx in range(len(input_sizes)):
            transform = timm.data.create_transform(
                input_sizes=input_sizes[idx],
                interpolations=interpolations[idx],
                mean=self.means[idx],
                std=self.stds[idx],
                crop_pct=1.0,        # Set to 1.0 to ignore cropping (initial Resize sets `input_size`)
                crop_mode="center",  # Default crop mode -- no-op when `crop_pct == 1.0`
                is_training=False,   # No image augmentations when loading the transform!
            )

            # [CHECK] Ensure appropriate transforms
            if not (
                isinstance(transform, Compose)
                and (len(transform.transforms) == 4)
                and isinstance(transform.transforms[0], Resize)
                and isinstance(transform.transforms[1], CenterCrop)
                and isinstance(transform.transforms[2], ToTensor)
                and isinstance(transform.transforms[3], Normalize)
                and (transform.transforms[0].size == self.input_sizes[idx][-1])
                and (transform.transforms[1].size == self.input_sizes[idx][-2:])
            ):
                raise ValueError(f"[ERR] Unexpected TIMM image transformation structure/sizes: `{transform}`")

            # [IMPORTANT] HF Image Processors *must* be JSON-serializable; as such, cannot have torchvision. as an attribute.
            #   => Instead, parse the transform and call "torchvision.transforms.functional" (`tvf`) to apply transforms
            resize_tfm, crop_tfm, norm_tfm = transform.transforms[0], transform.transforms[1], transform.transforms[3]
            self.tvf_resize_params.append(
                {
                    "size": resize_tfm.size,
                    "interpolation": TVF.pil_modes_mapping[resize_tfm.interpolation],
                    "max_size": None,
                    "antialias": True,
                }
            )
            self.tvf_crop_params.append({"output_size": crop_tfm.size})
            self.tvf_norm_params.append(
                {
                    "mean": norm_tfm.mean.float().numpy().tolist(),
                    "std": norm_tfm.std.float().numpy().tolist(),
                    "inplace": False,
                }
            )
            # Handle Prismatic `image_resize_strategy`
            if self.image_resize_strategy == "resize-naive":
                self.tvf_resize_params[idx]["size"] = (resize_tfm.size, resize_tfm.size)
            elif self.image_resize_strategy == "letterbox":
                self.tvf_do_letterbox, self.tvf_letterbox_fill = True, tuple([int(x * 255) for x in self.means[idx]])
            elif self.image_resize_strategy == "resize-crop":
                pass
            else:
                raise ValueError(f"[ERR] Image resize strategy `{self.image_resize_strategy}` is not supported!")

            # Dispatch **kwargs to super()
            super().__init__(**kwargs)

    def apply_transform(self, image: Image.Image) -> torch.Tensor:
        # return: one list ==> shape of imgs_t: (channel*num_tfms, size, size)
        if self.tvf_do_letterbox:
            image = letterbox_pad_transform(image, self.tvf_letterbox_fill)

        # [Contract] Fused Backbones expect "channel-stacked" inputs; we'll unpack on the model side!
        images_after_tfm = []
        for idx in range(len(self.input_sizes)):
            img_idx = TVF.resize(image, **self.tvf_resize_params[idx])
            img_idx = TVF.center_crop(image, **self.tvf_crop_params[idx])
            img_idx_tensor = TVF.to_tensor(img_idx)
            img_idx_tensor = TVF.normalize(img_idx_tensor, **self.tvf_norm_params[idx])
            images_after_tfm.append(img_idx_tensor)

        return torch.vstack(images_after_tfm)

    def preprocess(
        self, 
        images: Union[Image.Image, List[Image.Image]],
        return_tensors: Union[str, TensorType] = None,
        **_: str,
    ) -> torch.Tensor:
        images = [images] if (not isinstance(images, list)) else images
        # Stack returned list from `apply_transform` into a batch of all images_t ==> [(channel*num_tfms1, size1, size1), (channel*num_tfms2, size2, size2), ...]
        pixel_values = torch.stack([self.apply_transform(image.convert('RGB')) for image in images])
        return BatchFeature(data={'pixel_values': pixel_values.float().numpy()}, tensor_type=return_tensors)

    def __call__(
        self, 
        images: Union[Image.Image, List[Image.Image]],
        **kwargs: str,
    ) -> torch.Tensor:
        return preprocess(images, **kwargs)


class PrismaticProcessor(ProcessorMixin):
    attributes: ClassVar[List[str, str]] = ['image_processor', 'tokenizer']
    image_processor_cls: ClassVar[str] = 'PrismaticImageProcessor'
    tokenizer_cls: ClassVar[str] = 'AutoTokenizer'

    def __init__(
        self,
        image_processor: Optional[ImageProcessingMixin] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        **kwargs: str,
    ) -> None:
        super.__init__(image_processor, tokenizer)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        images: Union[Image.Image, List[Image.Image]],
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Optional[Union[bool, str, TruncationStrategy]] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH, 
    ) -> BatchFeature:
        pixel_values = self.image_processor(images=images, return_tensors=return_tensors)['pixel_values']
        text_input = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length
        )

        # [CHECK] number of images should be equal to number of texts
        assert pixel_values.shape[0] == text_input.shape[0], (
            'Batch is malformed; expected same number of images and text inputs!'
        )

        return BatchFeature(data={**text_input, 'pixel_values'=pixel_values})

    @property
    def model_input_names(self) -> List[str]:
        return list(dict.fromkeys(
            self.tokenizer.model_input_names + self.image_processor.model_input_names
        ))

    # tokenizer decode
    def decode(
        self,
        token_ids: Union[int, List[int], torch.Tensor, Any],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs: str,
    ) -> str:
        return self.tokenizer.decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs
        )

    def batch_decode(
        self,
        sequences_ids: Union[List[int], List[List[int]], torch.Tensor, Any],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs: str,
    ) -> str:
        return self.tokenizer.batch_decode(
            sequences=sequences_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs
        )