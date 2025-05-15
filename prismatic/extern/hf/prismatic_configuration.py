'''
prismatic_configuration.py

HF-style configs for Prismatic VLMs, inheriting from `transformers.PretrainedConfig`
Default choice: `siglip-vit-so400m` + `vicuna-v15-7b`.
'''

from typing import Any, Dict, List, Optional

from transformers import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING

# === MAPPINT TAB ===
# fmt: on
VISION_BACKBONE_TO_RESOLUTIONS: Dict[str, Dict[str, List[int]]] = {
    'single': {
        "clip-vit-l": [224],
        "clip-vit-l-336px": [336],
        "siglip-vit-so400m": [224],
        "siglip-vit-so400m-384px": [384],
        "dinov2-vit-l": [224],
        "in1k-vit-l": [224],
    },
    'fuse': {
        "dinoclip-vit-l-336px": [336, 336],
        "dinosiglip-vit-so-224px": [224, 224],
        "dinosiglip-vit-so-384px": [384, 384],
    },
}
VISION_BACKBONE_TO_TIMM_IDS: Dict[str, Dict[str, List[int]]] = {
    'single': {
        "clip-vit-l": ["vit_large_patch14_clip_224.openai"],
        "clip-vit-l-336px": ["vit_large_patch14_clip_336.openai"],
        "siglip-vit-so400m": ["vit_so400m_patch14_siglip_224"],
        "siglip-vit-so400m-384px": ["vit_so400m_patch14_siglip_384"],
        "dinov2-vit-l": ["vit_large_patch14_reg4_dinov2.lvd142m"],
        "in1k-vit-l": ["vit_large_patch16_224.augreg_in21k_ft_in1k"],
    },
    'fuse': {
        "dinoclip-vit-l-336px": ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_large_patch14_clip_336.openai"],
        "dinosiglip-vit-so-224px": ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_224"],
        "dinosiglip-vit-so-384px": ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_384"],
    },
}
TIMM_OVERRIDE_ACT_LAYERS: Dict[str, Dict[str, List[int]]] = {
    'single': {
        "clip-vit-l": ["quick_gelu"],
        "clip-vit-l-336px": ["quick_gelu"],
        "siglip-vit-so400m": [None],
        "siglip-vit-so400m-384px": [None],
        "dinov2-vit-l": [None],
        "in1k-vit-l": [None],
    },
    'fuse': {
        "dinoclip-vit-l-336px": [None, "quick_gelu"],
        "dinosiglip-vit-so-224px": [None, None],
        "dinosiglip-vit-so-384px": [None, None],
    }
}

LLM_BACKBONE_TO_HF_PATHS: Dict[str, List[str]] = {
    "llama2-7b-pure": "meta-llama/Llama-2-7b-hf", 
    "llama2-13b-pure": "meta-llama/Llama-2-13b-hf", 
    "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf", 
    "llama2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",

    "vicuna-v15-7b": "lmsys/vicuna-7b-v1.5", 
    "vicuna-v15-13b": "lmsys/vicuna-13b-v1.5",

    "mistral-v0.1-7b-pure": "mistralai/Mistral-7B-v0.1", 
    "mistral-v0.1-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.1",

    "phi-2-3b": "microsoft/phi-2",
}
LLM_BACKBONE_TO_HF_METACLS: Dict[str, List[str]] = {
    "llama2-7b-pure": "llama", 
    "llama2-13b-pure": "llama", 
    "llama2-7b-chat": "llama", 
    "llama2-13b-chat": "llama",

    "vicuna-v15-7b": "llama", 
    "vicuna-v15-13b": "llama",

    "mistral-v0.1-7b-pure": "mistral", 
    "mistral-v0.1-7b-instruct": "mistral",

    "phi-2-3b": "phi",
}

SINGLE_VISION_BACKBONES = set(VISION_BACKBONE_TO_RESOLUTIONS['single'].keys())
FUSE_VISION_BACKBONES = set(VISION_BACKBONE_TO_RESOLUTIONS['fuse'].keys())

VALID_VISION_BACKBONES = SINGLE_VISION_BACKBONES | FUSE_VISION_BACKBONES
VALID_LLM_BACKBONES = set(LLM_BACKBONE_TO_HF_PATHS.keys())

def print_valid_backbones() -> None:
    print(f'======================\nvalid vision backbones:\n\tsingle arch:{SINGLE_VISION_BACKBONES}\n\tfuse arch:{FUSE_VISION_BACKBONES}')
    print(f'valid LLM backbones:\n\t{VALID_LLM_BACKBONES}\n======================')
# fmt: off

class PrismaticConfig(PretrainedConfig):
    model_type: str = 'prismatic'
    is_compositional: bool = False

    def __init__(
        self,
        use_fused_vision_backbone: Optional[bool] = False,
        vision_backbone_id: str = 'siglip-vit-so400m',
        llm_backbone_id: str = 'vicuna-v15-7b',
        image_resize_strategy: str = 'letterbox',
        llm_max_len: int = 2048,
        pad_token_id: int = 32000,
        pad_to_multiple_of: int = 64,
        output_projector_states: bool = False,
        text_config: Optional[Dict[str, Any]] = None,
        **kwargs: str,
    ) -> None:
        # Basic definitions of model backbone
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_backbone_id = vision_backbone_id
        self.llm_backbone_id = llm_backbone_id
        self.backbones_checking()

        self.vision_timm_ids = VISION_BACKBONE_TO_TIMM_IDS[self.vision_backbone_id]
        self.override_act_layers_lst = TIMM_OVERRIDE_ACT_LAYERS[self.vision_backbone_id]
        self.hf_llm_backbone = LLM_BACKBONE_TO_HF_PATHS[self.llm_backbone_id]   

        self.output_projector_states = output_projector_states
        self.image_sizes = VISION_BACKBONE_TO_RESOLUTIONS[self.vision_backbone_id]
        self.llm_max_len, self.pad_token_id, self.pad_to_multiple_of = llm_max_len, pad_token_id, pad_to_multiple_of     

        # [IMPORTANT] Config definition of `text_config`
        self.text_config = {
            CONFIG_MAPPING[LLM_BACKBONE_TO_HF_METACLS[llm_backbone_id]](**text_config)
            if text_config is not None
            else CONFIG_MAPPING[LLM_BACKBONE_TO_HF_METACLASS[self.llm_backbone_id]]()
        }
        
        # Dispatch **kwargs to super() =>> note that `pad_token_id` collides, so pass it in here as well
        super().__init__(pad_token_id=pad_token_id, **kwargs)

    def backbones_checking(self) -> None:
        assert self.vision_backbone_id in VALID_VISION_BACKBONES, f'[ERR] {self.vision_backbone_id} NOT FOUND in valid backbones!'
        assert self.llm_backbone_id in VALID_LLM_BACKBONES, f'[ERR] {self.llm_backbone_id} NOT FOUND in valid backbones!'       
        assert self.use_fused_vision_backbone and (vision_backbone_id in SINGLE_VISION_BACKBONES), f'[ERR] {self.vision_backbone_id} NOT FOUND in FUSE backbones!' 


class OpenVLAConfig(PrismaticConfig):
    model_type: str = 'openvla'

    def __init__(
        self,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
        n_action_bins: int = 256,
        vision_aggregate_type: str = 'moe',
        num_vision_queries: int = 32,
        use_regression: bool = False,
        use_mod: bool = False,
        mod_type: str = 'shiftedcos_decay_0.85_0.15',
        mod_average_router_factor: float = 0.5,
        mod_enable_film: bool = False,
        mod_share_router: bool = False,
        llm_use_film: bool = False,
        **kwargs: str,
    ) -> None:
        super.__init__(**kwargs)
        self.norm_stats, self.n_action_bins = norm_stats, n_action_bins
        self.vision_aggregate_type = vision_aggregate_type
        self.num_vision_queries = num_vision_queries
        self.use_reg = use_reg

        if not hasattr(self.text_config, 'use_mod'):
            self.text_config.use_mod = use_mod
            self.text_config.mod_type = mod_type
            self.text_config.mod_average_router_factor = mod_average_router_factor
            self.text_config.mod_share_router = mod_share_router
            self.text_config.mod_enable_film = mod_enable_film
            self.text_config.llm_use_film = llm_use_film

