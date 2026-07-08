from typing_extensions import Literal, TypeAlias
from ..models.wan_video_dit import WanModel
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vae import WanVideoVAE, WanVideoVAE38

model_loader_configs = [
    # These configs are provided for detecting model type automatically.
    # The format is (state_dict_keys_hash, state_dict_keys_hash_with_shape, model_names, model_classes, model_resource)
    (None, "9269f8db9040a9d860eaca435be61814", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "aafcfd9672c3a2456dc46e1cb6e52c70", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "6bfcfb3b342cb286ce886889d519a77e", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "6d6ccde6845b95ad9114ab993d917893", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "6bfcfb3b342cb286ce886889d519a77e", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "349723183fc063b2bfc10bb2835cf677", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "efa44cddf936c70abd0ea28b6cbe946c", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "3ef3b1f8e1dab83d5b71fd7b617f859f", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "70ddad9d3a133785da5ea371aae09504", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "26bde73488a92e64cc20b0a7485b9e5b", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "ac6a5aa74f4a0aab6f64eb9a72f19901", ["wan_video_dit"], [WanModel], "civitai"), 
    (None, "b61c605c2adbd23124d152ed28e049ae", ["wan_video_dit"], [WanModel], "civitai"), 
    (None, "1f5ab7703c6fc803fdded85ff040c316", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "5b013604280dd715f8457c6ed6d6a626", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "2267d489f0ceb9f21836532952852ee5", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "47dbeab5e560db3180adf51dc0232fb1", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "cb104773c6c2cb6df4f9529ad5c60d0b", ["wan_video_dit"], [WanModel], "diffusers"),
    (None, "9c8818c2cbea55eca56c7b447df170da", ["wan_video_text_encoder"], [WanTextEncoder], "civitai"),
    (None, "5941c53e207d62f20f9025686193c40b", ["wan_video_image_encoder"], [WanImageEncoder], "civitai"),
    (None, "1378ea763357eea97acdef78e65d6d96", ["wan_video_vae"], [WanVideoVAE], "civitai"),
    (None, "ccc42284ea13e1ad04693284c7a09be6", ["wan_video_vae"], [WanVideoVAE], "civitai"),
    (None, "e1de6c02cdac79f8b739f4d3698cd216", ["wan_video_vae"], [WanVideoVAE38], "civitai"),
]
huggingface_model_loader_configs = [
    # These configs are provided for detecting model type automatically.
    # The format is (architecture_in_huggingface_config, huggingface_lib, model_name, redirected_architecture)
    ("ChatGLMModel", "diffsynth.models.kolors_text_encoder", "kolors_text_encoder", None),
    ("MarianMTModel", "transformers.models.marian.modeling_marian", "translator", None),
    ("BloomForCausalLM", "transformers.models.bloom.modeling_bloom", "beautiful_prompt", None),
    ("Qwen2ForCausalLM", "transformers.models.qwen2.modeling_qwen2", "qwen_prompt", None),
    # ("LlamaForCausalLM", "transformers.models.llama.modeling_llama", "omost_prompt", None),
    ("T5EncoderModel", "diffsynth.models.flux_text_encoder", "flux_text_encoder_2", "FluxTextEncoder2"),
    ("CogVideoXTransformer3DModel", "diffsynth.models.cog_dit", "cog_dit", "CogDiT"),
    ("SiglipModel", "transformers.models.siglip.modeling_siglip", "siglip_vision_model", "SiglipVisionModel"),
    ("LlamaForCausalLM", "diffsynth.models.hunyuan_video_text_encoder", "hunyuan_video_text_encoder_2", "HunyuanVideoLLMEncoder"),
    ("LlavaForConditionalGeneration", "diffsynth.models.hunyuan_video_text_encoder", "hunyuan_video_text_encoder_2", "HunyuanVideoMLLMEncoder"),
    ("Step1Model", "diffsynth.models.stepvideo_text_encoder", "stepvideo_text_encoder_2", "STEP1TextEncoder"),
    ("Qwen2_5_VLForConditionalGeneration", "diffsynth.models.qwenvl", "qwenvl", "Qwen25VL_7b_Embedder"),
]
patch_model_loader_configs = [
    # These configs are provided for detecting model type automatically.
    # The format is (state_dict_keys_hash_with_shape, model_name, model_class, extra_kwargs)
]

preset_models_on_huggingface = {
    
}
preset_models_on_modelscope = {
    
}
Preset_model_id: TypeAlias = Literal[
    "HunyuanDiT",
    "stable-video-diffusion-img2vid-xt",
    "ExVideo-SVD-128f-v1",
    "ExVideo-CogVideoX-LoRA-129f-v1",
    "StableDiffusion_v15",
    "DreamShaper_8",
    "AingDiffusion_v12",
    "Flat2DAnimerge_v45Sharp",
    "TextualInversion_VeryBadImageNegative_v1.3",
    "StableDiffusionXL_v1",
    "BluePencilXL_v200",
    "StableDiffusionXL_Turbo",
    "ControlNet_v11f1p_sd15_depth",
    "ControlNet_v11p_sd15_softedge",
    "ControlNet_v11f1e_sd15_tile",
    "ControlNet_v11p_sd15_lineart",
    "AnimateDiff_v2",
    "AnimateDiff_xl_beta",
    "RIFE",
    "BeautifulPrompt",
    "opus-mt-zh-en",
    "IP-Adapter-SD",
    "IP-Adapter-SDXL",
    "StableDiffusion3",
    "StableDiffusion3_without_T5",
    "Kolors",
    "SDXL-vae-fp16-fix",
    "ControlNet_union_sdxl_promax",
    "FLUX.1-dev",
    "FLUX.1-schnell",
    "InstantX/FLUX.1-dev-Controlnet-Union-alpha",
    "jasperai/Flux.1-dev-Controlnet-Depth",
    "jasperai/Flux.1-dev-Controlnet-Surface-Normals",
    "jasperai/Flux.1-dev-Controlnet-Upscaler",
    "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha",
    "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta",
    "Shakker-Labs/FLUX.1-dev-ControlNet-Depth",
    "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro",
    "InstantX/FLUX.1-dev-IP-Adapter",
    "InfiniteYou",
    "SDXL_lora_zyd232_ChineseInkStyle_SDXL_v1_0",
    "QwenPrompt",
    "OmostPrompt",
    "ESRGAN_x4",
    "RIFE",
    "OmniGen-v1",
    "CogVideoX-5B",
    "Annotators:Depth",
    "Annotators:Softedge",
    "Annotators:Lineart",
    "Annotators:Normal",
    "Annotators:Openpose",
    "StableDiffusion3.5-large",
    "StableDiffusion3.5-medium",
    "HunyuanVideo",
    "HunyuanVideo-fp8",
    "HunyuanVideoI2V",
]
