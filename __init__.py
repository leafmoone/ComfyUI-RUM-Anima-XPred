from __future__ import annotations

import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


NODE_ROOT = Path(__file__).resolve().parent
XPRED_FOLDER = "anima_xpred"


def _backend_root() -> Path:
    candidates = []
    env_root = os.environ.get("RUM_ANIMA_XPRED_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend(
        [
            NODE_ROOT.parent / "RUM-anima-xpred",
            NODE_ROOT.parent / "RUM-anima-x",
            Path("/root/shared-nvme/RUM-anima-xpred"),
        ]
    )
    for candidate in candidates:
        if (candidate / "src" / "rum_xpred").is_dir():
            return candidate
    raise ImportError(
        "Could not locate the RUM Anima X-Pred backend. "
        "Set RUM_ANIMA_XPRED_ROOT to the repository containing src/rum_xpred."
    )


BACKEND_ROOT = _backend_root()
SRC_ROOT = BACKEND_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rum_xpred.adapters.anima_sd_scripts import create_adapter
from rum_xpred.anima import anima_euler_step, make_shifted_sigma_schedule, xpred_to_anima_v
from rum_xpred.cache_batches import make_seeded_eps_batch


def _register_model_folders() -> None:
    try:
        import folder_paths

        folder_paths.add_model_folder_path(XPRED_FOLDER, str(Path(folder_paths.models_dir) / XPRED_FOLDER))
    except Exception:
        pass


_register_model_folders()


def _model_names(folder_name: str, placeholder: str) -> list[str]:
    try:
        import folder_paths

        names = folder_paths.get_filename_list(folder_name)
        return names if names else [placeholder]
    except Exception:
        return [placeholder]


def _model_path(folder_name: str, filename: str) -> Path:
    try:
        import folder_paths

        return Path(folder_paths.get_full_path_or_raise(folder_name, filename))
    except Exception as exc:
        raise FileNotFoundError(f"Could not resolve {filename!r} from ComfyUI models/{folder_name}") from exc


def _sigma_like(value: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return value.reshape(1, 1, 1, 1).to(device=z.device, dtype=z.dtype).expand(z.shape[0], 1, 1, 1)


def _velocity_from_xpred(student_forward, z: torch.Tensor, sigma: torch.Tensor, eps_floor: float) -> torch.Tensor:
    x_pred = student_forward(z, sigma)
    return xpred_to_anima_v(z, x_pred, sigma, eps_floor)


def _sample_euler(
    student_forward,
    eps_latent: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    prediction_type: str,
    eps_floor: float,
) -> torch.Tensor:
    z = eps_latent
    for index, sigma_value in enumerate(sigmas[:-1]):
        sigma = _sigma_like(sigma_value, z)
        sigma_next = sigmas[index + 1].reshape(1, 1, 1, 1).to(device=z.device, dtype=z.dtype)
        if prediction_type == "x":
            v = _velocity_from_xpred(student_forward, z, sigma, eps_floor)
        else:
            v = student_forward(z, sigma)
        z = anima_euler_step(z, v, sigma, sigma_next)
        if not torch.isfinite(z).all():
            raise FloatingPointError(f"non-finite latent during {prediction_type}-pred Euler sampling")
    return z


def _sample_heun(
    student_forward,
    eps_latent: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    prediction_type: str,
    eps_floor: float,
) -> torch.Tensor:
    z = eps_latent
    if sigmas.numel() <= 2:
        return _sample_euler(student_forward, z, sigmas, prediction_type=prediction_type, eps_floor=eps_floor)

    for index in range(sigmas.numel() - 2):
        sigma = _sigma_like(sigmas[index], z)
        sigma_next = _sigma_like(sigmas[index + 1], z)
        if prediction_type == "x":
            v = _velocity_from_xpred(student_forward, z, sigma, eps_floor)
        else:
            v = student_forward(z, sigma)
        z_euler = anima_euler_step(z, v, sigma, sigma_next)
        if prediction_type == "x":
            v_next = _velocity_from_xpred(student_forward, z_euler, sigma_next, eps_floor)
        else:
            v_next = student_forward(z_euler, sigma_next)
        z = anima_euler_step(z, 0.5 * (v + v_next), sigma, sigma_next)
        if not torch.isfinite(z).all():
            raise FloatingPointError(f"non-finite latent during {prediction_type}-pred Heun sampling")

    sigma = _sigma_like(sigmas[-2], z)
    sigma_next = sigmas[-1].reshape(1, 1, 1, 1).to(device=z.device, dtype=z.dtype)
    if prediction_type == "x":
        v = _velocity_from_xpred(student_forward, z, sigma, eps_floor)
    else:
        v = student_forward(z, sigma)
    return anima_euler_step(z, v, sigma, sigma_next)


def _comfy_device() -> torch.device:
    try:
        import comfy.model_management as model_management

        return torch.device(model_management.get_torch_device())
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _comfy_intermediate_device() -> torch.device:
    try:
        import comfy.model_management as model_management

        return torch.device(model_management.intermediate_device())
    except Exception:
        return torch.device("cpu")


def _soft_empty_cache() -> None:
    gc.collect()
    try:
        import comfy.model_management as model_management

        model_management.soft_empty_cache()
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported precision: {name!r}")


@dataclass
class LoadedAnimaXPred:
    args: Any
    adapter: Any
    student: torch.nn.Module
    device: torch.device
    dtype: torch.dtype
    prediction_type: str


class AnimaXPredModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint": (_model_names(XPRED_FOLDER, "put_xpred_checkpoint_in_models_anima_xpred.safetensors"),),
                "text_encoder": (_model_names("text_encoders", "put_qwen_text_encoder_in_models_text_encoders.safetensors"),),
                "vae": (_model_names("vae", "put_qwen_image_vae_in_models_vae.safetensors"),),
                "base_dit": (_model_names("diffusion_models", "put_anima_base_dit_in_models_diffusion_models.safetensors"),),
                "prediction_type": (["x", "v"], {"default": "x"}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "attn_mode": (["torch", "flash", "sageattn", "xformers"], {"default": "flash"}),
                "text_encoder_cpu": ("BOOLEAN", {"default": False}),
                "fp8": ("BOOLEAN", {"default": False}),
                "fp8_scaled": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("RUM_ANIMA_XPRED",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "RUM/Anima XPred"

    def load_model(
        self,
        checkpoint: str,
        text_encoder: str,
        vae: str,
        base_dit: str,
        prediction_type: str,
        precision: str,
        attn_mode: str,
        text_encoder_cpu: bool,
        fp8: bool,
        fp8_scaled: bool,
    ):
        checkpoint_path = _model_path(XPRED_FOLDER, checkpoint)
        text_encoder_path = _model_path("text_encoders", text_encoder)
        vae_path = _model_path("vae", vae)
        base_dit_path = _model_path("diffusion_models", base_dit)
        for label, path in {
            "checkpoint": checkpoint_path,
            "text_encoder": text_encoder_path,
            "vae": vae_path,
            "base_dit": base_dit_path,
        }.items():
            if not path.is_file():
                raise FileNotFoundError(f"{label} not found: {path}")

        device = _comfy_device()
        dtype = _dtype_from_name(precision)
        args = SimpleNamespace(
            dit=str(base_dit_path),
            student_init=str(checkpoint_path),
            text_encoder=str(text_encoder_path),
            vae=str(vae_path),
            output_dir=str(NODE_ROOT / "outputs/comfyui-rum-anima-xpred"),
            negative_prompt="",
            flow_shift=3.0,
            teacher_steps=40,
            attn_mode=attn_mode,
            fp8=fp8,
            fp8_scaled=fp8_scaled,
            text_encoder_cpu=text_encoder_cpu,
            teacher_lora=None,
            teacher_lora_weight=1.0,
            vae_spatial_chunk_size=None,
            vae_disable_cache=False,
        )
        adapter = create_adapter(args, device=device, dtype=dtype)
        student = adapter.load_student_xpred(init_checkpoint=str(checkpoint_path))
        student.to(device=device, dtype=dtype).eval().requires_grad_(False)
        return (LoadedAnimaXPred(args=args, adapter=adapter, student=student, device=device, dtype=dtype, prediction_type=prediction_type),)


class AnimaXPredSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("RUM_ANIMA_XPRED",),
                "prompt": ("STRING", {"multiline": True, "default": "hatsune miku, 1girl, masterpiece, best quality"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 20260701, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 10, "min": 1, "max": 200}),
                "sampler": (["heun", "euler"], {"default": "heun"}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16}),
                "flow_shift": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "eps_floor": ("FLOAT", {"default": 1e-4, "min": 1e-8, "max": 1.0, "step": 1e-5}),
                "offload_after_sample": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "RUM_ANIMA_LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "sample"
    CATEGORY = "RUM/Anima XPred"

    def sample(
        self,
        model: LoadedAnimaXPred,
        prompt: str,
        negative_prompt: str,
        seed: int,
        steps: int,
        sampler: str,
        cfg: float,
        width: int,
        height: int,
        batch_size: int,
        flow_shift: float,
        eps_floor: float,
        offload_after_sample: bool,
    ):
        if width % 8 != 0 or height % 8 != 0:
            raise ValueError("width and height must be divisible by 8")

        model.args.prompt = prompt
        model.args.negative_prompt = negative_prompt
        model.args.width = width
        model.args.height = height
        model.args.flow_shift = flow_shift
        model.student.to(device=model.device, dtype=model.dtype).eval()

        with torch.no_grad():
            sigmas = make_shifted_sigma_schedule(steps, flow_shift, device=model.device, dtype=model.dtype)
            eps_latent = make_seeded_eps_batch(
                list(range(batch_size)),
                seed=int(seed),
                height=height,
                width=width,
                device=model.device,
                dtype=model.dtype,
            )
            cond_embed, uncond_embed = model.adapter._encode_prompts([prompt] * batch_size, cfg=cfg, anima_model=model.student)
            text_conditioning = {
                "prompt_embeds": cond_embed.detach(),
                "negative_prompt_embeds": uncond_embed.detach(),
            }

            def student_forward(z: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
                return model.adapter.student_forward_xpred(
                    model.student,
                    z,
                    sigma,
                    text_conditioning,
                    guidance_scale=cfg,
                )

            if sampler == "euler":
                latent = _sample_euler(
                    student_forward,
                    eps_latent,
                    sigmas,
                    prediction_type=model.prediction_type,
                    eps_floor=eps_floor,
                )
            else:
                latent = _sample_heun(
                    student_forward,
                    eps_latent,
                    sigmas,
                    prediction_type=model.prediction_type,
                    eps_floor=eps_floor,
                )

            vae = model.adapter._load_vae()
            vae.to(model.device)
            pixels = vae.decode_to_pixels(latent.to(model.device, dtype=vae.dtype))
            if pixels.ndim == 5:
                pixels = pixels.squeeze(2)
            images = ((pixels.clamp(-1.0, 1.0) + 1.0) * 0.5).to(torch.float32)
            images = images.movedim(1, -1).to(device=_comfy_intermediate_device())
            latent_out = {
                "samples": latent.detach().to(device=_comfy_intermediate_device(), dtype=torch.float32),
                "sigmas": sigmas.detach().to(device="cpu", dtype=torch.float32),
            }

        if offload_after_sample:
            model.student.to("cpu")
            if model.adapter.vae is not None:
                model.adapter.vae.to("cpu")
            _soft_empty_cache()

        return (images, latent_out)


NODE_CLASS_MAPPINGS = {
    "AnimaXPredModelLoader": AnimaXPredModelLoader,
    "AnimaXPredSampler": AnimaXPredSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaXPredModelLoader": "Load Anima XPred Model",
    "AnimaXPredSampler": "Sample Anima XPred",
}
