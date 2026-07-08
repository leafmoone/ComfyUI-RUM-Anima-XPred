from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_node_module():
    spec = importlib.util.spec_from_file_location("rum_anima_xpred_nodes", ROOT / "__init__.py")
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DummyAdapter:
    def __init__(self, args, device, dtype):
        self.args = args
        self.device = device
        self.dtype = dtype

    def load_student_xpred(self, init_checkpoint):
        self.init_checkpoint = init_checkpoint
        return torch.nn.Linear(1, 1)


def test_lora_loader_returns_new_rum_model(monkeypatch, tmp_path):
    module = load_node_module()
    lora_path = tmp_path / "style.safetensors"
    lora_path.write_bytes(b"placeholder")

    created = {}

    def fake_model_path(folder_name, filename):
        assert folder_name == "loras"
        assert filename == "style.safetensors"
        return lora_path

    def fake_create_adapter(args, device, dtype):
        created["args"] = args
        return DummyAdapter(args, device, dtype)

    monkeypatch.setattr(module, "_model_path", fake_model_path)
    monkeypatch.setattr(module, "create_adapter", fake_create_adapter)

    original_args = SimpleNamespace(
        dit="/models/diffusion_models/base.safetensors",
        student_init="/models/diffusion_models/base.safetensors",
        text_encoder="/models/text_encoders/qwen.safetensors",
        vae="/models/vae/qwen_vae.safetensors",
        teacher_lora=None,
        teacher_lora_weight=1.0,
    )
    original = module.LoadedAnimaXPred(
        args=original_args,
        adapter=object(),
        student=torch.nn.Linear(1, 1),
        device=torch.device("cpu"),
        dtype=torch.float32,
        prediction_type="x",
    )

    (loaded,) = module.AnimaXPredLoraLoader().load_lora(original, "style.safetensors", 0.75)

    assert loaded is not original
    assert loaded.args is not original_args
    assert loaded.args.teacher_lora == str(lora_path)
    assert loaded.args.teacher_lora_weight == 0.75
    assert loaded.args.student_init == original_args.student_init
    assert loaded.prediction_type == "x"
    assert created["args"] is loaded.args
