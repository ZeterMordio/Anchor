import importlib
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def import_qwen35(monkeypatch, **env):
    return import_training_module(monkeypatch, "train_negotiation_sdpo_qwen35", **env)


def import_qwen3(monkeypatch, **env):
    return import_training_module(monkeypatch, "train_negotiation_sdpo", **env)


def import_training_module(monkeypatch, module_name, **env):
    monkeypatch.syspath_prepend(str(PROJECT_ROOT))
    for key in [
        "USE_LORA",
        "LORA_R",
        "LORA_ALPHA",
        "LORA_DROPOUT",
        "LORA_TARGET_MODULES",
        "OPTIMIZER",
        "ROLLOUT_TOKEN_TELEMETRY",
        "USE_LIGER",
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("USE_LIGER", "0")
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


class DummyQwen35(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.layers = torch.nn.ModuleList([DummyQwen35Layer()])
        self.model.vision_model = torch.nn.Module()
        self.model.vision_model.layers = torch.nn.ModuleList([DummyVisionLayer()])
        self.mtp = torch.nn.Module()
        self.mtp.layers = torch.nn.ModuleList([DummyQwen35Layer()])
        self.lm_head = torch.nn.Linear(4, 4, bias=False)


class DummyQwen3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([DummyQwen3Layer()])
        self.lm_head = torch.nn.Linear(4, 4, bias=False)


class DummyQwen3Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.q_proj = torch.nn.Linear(4, 4, bias=False)
        self.self_attn.k_proj = torch.nn.Linear(4, 4, bias=False)
        self.self_attn.v_proj = torch.nn.Linear(4, 4, bias=False)
        self.self_attn.o_proj = torch.nn.Linear(4, 4, bias=False)
        self.mlp = torch.nn.Module()
        self.mlp.gate_proj = torch.nn.Linear(4, 4, bias=False)
        self.mlp.up_proj = torch.nn.Linear(4, 4, bias=False)
        self.mlp.down_proj = torch.nn.Linear(4, 4, bias=False)


class DummyQwen35Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_attn = torch.nn.Module()
        self.linear_attn.in_proj_qkv = torch.nn.Linear(4, 4, bias=False)
        self.linear_attn.in_proj_a = torch.nn.Linear(4, 4, bias=False)
        self.linear_attn.in_proj_b = torch.nn.Linear(4, 4, bias=False)
        self.linear_attn.in_proj_z = torch.nn.Linear(4, 4, bias=False)
        self.linear_attn.out_proj = torch.nn.Linear(4, 4, bias=False)
        self.self_attn = torch.nn.Module()
        self.self_attn.q_proj = torch.nn.Linear(4, 4, bias=False)
        self.self_attn.k_proj = torch.nn.Linear(4, 4, bias=False)
        self.self_attn.v_proj = torch.nn.Linear(4, 4, bias=False)
        self.self_attn.o_proj = torch.nn.Linear(4, 4, bias=False)
        self.mlp = torch.nn.Module()
        self.mlp.gate_proj = torch.nn.Linear(4, 4, bias=False)
        self.mlp.up_proj = torch.nn.Linear(4, 4, bias=False)
        self.mlp.down_proj = torch.nn.Linear(4, 4, bias=False)


class DummyVisionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.q_proj = torch.nn.Linear(4, 4, bias=False)
        self.mlp = torch.nn.Module()
        self.mlp.gate_proj = torch.nn.Linear(4, 4, bias=False)


def test_qwen35_lora_targets_only_text_transformer_layers(monkeypatch):
    mod = import_qwen35(monkeypatch, USE_LORA=1)

    targets = mod._infer_lora_target_modules(DummyQwen35())

    assert "model.language_model.layers.0.linear_attn.in_proj_qkv" in targets
    assert "model.language_model.layers.0.linear_attn.out_proj" in targets
    assert "model.language_model.layers.0.self_attn.q_proj" in targets
    assert "model.language_model.layers.0.mlp.gate_proj" in targets
    assert "model.vision_model.layers.0.self_attn.q_proj" not in targets
    assert "model.vision_model.layers.0.mlp.gate_proj" not in targets
    assert "mtp.layers.0.self_attn.q_proj" not in targets
    assert "lm_head" not in targets


def test_lora_defaults_switch_to_cuda_optimizer_and_run_name(monkeypatch):
    mod = import_qwen35(monkeypatch, USE_LORA=1, LORA_R=32, LORA_ALPHA=64)

    assert mod.OPTIMIZER == "adamw_cuda"
    assert mod.training_adapter_slug() == "lora-r32-a64"
    assert "lora-r32-a64" in mod.default_run_name()


def test_dense_defaults_keep_pre_lora_run_identity(monkeypatch):
    mod = import_qwen35(monkeypatch)

    assert mod.OPTIMIZER == "adamw_cpu"
    assert "__fullft" not in mod.default_run_name()
    assert "__lora-" not in mod.default_run_name()
    assert "__fullft" not in mod.default_wandb_group()
    assert "__lora-" not in mod.default_wandb_group()


def test_qwen3_fallback_lora_targets_and_defaults(monkeypatch):
    mod = import_qwen3(monkeypatch, USE_LORA=1, LORA_R=32, LORA_ALPHA=64)

    targets = mod._infer_lora_target_modules(DummyQwen3())

    assert mod.OPTIMIZER == "adamw_cuda"
    assert mod.training_adapter_slug() == "lora-r32-a64"
    assert "model.layers.0.self_attn.q_proj" in targets
    assert "model.layers.0.mlp.down_proj" in targets
    assert "lm_head" not in targets


def test_qwen3_dense_defaults_keep_pre_lora_run_identity(monkeypatch):
    mod = import_qwen3(monkeypatch)

    assert mod.OPTIMIZER == "adamw_cpu"
    assert "__fullft" not in mod.default_run_name()
    assert "__lora-" not in mod.default_run_name()
    assert "__fullft" not in mod.default_wandb_group()
    assert "__lora-" not in mod.default_wandb_group()


def test_qwen35_text_loader_falls_back_to_tokenizer_without_image_deps(monkeypatch):
    mod = import_qwen35(monkeypatch)

    class DummyConfig:
        model_type = "qwen3_5"
        vision_config = object()

    class FailingProcessor:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise ImportError("PIL and torchvision are unavailable")

    class DummyTokenizer:
        pass

    class DummyImageTextModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return "model"

    tokenizer = DummyTokenizer()
    monkeypatch.setattr(mod.AutoConfig, "from_pretrained", lambda *args, **kwargs: DummyConfig())
    monkeypatch.setattr(mod, "AutoProcessor", FailingProcessor)
    monkeypatch.setattr(mod.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(mod, "AutoModelForImageTextToText", DummyImageTextModel)

    model, processor, returned_tokenizer = mod._load_text_or_image_text_stack("dummy-qwen35")

    assert model == "model"
    assert processor is tokenizer
    assert returned_tokenizer is tokenizer


def test_rollout_token_telemetry_counts_finalizer_and_tail(monkeypatch):
    mod = import_qwen35(monkeypatch)

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [tok for tok in text.replace("\n", " ").split(" ") if tok]

    telemetry = mod._new_rollout_token_telemetry()
    decoded = "<think>private</think>\nTalk: ok\nAction: [BUY] $12.00 (1x sku)\nextra tail"

    mod._record_rollout_token_result(
        telemetry,
        "buyer",
        FakeTokenizer(),
        prompt_tokens=10,
        first_pass_generated_tokens=8,
        first_pass_had_action=False,
        finalizer_generated_tokens=3,
        finalizer_used=True,
        decoded=decoded,
    )

    summary = mod._summarize_rollout_token_telemetry(telemetry)

    assert summary["rollout_token_buyer_sequences"] == 1
    assert summary["rollout_token_buyer_prompt_mean"] == 10
    assert summary["rollout_token_buyer_total_generated_mean"] == 11
    assert summary["rollout_token_buyer_finalizer_rate"] == 1
    assert summary["rollout_token_buyer_first_pass_action_rate"] == 0
    assert summary["rollout_token_buyer_parseable_action_rate"] == 1
    assert summary["rollout_token_buyer_tail_after_action_mean"] > 0
    assert summary["rollout_token_total_sequences"] == 1


def test_rollout_token_telemetry_can_be_disabled(monkeypatch):
    mod = import_qwen35(monkeypatch, ROLLOUT_TOKEN_TELEMETRY=0)

    telemetry = mod._new_rollout_token_telemetry()

    assert mod._summarize_rollout_token_telemetry(telemetry) == {}
