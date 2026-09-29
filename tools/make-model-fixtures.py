"""Build tiny random models with mlx-lm, for bit-exact parity tests.

For each architecture, mlx-lm creates a small random model (bf16, 2 layers,
hidden 64; MoE: 8 experts, top-4).  Its weights, config and mlx-lm's own
logits for a fixed token sequence go to tests/fixtures/tiny/<name>/:

  config.json, model.safetensors
  logits-full.npy          one forward pass over all 16 tokens
  logits-incremental.npy   12-token prefill, then 4 single-token steps
                           (rows = logits of the last position each call)

tests/llm.lisp loads each model with mlx.llm:load-model and requires the
same logits.  The 16-token pass routes 16 x 4 = 64 expert slots, exercising
SwitchGLU's sorted dispatch; single tokens take the unsorted path.

Exact agreement depends on the MLX version: use the one mlx-c is built on
(see (mlx:version)).  The version used is recorded in tiny/MLX_VERSION.

Usage (a Python environment with mlx-lm and that MLX version):
  python tools/make-model-fixtures.py
"""

import importlib
import json
import os
import shutil

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.cache import make_prompt_cache

ROOT = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "tiny")
TOKENS = [5, 17, 3, 42, 8, 60, 1, 33, 21, 9, 50, 12, 7, 44, 2, 30]

COMMON = {
    "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
    "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 64,
    "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": False,
    "max_position_embeddings": 256,
}
MOE = {"num_experts_per_tok": 4, "norm_topk_prob": True}

FAMILIES = {
    "qwen3": {"model_type": "qwen3", "intermediate_size": 64},
    "mixtral": {"model_type": "mixtral", "intermediate_size": 16, "num_local_experts": 8,
                "num_experts_per_tok": 4},
    "qwen2_moe": {"model_type": "qwen2_moe", "intermediate_size": 64, "moe_intermediate_size": 16,
                  "shared_expert_intermediate_size": 32, "num_experts": 8, **MOE},
    "qwen3_moe": {"model_type": "qwen3_moe", "intermediate_size": 64, "moe_intermediate_size": 16,
                  "num_experts": 8, "decoder_sparse_step": 1, "mlp_only_layers": [1], **MOE},
    "olmoe": {"model_type": "olmoe", "intermediate_size": 16, "num_experts": 8,
              "num_experts_per_tok": 4, "norm_topk_prob": False},
    "mixtral-4bit": {"model_type": "mixtral", "intermediate_size": 64, "num_local_experts": 8,
                     "num_experts_per_tok": 4, "quantization": {"group_size": 64, "bits": 4}},
}


def build(config):
    module = importlib.import_module(f"mlx_lm.models.{config['model_type']}")
    args = module.ModelArgs.from_dict(config)
    model = module.Model(args)
    model.set_dtype(mx.bfloat16)
    if "quantization" in config:
        q = config["quantization"]
        nn.quantize(model, group_size=q["group_size"], bits=q["bits"],
                    class_predicate=lambda p, m: hasattr(m, "to_quantized"))
    mx.eval(model.parameters())
    return model


def main():
    os.makedirs(ROOT, exist_ok=True)
    with open(os.path.join(ROOT, "MLX_VERSION"), "w") as f:
        f.write(mx.__version__ + "\n")
    for name, overrides in FAMILIES.items():
        mx.random.seed(0)
        config = {**COMMON, **overrides}
        model = build(config)
        out = os.path.join(ROOT, name)
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(out)
        with open(os.path.join(out, "config.json"), "w") as f:
            json.dump(config, f, indent=1)
        mx.save_safetensors(os.path.join(out, "model.safetensors"), dict(tree_flatten(model.parameters())))

        full = model(mx.array([TOKENS]))[0].astype(mx.float32)
        cache = make_prompt_cache(model)
        rows = [model(mx.array([TOKENS[:12]]), cache=cache)[0, -1]]
        for t in TOKENS[12:15]:
            rows.append(model(mx.array([[t]]), cache=cache)[0, -1])
        mx.save(os.path.join(out, "logits-full.npy"), full)
        mx.save(os.path.join(out, "logits-incremental.npy"), mx.stack(rows).astype(mx.float32))
        size = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out))
        print(f"{name:14s} {size / 1024:6.0f} KB")


if __name__ == "__main__":
    main()
