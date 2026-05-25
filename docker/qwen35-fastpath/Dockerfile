FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN /usr/bin/python -m pip install --no-cache-dir -q \
    numpy \
    packaging \
    ninja \
    wheel \
    setuptools \
    "transformers==5.9.0" \
    "accelerate==1.13.0" \
    "flash-linear-attention==0.5.0" \
    safetensors \
    "wandb==0.27.0" \
    "huggingface_hub==1.16.1" \
    pillow \
 && /usr/bin/python -m pip install --no-cache-dir -q --no-build-isolation \
    "causal-conv1d==1.6.2.post1" \
 && /usr/bin/python - <<'PY'
import importlib
import importlib.metadata as metadata

required = [
    ("causal_conv1d", "causal_conv1d_fn"),
    ("causal_conv1d", "causal_conv1d_update"),
    ("fla.modules", "FusedRMSNormGated"),
    ("fla.ops.gated_delta_rule", "chunk_gated_delta_rule"),
    ("fla.ops.gated_delta_rule", "fused_recurrent_gated_delta_rule"),
]
for module_name, attr_name in required:
    module = importlib.import_module(module_name)
    if getattr(module, attr_name, None) is None:
        raise SystemExit(f"missing {module_name}.{attr_name}")

for package in [
    "torch",
    "transformers",
    "accelerate",
    "flash-linear-attention",
    "causal-conv1d",
    "wandb",
    "huggingface_hub",
]:
    print(f"{package}={metadata.version(package)}")
PY

WORKDIR /workspace
EXPOSE 7860
CMD ["/usr/bin/python", "-c", "from http.server import HTTPServer, SimpleHTTPRequestHandler; print('serving on 0.0.0.0:7860', flush=True); HTTPServer(('0.0.0.0', 7860), SimpleHTTPRequestHandler).serve_forever()"]
