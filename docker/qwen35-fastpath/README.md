# Qwen3.5 Fastpath Image

Prebuilt dependency image for dense Qwen3.5 SDPO HF Jobs.

Why it exists:

- Keep `a100-large` as the only current dense Qwen3.5 hardware target.
- Avoid per-job source builds for `causal-conv1d`.
- Preserve the validated fastpath stack from job `6a11339eb33ece92698c11b9`.
- Keep training code outside the image; launcher uploads current script snapshot to Hub per run.

HF Jobs must be able to pull the image. Use a Docker Hub repo or a Hugging Face
Space image (`hf.co/spaces/<owner>/<space>`); a local-only tag such as
`anchor-qwen35-fastpath:...` will not work remotely.

Canonical image for this project:

```text
hf.co/spaces/ZeterMordio/anchor-qwen35-fastpath
```

Publish/update the canonical Space image:

```bash
hf repo create ZeterMordio/anchor-qwen35-fastpath --type space --space-sdk docker --public --exist-ok
hf upload ZeterMordio/anchor-qwen35-fastpath \
  docker/qwen35-fastpath/Dockerfile Dockerfile \
  --repo-type space \
  --commit-message "Update Qwen3.5 fastpath Docker image"
```

Build locally:

```bash
docker build -t <registry>/anchor-qwen35-fastpath:torch2.10-cu128-fla0.5.0 docker/qwen35-fastpath
```

Optional Docker Hub/GHCR push after choosing a public/pullable target:

```bash
docker push <registry>/anchor-qwen35-fastpath:torch2.10-cu128-fla0.5.0
```

Example Docker Hub shape:

```bash
docker tag anchor-qwen35-fastpath:torch2.10-cu128-fla0.5.0 \
  zetermordio/anchor-qwen35-fastpath:torch2.10-cu128-fla0.5.0
docker push zetermordio/anchor-qwen35-fastpath:torch2.10-cu128-fla0.5.0
```

Use with HF Jobs:

```bash
python tools/launch_qwen35_fastpath_sdpo.py \
  --dry-run
```

Then rerun with `--execute` when ready. Conceptually:

- `--dry-run`: preview exact HF upload/job commands, start nothing.
- `--execute`: upload current training-script snapshot, then start the HF Job.

Install invariant:

- `causal-conv1d` must be installed with `--no-build-isolation`.
- Build isolation previously pulled `torch 2.12.0+cu130` and failed against the CUDA 12.8 image.

Current pinned stack:

- `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel`
- `transformers==5.9.0`
- `accelerate==1.13.0`
- `flash-linear-attention==0.5.0`
- `causal-conv1d==1.6.2.post1`
- `wandb==0.27.0`
- `huggingface_hub==1.16.1`
