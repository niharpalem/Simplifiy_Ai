# Qwen-Image-Edit: Setup & Usage Guide

## Overview

Qwen-Image-Edit is a **diffusion model** for image editing with ~28B total parameters:
- **Transformer**: 20.4B params (main diffusion model)
- **Text Encoder**: 8.3B params (converts text to embeddings)
- **VAE**: 0.1B params (encodes/decodes images)

## Architecture Flow

```
Text Prompt                    Input Image
     ↓                              ↓
Text Encoder (8B)              VAE Encoder
     ↓                              ↓
 text embeddings              image latents
     ↓                              ↓
     └──────────┬───────────────────┘
                ↓
        TRANSFORMER (20B)
        (iterative denoising)
                ↓
           VAE Decoder
                ↓
          Output Image
```

## Installation

```bash
# Python 3.12+ recommended
python3.12 -m venv ~/py312env
source ~/py312env/bin/activate

pip install torch diffusers transformers accelerate peft bitsandbytes
```

## Loading the Model

### Option 1: CPU Offload (22GB+ VRAM)
Loads components to GPU one at a time. Slower but fits in memory.

```python
from diffusers import QwenImageEditPipeline
import torch

pipe = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()
```

### Option 2: 4-bit Quantization (~14GB VRAM)
Compresses model weights. Faster inference but slight quality loss.

```python
from diffusers import QwenImageEditPipeline, PipelineQuantizationConfig
import torch

quantization_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.bfloat16}
)

pipe = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

## Basic Usage

```python
from PIL import Image
import requests

# Load image (must be RGB)
url = "https://example.com/image.png"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB").resize((512, 512))

# Run edit
output = pipe(
    image=image,
    prompt="make the cat wear sunglasses",
    num_inference_steps=20
).images[0]

output.save("edited.png")
```

## Hyperparameters

| Parameter | Description | Range | Default |
|-----------|-------------|-------|---------|
| `num_inference_steps` | Denoising iterations. More = better quality, slower | 10-50 | 20 |
| `guidance_scale` | Prompt adherence. Higher = stricter to prompt | 1.0-10.0 | 3.5 |
| `num_images_per_prompt` | Number of output variations | 1-4 | 1 |
| `generator` | Random seed for reproducibility | torch.Generator | None |
| `height` / `width` | Output image dimensions | 256-1024 | 512 |

### Example with All Parameters

```python
output = pipe(
    image=image,
    prompt="make the cat wear sunglasses",
    num_inference_steps=20,
    guidance_scale=3.5,
    height=512,
    width=512,
    num_images_per_prompt=1,
    generator=torch.Generator().manual_seed(42)  # Reproducible
).images[0]
```

## Understanding Key Concepts

### Inference Steps
More steps = better denoising from noise to clean image:
```
Step 1:  Pure noise      ████████████
Step 10: Rough shape     ▓▓▓▓████████
Step 20: Clean image     ░░░░░░░░░░░░
```

### Guidance Scale
- **Low (1-3)**: Creative, may deviate from prompt
- **Medium (3-5)**: Balanced (recommended)
- **High (7-10)**: Strict to prompt, can look artificial

### CPU Offload Mechanism
```
CPU RAM                     GPU VRAM
┌──────────────────┐       ┌──────────────┐
│ VAE              │       │              │
│ Text Encoder     │ ────► │ Active Model │
│ Transformer      │       │              │
└──────────────────┘       └──────────────┘

Only ONE component on GPU at a time.
Peak VRAM = largest component (20GB), not total (28GB).
```

## Inspecting Model Architecture

```python
# View all components
print(pipe.components.keys())

# Count parameters per component
for name, component in pipe.components.items():
    if hasattr(component, 'parameters'):
        params = sum(p.numel() for p in component.parameters())
        print(f"{name}: {params:,} parameters")

# View transformer architecture
print(pipe.transformer)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA OOM | Use `enable_model_cpu_offload()` or quantization |
| Black output | Increase `num_inference_steps` (min 10-20) |
| 4 channels error | Convert image: `image.convert("RGB")` |
| Import errors | Upgrade: `pip install --upgrade transformers diffusers` |

## Memory Requirements

| Method | VRAM Needed | Speed |
|--------|-------------|-------|
| Full model | ~56GB | Fastest |
| 4-bit quantized | ~14GB | Fast |
| CPU offload | ~20GB peak | Slow |
---
*Generated from hands-on exploration of Qwen-Image-Edit*
"""


