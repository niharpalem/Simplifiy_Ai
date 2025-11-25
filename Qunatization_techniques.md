# 🔢 Model Quantization

Reduce model precision to save memory and increase inference speed.

---

## Overview

**Goal:** Compress model weights and activations from high precision (FP32) to lower precision (FP16, INT8, INT4, etc.)

**Benefits:**
- ✅ Lower memory usage (2-8x compression)
- ✅ Faster inference (integer operations are faster)
- ✅ Enables deployment on smaller devices

**Trade-off:**
- ❌ Slight accuracy loss (typically 1-5%)

---

## Quantization Approaches

### 1. Post-Training Quantization (PTQ)
- Load pre-trained model and quantize weights
- **No training or fine-tuning required**
- Fast and simple
- Slight accuracy drop

```python
# Example: Load model in INT8
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    load_in_8bit=True
)
```

### 2. Quantization-Aware Training (QAT)
- Train or fine-tune with quantization in mind
- Model learns to adapt to lower precision
- **Requires training compute**
- Better accuracy than PTQ

```python
# Example: Train with quantization
from torch.quantization import quantize_qat

model = quantize_qat(model)
# ... training loop ...
```

---

## Precision Formats

### Understanding Floating-Point Representation

Every floating-point number has 3 components:

```
┌────┬──────────┬─────────────────┐
│Sign│ Exponent │   Significand   │
└────┴──────────┴─────────────────┘

Sign (1 bit): Positive (0) or Negative (1)
Exponent: Controls RANGE (how big/small)
Significand: Controls PRECISION (how accurate)
```

**Example:**
```
Number: -3.14159 × 10²

Sign: - (negative)
Significand: 3.14159 (the digits)
Exponent: 2 (power of 10)
```

---

### FP32 (Full Precision)

```
┌─┬────────┬───────────────────────────┐
│1│   8    │            23             │
│S│ Exp    │       Significand         │
└─┴────────┴───────────────────────────┘
32 bits total

Exponent: 8 bits
  Range: ±10^38 (huge!)
  
Significand: 23 bits
  Precision: ~7 decimal digits
  Example: 3.141592 ✓

Memory: 4 bytes per parameter
```

**Characteristics:**
- ✅ Very wide range
- ✅ High precision
- ❌ Memory intensive
- ❌ Slower computation

**Use:** Training, reference quality

---

### FP16 (Half Precision)

```
┌─┬──────┬────────────┐
│1│  5   │     10     │
│S│ Exp  │Significand │
└─┴──────┴────────────┘
16 bits total

Exponent: 5 bits
  Range: ±65,504 (small!)
  
Significand: 10 bits
  Precision: ~3 decimal digits
  Example: 3.14 ✓

Memory: 2 bytes per parameter (50% savings)
```

**Problem:**
```
Training example:
  Gradient = 100,000
  → Overflow! (max is 65,504)
  → Becomes Infinity
  → Training crashes ❌
```

**Characteristics:**
- ✅ 50% memory savings vs FP32
- ✅ Good precision (10 bits)
- ❌ Narrow range (overflow issues)
- ❌ Not stable for training

**Use:** Inference only (with mixed precision)

---

### BF16 (Brain Float 16)

```
┌─┬────────┬───────┐
│1│   8    │   7   │
│S│ Exp    │  Sig  │
└─┴────────┴───────┘
16 bits total

Exponent: 8 bits (same as FP32!)
  Range: ±10^38 (same as FP32!)
  
Significand: 7 bits
  Precision: ~2 decimal digits
  Example: 3.1 ✓

Memory: 2 bytes per parameter (50% savings)
```

**Why it's better for deep learning:**
```
Training example:
  Gradient = 100,000
  → Fits in range! ✓
  → Stored as ~100,000 (slight approximation)
  → Training continues ✓
```

**Characteristics:**
- ✅ 50% memory savings vs FP32
- ✅ Same range as FP32 (no overflow!)
- ✅ Stable for training
- ❌ Lower precision (7 bits)

**Use:** Training and inference (recommended!)

---

### Comparison Table

| Format | Bits | Sign | Exponent | Significand | Range | Precision | Memory |
|--------|------|------|----------|-------------|-------|-----------|--------|
| **FP32** | 32 | 1 | 8 | 23 | 10^-38 to 10^38 | ~7 decimals | 4 bytes |
| **FP16** | 16 | 1 | 5 | 10 | 10^-5 to 10^5 | ~3 decimals | 2 bytes |
| **BF16** | 16 | 1 | 8 | 7 | 10^-38 to 10^38 | ~2 decimals | 2 bytes |

**Key Insight:**
```
Deep Learning Priorities:
  RANGE > Precision

Why? Large gradients are common, but don't need extreme precision.

FP16: Good precision, narrow range → Overflows ❌
BF16: Lower precision, wide range → Stable ✅
```

---

## INT8 Quantization

Convert floating-point weights to 8-bit integers: **-128 to 127**

### Basic Concept

```
Floating-point weights: [0.1, 0.5, -0.3, 0.8, ...]
                        ↓ Quantize
Integer weights: [12, 63, -38, 102, ...]

Memory: 8 bits per parameter (75% savings vs FP32!)
```

---

### Method 1: Absmax Quantization

**Formula:** `X_quant = round(127 × X / max|X|)`

**Example:**
```
Weights: [0.0, 1.0, 2.0, 3.0]
max|X| = 3.0

Quantization:
  0.0 → round(127 × 0.0 / 3.0) = 0
  1.0 → round(127 × 1.0 / 3.0) = 42
  2.0 → round(127 × 2.0 / 3.0) = 85
  3.0 → round(127 × 3.0 / 3.0) = 127

Result: [0, 42, 85, 127]
Storage: 1 byte each + 1 float for max (scale factor)
```

**Dequantization:** `X_dequant = (max|X| × X_quant) / 127`

```
Dequantize 42:
  X = (3.0 × 42) / 127 = 0.992 ≈ 1.0
  Error: 0.008 (acceptable!)
```

**Problem with Absmax:**
```
If data is asymmetric (e.g., all positive):

Weights: [0.0, 1.0, 2.0, 3.0]

Maps to: [0, 42, 85, 127]
Uses: 0 to 127 only
Wastes: -128 to -1 (negative range unused!)

Precision: 3.0 / 127 = 0.024 per step
```

---

### Method 2: Zero-Point Quantization

**Better for asymmetric data** - uses full INT8 range!

**Formula:** `X_quant = round(X / scale) + zero_point`

**Example:**
```
Weights: [0.0, 1.0, 2.0, 3.0]

Calculate scale:
  scale = (max - min) / 255
        = (3.0 - 0.0) / 255
        = 0.0118

Calculate zero-point:
  zero_point = -128 (maps 0.0 to -128)

Quantization:
  0.0 → round(0.0 / 0.0118) + (-128) = -128
  1.0 → round(1.0 / 0.0118) + (-128) = -43
  2.0 → round(2.0 / 0.0118) + (-128) = 42
  3.0 → round(3.0 / 0.0118) + (-128) = 127

Result: [-128, -43, 42, 127]
Uses: Full -128 to 127 range! ✅

Precision: 3.0 / 255 = 0.0118 per step (2x better!)
```

**Dequantization:** `X_dequant = (X_quant - zero_point) × scale`

```
Dequantize -43:
  X = (-43 - (-128)) × 0.0118
    = 85 × 0.0118
    = 1.003 ≈ 1.0
```

---

### Comparison: Absmax vs Zero-Point

| Aspect | Absmax | Zero-Point |
|--------|---------|-----------|
| **Formula** | `round(127 × X / max)` | `round(X / scale) + zero_point` |
| **Parameters** | 1 (max value) | 2 (scale + zero_point) |
| **Range Used** | Often partial (-127 to 127) | Full (-128 to 127) |
| **Best For** | Symmetric data | Any data (especially asymmetric) |
| **Precision** | Lower | Higher (2x better for asymmetric) |

**Visual Comparison:**
```
Data: [0.0 ────── 3.0] (all positive)

Absmax:
  INT8: [═══════╱unused╱═════] 
         -127    0      127
  Wastes negative range

Zero-Point:
  INT8: [═══════════════════]
        -128              127
  Uses full range!
```

---

## LLM.int8() - Mixed Precision Quantization

### The Problem: Outliers in LLMs

```
Typical weights: [-1.0, 0.5, 0.8, -0.3, 0.2, ...]
Outliers:        [50.0, -45.0] ← Very large!

Regular INT8 quantization:
  scale = 50.0 / 127 = 0.39
  
  Large weight 50.0 → 127 ✓
  Small weight 0.1 → round(0.1 / 0.39) = 0 ❌
  
  Problem: Small weights disappear!
```

**Why outliers exist in LLMs:**
- Certain features consistently have large magnitudes
- Critical for model quality
- Appear in ~0.1% of weights but dominate quantization

---

### The Solution: Mixed Precision

**LLM.int8() separates computation:**

```
Step 1: Identify outliers
  Threshold: |weight| > 6.0
  
  Normal weights (99.9%): [-1.0, 0.5, 0.8, -0.3, ...]
  Outliers (0.1%):        [50.0, -45.0]

Step 2: Quantize separately
  Normal weights → INT8 (quantized)
  Outliers → FP16 (keep full precision)

Step 3: Compute separately
  INT8 matmul (fast, 99.9% of work)
  + FP16 matmul (accurate, 0.1% of work)
  
Step 4: Combine results
  Final output = INT8 result + FP16 result
```

---

### Visual Process

```
┌────────────────────────────────────────────┐
│ Weight Matrix (4096 × 4096)                │
│                                            │
│ [0.5, 0.3, 50.0, -0.2, 0.8, -45.0, ...]  │
└────────────────────────────────────────────┘
                    ↓
        ┌───────────┴───────────┐
        │                       │
  ┌─────▼──────┐      ┌────────▼────────┐
  │ Normal     │      │ Outliers        │
  │ 99.9%      │      │ 0.1%            │
  │ [0.5, 0.3, │      │ [50.0, -45.0]   │
  │  -0.2, 0.8]│      │                 │
  └─────┬──────┘      └────────┬────────┘
        │                      │
  ┌─────▼──────┐      ┌────────▼────────┐
  │ Quantize   │      │ Keep FP16       │
  │ to INT8    │      │ precision       │
  └─────┬──────┘      └────────┬────────┘
        │                      │
  ┌─────▼──────┐      ┌────────▼────────┐
  │ INT8 matmul│      │ FP16 matmul     │
  │ (fast)     │      │ (accurate)      │
  └─────┬──────┘      └────────┬────────┘
        │                      │
        └───────────┬───────────┘
                    ↓
           ┌────────────────┐
           │ Combine Results│
           │ (Final Output) │
           └────────────────┘
```

---

### Benefits

```
Memory:
  99.9% weights in INT8: 1 byte each
  0.1% weights in FP16: 2 bytes each
  Average: ~1.002 bytes per weight
  
  Savings: 75% vs FP32! (almost as good as pure INT8)

Accuracy:
  Outliers preserved in FP16 → No quality loss
  Normal weights quantized → Small impact
  
  Result: Near-FP16 quality with INT8 memory!

Speed:
  99.9% of work in fast INT8
  0.1% of work in FP16
  
  Result: ~2-3x faster than FP16
```

---

### Implementation

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    device_map="auto",
    load_in_8bit=True  # Enables LLM.int8()
)
```

**What happens:**
- Automatically detects outliers
- Splits computation into INT8 + FP16
- Combines results transparently

---

### Key Distinction

```
INT8 (data format):
  - 8-bit integer number representation
  - Range: -128 to 127
  - Just a storage format

LLM.int8() (quantization method):
  - Smart mixed-precision approach
  - Uses INT8 for most weights
  - Uses FP16 for outliers
  - Complete quantization strategy
```

---

## INT4 Quantization

Even more aggressive compression: **-8 to 7** (4 bits)

```
FP32: 32 bits per weight
INT8: 8 bits per weight (4x compression)
INT4: 4 bits per weight (8x compression!)

Model: LLaMA-7B
  FP32: 28 GB
  INT8: 7 GB
  INT4: 3.5 GB
```

### Challenge: Extreme Precision Loss

```
INT8 range: -128 to 127 (256 values)
INT4 range: -8 to 7 (16 values only!)

Example weights: [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

INT8: Can represent all distinctly
INT4: Must round heavily → large errors
```

**Solution:** Advanced techniques like QLoRA, GGUF, GPTQ, AWQ

---

## llama.cpp & GGUF

### What It Is

- **llama.cpp:** C++ implementation for running LLMs
- **Key feature:** Runs on **CPU** (no GPU required!)
- **GGUF:** GPT-Generated Unified Format (quantization format)

```
Enables: Running 7B-70B models on laptops and consumer hardware
```

---

### Quantization Levels

Multiple precision options (trade memory for quality):

| Format | Bits/Weight | Memory (7B) | Quality vs FP16 | Best For |
|--------|-------------|-------------|-----------------|----------|
| **FP16** | 16 | ~14 GB | 100% (reference) | GPU inference |
| **Q8_0** | 8 | ~7 GB | 99% | High quality needed |
| **Q6_K** | ~6 | ~5.5 GB | 97% | Balanced |
| **Q5_K_M** | ~5 | ~5 GB | 96% | Good quality |
| **Q4_K_M** | ~4.5 | ~4 GB | 95% | **Recommended!** |
| **Q4_0** | 4 | ~3.5 GB | 90% | Minimum quality |
| **Q3_K_M** | ~3 | ~3 GB | 85% | Low memory |
| **Q2_K** | ~2.5 | ~2 GB | 70% | Maximum compression |

**Format naming:**
- **Q4** = 4-bit quantization
- **K** = Uses k-quants (improved method)
- **M** = Medium (balanced size/quality)
- **S** = Small (more compression)
- **L** = Large (better quality)

---

### How It Works: Block-wise Quantization

**Key Innovation:** Each block has its own scale factor

```
Traditional INT4:
  Quantize entire model with one scale
  → Poor precision for varied weight distributions

Block-wise INT4:
  Split weights into blocks (32-256 values)
  Each block gets own scale
  → Much better precision!
```

#### Example: Q4_0

```
Weight block (32 values):
  Original (FP16):
    [0.5, 0.3, -0.2, 0.8, 0.1, ..., 0.4]
    32 weights × 16 bits = 512 bits

  Quantized (Q4_0):
    scale: 16 bits (one per block)
    values: 32 × 4 bits = 128 bits
    Total: 16 + 128 = 144 bits

Compression: 512 → 144 bits (3.6x!)
```

**Why block-wise is better:**
```
Weights vary across model:
  Block 1: [0.1, 0.15, 0.2, ...] (small values)
  Block 2: [5.0, 6.2, 4.8, ...] (large values)

Single scale:
  scale = 6.2 / 7 = 0.89
  Block 1: 0.1 → 0 (lost precision!)

Block scales:
  Block 1 scale = 0.2 / 7 = 0.029
  Block 2 scale = 6.2 / 7 = 0.89
  Block 1: 0.1 → round(0.1/0.029) = 3 ✓
```

---

### Benefits

- ✅ **Runs on CPU** - no GPU needed!
- ✅ **Multiple quality levels** - choose memory vs accuracy
- ✅ **Fast inference** - optimized C++ code
- ✅ **Cross-platform** - Windows, Mac (Apple Silicon!), Linux
- ✅ **Smart quantization** - block-wise is better than naive INT4

### Real-World Performance

```
Hardware: MacBook Pro M1 Max (64GB RAM)
Model: LLaMA-7B Q4_K_M

Memory: 4 GB
Speed: 25-30 tokens/second
Quality: 95% of FP16

Can run 13B models at ~15 tokens/sec!
```

---

## GPTQ & EXL2

### GPTQ (Post-Training Quantization for GPT models)

**Key difference from llama.cpp:** Requires GPU (CUDA)

**How it works:**
1. Layer-wise quantization with calibration data
2. Minimize reconstruction error
3. Optimize quantization parameters per layer

```
Process:
  For each layer:
    1. Run calibration data through layer
    2. Measure activation patterns
    3. Quantize weights to minimize error
    4. Move to next layer

Result: Better quality than naive quantization
```

**Benefits:**
- ✅ Better quality than simple quantization
- ✅ 4-bit with minimal accuracy loss
- ✅ GPU acceleration

**Limitations:**
- ❌ Requires CUDA GPU
- ❌ Slower than llama.cpp on CPU

---

### EXL2 (Improved GPTQ)

**Enhanced version of GPTQ with:**
- ✅ Better quality at same bit-width
- ✅ Mixed bit-width (different layers can use different precision)
- ✅ Faster inference than GPTQ

```
Example: 7B model
  Attention layers: 4-bit (less critical)
  FFN layers: 6-bit (more critical)
  
  Average: ~4.5 bits
  Quality: Better than uniform 5-bit!
```

---

### Comparison

| Method | Hardware | Quality | Speed | Flexibility |
|--------|----------|---------|-------|-------------|
| **llama.cpp** | CPU + GPU | Good | Fast (CPU) | Multiple formats |
| **GPTQ** | GPU only | Better | Fast (GPU) | 4-bit focus |
| **EXL2** | GPU only | Best | Fastest | Mixed precision |

---

## AWQ (Activation-aware Weight Quantization)

### The Key Innovation

**Traditional:** Protect large weights during quantization  
**AWQ:** Protect weights with large **activations**

### Why Activations Matter

```
Impact = Weight × Activation

Example:
  Neuron A: weight = 10.0, activation = 0.01
            impact = 0.1 (small!)
  
  Neuron B: weight = 0.5, activation = 100
            impact = 50 (huge!)

Traditional: Protects neuron A (large weight)
AWQ: Protects neuron B (large impact!) ✓
```

---

### How It Works

#### Step 1: Measure Activations

Run calibration data through model and record activation magnitudes:

```
Calibration: ["Hello world", "The cat sat", ...]

Layer 5, position 0:
  Average activation: 100.5
  
Layer 5, position 1:
  Average activation: 0.3
  
Layer 5, position 2:
  Average activation: 50.2
```

#### Step 2: Calculate Importance

```
For each weight channel:
  importance = weight_magnitude × avg_activation

Example:
  Weight column 0: 2.0 × 100.5 = 201 (high importance!)
  Weight column 1: 8.0 × 0.3 = 2.4 (low importance)
  Weight column 2: 1.0 × 50.2 = 50.2 (medium importance)
```

#### Step 3: Scale Weights

```
Apply per-channel scaling before quantization:

High importance channels:
  Scale UP → preserve more precision during quantization
  
Low importance channels:
  Scale DOWN → allow more quantization error
```

#### Step 4: Quantize

Standard INT4 quantization, but scaled weights preserve important channels better.

---

### Visual Example

```
Layer with 4 weights:

Weights:      [0.5,  0.3,  8.0,  0.1]
Activations:  [100,  80,   0.01, 90]
Impact:       [50,   24,   0.08, 9]
                ↑     ↑     ↑     ↑
             Most  Med  Least  Low
```

**Traditional INT4:**
```
Protects weight 8.0 (largest weight)
But activation is 0.01 → impact is tiny!

Result: Wastes precision on unimportant weight
```

**AWQ:**
```
Protects weight 0.5 (largest impact: 50)
Even though weight is small, activation is huge!

Result: Preserves what actually matters for output
```

---

### Benefits

```
Quality:
  AWQ 4-bit ≈ Naive 6-bit quality
  Better accuracy at same compression!

Memory:
  Still 4 bits per weight
  Same memory savings as regular INT4

Real Results (LLaMA-7B):
  Naive INT4: 85% quality
  GPTQ INT4: 92% quality
  AWQ INT4: 95% quality ← Best!
```

---

### Implementation

```python
from awq import AutoAWQForCausalLM

# Load model
model = AutoAWQForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

# Prepare calibration data
calib_data = ["sample 1", "sample 2", ...] # 128-512 samples

# Quantize with AWQ
model.quantize(
    calib_data=calib_data,
    w_bit=4,  # 4-bit quantization
)

# Save quantized model
model.save_quantized("llama-7b-awq")
```

---

### What Makes AWQ Special

```
Activation-aware = Considers actual model behavior

Other methods:
  - Look only at weight values
  - Miss the bigger picture
  
AWQ:
  - Measures what weights multiply with
  - Protects weights that matter for output
  - Results in better quality
```

---

## Quantization Methods Summary

### Quick Comparison

| Method | Bits | Hardware | Quality | Speed | Calibration | Use Case |
|--------|------|----------|---------|-------|-------------|----------|
| **FP16** | 16 | GPU | 100% | Fast | None | Baseline |
| **BF16** | 16 | GPU/TPU | 100% | Fast | None | Training |
| **LLM.int8()** | 8 | GPU | 99% | 2x | None | Easy INT8 |
| **llama.cpp Q4** | 4 | CPU/GPU | 95% | Very fast (CPU) | None | Consumer hardware |
| **GPTQ** | 4 | GPU | 92% | Fast | Yes | GPU servers |
| **EXL2** | 3-6 | GPU | 96% | Very fast | Yes | Best GPU quality |
| **AWQ** | 4 | GPU | 95% | Fast | Yes | Best INT4 quality |

---

### Decision Tree

```
Need to run on CPU?
  → llama.cpp (GGUF format)

Have GPU, want best quality?
  → AWQ or EXL2

Have GPU, want easy setup?
  → LLM.int8()

Have GPU, want maximum compression?
  → GPTQ or llama.cpp Q2_K

Need to train?
  → BF16 or FP32
```

---

## Real-World Examples

### Example 1: LLaMA-7B Compression

```
Original (FP32): 28 GB
  ❌ Doesn't fit on consumer GPU

FP16: 14 GB
  ✓ Fits on 16GB GPU
  Speed: 20 tokens/sec

LLM.int8(): 7 GB
  ✓ Fits on 8GB GPU
  Speed: 35 tokens/sec
  Quality: 99%

AWQ INT4: 3.5 GB
  ✓ Fits on 4GB GPU (or CPU!)
  Speed: 50 tokens/sec
  Quality: 95%

llama.cpp Q2_K: 2 GB
  ✓ Runs on laptop CPU
  Speed: 15 tokens/sec
  Quality: 70%
```

### Example 2: Deployment Trade-offs

```
High-quality API service:
  → FP16 or BF16 on A100 GPU
  Max quality, fast, expensive

Cloud inference (cost-conscious):
  → LLM.int8() on L4 GPU
  Good quality, 2x cheaper

Edge device (phone, laptop):
  → llama.cpp Q4_K_M
  Decent quality, runs anywhere

Extreme compression:
  → llama.cpp Q2_K or EXL2 mixed
  Lower quality, tiny size
```

---

## Best Practices

### 1. Choose Right Precision

```
Training: FP32 or BF16
Inference (quality critical): FP16 or LLM.int8()
Inference (resource limited): AWQ or llama.cpp Q4
Maximum compression: llama.cpp Q2_K
```

### 2. Always Validate

```
After quantization:
  1. Test on evaluation set
  2. Compare perplexity to baseline
  3. Test real-world prompts
  4. Check for specific failure modes
```

### 3. Calibration Data

```
For methods needing calibration (AWQ, GPTQ):
  - Use diverse, representative samples
  - 128-512 samples usually sufficient
  - Match your actual use case
  - Include edge cases
```

### 4. Monitor Quality

```
Acceptable quality loss: 1-5%
Warning signs:
  - Repetitive outputs
  - Nonsensical responses
  - Failure on specific tasks
  
If quality drops >5%: Try higher precision or better method
```

---

## Summary

**Quantization enables:**
- ✅ Running large models on smaller hardware
- ✅ Faster inference (2-4x speedup)
- ✅ Lower deployment costs
- ✅ Edge device deployment

**Key techniques:**
- **FP16/BF16:** Standard precision reduction
- **LLM.int8():** Mixed precision for LLMs
- **llama.cpp:** CPU-friendly quantization
- **GPTQ/EXL2:** GPU-optimized INT4
- **AWQ:** Activation-aware INT4

**Choose based on:**
- Hardware constraints (CPU vs GPU)
- Quality requirements
- Deployment target
- Cost considerations

**Remember:** Always validate quantized models on your specific use case!