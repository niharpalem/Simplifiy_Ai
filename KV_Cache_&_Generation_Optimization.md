# 🔄 KV Cache & Generation Optimization

## The Core Problem

Decoder-only networks generate tokens **sequentially** - this process cannot be parallelized, making generation slow.

---

## 1. KV Cache

### What It Does
Stores Key-Value pairs from previous tokens to avoid recomputing them for every new token.

### Without KV Cache ❌
```
Token 1: Compute attention for token 1
Token 2: Recompute token 1 + compute token 2
Token 3: Recompute tokens 1,2 + compute token 3
Token 4: Recompute tokens 1,2,3 + compute token 4

Complexity: O(n²) where n is sequence length
```

### With KV Cache ✅
```
Token 1: Compute & store K,V
Token 2: Reuse K,V from token 1 + compute token 2
Token 3: Reuse K,V from tokens 1,2 + compute token 3
Token 4: Reuse K,V from tokens 1,2,3 + compute token 4

Complexity: O(n) - much faster!
```

### Visual Representation
```
Generation Step 1:
  Query₁ attends to → Key₁, Value₁
  Store: K₁, V₁

Generation Step 2:
  Query₂ attends to → Key₁, Value₁ (from cache)
                     Key₂, Value₂ (new)
  Store: K₁, V₁, K₂, V₂

Generation Step 3:
  Query₃ attends to → Key₁, Value₁ (from cache)
                     Key₂, Value₂ (from cache)
                     Key₃, Value₃ (new)
  Store: K₁, V₁, K₂, V₂, K₃, V₃
```

### The Challenge
**KV cache grows with sequence length** → Memory issues!

```
Model: LLaMA-7B
Layers: 32
Heads: 32
Head dimension: 128

KV cache per token = 2 (K+V) × 32 layers × 32 heads × 128 dim
                   = 262,144 values per token
                   
For 2048 tokens: ~2 GB just for KV cache!
```

---

## 2. Static Cache

### The Problem
Dynamic KV cache prevents using `torch.compile` optimization because the cache size changes with each generation step.

### Solution
Pre-allocate KV cache to a **fixed maximum size**.

```python
# Dynamic cache (grows)
cache_size = current_tokens  # Changes each step ❌

# Static cache (fixed)
cache_size = max_tokens  # Fixed size ✅
```

### Benefits
- ✅ Enables `torch.compile` optimization
- ✅ Up to **4x speedup** in forward pass
- ✅ Predictable memory usage

### Trade-off
- ❌ Wastes memory if you allocate for 2048 but only use 100 tokens
- But the speed gain often outweighs the memory waste!

---

## 3. Continuous Batching

### Traditional Batching Problem

```
Batch: [Request A (100 tokens), Request B (10 tokens), Request C (50 tokens)]

Timeline:
0-10s:  All 3 requests processing
10s:    Request B finishes → GPU idle for B's slot
10-50s: Requests A, C still processing → B's slot wasted
50s:    Request C finishes → GPU idle for C's slot
50-100s: Request A still processing → B and C slots wasted

Result: GPU underutilized, waiting for longest request
```

### Continuous Batching Solution

```
Time 0: Batch = [Request A, Request B, Request C]
        All GPUs busy ✅

Time 1: Request B finishes (10 tokens)
        Immediately: Batch = [Request A, Request D, Request C]
        Replace B with new Request D ✅

Time 2: Request C finishes (50 tokens)
        Immediately: Batch = [Request A, Request D, Request E]
        Replace C with new Request E ✅

Result: GPU always processing full batch!
```

### Visual Comparison

**Traditional Batching:**
```
Request A: [████████████████████] (100 tokens)
Request B: [██]                    (10 tokens) → idle
Request C: [████████]              (50 tokens) → idle
           └────── GPU utilization: 60% ──────┘
```

**Continuous Batching:**
```
Request A: [████████████████████] (100 tokens)
Request B: [██]                    (10 tokens)
Request D:     [████]              (20 tokens)
Request E:         [████████]      (40 tokens)
Request C: [████████]              (50 tokens)
Request F:         [██████]        (30 tokens)
           └────── GPU utilization: 95%+ ──────┘
```

### Key Benefits
- ✅ **Maximizes GPU utilization** - always processing full batch
- ✅ **Reduces latency** - new requests start immediately
- ✅ **Higher throughput** - more requests processed per second
- ✅ **Better for production** - handles variable-length requests efficiently

### Implementation Note
Most modern inference frameworks (vLLM, TGI, TensorRT-LLM) implement continuous batching natively.

---

## 4. Speculative Decoding

### The Problem
Even with continuous batching, we're still generating **one token at a time**.

```
Traditional Generation:
Step 1: Generate token 1 (large model) → 100ms
Step 2: Generate token 2 (large model) → 100ms
Step 3: Generate token 3 (large model) → 100ms
Step 4: Generate token 4 (large model) → 100ms

Total: 400ms for 4 tokens
```

### The Solution
Use a **small, fast model** to draft multiple tokens, then **large model** verifies all at once.

```
Speculative Decoding:
Step 1: Small model drafts 4 tokens → 20ms
        Draft: ["The", "cat", "sat", "down"]

Step 2: Large model verifies all 4 in parallel → 100ms
        ✓ "The" - Accept
        ✓ "cat" - Accept
        ✓ "sat" - Accept
        ✗ "down" - Reject, should be "on"
        
        Result: Accept 3, regenerate 1

Total: 120ms for 3-4 tokens
Speedup: 2-3x faster!
```

### How It Works

```
┌─────────────────────────────────────────────┐
│ Small Model (Draft)                         │
│ - Fast (1-2B parameters)                    │
│ - Lower quality                             │
│ - Generates N candidate tokens              │
└─────────────────────────────────────────────┘
                    ↓
            [Draft tokens]
                    ↓
┌─────────────────────────────────────────────┐
│ Large Model (Verify)                        │
│ - Slower (7B-70B parameters)                │
│ - High quality                              │
│ - Verifies all N tokens in PARALLEL        │
│ - Accepts correct, rejects incorrect        │
└─────────────────────────────────────────────┘
```

### Why It's Fast

**Key Insight:** Large models are slow at **generating** but fast at **evaluating** (parallel verification).

```
Serial generation (traditional):
Token 1: [████] 100ms
Token 2: [████] 100ms
Token 3: [████] 100ms
Token 4: [████] 100ms
Total: 400ms

Parallel verification (speculative):
Tokens 1-4: [████] 100ms (all at once!)
Total: 100ms + draft overhead (~20ms) = 120ms
```

### Acceptance Rate

The effectiveness depends on how often the small model matches the large model:

```
High acceptance (80%+):
  Draft: ["The", "cat", "sat", "on"]
  Large: Accept all 4 ✓
  Speedup: 4x

Medium acceptance (50%):
  Draft: ["The", "cat", "sat", "down"]
  Large: Accept 3, reject 1 ✓✓✓✗
  Speedup: 2-3x

Low acceptance (25%):
  Draft: ["The", "dog", "ran", "quickly"]
  Large: Accept 1, reject 3 ✓✗✗✗
  Speedup: ~1.2x (barely worth it)
```

### Implementation

```python
# Using HuggingFace Transformers
from transformers import AutoModelForCausalLM

# Load large model
large_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

# Load small draft model
small_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-1.1b")

# Generate with speculative decoding
output = large_model.generate(
    prompt,
    assistant_model=small_model,      # Small model for drafting
    do_sample=True,
    temperature=0.7,
)
```

**Alternative: Prompt Lookup Decoding**
```python
# Uses n-gram matching instead of small model
output = model.generate(
    prompt,
    prompt_lookup_num_tokens=4,  # Look ahead 4 tokens
)
```

### Best Practices

1. **Choose the right draft model:**
   - Same architecture family as large model (e.g., both LLaMA)
   - 5-10x smaller than target model
   - Examples: LLaMA-1B drafting for LLaMA-7B

2. **When to use:**
   - ✅ Predictable text (code, structured output)
   - ✅ High temperature sampling (creative writing)
   - ❌ Low temperature (factual answers) - lower acceptance

3. **Tune draft length:**
   - Too short (2-3): Less speedup potential
   - Too long (10+): Higher rejection rate
   - Sweet spot: 4-6 tokens

---

## Summary Comparison

| Technique | What It Optimizes | Speedup | Memory Impact |
|-----------|------------------|---------|---------------|
| **KV Cache** | Avoid recomputation | 10-100x | Increases memory |
| **Static Cache** | Enable torch.compile | 4x | Fixed memory |
| **Continuous Batching** | GPU utilization | 2-3x throughput | Neutral |
| **Speculative Decoding** | Token generation | 2-4x | Slight increase |

---

## Combining Techniques

All these techniques work together!

```
Production LLM Inference Setup:

1. KV Cache: Store previous tokens ✅
2. Static Cache: Pre-allocate for torch.compile ✅
3. Continuous Batching: Always process full batch ✅
4. Speculative Decoding: Generate multiple tokens ✅

Result: 10-20x faster than naive implementation!
```

---

## Real-World Impact

### Example: Serving LLaMA-7B

**Baseline (no optimizations):**
- Throughput: 10 tokens/second
- Batch size: 1
- Latency: 100ms per token

**With all optimizations:**
- Throughput: 100-200 tokens/second
- Batch size: 32+
- Latency: 10-20ms per token

**10-20x improvement!**