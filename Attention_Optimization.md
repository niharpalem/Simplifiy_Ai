# ⚡ Attention Optimization Techniques

## The Core Problem

Attention mechanism has **quadratic complexity** with respect to sequence length.

```
Attention computation: Q @ K^T @ V

Sequence length: n tokens
Attention matrix size: n × n

Memory: O(n²)
Computation: O(n²)
```

### Why It's Quadratic

```
For 1000 tokens:
Attention matrix = 1000 × 1000 = 1,000,000 elements

For 2000 tokens:
Attention matrix = 2000 × 2000 = 4,000,000 elements

Double the sequence → 4x the memory and computation!
```

### Real-World Impact

| Sequence Length | Attention Matrix Size | Memory (FP16) |
|-----------------|----------------------|---------------|
| 512 tokens | 512 × 512 | ~0.5 MB |
| 2,048 tokens | 2,048 × 2,048 | ~8 MB |
| 8,192 tokens | 8,192 × 8,192 | ~134 MB |
| 32,768 tokens | 32,768 × 32,768 | ~2.1 GB |

**Per layer, per attention head!** A model with 32 layers × 32 heads needs this 1024 times.

---

## 1. Paged Attention

### The Problem: Memory Fragmentation

**Traditional KV Cache Requirements:**
- Needs **contiguous memory** allocation
- Must allocate **maximum possible size** upfront
- Leads to massive memory waste

```
Example: Request needs 100 tokens

Traditional allocation:
  Allocate: 2048 contiguous slots (max length)
  Use: 100 slots
  Waste: 1948 slots (95% wasted!)

GPU Memory State:
[Used 500][Free 600][Used 300][Free 500][Used 200]

Need: 1000 contiguous slots
Available: 1100 total free (but scattered)
Result: ❌ OUT OF MEMORY (even though enough space exists!)
```

### The Solution: Block-Based Memory

**Paged Attention** partitions KV cache into fixed-size blocks that can be placed anywhere in memory.

```
Traditional (Contiguous):
Need 1000 tokens:
[═══════════════ 1000 slots ═══════════════]
Must be one continuous block

Paged (Blocks):
Need 1000 tokens (block size = 16):
[Block 5][Other][Block 12][Other][Block 8][Block 23]...
   16      data     16      data     16       16
Blocks can be scattered anywhere!
```

### How It Works

#### Step 1: Fixed Block Size
```
Block size: 16 tokens (typical)

Each block stores:
- Keys for 16 tokens
- Values for 16 tokens
```

#### Step 2: Block Table (Memory Map)
```
Request's logical view:
[Token 0-15][Token 16-31][Token 32-47]

Physical GPU memory:
Address 0x1000: [Other data]
Address 0x2000: [Block 1: tokens 16-31] ← Request's block
Address 0x3000: [Other data]
Address 0x4000: [Block 0: tokens 0-15]  ← Request's block
Address 0x5000: [Other data]
Address 0x6000: [Block 2: tokens 32-47] ← Request's block

Block Table:
  Block 0 → 0x4000
  Block 1 → 0x2000
  Block 2 → 0x6000
```

#### Step 3: Dynamic Allocation
```
Generation process:

Token 1-16:   [Block A: ████████████████] Full
              Allocate Block B

Token 17-32:  [Block B: ████████████████] Full
              Allocate Block C

Token 33-40:  [Block C: ████████________] Filling...
```

### Key Benefits

#### 1. No Fragmentation
```
Fragmented memory:
[Used][Free 100][Used][Free 150][Used][Free 80]

Traditional: Can't allocate 200 contiguous ❌

Paged: Can allocate 13 blocks (208 slots) ✅
       Blocks scattered: [100 space] + [150 space] + [80 space]
```

#### 2. Memory Efficiency
```
Request generates 100 tokens:

Traditional:
  Allocate: 2048 slots
  Use: 100 slots
  Waste: 95%

Paged (block size 16):
  Allocate: 7 blocks = 112 slots
  Use: 100 slots
  Waste: 10%
```

#### 3. Prefix Sharing
```
System prompt: "You are a helpful assistant."
(Used by all requests)

Traditional:
  Request 1: [System prompt copy] + [User 1 query]
  Request 2: [System prompt copy] + [User 2 query]
  Request 3: [System prompt copy] + [User 3 query]
  
  Stores system prompt 3 times!

Paged:
  [System prompt blocks] ← Shared (read-only)
       ↓         ↓         ↓
  Request 1  Request 2  Request 3
  [User blocks] [User blocks] [User blocks]
  
  Stores system prompt once!
```

### Visual Example

```
2 Requests with shared prefix:

Request A: "Translate to French: Hello, how are you?"
Request B: "Translate to French: Hello, what's your name?"

Common prefix: "Translate to French: Hello,"

Memory layout:
┌─────────────────────────────────────────┐
│ Shared Blocks (Read-only)              │
│ Block 5: "Translate to French: Hello," │
└─────────────────────────────────────────┘
         ↓                    ↓
┌──────────────┐      ┌──────────────┐
│ Request A    │      │ Request B    │
│ Block 10:    │      │ Block 15:    │
│ "how are you"│      │ "what's your"│
└──────────────┘      └──────────────┘

Block tables:
  Request A: [Block 5, Block 10]
  Request B: [Block 5, Block 15]
              ↑
         Both reference same block!
```

### What Paged Attention Optimizes

**Summary:**
- ✅ **Memory allocation** - use fragmented space efficiently
- ✅ **Memory utilization** - 90% less waste
- ✅ **Prefix sharing** - deduplicate common content
- ❌ Does NOT speed up attention computation itself
- ❌ Does NOT reduce bandwidth usage

**Impact:** Enables larger batch sizes, more concurrent requests

---

## 2. Flash Attention

### The Problem: Memory Bandwidth Bottleneck

**GPU Memory Hierarchy:**
```
┌────────────────────────────────────────┐
│ GPU Chip                               │
│                                        │
│  ┌──────────────────┐                 │
│  │ Processing Units │ ← Compute       │
│  │ (SM Cores)       │                 │
│  └────────┬─────────┘                 │
│           │                            │
│           ↓ ~20 TB/s                   │
│  ┌──────────────────┐                 │
│  │   SRAM           │ ← Fast, Tiny    │
│  │ (On-chip cache)  │   (~20 MB)      │
│  │                  │                 │
│  └────────┬─────────┘                 │
│           │                            │
│           ↓ ~1.5 TB/s (BOTTLENECK!)   │
│  ┌──────────────────┐                 │
│  │   HBM            │ ← Slow, Large   │
│  │ (Main memory)    │   (~80 GB)      │
│  └──────────────────┘                 │
└────────────────────────────────────────┘

Problem: HBM ↔ SRAM transfer is 10-20x slower than computation!
```

### Traditional Attention Data Movement

```
Computing attention for 1024 tokens:

Step 1: Load Q from HBM → SRAM (4 MB)
        Transfer time: 4 MB / 1.5 TB/s = 2.7 μs

Step 2: Load K from HBM → SRAM (4 MB)
        Transfer time: 2.7 μs

Step 3: Compute scores = Q @ K^T in SRAM
        Compute time: 0.5 μs ⚡ (FAST!)

Step 4: Save scores to HBM (4 MB)
        Transfer time: 2.7 μs

Step 5: Load scores from HBM → SRAM (4 MB)
        Transfer time: 2.7 μs

Step 6: Load V from HBM → SRAM (4 MB)
        Transfer time: 2.7 μs

Step 7: Compute output = scores @ V
        Compute time: 0.5 μs ⚡ (FAST!)

Step 8: Save output to HBM (4 MB)
        Transfer time: 2.7 μs

Total time: ~19 μs
Breakdown: 90% memory transfer, 10% computation!
```

**The processor sits idle waiting for data!**

### Flash Attention Solution

**Key Idea:** Chunk computation into pieces that fit entirely in SRAM.

```
Instead of loading entire attention matrix:

Traditional:
  Full matrix: 1024 × 1024 = 1M elements = 4 MB
  Doesn't fit well in SRAM → constant HBM access

Flash Attention:
  Tiles: 128 × 128 = 16K elements = 64 KB per tile
  Fits perfectly in SRAM!
```

### How It Works

#### Tiled Computation

```
Attention matrix (1024 × 1024) divided into tiles:

┌─────┬─────┬─────┬─────┐
│ T00 │ T01 │ T02 │ T03 │  Each tile: 128×128
├─────┼─────┼─────┼─────┤
│ T10 │ T11 │ T12 │ T13 │  Process one at a time
├─────┼─────┼─────┼─────┤
│ T20 │ T21 │ T22 │ T23 │  Keep in SRAM
├─────┼─────┼─────┼─────┤
│ T30 │ T31 │ T32 │ T33 │
└─────┴─────┴─────┴─────┘

Process tile by tile, accumulating results
```

#### Step-by-Step Process

```
For each tile:

1. Load Q chunk (128 queries) → SRAM
2. Load K chunk (128 keys) → SRAM
3. Compute scores = Q @ K^T → KEEP IN SRAM
4. Apply softmax → KEEP IN SRAM
5. Load V chunk (128 values) → SRAM
6. Compute output → KEEP IN SRAM
7. Update running accumulator

Key: Never write intermediate scores to HBM!
```

### Data Movement Comparison

#### Traditional Attention (1024 tokens)

```
HBM Reads:
  Q: 4 MB
  K: 4 MB
  Scores from HBM: 4 MB (after writing)
  V: 4 MB

HBM Writes:
  Scores: 4 MB
  Output: 4 MB

Total HBM traffic: 24 MB
Time: 24 MB / 1.5 TB/s = 16 μs
```

#### Flash Attention (1024 tokens, 128×128 tiles)

```
For each tile (64 tiles total):
  Load Q chunk: 0.5 MB
  Load K chunk: 0.5 MB
  Load V chunk: 0.5 MB
  (No intermediate writes!)

Total HBM traffic per tile: 1.5 MB
Total for all tiles: 8 MB

Time: 8 MB / 1.5 TB/s = 5 μs

Speedup: 16 / 5 = 3.2x faster!
```

### Why It's Faster

```
Traditional:
  ████████████████████ (90% memory, 10% compute)
  
Flash Attention:
  ████████ (70% memory, 30% compute)

Same computation, but:
  - 3x less data moved through HBM ↔ SRAM
  - Intermediate results stay in fast SRAM
  - Better compute utilization
```

### Additional Benefits

1. **Longer Context Windows**
```
Traditional: Limited by memory for full attention matrix
Flash: Can handle much longer sequences (100K+ tokens)
```

2. **Lower Memory Footprint**
```
Traditional: O(n²) memory for attention matrix
Flash: O(n) memory (processes tiles incrementally)
```

3. **Hardware Efficiency**
```
Better cache utilization
Reduced memory bandwidth pressure
Higher GPU utilization
```

### Installation

```bash
pip install flash-attn --no-build-isolation
```

### Usage

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    attn_implementation="flash_attention_2",  # Enable Flash Attention
    torch_dtype=torch.float16,
    device_map="auto"
)
```

### What Flash Attention Optimizes

**Summary:**
- ✅ **Memory bandwidth** - reduces HBM ↔ SRAM transfers
- ✅ **Computation speed** - 3-4x faster attention
- ✅ **Memory usage** - O(n) instead of O(n²)
- ✅ **Longer contexts** - enables 100K+ token sequences
- ❌ Does NOT affect memory allocation
- ❌ Does NOT change model quality

**Impact:** Faster generation, longer contexts, better GPU utilization

---

## Paged Attention vs Flash Attention

### Quick Comparison

| Aspect | Paged Attention | Flash Attention |
|--------|----------------|-----------------|
| **Problem Solved** | Memory fragmentation | Bandwidth bottleneck |
| **What It Chunks** | KV cache storage | Attention computation |
| **Where Chunks Go** | Anywhere in HBM (scattered) | Processed in SRAM |
| **Memory Level** | HBM only | HBM ↔ SRAM transfers |
| **Speed Impact** | Indirect (enables larger batches) | Direct (3-4x faster) |
| **Memory Impact** | Direct (90% less waste) | Indirect (lower peak usage) |
| **Use Case** | Production serving | Any attention computation |

### Visual Distinction

```
Paged Attention:
┌────────────────────────────────────┐
│         HBM Memory                 │
│                                    │
│ [Block 5][Data][Block 2][Block 8] │ ← Storage layout
│                                    │
│ Optimization: Where to STORE cache │
└────────────────────────────────────┘

Flash Attention:
┌────────────────────────────────────┐
│ HBM Memory                         │
│ [Full attention data]              │
│         ↓ minimize transfers       │
│ SRAM (compute tiles here)          │
│         ↓                          │
│ Processing Units                   │
│                                    │
│ Optimization: How to COMPUTE       │
└────────────────────────────────────┘
```

### Can You Use Both?

**YES!** They're complementary.

```
Modern Inference System:

1. Paged Attention: Stores KV cache efficiently
   - Blocks scattered in HBM
   - No fragmentation
   - Shared prefixes

2. Flash Attention: Computes attention efficiently
   - Tiles processed in SRAM
   - Less bandwidth usage
   - Faster computation

Result: Best memory AND compute efficiency!
```

---

## Real-World Impact

### Example: LLaMA-7B with 2048 Context

**Baseline (No Optimizations):**
- Memory: 14 GB (model) + 8 GB (attention) = 22 GB
- Speed: 20 tokens/second
- Batch size: 4

**With Paged Attention:**
- Memory: 14 GB (model) + 1 GB (efficient cache) = 15 GB
- Speed: 20 tokens/second (same)
- Batch size: 16 (4x larger!)

**With Flash Attention:**
- Memory: 14 GB (model) + 8 GB (attention) = 22 GB
- Speed: 60 tokens/second (3x faster!)
- Batch size: 4 (same)

**With Both:**
- Memory: 14 GB (model) + 1 GB (efficient cache) = 15 GB
- Speed: 60 tokens/second (3x faster!)
- Batch size: 16 (4x larger!)

**Combined: 12x more throughput!**

---

## Summary

### Paged Attention
- **Solves:** Memory fragmentation and waste
- **How:** Block-based storage in HBM
- **Benefit:** Larger batches, better memory utilization
- **Use:** Production serving with many concurrent requests

### Flash Attention
- **Solves:** Memory bandwidth bottleneck
- **How:** Tiled computation in SRAM
- **Benefit:** 3-4x faster attention, longer contexts
- **Use:** Any attention computation (training or inference)

### Together
- **Paged:** Efficient storage
- **Flash:** Efficient computation
- **Result:** Maximum throughput and efficiency