# 🔀 Model Parallelism

When models are too large to fit on a single GPU, we split them across multiple GPUs using parallelism strategies.

---

## Overview of Parallelism Types

| Type | What It Splits | Best For | GPU Utilization | Communication |
|------|---------------|----------|-----------------|---------------|
| **Data Parallel** | Batches | Training | Good | Low frequency |
| **Pipeline Parallel** | Layers | Inference (deep models) | Poor (bubbles) | Medium |
| **Tensor Parallel** | Weights | Inference (wide layers) | Excellent | High frequency |

---

## 1. Data Parallelism (DP)

### Concept
Replicate the entire model on each GPU. Each GPU processes different data batches.

### How It Works

```
Model: 12-layer transformer
GPUs: 3
Batch size: 96

GPU 0: [Full Model Copy] → Process batch 1-32
GPU 1: [Full Model Copy] → Process batch 33-64
GPU 2: [Full Model Copy] → Process batch 65-96

All GPUs compute independently
```

### Training Steps

```
Step 1: Forward Pass (Parallel)
  GPU 0: batch 1-32 → loss_0, gradients_0
  GPU 1: batch 33-64 → loss_1, gradients_1
  GPU 2: batch 65-96 → loss_2, gradients_2

Step 2: All-Reduce Gradients
  Average gradients across all GPUs:
  gradient_avg = (gradients_0 + gradients_1 + gradients_2) / 3

Step 3: Update Weights (Synchronized)
  All GPUs: weights = weights - lr × gradient_avg
  
Result: All GPUs have identical weights ✓
```

### Visual Representation

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   GPU 0      │  │   GPU 1      │  │   GPU 2      │
│              │  │              │  │              │
│ Full Model   │  │ Full Model   │  │ Full Model   │
│ (copy)       │  │ (copy)       │  │ (copy)       │
│              │  │              │  │              │
│ Batch 1-32   │  │ Batch 33-64  │  │ Batch 65-96  │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └─────────────────┴─────────────────┘
                         │
                   All-Reduce
                  (Average Gradients)
                         │
       ┌─────────────────┴─────────────────┐
       │                 │                 │
       ↓                 ↓                 ↓
  Update GPU 0      Update GPU 1      Update GPU 2
```

### Benefits
- ✅ Simple to implement
- ✅ Linear scaling with number of GPUs (ideally)
- ✅ No idle GPUs (all compute simultaneously)
- ✅ Good for large batch training

### Limitations
- ❌ Model must fit on single GPU
- ❌ Gradient synchronization overhead
- ❌ Memory duplication (each GPU stores full model)
- ❌ Not efficient for inference

### Use Cases
- **Training** large batch size models
- When batch size >> model size
- Distributed training across many nodes

---

## 2. Pipeline Parallelism (PP)

### Concept
Split model into sequential stages (layer groups). Each GPU handles different layers.

### How It Works

```
Model: 12 layers
GPUs: 3

GPU 0: Layers 1-4
GPU 1: Layers 5-8
GPU 2: Layers 9-12

Data flows sequentially:
Input → GPU 0 → GPU 1 → GPU 2 → Output
```

### Forward Pass Flow

```
Batch 1:

Time 0-10ms:  GPU 0 [████] Layers 1-4
              GPU 1 [    ] Idle
              GPU 2 [    ] Idle

Time 10-20ms: GPU 0 [    ] Idle
              GPU 1 [████] Layers 5-8
              GPU 2 [    ] Idle

Time 20-30ms: GPU 0 [    ] Idle
              GPU 1 [    ] Idle
              GPU 2 [████] Layers 9-12

Result: 30ms total, but 66% GPU idle time! (GPU Bubble)
```

### The GPU Bubble Problem

```
Sequential Processing:
┌──────────────────────────────────────┐
│ GPU 0: [████]____                    │
│ GPU 1:       ____[████]              │ ← Idle waiting
│ GPU 2:             ____[████]        │ ← Idle waiting
└──────────────────────────────────────┘

Wasted compute time = GPU bubbles
```

### Solution: Micro-Batching

Split each batch into smaller micro-batches and pipeline them:

```
3 micro-batches:

GPU 0: [MB1][MB2][MB3]____
GPU 1:     [MB1][MB2][MB3]____
GPU 2:         [MB1][MB2][MB3]

Timeline:
Time 0-3:  GPU 0 processes MB1
Time 3-6:  GPU 0 processes MB2, GPU 1 processes MB1
Time 6-9:  GPU 0 processes MB3, GPU 1 processes MB2, GPU 2 processes MB1
Time 9-12: GPU 0 idle, GPU 1 processes MB3, GPU 2 processes MB2
Time 12-15: GPU 0 idle, GPU 1 idle, GPU 2 processes MB3

Better utilization, but still has bubbles at start/end
```

### Visual with Micro-Batching

```
Without Micro-Batching:
GPU 0: [████████]________
GPU 1:         [████████]________
GPU 2:                 [████████]
Utilization: 33%

With Micro-Batching (4 micro-batches):
GPU 0: [██][██][██][██]____
GPU 1:    [██][██][██][██]____
GPU 2:       [██][██][██][██]
Utilization: ~66% (better!)
```

### Benefits
- ✅ Can handle very deep models
- ✅ Each GPU only stores part of model (lower memory per GPU)
- ✅ Good for inference (forward pass only)

### Limitations
- ❌ GPU bubbles (idle time)
- ❌ Sequential dependencies
- ❌ Requires careful micro-batch tuning
- ❌ Communication overhead between stages

### Use Cases
- **Inference** for extremely deep models
- When model layers >> GPU memory
- Models with clear layer boundaries

---

## 3. Tensor Parallelism (TP)

### Concept
Split weight matrices **within each layer** across GPUs. All GPUs work on the same layer simultaneously.

### How It Works

```
Single layer (e.g., MLP):
  Weight matrix: 4096 × 4096

Split across 2 GPUs:
  GPU 0: Rows 0-2047 (top half)
  GPU 1: Rows 2048-4095 (bottom half)

Both process same input at same time!
```

### Detailed Example

```
Layer weight matrix (4×4):
        ┌──────────────┐
GPU 0 → │ 1   2   3   4│ ← Row 0
GPU 0 → │ 5   6   7   8│ ← Row 1
        ├──────────────┤ ← Tensor Split
GPU 1 → │ 9  10  11  12│ ← Row 2
GPU 1 → │13  14  15  16│ ← Row 3
        └──────────────┘

Input: x = [1, 2, 3, 4] (same to both GPUs)
```

### Step-by-Step Process

```
Step 1: Broadcast Input
  GPU 0 receives: [1, 2, 3, 4]
  GPU 1 receives: [1, 2, 3, 4]
  (Same input to both!)

Step 2: Local Computation (Parallel!)
  GPU 0 computes:
    Row 0: 1×1 + 2×2 + 3×3 + 4×4 = 30
    Row 1: 5×1 + 6×2 + 7×3 + 8×4 = 70
    → Partial output: [30, 70]
  
  GPU 1 computes:
    Row 2: 9×1 + 10×2 + 11×3 + 12×4 = 110
    Row 3: 13×1 + 14×2 + 15×3 + 16×4 = 150
    → Partial output: [110, 150]

Step 3: All-Reduce (Gather/Concatenate)
  Combine results:
  GPU 0 gets: [30, 70, 110, 150]
  GPU 1 gets: [30, 70, 110, 150]
  
  ✓ Both have complete output!
```

### Timeline Comparison

```
Pipeline Parallelism (Sequential):
GPU 0: [Layer 1-4 ████]____
GPU 1:                [Layer 5-8 ████]____

Tensor Parallelism (Parallel):
GPU 0: [Layer 1 ██][Layer 2 ██][Layer 3 ██]
GPU 1: [Layer 1 ██][Layer 2 ██][Layer 3 ██]
       ↑ Same layer, different weights!
       
No bubbles - all GPUs busy!
```

### Communication Pattern

```
For EACH layer:

1. Broadcast input (small, fast)
   GPU 0 ← input
   GPU 1 ← input

2. Compute locally (parallel, no communication)
   GPU 0: [computing...]
   GPU 1: [computing...]

3. All-reduce output (gather results)
   GPU 0 ↔ GPU 1 (exchange partial outputs)
   
Communication happens EVERY layer!
```

### Benefits
- ✅ No GPU bubbles (perfect parallelism)
- ✅ All GPUs always working
- ✅ Handles huge layers (split giant matrices)
- ✅ Better GPU utilization than pipeline

### Limitations
- ❌ High communication frequency (every layer)
- ❌ Requires fast GPU interconnect (NVLink, NVSwitch)
- ❌ Can't split all layers (LayerNorm, Dropout need special handling)
- ❌ Communication overhead can dominate for small models

### Sequence Parallelism

Some layers can't be split by weights (LayerNorm, Dropout):

```
Solution: Split along sequence dimension instead

LayerNorm on 2048 tokens:
  GPU 0: Process tokens 0-1023
  GPU 1: Process tokens 1024-2047
  
Then gather results
```

### Use Cases
- **Inference** for models with very wide layers
- Models with huge hidden dimensions
- When GPUs have fast interconnect (same node)

---

## 4. Combining All Three (3D Parallelism)

### The Power of Combination

```
Each parallelism solves different bottlenecks:
- Data Parallel: Scale batch size
- Pipeline Parallel: Scale model depth
- Tensor Parallel: Scale layer width

Combined: Scale everything!
```

### Configuration Example

```
Setup: 8 GPUs, 12-layer model

Data Parallel: 2 replicas
Pipeline Parallel: 2 stages
Tensor Parallel: 2-way split

Math: 2 (DP) × 2 (PP) × 2 (TP) = 8 GPUs ✓
```

### GPU Layout

```
                Replica 0              Replica 1
               (Batch 1-16)          (Batch 17-32)
               
Pipeline   ┌──────────┬──────────┐ ┌──────────┬──────────┐
Stage 0    │  GPU 0   │  GPU 1   │ │  GPU 4   │  GPU 5   │
(Layers    │  Tensor  │  Tensor  │ │  Tensor  │  Tensor  │
1-6)       │  Part A  │  Part B  │ │  Part A  │  Part B  │
           └──────────┴──────────┘ └──────────┴──────────┘
                      │                        │
           Pipeline Pass (activations)  Pipeline Pass
                      ↓                        ↓
Pipeline   ┌──────────┬──────────┐ ┌──────────┬──────────┐
Stage 1    │  GPU 2   │  GPU 3   │ │  GPU 6   │  GPU 7   │
(Layers    │  Tensor  │  Tensor  │ │  Tensor  │  Tensor  │
7-12)      │  Part A  │  Part B  │ │  Part A  │  Part B  │
           └──────────┴──────────┘ └──────────┴──────────┘
```

### Execution Flow

```
Batch input (32 samples):

Split by Data Parallel:
  Replica 0: samples 1-16
  Replica 1: samples 17-32

Each replica processes independently:

Replica 0 (GPU 0-3):
  Stage 0 (Layers 1-6):
    GPU 0 + GPU 1 (Tensor Parallel):
      - Both receive same input
      - GPU 0: Left half of weights
      - GPU 1: Right half of weights
      - All-reduce: Combine results
    
    Output → Pass to Stage 1
  
  Stage 1 (Layers 7-12):
    GPU 2 + GPU 3 (Tensor Parallel):
      - Both receive input from Stage 0
      - GPU 2: Left half of weights
      - GPU 3: Right half of weights
      - All-reduce: Combine results
    
    Final output for Replica 0

Replica 1 (GPU 4-7):
  (Same process with GPUs 4-7)
```

### Communication Patterns

```
1. Tensor Parallel (Highest Frequency):
   Within each layer:
   GPU 0 ↔ GPU 1  (all-reduce)
   GPU 2 ↔ GPU 3  (all-reduce)
   GPU 4 ↔ GPU 5  (all-reduce)
   GPU 6 ↔ GPU 7  (all-reduce)
   
   Frequency: Every layer (12 times per forward pass)
   Data size: Layer output (small-medium)
   Requirement: Fast interconnect (NVLink)

2. Pipeline Parallel (Medium Frequency):
   Between stages:
   GPU 0,1 → GPU 2,3  (pass activations)
   GPU 4,5 → GPU 6,7  (pass activations)
   
   Frequency: Once per stage (2 times per forward pass)
   Data size: Stage output (medium)
   Requirement: Medium bandwidth

3. Data Parallel (Lowest Frequency):
   Between replicas (after backward pass):
   GPU 0,1,2,3 ↔ GPU 4,5,6,7  (gradient sync)
   
   Frequency: Once per training step
   Data size: All gradients (large)
   Requirement: Can use slower interconnect
```

### Memory Distribution

```
Each GPU stores:

GPU 0:
  - 1/2 of layers 1-6 weights (Tensor split)
  - Activations for 1/2 of batch (Data split)
  - Optimizer states for its weights
  
  Total: ~12.5% of full model
  
All GPUs combined: 100% of model, can process 2x batch size
```

### Real-World Example: GPT-3 Scale

```
Model: 175B parameters, 96 layers
GPUs: 64 (8 nodes × 8 GPUs)

Configuration:
- Data Parallel: 2 replicas (2 separate batches)
- Pipeline Parallel: 8 stages (12 layers per stage)
- Tensor Parallel: 4-way (split each layer 4 ways)

Math: 2 (DP) × 8 (PP) × 4 (TP) = 64 GPUs

Per GPU:
  Layers: 12 (1/8 of model)
  Weights: 1/4 of those 12 layers (TP)
  Batch: 1/2 of total batch (DP)
  
  Memory per GPU: ~550GB / 64 ≈ 8.6 GB
  Fits on A100 40GB! ✓
```

### Choosing Dimensions

**Rules of Thumb:**

1. **Tensor Parallelism (TP):**
   - Limited by communication
   - Use 2-8 GPUs within same node
   - Requires fast interconnect (NVLink)

2. **Pipeline Parallelism (PP):**
   - Limited by bubble overhead
   - Use 2-8 stages
   - More stages = more bubbles

3. **Data Parallelism (DP):**
   - Almost unlimited
   - Use remaining GPUs
   - Scales across nodes easily

**Example Configurations:**

```
16 GPUs (2 nodes):
  TP = 8 (within node, need NVLink)
  PP = 2 (avoid too many stages)
  DP = 1 (no data replication)
  
64 GPUs (8 nodes):
  TP = 4 (conservative for cross-node)
  PP = 4 (manageable bubbles)
  DP = 4 (4 replicas)
  
256 GPUs (32 nodes):
  TP = 8 (within node)
  PP = 8 (more stages acceptable)
  DP = 4 (4 replicas)
```

---

## Summary Comparison

### Quick Reference

| Parallelism | Batch | Layers | Weights | Timing | Bubbles | Communication |
|-------------|-------|--------|---------|--------|---------|---------------|
| **Data** | ✅ Split | ❌ Replicate | ❌ Replicate | Parallel | No | Low (gradients) |
| **Pipeline** | ❌ Same | ✅ Split | ❌ All in stage | Sequential | Yes | Medium (activations) |
| **Tensor** | ❌ Same | ❌ Same | ✅ Split | Parallel | No | High (all-reduce) |

### When to Use Each

```
Use Data Parallel when:
  ✅ Training (need gradient averaging)
  ✅ Model fits on single GPU
  ✅ Want to scale batch size

Use Pipeline Parallel when:
  ✅ Model too deep for single GPU
  ✅ Clear layer boundaries
  ✅ Can tolerate some inefficiency

Use Tensor Parallel when:
  ✅ Layers too large for single GPU
  ✅ Have fast GPU interconnect
  ✅ Need maximum efficiency

Use 3D Parallelism when:
  ✅ Training massive models (100B+ parameters)
  ✅ Have many GPUs available
  ✅ Need to scale everything (batch, depth, width)
```

---

## Implementation Notes

### PyTorch FSDP (Fully Sharded Data Parallel)
```python
from torch.distributed.fsdp import FullyShardedDataParallel

model = FullyShardedDataParallel(
    model,
    # Automatic sharding and gradient reduction
)
```

### DeepSpeed Pipeline
```python
from deepspeed import PipelineModule

model = PipelineModule(
    layers=model.layers,
    num_stages=4,  # 4 pipeline stages
)
```

### Megatron Tensor Parallel
```python
# Model parallelism across GPUs
tensor_model_parallel_size = 4
pipeline_model_parallel_size = 2
```

---

## Real-World Impact

### Example: Training 70B Model

**Without Parallelism:**
- Impossible on single GPU (needs ~140 GB)

**With Data Parallel Only:**
- Still impossible (model doesn't fit)

**With 3D Parallelism (16 GPUs):**
- TP = 4 (split layers)
- PP = 2 (split depth)
- DP = 2 (2 batches)
- Per GPU: ~9 GB model + activations
- Fits on A100 40GB! ✓
- Training speed: ~100 TFLOPs/GPU

**Achieved: Training massive models efficiently across many GPUs!**