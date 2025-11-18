
# DPO Loss Formula Breakdown

**The formula:**
```
L_DPO(πθ; πref) = -log σ(β log[πθ(yw|x)/πref(yw|x)] - β log[πθ(yl|x)/πref(yl|x)])
```

**Step-by-step with our example:**

### Given:
- Prompt x: "What is 2+2?"
- Chosen answer yw: "4"
- Rejected answer yl: "3"
- β = 1.0 (temperature parameter)

### Step 1: Get log probabilities from both models

**Reference model πref (fixed):**
- log πref(yw|x) = -1.609
- log πref(yl|x) = -2.303

**Policy model πθ (trainable):**
- log πθ(yw|x) = -2.996
- log πθ(yl|x) = -1.897

### Step 2: Compute the log ratios

**For chosen answer:**
```
β log[πθ(yw|x)/πref(yw|x)] = 1.0 × [log πθ(yw|x) - log πref(yw|x)]
                             = 1.0 × [-2.996 - (-1.609)]
                             = 1.0 × [-1.387]
                             = -1.387
```

**For rejected answer:**
```
β log[πθ(yl|x)/πref(yl|x)] = 1.0 × [log πθ(yl|x) - log πref(yl|x)]
                             = 1.0 × [-1.897 - (-2.303)]
                             = 1.0 × [+0.406]
                             = +0.406
```

### Step 3: Compute the difference Δ

```
Δ = β log[πθ(yw|x)/πref(yw|x)] - β log[πθ(yl|x)/πref(yl|x)]
  = -1.387 - 0.406
  = -1.793
```

**Interpretation:** Δ is negative, meaning our policy πθ currently prefers the wrong answer (yl) relative to the reference model!

### Step 4: Apply sigmoid function

```
σ(Δ) = 1 / (1 + exp(-Δ))
     = 1 / (1 + exp(-(-1.793)))
     = 1 / (1 + exp(1.793))
     = 1 / (1 + 6.00)
     ≈ 0.143
```

**Interpretation:** Only 14.3% probability that chosen answer is preferred - bad!

### Step 5: Compute the loss

```
L_DPO = -log σ(Δ)
      = -log(0.143)
      ≈ 1.946
```

**This is the loss we use for gradient descent!**

### Step 6: What happens during training?

The gradient ∂L/∂θ will:
- **Increase** log πθ(yw|x) (make "4" more likely)
- **Decrease** log πθ(yl|x) (make "3" less likely)

### After one gradient update:

**New policy model:**
- log πθ(yw|x) = -1.715 (improved from -2.996!)
- log πθ(yl|x) = -2.526 (worsened from -1.897!)

**New loss calculation:**
- New log ratio for yw: -1.715 - (-1.609) = -0.106
- New log ratio for yl: -2.526 - (-2.303) = -0.223
- New Δ = -0.106 - (-0.223) = +0.118 (now positive!)
- New σ(Δ) ≈ 0.529 (now 52.9% - better than 14.3%!)
- **New loss = 0.635** (down from 1.946!)

---

**Key insight:** The entire formula is computed in one go during the loss calculation. You don't compute advantages separately and then loss - it's all part of evaluating that single DPO loss equation. The "advantages" A(y) = log πθ(y|x) - log πref(y|x) are just a convenient way to understand what's inside the formula, but they're computed as part of the log ratio terms.