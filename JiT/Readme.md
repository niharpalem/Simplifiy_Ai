# JiT: Just image Transformers

> **Paper Notes**: "Back to Basics: Let Denoising Generative Models Denoise"  
> Tianhong Li, Kaiming He (MIT) • arXiv:2511.13720 • November 2025

---

## 📌 One-Line Summary

**Predict the clean image directly (x-prediction) instead of noise — it works because real images lie on a low-dimensional manifold.**

---

## 🎯 Key Insight

| Prediction Target | Lives Where? | Network Requirement | High-Dim Patches |
|-------------------|--------------|---------------------|------------------|
| **x** (clean image) | ON manifold | Limited capacity OK | ✅ Works |
| **ε** (noise) | OFF manifold | Needs full capacity | ❌ Fails |
| **v** (velocity) | OFF manifold | Needs full capacity | ❌ Fails |

---

## 🔄 How JiT Works

### Training
```
1. Take real image x
2. Add noise: z_t = t·x + (1-t)·ε
3. Network predicts clean image: x_pred = ViT(z_t, t)
4. Convert to velocity: v_pred = (x_pred - z_t) / (1-t)
5. Loss = ‖v_true - v_pred‖²
```

### Sampling
```
1. Start with random noise z, t=0
2. Predict clean image: x_pred = ViT(z, t)
3. Compute velocity: v = (x_pred - z) / (1-t)
4. Take step: z = z + Δt · v
5. Repeat until t=1 → Generated image!
```

---

## 🏗️ Architecture

- **Plain ViT** on pixel patches (no special modifications)
- **No VAE** — works directly in pixel space
- **No pre-training** — trained from scratch
- **No extra losses** — just diffusion loss
- Uses **adaLN-Zero** for timestep/class conditioning

---

## 📊 Results

| Model | ImageNet 256×256 | ImageNet 512×512 | Params |
|-------|------------------|------------------|--------|
| JiT-B | 3.66 FID | 4.02 FID | 131M |
| JiT-L | 2.36 FID | 2.53 FID | 459M |
| JiT-H | 1.86 FID | 1.94 FID | 953M |
| JiT-G | 1.82 FID | 1.78 FID | 2B |

---

## 🆚 JiT vs Latent Diffusion

| Aspect | Stable Diffusion | JiT |
|--------|------------------|-----|
| Space | Latent (64×64×4) | Pixel (512×512×3) |
| Needs VAE? | Yes | No |
| Self-contained? | No | Yes |
| Prediction | ε or v | x |

---

## 📝 Detailed Notes

👉 **[View Complete HTML Notes](https://claude.ai/public/artifacts/d7251259-9242-4afa-85aa-183c2ba4283f)**

---

## 📚 References

- **Paper**: [arXiv:2511.13720](https://arxiv.org/abs/2511.13720)
- **Authors**: Tianhong Li, Kaiming He
- **Institution**: MIT

---

## 🧠 Core Concepts Covered

- [x] Manifold Hypothesis
- [x] x-prediction vs ε-prediction vs v-prediction
- [x] Loss space vs Output space decoupling
- [x] Flow-based formulation (ODE)
- [x] Pixel-space vs Latent-space diffusion
- [x] High-dimensional diffusion challenges
- [x] Training algorithm walkthrough
- [x] Sampling algorithm walkthrough

---

<p align="center">
  <i>The key insight: Let the network predict what it's good at — structured, on-manifold data.</i>
</p>