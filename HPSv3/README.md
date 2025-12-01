# HPSv3: Human Preference Score v3

**Paper:** [arxiv.org/pdf/2508.03789](https://arxiv.org/pdf/2508.03789)  
**Date:** December 1, 2025

---

## 🎯 TLDR

HPSv3 is a **trained rating system** that evaluates text-to-image generation models the way humans do. It's built on:
- **HPDv3 Dataset:** 1.17M human preference comparisons
- **Better Architecture:** QWen2-VL encoder + uncertainty-aware ranking
- **CoHP:** Chain of Human Preference for iterative quality improvement

**Key Innovation:** Models uncertainty in human preferences using Gaussian distributions (μ, σ) instead of fixed scores, preventing overfitting to noisy labels.

---

## 📋 What's the Problem?

Current evaluation metrics (HPS, ImageReward, PickScore, MPS) have limitations:

| Problem | Impact |
|---------|--------|
| **Limited Data Distribution** | Poor generalization to new styles/domains |
| **CLIP/BLIP Limitations** | Can't distinguish quality - sees "a dog" and "poorly drawn dog" as similar |
| **KL Divergence Issues** | Overfits to noisy labels, doesn't handle annotation disagreement |

---

## ✅ HPSv3 Solution

### 1. **HPDv3 Dataset**

**Three Data Sources:**
- **HPDv2 + 10 New Models:** Diverse model outputs (SDXL, Playground, etc.)
- **Real Photos + VLM Captions:** Real-world quality standards
- **Midjourney Data:** 4 images/prompt with user-selected preferences

**Quality Control:**
- 9-19 annotators per sample
- 90% agreement threshold
- Annotators must pass expert validation test (16/20 correct)

**Categories:** 12 prompt categories (Animals, Architecture, Arts, Characters, Design, Food, Natural scenery, Plants, Products, Transportation, Science, Others)

### 2. **Model Architecture**

```
(Image, Prompt) → QWen2-VL Encoder → MLP → (μ, σ)
```

- **QWen2-VL:** Powerful vision-language model (better than CLIP/BLIP)
- **MLP:** Outputs mean (μ) and uncertainty (σ) instead of single score

### 3. **Uncertainty-Aware Ranking**

**Traditional Approach:**
```
r1 = 6.0, r2 = 5.0
P(x1 > x2) = sigmoid(6.0 - 5.0) = 0.73
→ Always equally confident
```

**Uncertainty-Aware:**
```
μ1=6.0, σ1=2.0 | μ2=5.0, σ2=2.0
r1 ~ N(μ1, σ1), r2 ~ N(μ2, σ2)
P(x1 > x2) = ∫∫ sigmoid(r1-r2) × N(r1|μ1,σ1) × N(r2|μ2,σ2) dr1 dr2
→ Confidence adapts to annotation agreement
```

**Why This Works:**
- Large σ = humans disagree = low confidence
- Small σ = humans agree = high confidence
- Prevents overfitting to noisy labels

### 4. **Chain of Human Preference (CoHP)**

Inspired by Chain-of-Thought, CoHP has two stages:

**Stage 1 - Model-wise Preference:**
1. Each model generates N images for prompt
2. HPSv3 scores all images
3. Select model with highest average score as "golden model"

**Stage 2 - Sample-wise Preference:**
1. Golden model generates B images
2. HPSv3 picks winner
3. **Add noise to winner** → use as input for next round
4. Repeat S times
5. Select best image across all rounds

**Why Add Noise?** Progressive refinement - keeps good features while exploring variations

---

## 📊 Key Results

| Metric | HPDv2 | HPDv3 |
|--------|-------|-------|
| Average Convergence | 59.9% | 76.5% |
| Total Comparisons | - | 1.17M |
| Annotators per Sample | - | 9-19 |
| Agreement Threshold | - | 90% |

---

## 💡 Key Takeaways

1. **HPSv3 is a trained rating system** that predicts human preferences
2. **Uncertainty modeling** handles annotation disagreement naturally
3. **Better encoder** (QWen2-VL) captures quality beyond semantic matching
4. **CoHP** enables iterative improvement without retraining
5. **High-quality dataset** with real photos + diverse models

---

## 🔗 Resources

- **Paper:** [arxiv.org/pdf/2508.03789](https://arxiv.org/pdf/2508.03789)
- **Full Notes:** [hpsv3_notes.html](https://claude.ai/public/artifacts/2dcd341e-d3d0-4e5d-a725-95760011faf5)
- **Flowchart:** [cohp_flowchart.svg](cohp_flowchart.svg)

---

## 📝 Notes Structure

```
HPSv3 Paper
├── Problem: Current metrics can't capture human preferences
├── Solution: Better data + architecture + uncertainty modeling
├── Dataset: HPDv3 (1.17M comparisons, 90% agreement)
├── Model: QWen2-VL + Uncertainty-Aware Ranking
└── CoHP: Iterative refinement process
```

---

**Simplifiy_AI** | Making complex AI research accessible
