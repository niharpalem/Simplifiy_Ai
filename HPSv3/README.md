# HPSv3: Human Preference Score v3

**Paper:** [arxiv.org/pdf/2508.03789](https://arxiv.org/pdf/2508.03789)  
**Date:** December 1, 2025

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'fontSize':'13px', 'fontFamily':'arial'}, 'flowchart':{'nodeSpacing': 50, 'rankSpacing': 100, 'padding': 20, 'useMaxWidth':true}}}%%
flowchart TD
    Start["<b>HPSv3: Complete Pipeline</b><br/><br/>From Data Curation to Deployment"]
    
    %% ==========================================
    %% PHASE 1: DATA CURATION
    %% ==========================================
    
    Start --> P1["<b>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</b><br/><b>PHASE 1: DATA CURATION</b><br/><b>HPDv3 Dataset</b><br/><b>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</b>"]
    
    P1 --> S1["<b>📊 Source 1: HPDv2 + 10 Models</b><br/><br/>• Expand HPDv2 with new prompts<br/>• 10 SOTA models generate images:<br/>   SDXL, Flux, SD3, Playground...<br/>• Diverse model capabilities<br/>• Different failure modes<br/>• Coverage of generation approaches<br/><br/><b>✅ Variety in outputs</b>"]
    
    P1 --> S2["<b>📸 Source 2: Real Photos + VLM</b><br/><br/>• Collect high-quality real photos<br/>• 12 categories organized:<br/>   Animals, Architecture, Arts...<br/>• VLM generates captions<br/>• Gen AI creates synthetic versions<br/>• Compare: Real vs AI-generated<br/><br/><b>✅ Real-world standards</b>"]
    
    P1 --> S3["<b>🎨 Source 3: Midjourney</b><br/><br/>• User-written prompts<br/>• 4 images generated per prompt<br/>• User selects favorite image<br/>• 331,955 user-generated images<br/>• Collected from Discord<br/>• Natural preference signals<br/><br/><b>✅ Real user choices</b>"]
    
    %% ==========================================
    %% PHASE 2: FILTERING & CURATION
    %% ==========================================
    
    S1 --> P2["<b>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</b><br/><b>PHASE 2: FILTERING & CURATION</b><br/><b>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</b>"]
    S2 --> P2
    S3 --> P2
    
    P2 --> F1["<b>① Prompt Categorization</b><br/><br/>• Organize into 12 distinct categories:<br/>   Animals, Architecture, Arts,<br/>   Characters, Design, Food,<br/>   Natural scenery, Plants, Products,<br/>   Transportation, Science, Others"]
    
    P2 --> F2["<b>② Distribution Alignment</b><br/><br/>• Match JourneyDB distribution<br/>• Each category reflects<br/>   real-world usage<br/>• Balanced representation<br/>   across categories<br/><br/><b>✅ Authentic usage patterns</b>"]
    
    F1 --> F3["<b>③ Aesthetic Filtering</b><br/><br/>• Aesthetic Predictor<br/>   scores all images<br/>• Select top 10% per category<br/>• Prevents category bias<br/><br/><b>✅ 57,759 high-quality images</b>"]
    
    F2 --> F4["<b>④ Captioning & Generation</b><br/><br/>• VLM generates<br/>   detailed descriptions<br/>• Multiple generative models<br/>   create images<br/>• Form pairwise comparisons<br/><br/><b>✅ Ready for annotation</b>"]
    
    %% ==========================================
    %% PHASE 3: HUMAN ANNOTATION
    %% ==========================================
    
    F3 --> P3["<b>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</b><br/><b>PHASE 3: HUMAN ANNOTATION</b><br/><b>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</b>"]
    F4 --> P3
    
    P3 --> A1["<b>👥 Annotator Qualification</b><br/><br/>1. Create validation set with<br/>    600 image pairs<br/>2. Expert annotation by<br/>    20 professional artists<br/>3. Achieve 80% convergence rate<br/>4. New annotators must score<br/>    ≥16/20 to qualify<br/>5. Ongoing quality control<br/><br/><b>✅ Ensures annotator proficiency</b>"]
    
    P3 --> A2["<b>📝 Annotation Process</b><br/><br/>• Each pair rated by<br/>   9-19 annotators<br/>• Select preferred image<br/>   from each pair<br/>• Rating criteria:<br/>   Aesthetics, Semantic similarity,<br/>   Overall coherence<br/>• <b>90% agreement threshold</b><br/>   required<br/><br/><b>✅ 1.17M high-confidence<br/>comparisons</b>"]
    
    A1 --> Dataset["<b>High-Quality Annotated Dataset</b><br/><br/>Ready for Model Training"]
    A2 --> Dataset
    
    %% ==========================================
    %% PHASE 4: MODEL TRAINING
    %% ==========================================
    
    Dataset ==> P4["<b>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</b><br/><b>PHASE 4: HPSv3 MODEL TRAINING</b><br/><b>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</b>"]
    
    P4 --> M1["<b>🏗️ Model Architecture</b><br/><br/><b>Input:</b><br/>(Image, Text Prompt)<br/><br/>⬇<br/><br/><b>QWen2-VL Encoder</b><br/>Powerful vision-language model<br/><br/>⬇<br/><br/><b>MLP →</b><br/>μ (mean), σ (uncertainty)"]
    
    P4 --> M2["<b>🎯 Uncertainty-Aware<br/>Ranking Loss</b><br/><br/><b>Traditional Approach:</b><br/>• r ~ single fixed score<br/>• Always equally confident<br/>   (problematic)<br/><br/><b>Uncertainty-Aware:</b><br/>• r ~ N(μ, σ) Gaussian distribution<br/>• Large σ = humans disagree<br/>   (low confidence)<br/>• Small σ = humans agree<br/>   (high confidence)"]
    
    M1 --> Trained["<b>✅ Trained HPSv3 Model</b><br/><br/>Predicts human preference scores<br/>with uncertainty estimation<br/><br/>Ready for deployment and<br/>iterative refinement (CoHP)"]
    M2 --> Trained
    
    %% ==========================================
    %% PHASE 5: CoHP
    %% ==========================================
    
    Trained ==> P5["<b>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</b><br/><b>PHASE 5: CHAIN OF HUMAN PREFERENCE</b><br/><b>(CoHP)</b><br/><b>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</b>"]
    
    P5 --> C1["<b>⭐ Stage 1: Model-wise Preference</b><br/><br/><b>Goal:</b><br/>Select the best model for<br/>the given prompt<br/><br/><b>Process:</b><br/>1. Each candidate model<br/>    generates N images<br/>2. HPSv3 scores all<br/>    generated images<br/>3. Calculate average score:<br/>    r̄ᵢ = (1/N) Σ rᵢ,ⱼ<br/>4. Select golden model:<br/>    m* = argmax(r̄ᵢ)<br/><br/><b>✅ Optimal model for<br/>this specific prompt</b>"]
    
    C1 --> C2["<b>🔄 Stage 2: Sample-wise Preference</b><br/><br/><b>Goal:</b><br/>Iteratively refine image quality<br/><br/><b>Process:</b><br/>1. Golden model m* generates<br/>    B images (batch)<br/>2. HPSv3 evaluates and<br/>    selects winner<br/>3. Add noise to winner +<br/>    original prompt<br/>4. Use as input for next round,<br/>    repeat S times<br/><br/><b>✅ Progressive quality<br/>improvement</b>"]
    
    C1 --> Final["<b>🏆 FINAL OUTPUT</b><br/><b>Best Quality Image</b><br/><br/>I* = argmax(rₙ,ₖ)<br/>across ALL rounds<br/><br/>Select the single best image<br/>from all refinement rounds<br/><br/><b>Result:</b><br/>Highest quality image<br/>aligned with human preferences"]
    C2 --> Final
    
    %% ==========================================
    %% STYLING
    %% ==========================================
    
    classDef titleStyle fill:#667eea,stroke:#667eea,color:#fff,stroke-width:4px,padding:20px
    classDef dataStyle fill:#e3f2fd,stroke:#2196f3,stroke-width:4px,color:#333,padding:20px
    classDef filterStyle fill:#fff3e0,stroke:#ff9800,stroke-width:4px,color:#333,padding:20px
    classDef annotateStyle fill:#f3e5f5,stroke:#9c27b0,stroke-width:4px,color:#333,padding:20px
    classDef modelStyle fill:#e8f5e9,stroke:#4caf50,stroke-width:4px,color:#333,padding:20px
    classDef trainStyle fill:#fce4ec,stroke:#e91e63,stroke-width:4px,color:#333,padding:20px
    classDef cohpStyle fill:#fff9c4,stroke:#fbc02d,stroke-width:4px,color:#333,padding:20px
    classDef outputStyle fill:#ffebee,stroke:#f44336,stroke-width:4px,color:#333,padding:20px
    classDef phaseStyle fill:#667eea,stroke:#667eea,color:#fff,stroke-width:4px,padding:20px
    classDef phase2Style fill:#ff9800,stroke:#ff9800,color:#fff,stroke-width:4px,padding:20px
    classDef phase3Style fill:#9c27b0,stroke:#9c27b0,color:#fff,stroke-width:4px,padding:20px
    classDef phase4Style fill:#4caf50,stroke:#4caf50,color:#fff,stroke-width:4px,padding:20px
    classDef phase5Style fill:#fbc02d,stroke:#fbc02d,color:#fff,stroke-width:4px,padding:20px
    
    class Start titleStyle
    class P1 phaseStyle
    class S1,S2,S3 dataStyle
    class P2 phase2Style
    class F1,F2,F3,F4 filterStyle
    class P3 phase3Style
    class A1,A2,Dataset annotateStyle
    class P4 phase4Style
    class M1,M2 modelStyle
    class Trained trainStyle
    class P5 phase5Style
    class C1,C2 cohpStyle
    class Final outputStyle
```
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
