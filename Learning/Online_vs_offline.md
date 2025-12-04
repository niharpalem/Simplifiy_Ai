# Online vs Offline Learning

A visual guide to understanding when models generate during training vs learning from fixed datasets.

## TL;DR

| Aspect | Offline Learning | Online Learning |
|--------|------------------|-----------------|
| **Data** | Fixed dataset collected upfront | Generated during training |
| **Interaction** | None during training | Continuous feedback loop |
| **Speed** | ⚡ Fast | 🐢 Slower |
| **Cost** | 💰 Cheaper | 💸 Expensive |
| **Adaptability** | Static | Adapts to improvements |
| **Examples** | SFT, Diffusion pretraining | RLHF, Online DPO |

---

## Flowchart Overview

```mermaid
flowchart LR
    subgraph OFFLINE["🗄️ OFFLINE LEARNING"]
        direction TB
        O1[1️⃣ Collect data FIRST]
        O2[2️⃣ Train on fixed dataset]
        O3[3️⃣ No generation during training]
        O1 --> O2 --> O3
    end
    
    subgraph ONLINE["🔄 ONLINE LEARNING"]
        direction TB
        N1[1️⃣ Generate DURING training]
        N2[2️⃣ Get immediate feedback]
        N3[3️⃣ Constant interaction]
        N1 --> N2 --> N3
    end
    
    style OFFLINE fill:#e0f7fa,stroke:#00838f
    style ONLINE fill:#fce4ec,stroke:#c2185b
```

---

## The Core Difference

```mermaid
flowchart TD
    Q["❓ Does the model generate<br/>new outputs during training?"]
    
    Q -->|"❌ No - Uses pre-collected data"| OFF[OFFLINE]
    Q -->|"✅ Yes - Generates & learns"| ON[ONLINE]
    
    OFF --> OFF_EX["SFT, Pretraining,<br/>Offline RL"]
    ON --> ON_EX["RLHF, Online DPO,<br/>Active Learning"]
    
    style OFF fill:#e0f7fa
    style ON fill:#fce4ec
```

---

## Offline Learning

The model learns from a **fixed dataset** collected before training begins.

```mermaid
flowchart LR
    subgraph Phase1["📦 Phase 1: Data Collection (Once)"]
        A[Generate images] --> B[Human annotation]
        B --> C[Store in dataset]
    end
    
    subgraph Phase2["🎯 Phase 2: Training (Many epochs)"]
        D[Load fixed dataset] --> E[Compute loss]
        E --> F[Update model]
        F --> D
    end
    
    Phase1 --> Phase2
    
    style Phase1 fill:#b2ebf2
    style Phase2 fill:#e0f7fa
```

### Advantages

- ⚡ **Very fast training** — no generation overhead
- ♻️ **Efficient** — reuse same data many times
- 📈 **Scalable** — can leverage massive datasets
- 💰 **Cheaper** — no constant human feedback needed

### Pseudocode

```python
# Step 1: Collect data (once)
dataset = []
for prompt in prompts:
    images = generate_many(prompt)
    best, worst = human_annotate(images)
    dataset.add((prompt, best, worst))

# Step 2: Train on fixed dataset
for epoch in epochs:
    for (prompt, best, worst) in dataset:
        loss = compute_loss(prompt, best, worst)
        update_model(loss)
        # ❌ No new generation happens here!
```

---

## Online Learning

The model **generates new outputs at each training step** and learns from immediate feedback.

```mermaid
flowchart LR
    subgraph Loop["🔄 Training Loop (Every Step)"]
        A[Sample prompt] --> B[Generate with<br/>CURRENT model]
        B --> C[Get feedback<br/>reward model / human]
        C --> D[Compute loss]
        D --> E[Update model]
        E --> A
    end
    
    style Loop fill:#fce4ec
```

### Advantages

- 🎯 **Always on-policy** — training on current model's outputs
- 🔍 **Can explore** — discover new strategies
- 📈 **Better final performance** — potentially
- 🔄 **Adapts** — adjusts to model improvements

### Disadvantages

- 🐢 **Much slower** — constant generation overhead
- 💸 **Expensive** — more GPU compute
- 🖥️ **Resource heavy** — requires more infrastructure

### Pseudocode

```python
for training_step in steps:
    # Step 1: Generate NEW images
    prompt = sample_prompt()
    images = current_model.generate(prompt)
    
    # Step 2: Get feedback
    reward = reward_model(images)
    # or: best, worst = human_feedback(images)
    
    # Step 3: Update model
    loss = compute_loss(prompt, images, reward)
    update_model(loss)
    # ✅ Repeat with updated model!
```

---

## Decision Flowchart

```mermaid
flowchart TD
    START[Choose Training Approach] --> Q1{Have large<br/>existing dataset?}
    
    Q1 -->|Yes| Q2{Need model to<br/>adapt during training?}
    Q1 -->|No| ONLINE[Online Learning 🔄]
    
    Q2 -->|No| OFFLINE[Offline Learning 🗄️]
    Q2 -->|Yes| Q3{Budget for<br/>compute?}
    
    Q3 -->|Limited| OFFLINE
    Q3 -->|Available| ONLINE
    
    OFFLINE --> EX1["SFT, DPO, Pretraining"]
    ONLINE --> EX2["RLHF, Online DPO, PPO"]
    
    style OFFLINE fill:#e0f7fa
    style ONLINE fill:#fce4ec
```

---

## Real-World Examples

| Method | Type | Description |
|--------|------|-------------|
| **SFT** | Offline | Fine-tune on curated prompt-response pairs |
| **DPO** | Offline | Learn from fixed preference dataset |
| **RLHF** | Online | Generate → get reward → update (PPO) |
| **Online DPO** | Online | Generate pairs on-the-fly, update preferences |

---

## Intuitive Analogy

**Offline Learning** = Studying from a textbook
- All examples are pre-written
- You review the same problems multiple times
- Fast, cheap, but can't ask new questions

**Online Learning** = Learning with a tutor
- You attempt problems, get immediate feedback
- Tutor adapts to your current skill level
- Slower, expensive, but more personalized

---
