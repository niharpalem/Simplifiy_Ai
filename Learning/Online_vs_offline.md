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
%%{init: {'themeVariables': {'fontSize': '14px'}}}%%
flowchart TB
    subgraph OFF["OFFLINE LEARNING"]
        direction TB
        O1["1. Collect data FIRST"]
        O2["2. Train on fixed dataset"]
        O3["3. No generation during training"]
        O1 --> O2 --> O3
    end
    
    subgraph ON["ONLINE LEARNING"]
        direction TB
        N1["1. Generate DURING training"]
        N2["2. Get immediate feedback"]
        N3["3. Constant interaction"]
        N1 --> N2 --> N3
    end
    
    style OFF fill:#e0f7fa,stroke:#00838f
    style ON fill:#fce4ec,stroke:#c2185b
```

---

## The Core Difference

```mermaid
%%{init: {'themeVariables': {'fontSize': '14px'}}}%%
flowchart TD
    %%{init: {'themeVariables': {'fontSize': '14px'}}}%%
    Q{{"Does model generate new outputs during training?"}}
    
    Q -->|No| OFF["OFFLINE"]
    Q -->|Yes| ON["ONLINE"]
    
    OFF --> OFF_EX["SFT, Pretraining, Offline RL"]
    ON --> ON_EX["RLHF, Online DPO, Active Learning"]
    
    style OFF fill:#e0f7fa,stroke:#00838f
    style ON fill:#fce4ec,stroke:#c2185b
```

---

## Offline Learning

The model learns from a **fixed dataset** collected before training begins.

```mermaid
%%{init: {'themeVariables': {'fontSize': '14px'}}}%%
flowchart LR
    subgraph P1["Phase 1: Data Collection"]
        A["Generate images"] --> B["Human annotates"]
        B --> C["Store in dataset"]
    end
    
    subgraph P2["Phase 2: Training"]
        D["Load dataset"] --> E["Compute loss"]
        E --> F["Update model"]
        F -.->|repeat| D
    end
    
    P1 --> P2
    
    style P1 fill:#b2ebf2,stroke:#00838f
    style P2 fill:#e0f7fa,stroke:#00838f
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
%%{init: {'themeVariables': {'fontSize': '14px'}}}%%
flowchart LR
    subgraph LOOP["Training Loop - Every Step"]
        A["Sample prompt"] --> B["Generate with current model"]
        B --> C["Get feedback"]
        C --> D["Compute loss"]
        D --> E["Update model"]
        E -.->|repeat| A
    end
    
    style LOOP fill:#fce4ec,stroke:#c2185b
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
%%{init: {'themeVariables': {'fontSize': '14px'}}}%%
flowchart TD
    START(["Choose Training Approach"]) --> Q1{"Have large dataset?"}
    
    Q1 -->|Yes| Q2{"Need adaptation?"}
    Q1 -->|No| ON["ONLINE"]
    
    Q2 -->|No| OFF["OFFLINE"]
    Q2 -->|Yes| Q3{"Compute budget?"}
    
    Q3 -->|Limited| OFF
    Q3 -->|Available| ON
    
    OFF --> EX1["SFT, DPO, Pretraining"]
    ON --> EX2["RLHF, Online DPO, PPO"]
    
    style OFF fill:#e0f7fa,stroke:#00838f
    style ON fill:#fce4ec,stroke:#c2185b
    style START fill:#fff,stroke:#333
```

---

## Side-by-Side Comparison

```mermaid
%%{init: {'themeVariables': {'fontSize': '14px'}}}%%
flowchart TB
    subgraph OFFLINE["OFFLINE"]
        direction LR
        A1[("Dataset")] --> A2["Model"]
        A2 --> A3(["Trained Model"])
    end
    
    subgraph ONLINE["ONLINE"]
        direction LR
        B1["Model"] --> B2(["Output"])
        B2 --> B3{{"Reward"}}
        B3 --> B1
    end
    
    style OFFLINE fill:#e0f7fa,stroke:#00838f
    style ONLINE fill:#fce4ec,stroke:#c2185b
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

