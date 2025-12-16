# Attention Sink - TLDR

## What is it?
**Attention sink** = LLMs dump 30-40% of attention on first tokens (BOS) even though they're semantically meaningless.

## Why does this happen?
1. **Softmax constraint**: Attention must sum to 1.0
2. **No place to dump unused attention** → Model uses first tokens as "numerical sink"
3. **Global visibility**: BOS token visible to all positions in causal attention
4. **Learned behavior**: Safe no-op that doesn't hurt performance

## The Problem
```
Token Position:  [BOS]  T1   T2   T3   T4   T5  ...
Attention:       35%   15%   8%   6%   5%   4%  ...
                 ↑
            Attention Sink!
```

## Why it matters?

### KV Cache Issue
- **Problem**: Evicting first tokens from cache → performance collapse
- **Solution (StreamingLLM)**: Always keep first 4 tokens + recent window
- Enables infinite-length generation with constant memory

### Actually Helpful
✅ Numerical stability  
✅ Prevents attention collapse  
✅ Cleaner gradients  
✅ Model can ignore irrelevant tokens  

## Solutions

1. **Gated Attention (Qwen)**: Apply sigmoid gate to "mute" attention sink
   - Formula: 'Y = Y ⊙ σ(XW)'
   - Gate reduces sink to near-zero when info not useful
   - [Paper](https://arxiv.org/pdf/2505.06708)

2. **StreamingLLM**: Keep first 4 tokens + sliding window
   - Enables infinite context
   - Constant memory usage

3. **Alternative attention mechanisms**: Linear attention, learned sink tokens

## Key Takeaway
**Attention sink is a feature, not a bug.** Critical for long-context inference - never evict first 4-6 tokens from KV cache!

---

**Project**: Simplifiy_Ai  
**Date**: December 2025  
**Full Article**: [attention_sink.html](https://claude.ai/public/artifacts/311cf2a0-b1f1-4c8d-90b8-b7747c47429f)