# Variational Auto Encoders (VAE) — Simplified

A visual, beginner-friendly guide to understanding VAEs.

## TL;DR

VAE = Autoencoder + **organized latent space**

| Component | What it does |
|-----------|--------------|
| **Encoder** | Image → μ, σ (distribution, not a point) |
| **KL Loss** | Forces distributions toward N(0,1) |
| **Reparameterization** | z = μ + σ×ε (makes sampling differentiable) |
| **Decoder** | z → reconstructed image |
| **Generation** | Sample z ~ N(0,1) → Decoder → **new image** |

**Key insight:** KL loss organizes latent space so random samples land in meaningful regions.

## 📄 Files

- [`vae_notes_detailed.html`](./vae_notes_detailed.html) — Full walkthrough with examples

## 🔗 Quick Links

- [Interactive VAE Explainer](https://xnought.github.io/vae-explainer/) — Visual demo
- [Claude Artifact Preview](https://claude.ai/public/artifacts/ca1a0c2c-f584-422f-b229-6a8a46fa4e80) — View online

## Tags

`#VAE` `#generative-ai` `#deep-learning` `#autoencoder` `#latent-space` `#tutorial`