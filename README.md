# One Layer Is Enough: Adapting Pretrained Visual Encoders for Image Generation

<img src="images/fae_compare.png" alt="Diagram comparing SD-VAE, VA-VAE, RAE, and FAE" width="500">

Latent Diffusion Models reduced generation costs by shifting computation from raw pixels into a compressed latent space. A central question in recent research is whether we can skip training VAEs from scratch and instead reuse pretrained encoders like DINOv2 or SigLIP that already possess rich visual representations.

However, features optimized for understanding images differ fundamentally from those required to generate them:

Before we start, these are a few terms we'll keep reusing:

- $x$: A patch embedding coming out of a frozen, pretrained encoder (e.g., DINOv2).
- $z$: The compact latent that we compress $x$ into.
- $\hat{x}$: The reconstructed version of $x$, decoded back out of $z$.

## The Core Problem

Self-supervised encoders like DINOv2 use masked prediction to infer missing patches. To preserve high-level semantic options, these models output high-dimensional embeddings (e.g., 1,536 dimensions in DINOv2-g).

Generative models require the opposite. Tracking noisy inputs across hundreds of denoising steps becomes unstable in high-dimensional spaces, making low-dimensional latents (typically 4 to 64 dimensions) necessary. Keeping pretrained features large destabilizes the generator; over-compressing them discards the semantic rich priors we wanted to reuse in the first place.

## Phase 1: Architecture Components

<img src="images/fae_arch.png" alt="Diagram of the FAE architecture" width="200">

### 1. The Single-Attention Encoder

This is the whole trick of the paper. Instead of building a deep, multi-layer network to compress the pretrained embedding $x$ down into a small latent $z$, the authors use **one self-attention layer followed by a linear projection**. 
Because feature compression is far simpler than self-supervised pretraining and deep adapters easily overfit (altering features to minimise reconstruction loss at the expense of general semantics). A shallow adapter keeps $z$ aligned with the original embedding space. The single attention layer enables inter-patch communication to remove redundant global information across the image.

### 2. The Feature Decoder

The feature decoder reconstructs the original pretrained embedding $\hat{x}$ from $z$:

$$\mathcal{L}_{VAE} = \| \hat{x} - x \|_2^2 + \beta\, \text{KL}\big(q(z \mid x) \,\|\, p(z)\big)$$

This is a 6-layer transformer, built with RoPE, RMSNorm, and SwiGLU for stability, matched to the hidden dimension of whatever backbone it's paired with (DINOv2, SigLIP, etc.). The MSE + KL objective keeps $\hat{x}$ aligned with the pretrained feature space.

### 3. The Pixel Decoder

A second decoder maps $\hat{x}$ to RGB pixels using adversarial, perceptual, and pixel-reconstruction losses. Uncoupling feature reconstruction from pixel synthesis ensures feature alignment without forcing the generator to decode raw pixels directly.

### 4. The Generative Model

With $z$ fixed, standard generative models (such as SiT or STARFlow) train directly on the latent space without architectural modifications.

## Phase 2: How It All Runs Together

<img src="images/fae_stages.png" alt="Diagram showing FAE training stages" width="500">

1. A frozen pretrained encoder (DINOv2, SigLIP, whichever) turns an image into a high-dimensional embedding $x$.
2. The **single-attention encoder** compresses $x$ into a small latent $z$ (this is the only new trainable piece added to the frozen backbone).
3. The **feature decoder** reconstructs $\hat{x}$ from $z$, trained with a simple VAE objective to stay close to the original embedding.
4. The **pixel decoder** turns $\hat{x}$ into a real image, learning this mapping first from noised embeddings (so it becomes robust) and then fine-tuning on the actual reconstructed $\hat{x}$.
5. Separately, a diffusion model or normalizing flow model is trained to generate new samples of $z$ iteratively from sampling noise to denoising step by step.
6. At generation time: sample noise in latent space → denoise into a clean $z$ → decode through the feature decoder to get $\hat{x}$ → decode through the pixel decoder to get pixels.

## Comparison with Existing Approaches

- **Feature alignment** (e.g., REPA, VA-VAE): train the generator or its VAE to align with a pretrained encoder's features. This needs carefully tuned alignment losses and extra training stages, and still discards information that doesn't fit the generator's native architecture.
  
- **Direct modeling** (e.g., RAE): use the pretrained embeddings as the latent space directly. This avoids alignment losses but forces the generator to use wider channels, extra heads to handle the huge embedding dimension.

- **FAE**: Keeps the generator small by using a low-dimensional latent space, while preserving semantic richness through minimal adapter capacity. This yields competitive FID scores on ImageNet and MS-COCO with significantly fewer training epochs.
  
## Two-Cents

FAE demonstrates that minimal adapter capacity prevents feature distortion when mapping pretrained visual representations to generative latents. Splitting feature restoration from pixel synthesis cleanly isolates high-level semantics from low-level detail. While its reconstruction FID slightly trails pixel-first methods like VA-VAE, FAE effectively trades marginal pixel precision for structural simplicity and lower compute requirements.
