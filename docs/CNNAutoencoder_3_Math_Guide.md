# CNNAutoencoder_3: A Mathematical and Architectural Study Guide

This document provides a rigorous, equation-driven description of CNNAutoencoder_3 for histopathology patch reconstruction. It details all layers, the encoder/decoder mapping, attention, skip connections, and loss components with exact formulas and dimension tracking.

---

## 1. Problem Setup and Notation

- Input image: x ∈ R^{C×H×W}, with C=3, H=W=150.
- Autoencoder: fθ = gθ_d ∘ hθ_e, where h is encoder, g is decoder.
- Reconstruction: x̂ = fθ(x) ≈ x.

Objective (generic multi-term):
L(x, x̂) = α · L_MSE(x, x̂) + β · L1(x, x̂) + γ · L_SSIM(x, x̂) + λ · Ω(θ)

- L_MSE(x, x̂) = 1/(C·H·W) ∥x − x̂∥_2^2
- L1(x, x̂)   = 1/(C·H·W) ∥x − x̂∥_1
- Ω(θ)        = weight decay (e.g., ∑‖W‖_2^2)

Note: In practice we normalize inputs to y = (x − 0.5)/0.5 ∈ [−1, 1], and use tanh at the output.

---

## 2. Convolutional Building Blocks

### 2.1 2D Convolution (Conv2d)
Given F_in ∈ R^{C_in×H×W}, weight W ∈ R^{C_out×C_in×k×k}, bias b ∈ R^{C_out}, stride s, padding p:
Y[c_out, i, j] = b[c_out] + ∑_{c=0}^{C_in−1} ∑_{u=0}^{k−1} ∑_{v=0}^{k−1} W[c_out,c,u,v] · F_in[c, s·i + u − p, s·j + v − p]

Spatial size:
H_out = ⌊(H + 2p − k)/s⌋ + 1,  W_out = ⌊(W + 2p − k)/s⌋ + 1

### 2.2 Transposed Convolution (ConvTranspose2d)
Given X ∈ R^{C_in×H×W}, W ∈ R^{C_in×C_out×k×k}, stride s, padding p, output_padding op:
H_out = (H − 1)·s − 2p + k + op,  W_out = (W − 1)·s − 2p + k + op

### 2.3 Batch Normalization (per-channel)
For activations a (per batch and spatially):
BN(a) = γ · (a − μ)/√(σ^2 + ε) + β
where μ,σ^2 are batch statistics, and γ,β are learnable.

### 2.4 GELU Activation
GELU(x) = x · Φ(x) = x · 0.5 · (1 + erf(x/√2))
Smooth, non-monotonic, improves gradient flow.

### 2.5 Global Pooling
- Global average: GAP_c = (1/(H·W)) ∑_{i,j} F[c,i,j]
- Global max:     GMP_c = max_{i,j} F[c,i,j]

### 2.6 Bilinear Interpolation (for resizing)
Given a 2×2 neighborhood a00,a10,a01,a11 and local coords (u,v)∈[0,1]^2:
B(u,v) = a00(1−u)(1−v) + a10 u(1−v) + a01 (1−u) v + a11 u v

---

## 3. Channel Attention Block (Squeeze-and-Excitation style)
Input F ∈ R^{C×H×W}.

1) Squeeze (global descriptors):
 z^avg_c = (1/(H·W)) ∑_{i,j} F[c,i,j],   z^max_c = max_{i,j} F[c,i,j]

2) Excitation via 1×1 conv MLP (reduction r=16):
 s^avg = σ(W2 · δ(W1 · z^avg)),   s^max = σ(W2 · δ(W1 · z^max))
 s = σ(s^avg + s^max)  ∈ (0,1)^C

Here W1 ∈ R^{C/r × C}, W2 ∈ R^{C × C/r}, δ = ReLU, σ = sigmoid.

3) Scale:
 F̃[c,i,j] = s_c · F[c,i,j]

Purpose: learn channel-wise importance; preserve discriminative channels.

---

## 4. Encoder hθ_e: Layer-by-Layer
We trace shapes for x ∈ R^{3×150×150}. Each block uses Conv2d (k=3,s=2,p=1), BN, GELU, optional Dropout, and Channel Attention.

### Block E1
- Conv: 3→64, H×W: 150→ ⌊(150+2−3)/2⌋+1 = 75
  Output: F1 ∈ R^{64×75×75}
- BN + GELU
- Channel Attention on 64 channels → F1̃ ∈ R^{64×75×75}

### Block E2
- Conv: 64→128, 75→38 (⌊75/2⌋+1 = 38)
  Output: F2 ∈ R^{128×38×38}
- BN + GELU, optional Dropout p≈0.1
- Channel Attention → F2̃ ∈ R^{128×38×38}

### Block E3
- Conv: 128→256, 38→19
  Output: F3 ∈ R^{256×19×19}
- BN + GELU, optional Dropout
- Channel Attention → F3̃ ∈ R^{256×19×19}

### Block E4 (Bottleneck)
- Conv: 256→256, 19→10
  Output: Z ∈ R^{256×10×10}
- BN + GELU

Bottleneck dimensionality: 10×10×256 = 25,600.

---

## 5. Decoder gθ_d: Layer-by-Layer with Skips
We upsample via ConvTranspose2d (k=3,s=2,p=1,op=1) then concatenate resized encoder features (skip connections), fuse with Conv+BN+GELU.

### D1: 10→20, fuse with E3 (19)
- Upsample: U1 = ConvT(Z): 10→20, channels 256→256; U1 ∈ R^{256×20×20}
  (size formula: (10−1)·2−2+3+1=20)
- Resize encoder F3̃ (19→20) by bilinear: Ē3 = Resize(F3̃) ∈ R^{256×20×20}
- Concatenate: C1 = [U1; Ē3] ∈ R^{512×20×20}
- Fuse: Conv(512→256, k=3,s=1,p=1) + BN + GELU → D1 ∈ R^{256×20×20}

### D2: 20→40, fuse with E2 (38)
- Upsample: U2 = ConvT(D1): 20→40, 256→128; U2 ∈ R^{128×40×40}
- Resize Ē2 = Resize(F2̃, 38→40) ∈ R^{128×40×40}
- Concatenate: C2 = [U2; Ē2] ∈ R^{256×40×40}
- Fuse: Conv(256→128) + BN + GELU → D2 ∈ R^{128×40×40}

### D3: 40→80, fuse with E1 (75)
- Upsample: U3 = ConvT(D2): 40→80, 128→64; U3 ∈ R^{64×80×80}
- Resize Ē1 = Resize(F1̃, 75→80) ∈ R^{64×80×80}
- Concatenate: C3 = [U3; Ē1] ∈ R^{128×80×80}
- Fuse: Conv(128→64) + BN + GELU → D3 ∈ R^{64×80×80}

### D4: 80→160 → refine → 150
- Upsample: U4 = ConvT(D3): 80→160, 64→32; U4 ∈ R^{32×160×160}
- Refinement: R = Conv(32→3, k=3,s=1,p=1) + Tanh → R ∈ R^{3×160×160}
- Final size correction (GPU-safe):
  x̂ = Interp(R, size=(150,150), mode='bilinear', align_corners=False)

Note: Using bilinear interpolation keeps gradients well-behaved and is supported on MPS.

---

## 6. End-to-End Mapping and Gradients

### 6.1 Forward map
x ∈ R^{3×150×150}
→ E1 → F1̃ ∈ R^{64×75×75}
→ E2 → F2̃ ∈ R^{128×38×38}
→ E3 → F3̃ ∈ R^{256×19×19}
→ E4 → Z ∈ R^{256×10×10}
→ D1(Z, F3̃) → D1 ∈ R^{256×20×20}
→ D2(D1, F2̃) → D2 ∈ R^{128×40×40}
→ D3(D2, F1̃) → D3 ∈ R^{64×80×80}
→ D4(D3) → R ∈ R^{3×160×160}
→ x̂ = Interp(R, 150×150) ∈ R^{3×150×150}

### 6.2 Backpropagation (sketch)
Let L be the loss. For a generic layer y = φ(W * x):
∂L/∂W = (∂L/∂y) * ∂y/∂W,    ∂L/∂x = (∂L/∂y) * ∂y/∂x

- Convolution gradient: correlation with input patches.
- BN: gradients through affine and normalization stats.
- GELU: d/dx GELU(x) = Φ(x) + x · ϕ(x), where ϕ is N(0,1) pdf.
- Attention: gradients flow through MLP and sigmoid to scaling factors s.
- Skips: gradients split across both paths (decoder and encoder features).
- Interp: bilinear interpolation is differentiable almost everywhere.

---

## 7. Losses and Their Roles

### 7.1 Mean Squared Error (MSE)
L_MSE = 1/(C·H·W) ∑_{c,i,j} (x[c,i,j] − x̂[c,i,j])^2
Penalizes large deviations; smooth, convex in x̂.

### 7.2 L1 Loss
L1 = 1/(C·H·W) ∑_{c,i,j} |x[c,i,j] − x̂[c,i,j]|
Robust to outliers; encourages sparse residuals.

### 7.3 SSIM (structural similarity) — optional
For local windows, with means μx, μy, variances σx^2, σy^2, covariance σxy:
SSIM(x,y) = ((2 μx μy + C1)(2 σxy + C2))/((μx^2 + μy^2 + C1)(σx^2 + σy^2 + C2))
L_SSIM = 1 − SSIM(x, x̂)
Preserves perceived structure.

---

## 8. Design Rationale and Best Practices

- Odd sizes (150) + stride-2 yield off-by-one artifacts; we fix in decoder via output_padding and a final differentiable resize.
- Stop compressing at 10×10: retains semantic context while reducing spatial redundancy (25,600 latent).
- Channel attention improves SNR by reweighting informative channels.
- GELU + BN stabilize optimization; Dropout adds regularization where needed.
- Skip connections fuse multi-scale features, preserve edges and fine tissue textures.

---

## 9. Pseudocode (PyTorch-like)

```python
# Encoder blocks (example skeleton)
E1 = Conv2d(3,64,3,2,1) → BN → GELU → SE(64)
E2 = Conv2d(64,128,3,2,1) → BN → GELU → Dropout → SE(128)
E3 = Conv2d(128,256,3,2,1) → BN → GELU → Dropout → SE(256)
E4 = Conv2d(256,256,3,2,1) → BN → GELU

# Decoder with skips
U1 = ConvT2d(256,256,3,2,1,1)
D1 = Conv([U1; Resize(E3)], 512→256) → BN → GELU
U2 = ConvT2d(256,128,3,2,1,1)
D2 = Conv([U2; Resize(E2)], 256→128) → BN → GELU
U3 = ConvT2d(128,64,3,2,1,1)
D3 = Conv([U3; Resize(E1)], 128→64) → BN → GELU
U4 = ConvT2d(64,32,3,2,1,1)
R  = Conv(32→3,3,1,1) → Tanh
x_hat = Interpolate(R, size=(150,150), mode='bilinear', align_corners=False)
```

---

## 10. Complexity Estimates

Let k=3, typical conv: cost ≈ H_out·W_out·C_out·(k^2·C_in).
- E1 (3→64, 150→75): 75·75·64·(9·3) ≈ 9.7M MACs
- E2 (64→128, 75→38): 38·38·128·(9·64) ≈ 107M MACs
- E3 (128→256, 38→19): 19·19·256·(9·128) ≈ 107M MACs
- E4 (256→256, 19→10): 10·10·256·(9·256) ≈ 59M MACs
Decoder similar scale. Total per forward ≈ few hundred MFLOPs.

---

## 11. Practical Notes

- Normalization: Normalize RGB with mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5).
- Output activation: Tanh to match [-1,1].
- Optimizer: Adam (lr≈1e−3), ReduceLROnPlateau.
- Batch size: balance GPU/CPU memory; 16–64 typical for 150×150.
- Mixed precision can speed up if supported.

---

## 12. Summary Diagram (shapes)

3×150×150 → 64×75×75 → 128×38×38 → 256×19×19 → 256×10×10
                               ↓          ↓           ↓
                              skip       skip        bottleneck
                               ↓          ↓           ↓
               64×80×80 ← 128×40×40 ← 256×20×20 ← 256×10×10
                               ↓
                         3×160×160 → Interp → 3×150×150

This guide serves as a precise reference for the inner workings of CNNAutoencoder_3, including exact formulas, size transitions, and the rationale behind each architectural choice.
