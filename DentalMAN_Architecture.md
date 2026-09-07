# DentalMAN — Dental Manifold Attention Network
## Architecture Reference for Custom-Seg-Net (FDI 53-class Tooth Segmentation)

---

# 1. System Overview

DentalMAN is a point cloud segmentation architecture designed specifically for dental scan data. It replaces generic Euclidean grouping with **arch-aware** operators and uses **learnable tooth-class queries** to exploit dental domain structure.

## 1.1 File Map

| Layer | File | Class / Function |
|-------|------|------------------|
| **Model** | `networks/DentalMAN.py` | `DentalMAN` (main), `EncoderStage`, `DecoderStage`, `ManifoldVecAttn`, `HybridKNN`, `ToothQueryCrossAttention`, `DualStreamEncoder`, `MultiScaleSegHead` |
| **Data** | `dataloader/DataLoader_DentalMAN.py` | `ObjDataset_Arch`, `compute_arch_curve_params` |
| **Train** | `Train_DentalMAN.py` | `DentalMAN_Trainer`, `FocalLoss` |
| **Test** | `Test_DentalMAN.py` | `AccuracyOnlyTester` |

---

# 2. Full Data Flow (Tensor Shapes)

Below is the complete forward pass with every tensor shape annotated for `B=1, N=8192, num_classes=53`.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INPUT LAYER                                        │
│                                                                             │
│  pos:  (1, 8192, 3)    ← xyz coordinates (normalized, zero-centered)       │
│  x:    (1, 8192, 3)    ← rgb colors (normalized [0,1])                     │
│  arc_params: (1, 8192, 1)  ← arch arc-length parameter [0,1]              │
│                                                                             │
│  Total input channels: 3 (xyz) + 3 (rgb) + 1 (arc) = 7                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ① DUAL-STREAM INPUT ENCODER                              │
│                    `DualStreamEncoder` in DentalMAN.py:309-337               │
│                                                                             │
│  GEOMETRY STREAM:                                                           │
│    Input:  (1, 8192, 4)    ← torch.cat([pos, arc_params], dim=-1)          │
│    ┌─ Linear(4 → 64) ── GELU ── Linear(64 → 64) ── GELU ─┐               │
│    Output: (1, 8192, 64)                                                 │
│                                                                             │
│  APPEARANCE STREAM:                                                         │
│    Input:  (1, 3, 8192)    ← x.permute(0,2,1)                             │
│    ┌─ Conv1d(3→64,1) ── LN ── GELU ── Conv1d(64→64,1) ── LN ── GELU ─┐  │
│    └─ Conv1d(64→64,1) ── LN ── GELU ───────────────────────────────────────┘  │
│    Output: (1, 8192, 64)                                                 │
│                                                                             │
│  FUSION:                                                                    │
│    Input:  torch.cat([geom(8192,64), app(8192,64)]) → (1, 8192, 128)     │
│    ┌─ Linear(128 → 128) ── GELU ─┐                                       │
│    Output: (1, 8192, 128)                                                 │
│                                                                             │
│  Result:  x_fused = (1, 8192, 128)                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ② ENCODER STAGE 1                                        │
│                    `EncoderStage` in DentalMAN.py:249-272                    │
│                    npoint=2048, k=24, d_in=128, d_out=128, α_init=0.7       │
│                                                                             │
│  STEP 2a: FARTHEST POINT SAMPLING (FPS)                                     │
│    Input:  pos  (1, 8192, 3)                                               │
│    Output: pos1 (1, 2048, 3)   ← 2048 farthest points                      │
│    Index:  fps_idx (1, 2048)   ← indices into full set                     │
│                                                                             │
│  STEP 2b: INDEX + PROJECT                                                    │
│    x1  = index_points(x_fused, fps_idx)  → (1, 2048, 128)                  │
│    x1  = LayerNorm(x1)                     → (1, 2048, 128)                 │
│    arc1 = index_points(arc_params, fps_idx) → (1, 2048, 1)                 │
│                                                                             │
│  STEP 2c: HYBRID KNN (k=24)                                                 │
│    For each of 2048 query points, compute:                                  │
│      d_hybrid = sigmoid(α)·d_euclidean + (1-sigmoid(α))·d_arch              │
│                                                                             │
│    For N=2048 (≤ chunk_size=2048): use dense mode                           │
│    ┌─ Build (1, 2048, 2048) distance matrix ── topk(24) ─┐               │
│    → knn_idx1 (1, 2048, 24)                                                │
│                                                                             │
│  STEP 2d: MANIFOLD VEC ATTN × 2 blocks                                      │
│    Block input:  (1, 2048, 128)                                            │
│    Each block:                                                              │
│      ┌─ LayerNorm ── ManifoldVecAttn ── + ── LayerNorm ── FFN ── + ─┐    │
│    → x1 (1, 2048, 128)                                                   │
│                                                                             │
│  Output: x1=(1,2048,128), pos1=(1,2048,3), arc1=(1,2048,1)                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ③ ENCODER STAGE 2                                        │
│                    npoint=512, k=20, d_in=128, d_out=256, α_init=0.5        │
│                                                                             │
│  FPS:      pos1 (1,2048,3) → pos2 (1,512,3)                                │
│  Index:    x1 (1,2048,128) → (1,512,128)                                   │
│  Proj:     LayerNorm → Linear(128→256) → (1,512,256)                       │
│  HybridKNN:  k=20, dist_matrix (1,512,512) → knn_idx2 (1,512,20)           │
│  ManifoldVecAttn × 2: (1,512,256) → (1,512,256)                            │
│                                                                             │
│  Output: x2=(1,512,256), pos2=(1,512,3), arc2=(1,512,1)                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ④ ENCODER STAGE 3                                        │
│                    npoint=128, k=16, d_in=256, d_out=512, α_init=0.3        │
│                                                                             │
│  FPS:      pos2 (1,512,3) → pos3 (1,128,3)                                 │
│  Index:    x2 (1,512,256) → (1,128,256)                                    │
│  Proj:     LayerNorm → Linear(256→512) → (1,128,512)                       │
│  HybridKNN:  k=16, dist_matrix (1,128,128) → knn_idx3 (1,128,16)           │
│  ManifoldVecAttn × 3: (1,128,512) → (1,128,512)                            │
│                                                                             │
│  Output: x3=(1,128,512), pos3=(1,128,3), arc3=(1,128,1)                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ⑤ TOOTH QUERY CROSS-ATTENTION (BOTTLENECK)               │
│                    `ToothQueryCrossAttention` in DentalMAN.py:274-302        │
│                    num_queries=53, d_model=512, n_heads=8                    │
│                                                                             │
│  Learnable queries:  tooth_embeddings (1, 53, 512)                         │
│  Encoder features:   x3 (1, 128, 512)                                      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Q = tooth_embeddings        → (1, 53, 512)                         │   │
│  │  K = x3                      → (1, 128, 512)                        │   │
│  │  V = x3                      → (1, 128, 512)                        │   │
│  │                                                                     │   │
│  │  attn_out = MultiheadAttn(Q, K, V)  → (1, 53, 512)                  │   │
│  │  queries = LayerNorm(queries + attn_out)                            │   │
│  │  queries = queries + FFN(LayerNorm(queries))                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Output: tooth_context = (1, 53, 512)  ← 53 per-class feature vectors      │
│                                                                             │
│  FDI Query Mapping:                                                         │
│    idx 0:   FDI 0   (gingiva)                                               │
│    idx 1-8: FDI 11-18 (upper right permanent)                              │
│    idx 9-16: FDI 21-28 (upper left permanent)                              │
│    idx 17-24: FDI 31-38 (lower left permanent)                              │
│    idx 25-32: FDI 41-48 (lower right permanent)                             │
│    idx 33-37: FDI 51-55 (upper right deciduous)                            │
│    idx 38-42: FDI 61-65 (upper left deciduous)                             │
│    idx 43-47: FDI 71-75 (lower left deciduous)                             │
│    idx 48-52: FDI 81-85 (lower right deciduous)                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ⑥ DECODER STAGE 1                                        │
│                    `DecoderStage` in DentalMAN.py:319-339                    │
│                    d_in=512, d_out=256, d_skip=256, d_tooth=512             │
│                    N/64 → N/16  (128 → 512)                                │
│                                                                             │
│  Input:  x3=(1,128,512) at pos3, target positions pos2=(1,512,3)           │
│                                                                             │
│  STEP 6a: 3-NN INTERPOLATION                                                │
│    ┌─ For each target point in pos2 (512 pts)                             │
│    │   - Find 3 nearest neighbors in pos3 (128 pts)                        │
│    │   - Weight by inverse distance: wᵢ = (1/dᵢ) / Σ(1/dⱼ)               │
│    │   - Interpolate: interp = Σ wᵢ · x3[idxᵢ]                           │
│    └─ Result: (1, 512, 512)                                               │
│                                                                             │
│  STEP 6b: SKIP CONNECTION                                                    │
│    interp = torch.cat([interp, x2])   → (1, 512, 512+256) = (1,512,768)    │
│    x2 is the feature from encoder stage 2 at same resolution (N/16)        │
│                                                                             │
│  STEP 6c: TOOTH CONTEXT ATTENTION                                           │
│    `ToothContextDecoder` in DentalMAN.py:304-317                            │
│    ┌─ Input: tooth_context (1,53,512), interp_feat (1,512,768)            │
│    │   attn_logits = Linear(768→192) → GELU → Linear(192→53)               │
│    │   attn_weights = softmax(attn_logits, dim=-1)  → (1,512,53)          │
│    │   context = bmm(attn_weights, tooth_context)    → (1,512,512)        │
│    └─ Per-point weighted aggregation of the 53 tooth queries              │
│                                                                             │
│  STEP 6d: FUSE + MLP                                                        │
│    cat = [interp(512), context(512)]  → (1,512,1024)                       │
│    LayerNorm(1024)                                                          │
│    ┌─ Linear(1024→256) ── GELU ── Drop ── Linear(256→256) ── GELU ─┐    │
│    → d1 = (1, 512, 256)                                                 │
│                                                                             │
│  Output: d1=(1,512,256)                                                    │
│  Aux head applied: aux2 = Linear(256→128) → GELU → Linear(128→53)          │
│                                                              → (1,512,53)   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ⑦ DECODER STAGE 2                                        │
│                    d_in=256, d_out=128, d_skip=128, d_tooth=512             │
│                    N/16 → N/4  (512 → 2048)                                │
│                                                                             │
│  3-NN interp:  (1,512,256) at pos2 → (1,2048,256) at pos1                 │
│  Skip:        + x1 (1,2048,128) → (1,2048,384)                             │
│  ToothAttn:    context = attn(interp, tooth_context) → (1,2048,512)         │
│  Concat:       (384 + 512) = 896 → LayerNorm → MLP(896→256→128)            │
│  → d2 = (1,2048,128)                                                       │
│                                                                             │
│  Aux head: aux3 = Linear(128→64) → GELU → Linear(64→53) → (1,2048,53)     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ⑧ DECODER STAGE 3                                        │
│                    d_in=128, d_out=64, d_skip=128, d_tooth=512              │
│                    N/4 → N  (2048 → 8192)                                  │
│                                                                             │
│  3-NN interp:  (1,2048,128) at pos1 → (1,8192,128) at pos                 │
│  Skip:        + x_fused (1,8192,128) → (1,8192,256)                        │
│  ToothAttn:    context = attn(interp, tooth_context) → (1,8192,512)         │
│  Concat:       (256 + 512) = 768 → LayerNorm → MLP(768→128→64)             │
│  → d3 = (1,8192,64)                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ⑨ MULTI-SCALE SEGMENTATION HEAD                          │
│                    `MultiScaleSegHead` in DentalMAN.py:341-355               │
│                                                                             │
│  Gather decoder outputs (all permuted to Conv1d format B,C,N):              │
│    d1_t = d1.permute(0,2,1)  → (1, 256, 512)                               │
│    d2_t = d2.permute(0,2,1)  → (1, 128, 2048)                              │
│    d3_t = d3.permute(0,2,1)  → (1, 64, 8192)                               │
│                                                                             │
│  For each sample, we need same spatial dim (N=8192):                        │
│    d1_t interpolated from 512→8192 by nearest neighbor upsampling           │
│    d2_t interpolated from 2048→8192 by nearest neighbor upsampling          │
│    (Note: in practice, the decoder outputs are already at full N           │
│     resolution, so no interpolation is needed. d1/d2/d3 are all N.)         │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  concat = cat([d1, d2, d3], dim=1)  → (1, 256+128+64, 8192)         │   │
│  │                                      → (1, 896, 8192)                │   │
│  │                                                                     │   │
│  │  Conv1d(896 → 256, 1) ── BN ── ReLU ── Dropout(0.3)                 │   │
│  │  Conv1d(256 → 128, 1) ── BN ── ReLU ── Dropout(0.1)                 │   │
│  │  Conv1d(128 → 53, 1)   ── (no activation, raw logits)                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  logits = (1, 53, 8192) → permute(0,2,1) → (1, 8192, 53)                  │
│                                                                             │
│  During TRAINING, also returns:                                             │
│    aux_logits2 = (1, 512, 53)  (upsampled from decoder2)                   │
│    aux_logits3 = (1, 2048, 53) (upsampled from decoder3)                   │
│    Total loss = Focal(main) + 0.4·Focal(aux2) + 0.3·Focal(aux3)            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# 3. Component Deep-Dives

---

## 3.1 `compute_arch_curve_params` — Arch Curve Fitting
**File:** `dataloader/DataLoader_DentalMAN.py:12-72`

### Purpose
Computes a 1D arc-length parameter `t ∈ [0,1]` for each point, measuring how far along the dental arch it sits. Points at the leftmost molar → t=0, points at the rightmost molar → t=1.

### Algorithm (Step-by-Step)

```
Input:  points_3d  (N, 3)    ← raw xyz coordinates
Output: arc_params (N, 1)    ← [0, 1] parameter

┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: CENTER                                                      │
│   centroid = mean(points_3d, axis=0)                                │
│   centered = points_3d - centroid                                   │
│                                                                     │
│ Step 2: PCA TO 2D                                                    │
│   cov = centeredᵀ · centered / (N-1)                                 │
│   eigvals, eigvecs = eigh(cov)                                      │
│   main_axes = eigvecs[:, ::-1]        ← sort descending              │
│   proj_2d = centered · main_axes[:, :2]   ← (N, 2)                  │
│                                                                     │
│ Step 3: ANGLE-BASED SORTING                                          │
│   angles = arctan2(proj_2d[:,1], proj_2d[:,0])    ← [-π, π]         │
│   radii  = sqrt(proj_2d[:,0]² + proj_2d[:,1]²)                      │
│   sort_idx = argsort(angles)                                         │
│                                                                     │
│ Step 4: PIECEWISE SEGMENT AVERAGING (avoids Runge oscillation)      │
│   Divide [-π, π] into 40 uniform segments                           │
│   For each segment:                                                  │
│     - Average angles and radii of all points in that segment         │
│     - Handles empty segments by interpolation                       │
│   → segment_centers (40,), segment_radii (40,)                       │
│                                                                     │
│ Step 5: ARC-LENGTH PARAMETERIZATION                                  │
│   For each consecutive pair (θᵢ, rᵢ) → (θᵢ₊₁, rᵢ₊₁):                 │
│     dx = rᵢ₊₁·cos(θᵢ₊₁) - rᵢ·cos(θᵢ)                               │
│     dy = rᵢ₊₁·sin(θᵢ₊₁) - rᵢ·sin(θᵢ)                               │
│     segment_length = sqrt(dx² + dy²)                                 │
│   cumulative = cumsum(segment_lengths)                              │
│   normalize: t = cumulative / total_arc_length                       │
│                                                                     │
│ Step 6: INTERPOLATE BACK TO ORIGINAL POINTS                          │
│   For each original angle, find its t value:                         │
│     arc_params = interp(angles_original, theta_fine, t_fine)         │
│   → arc_params (N, 1)  in [0, 1]                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Decision
Uses **piecewise segment averaging** (40 segments) instead of `polyfit(deg=5)` to avoid **Runge's phenomenon** — high-degree polynomials oscillate wildly at boundaries. The piecewise approach is monotonic by construction and O(N) to compute.

---

## 3.2 `HybridKNN` — Hybrid Distance K-Nearest Neighbors
**File:** `networks/DentalMAN.py` class `HybridKNN`, lines 193-234

### Purpose
Replaces standard Euclidean ball query with a learned hybrid distance that blends Euclidean proximity with arch-geodesic proximity.

### Forward Pass

```
Input:  xyz       (B, N, 3)    ← coordinates
        arc_params (B, N, 1)   ← arc-length in [0,1]
Output: knn_idx   (B, N, k)    ← indices of k nearest neighbors

┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1: COMPUTE α (learnable gate per stage)                        │
│   α = sigmoid(self.alpha)    ← scalar in (0, 1)                    │
│   Stage 1: init=0.7 → mostly Euclidean                              │
│   Stage 2: init=0.5 → balanced                                     │
│   Stage 3: init=0.3 → mostly arch-aware                             │
│   (α is optimized via gradient descent during training)              │
│                                                                     │
│ STEP 2: COMPUTE DISTANCES                                            │
│   d_euclidean = sqrt(Σ(xyz_q - xyz_p)²)      ← (B, N, N)           │
│   d_arch      = |arc_params_q - arc_params_p|  ← (B, N, N)         │
│   d_hybrid    = α · d_euclidean + (1-α) · d_arch                    │
│                                                                     │
│ STEP 3: SELECT TOP-k                                                 │
│   _, knn_idx = topk(d_hybrid, k, largest=False)   ← (B, N, k)      │
└─────────────────────────────────────────────────────────────────────┘

MEMORY OPTIMIZATION (for N > chunk_size):
  Instead of building (B, N, N) distance matrix, process in chunks:
  For start in range(0, N, chunk_size):
    chunk_xyz = xyz[:, start : start+chunk_size]
    For each chunk, compute (B, chunk_size, N) distances.
    2048 ≤ chunk_size ≤ 65536: trades memory for sequential compute.
```

### Why Hybrid Distance?
Two points can be spatially close but belong to opposite arches (e.g., upper tooth 11 vs lower tooth 41). Euclidean distance alone cannot distinguish this. Arch distance separates the arches because the arc-length parameter traces along the gingival ridge — upper and lower teeth have very different arc-length values even when nearby in 3D.

---

## 3.3 `ManifoldVecAttn` — Manifold Vector Attention
**File:** `networks/DentalMAN.py` class `ManifoldVecAttn`, lines 78-140

### Purpose
Grouped vector attention (from Point Transformer v2) modified to include **arch-aware positional encoding**.

### Architecture Inside

```
Input:  x         (B, N, d_model)    ← features
        pos       (B, N, 3)          ← xyz
        arc_params (B, N, 1)         ← arc-length
        knn_idx   (B, N, k)          ← neighbor indices
Output: out       (B, N, d_model)    ← attended features

┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1: QKV PROJECTION                                              │
│   qkv = Linear(d_model → d_model*3)                                 │
│   q, k, v = reshape(qkv, 3, n_heads, head_dim)                     │
│   q = q · scale   where scale = 1/√(head_dim)                       │
│                                                                     │
│ STEP 2: GATHER NEIGHBOR FEATURES                                     │
│   k_n = gather_neighbors(k, knn_idx)   → (B, N, k, n_heads, h_dim) │
│   v_n = gather_neighbors(v, knn_idx)   → (B, N, k, n_heads, h_dim) │
│   pos_n = gather_neighbors(pos, knn_idx) → (B, N, k, 3)             │
│   arc_n = gather_neighbors(arc, knn_idx)  → (B, N, k, 1)            │
│                                                                     │
│ STEP 3: POSITIONAL ENCODING (ARCH-AWARE)                            │
│   Δxyz  = pos_q - pos_n                        → (B, N, k, 3)      │
│   Δarc  = arc_q - arc_n                        → (B, N, k, 1)      │
│   pos_enc_input = concat([Δxyz, Δarc])         → (B, N, k, 4)      │
│   pos_enc = Linear(4→h_dim) · GELU · Linear(h_dim→h_dim)           │
│                                                                     │
│ STEP 4: ATTENTION WEIGHTS                                            │
│   attn_input = concat([q - k_n, pos_enc])     → (B,N,k,n_heads,2h) │
│   attention = Linear(2h→h) · GELU · Linear(h→h)                    │
│   attn_w = softmax(attention, dim=2)          → (B,N,k,n_heads,h)  │
│                                                                     │
│ STEP 5: AGGREGATE                                                    │
│   out = Σⱼ(attn_wⱼ · v_nⱼ)                   → (B,N,n_heads,h)    │
│   out = reshape(out, B, N, d_model)                                 │
│   out = Linear(d_model→d_model) · Dropout                           │
└─────────────────────────────────────────────────────────────────────┘

Visualization of the position encoding difference:
                     ┌──────────────┐
                     │  Standard    │     ┌──────────────┐
                     │  PVTv2 pos   │     │  DentalMAN   │
                     │  encoding:   │     │  pos encoding│
                     │  Δxyz (3)    │     │  Δxyz (3)    │
                     │              │     │  + Δarc (1)  │
                     │              │     │  = 4 inputs  │
                     └──────────────┘     └──────────────┘

  Without Δarc: two points at same spatial offset but on different
  arch sides get IDENTICAL positional encoding.
  With Δarc: the arch-side difference appears in pos_enc → network
  can distinguish "same geometry, different arch position."
```

### Complexity
- **Grouped vector attention**: O(N · k · d) where k ≪ N (k=16-24, N=128-8192)
- **Euclidean attention**: O(N² · d)
- **DentalMAN**: O(N · k · d) — same as PVTv2, no asymptotic overhead

---

## 3.4 `EncoderStage` — Hierarchical Encoding with FPS + KNN + Attention
**File:** `networks/DentalMAN.py` class `EncoderStage`, lines 249-272

### Complete Forward Pass Diagram

```
Input:  x (B, N, d_in), pos (B, N, 3), arc (B, N, 1)
Output: x_new (B, N/npoint, d_out), pos_new, arc_new

                    ┌─────────┐
                    │  INPUT  │
                    │ (B,N,3) │
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │   FPS   │  furthest_point_sample()
                    │ N→N/4   │  → fps_idx (B, N/4)
                    └────┬────┘
                         │
              ┌──────────┼──────────┐
              │          │          │
         ┌────▼───┐ ┌───▼────┐ ┌───▼────┐
         │ index  │ │ index  │ │ index  │
         │ points │ │ points │ │ points │
         │ x→x_new│ │pos→new │ │arc→new │
         │(B,N/4) │ │(B,N/4) │ │(B,N/4) │
         └────┬───┘ └───┬────┘ └───┬────┘
              │         │          │
         ┌────▼─────────▼──────────▼────┐
         │       LayerNorm(x_new)       │
         │   Linear(d_in→d_out) if dim  │
         │        mismatch              │
         └────────────┬─────────────────┘
                      │
                 ┌────▼────┐
                 │Hybrid   │
                 │KNN      │  → knn_idx (B, N/4, k)
                 │pos, arc │
                 └────┬────┘
                      │
              ┌───────▼────────┐
              │ ManifoldVecAttn │  × n_blocks
              │    Block       │  (with stochastic depth)
              └───────┬────────┘
                      │
                  ┌───▼───┐
                  │OUTPUT │
                  │x_new  │
                  │pos_new│
                  │arc_new│
                  └───────┘

Point count per encoder stage (N=8192):
  Stage 1:  8192 → 2048   (N/4)    k=24   2 blocks
  Stage 2:  2048 →  512   (N/16)   k=20   2 blocks
  Stage 3:   512 →  128   (N/64)   k=16   3 blocks
```

---

## 3.5 `ToothQueryCrossAttention` — Per-Class Bottleneck
**File:** `networks/DentalMAN.py` class `ToothQueryCrossAttention`, lines 274-302

### Why 53 Queries?
Each query corresponds to one FDI tooth class. The network learns to "look for" each tooth type across the entire point cloud.

### Architecture

```
                    tooth_embeddings (1, 53, 512)
                             │
                         expand(B)
                             │
                         Q = W_q·emb   (B, 53, 512)
                   ┌────────┤
                   │        │
    x3 (B,128,512) │  K = W_k·x3   (B, 128, 512)
                   │  V = W_v·x3   (B, 128, 512)
                   │        │
                   └────────┤
                            │
              ┌─────────────▼──────────────┐
              │  attn_out = softmax(QKᵀ/√d) │
              │               · V           │
              │            (B, 53, 512)     │
              └─────────────┬──────────────┘
                            │
                    LayerNorm(queries + attn_out)
                            │
                    FFN(LayerNorm(...))
                            │
                    tooth_context (B, 53, 512)

  Visual interpretation of what each query learns:
  ┌──────────────────────────────────────────────────┐
  │  Query 0: "Find gingiva (background)"            │
  │    → attends to large continuous low-lying areas │
  │                                                   │
  │  Query 1: "Find FDI 11 (upper right central)"    │
  │    → attends to upper-right quadrant, incisor    │
  │                                                   │
  │  Query 23: "Find FDI 37 (lower left 2nd molar)" │
  │    → attends to lower-left quadrant, molar shape │
  │                                                   │
  │  Query 48: "Find FDI 81 (lower right deciduous)" │
  │    → attends to lower-right, small tooth profile │
  └──────────────────────────────────────────────────┘
```

---

## 3.6 `ToothContextDecoder` — Per-Point Query Attention
**File:** `networks/DentalMAN.py` class `ToothContextDecoder`, lines 304-317

### Why Learned Attention Instead of Mean?
The original design broadcasts `tooth_context.mean(dim=1)` to all decoder points, collapsing 53 class-specific vectors into one. This loses which tooth class is relevant where.

The optimized design learns **per-point attention weights** over the 53 queries:

```
Input:  tooth_context (B, 53, 512)
        interp_feat   (B, N_dec, d_concat)
Output: context       (B, N_dec, 512)

┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Compute query attention scores per point            │
│   attn_logits = Linear(d_concat → 128) · GELU               │
│               · Linear(128 → 53)                           │
│   → (B, N_dec, 53)  ← one score per tooth class            │
│                                                             │
│ STEP 2: Softmax over classes                                │
│   attn_weights = softmax(attn_logits, dim=-1)               │
│   → (B, N_dec, 53)  ← "how relevant each tooth is here"    │
│                                                             │
│ STEP 3: Weighted sum of tooth contexts                      │
│   context = bmm(attn_weights, tooth_context)                │
│   → (B, N_dec, 512)                                         │
│                                                             │
│ Example: A point on upper central incisor:                  │
│   attn_weights = [0.01, 0.85, 0.02, ..., 0.01]             │
│                  ↑background ↑FDI 11  ↑others              │
│   → context dominated by query 1 (FDI 11 features)         │
└─────────────────────────────────────────────────────────────┘
```

---

## 3.7 `DualStreamEncoder` — Geometry + Appearance Fusion
**File:** `networks/DentalMAN.py` class `DualStreamEncoder`, lines 309-337

### Stream Comparison

```
                    ┌───────────────────────┐
                    │     6D INPUT          │
                    │ pos (xyz) + x (rgb)   │
                    │ + arc_params          │
                    └───────┬───────┬───────┘
                            │       │
              ┌─────────────▼┐     ┌▼──────────────┐
              │  GEOMETRY    │     │  APPEARANCE    │
              │  STREAM      │     │  STREAM        │
              │              │     │                │
              │ Input (4):   │     │ Input (3):     │
              │  xyz + arc   │     │  rgb           │
              │              │     │                │
              │ Linear 4→64  │     │ Conv1d 3→64   │
              │ GELU         │     │ LayerNorm      │
              │ Linear 64→64 │     │ GELU           │
              │ GELU         │     │ Conv1d 64→64  │
              │              │     │ LayerNorm      │
              │              │     │ GELU           │
              │              │     │ Conv1d 64→64  │
              │              │     │ LayerNorm      │
              │              │     │ GELU           │
              │ Output (64)  │     │ Output (64)    │
              └──────┬───────┘     └───────┬────────┘
                     │                     │
                     └──────┬──────────────┘
                            │
                      ┌─────▼──────┐
                      │  FUSION    │
                      │ concat(64+ │
                      │ 64) → 128 │
                      │ Linear(128 │
                      │ → 128)    │
                      │ GELU      │
                      └─────┬──────┘
                            │
                     fused features (B, N, 128)

  GEOMETRY learns: "Where am I on the arch?"
    → How far along (arc), how far from center (xyz)

  APPEARANCE learns: "What color/texture surrounds me?"
    → Tooth vs gingiva (pink vs white), enamel vs decay
```

---

## 3.8 `MultiScaleSegHead` — Hierarchical Feature Aggregation
**File:** `networks/DentalMAN.py` class `MultiScaleSegHead`, lines 341-355

### Why Multi-Scale?
- Decoder 1 features (256-dim): capture large-scale arch structure (gingiva, whole quadrants)
- Decoder 2 features (128-dim): capture tooth-group patterns (molars vs incisors)
- Decoder 3 features (64-dim): capture fine boundary details (tooth margins, interproximal gaps)

```
              ┌────── d1 ──────┐  256-dim
              │  (coarse)      │
              │                │
              │ ───── d2 ──── │  128-dim
              │ (medium)       │
              │                │
              │  ─── d3 ────  │   64-dim
              │   (fine)       │
              └────────────────┘
                      │
                 concat(1)
                      │
                 896-dim
                      │
              ┌───────▼────────┐
              │ Conv1d(896,256) │  ← compress
              │ BN, ReLU, DO   │
              ├────────────────┤
              │ Conv1d(256,128) │  ← refine
              │ BN, ReLU, DO   │
              ├────────────────┤
              │ Conv1d(128,53) │  ← classify
              └───────┬────────┘
                      │
              per-point logits (B, N, 53)
```

---

# 4. Training Configuration

## 4.1 Loss Function

```
L_total = L_focal(main, γ=2.0)
        + 0.4 · L_focal(aux_decoder2, γ=2.0)
        + 0.3 · L_focal(aux_decoder3, γ=2.0)
```

Where:
```
L_focal(pred, target) = Σ -(1 - pt)^γ · log(pt) · α_target
  pt  = softmax(pred)[correct_class]
  γ   = 2.0 (focus on hard examples)
  α_c = weight(c) = 1 / log(1.02 + count(c))  (inverse frequency)
```

Deep supervision applies focal loss at two decoder stages, forcing intermediate features to be discriminative.

## 4.2 Optimizer

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingWarmRestarts(T_0=20, T_mult=2, eta_min=1e-6) |
| AMP | torch.cuda.amp.GradScaler + autocast |
| Batch size | 4 (configurable) |
| Epochs | 200 |

## 4.3 Data Augmentation

Applied in `ObjDataset_Arch.__getitem__`:
1. **Rotation**: Z-axis ±10° with 50% probability
2. **Scaling**: Anisotropic 0.9–1.1 per axis
3. **Color jitter**: RGB ±0.1 uniform
4. **Elastic distortion**: via `torch_points3d` granularity=[0.2,0.8], magnitude=[0.4,1.6]
5. **Subsampling**: Random choice of `num_points` from available vertices
6. **Normalization**: Zero-centering + unit sphere scaling

## 4.4 Regularization

| Technique | Value | Applied At |
|-----------|-------|------------|
| Stochastic depth | 0.05 (linear schedule) | Each ManifoldVecAttnBlock |
| Dropout | 0.1 | Attention FFN, positional MLP |
| Dropout | 0.3 → 0.1 | Segmentation head |
| Weight decay | 1e-4 | All parameters |
| Label smoothing | ε=0.1 | Loss computation |

---

# 5. Optimization Summary (vs Initial Implementation)

| Aspect | Initial Version | Optimized Version |
|--------|----------------|-------------------|
| **KNN memory** | O(N²) — all-pair distance matrix | O(N·chunk_size) — chunked for N > 2048 |
| **Attention gather** | 5 separate gather ops, dead tensors | Single `gather_neighbors()` helper, no waste |
| **Tooth context** | `.mean(dim=1)` collapses 53→1 | Learned per-point attention weights |
| **Arch fitting** | `polyfit(deg=5)` — oscillates | Piecewise segment averaging — monotonic |
| **Geometry encoder** | Only 1D arc input | 4D input: xyz(3) + arc(1) |
| **Normalization** | Missing in encoder/decoder | LayerNorm before every MLP |
| **Weight init** | `kaiming_normal_(relu)` — mismatch with GELU | `trunc_normal_(std=0.02)` — transformer standard |
| **Dropout** | Missing in attention MLPs | Dropout in pos_mlp and attn_mlp |

---

# 6. Comparison to Current HybridTransformer Baseline

| Property | HybridTransformer | DentalMAN | Expected Impact |
|----------|------------------|-----------|-----------------|
| Neighbor grouping | Euclidean ball query | Arch-distance KNN | Better boundary at arch crossings |
| Bottleneck | Self-attention (64 pts) | 53-class cross-attention | Better rare-class recall |
| Position encoding | None (implicit) | Δxyz + Δarc MLP | Arch-aware feature grouping |
| Multi-scale head | No | Yes (3 decoder scales) | Better gingiva + fine detail |
| Domain priors | None | Arch curve + FDI queries | Structured dental knowledge |
| Deep supervision | No | Aux heads on 2 decoders | Faster convergence |
| Input features | xyz + rgb | xyz + rgb + arc-length | Extra geometric prior |

---

# 7. References

1. **Point Transformer v2** — Wu et al., "Grouped Vector Attention and Partition-based Pooling for Point Cloud Understanding", CVPR 2022
2. **PointNet++** — Qi et al., "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space", NeurIPS 2017
3. **DETR** — Carion et al., "End-to-End Object Detection with Transformers", ECCV 2020
4. **PointNeXt** — Qian et al., "PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies", NeurIPS 2022
5. **THISNet** — Li et al., "Tooth Instance Segmentation on 3D Dental Models via Highlighting Tooth Regions", IEEE TCSVT 2023
