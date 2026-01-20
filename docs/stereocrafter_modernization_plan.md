# StereoCrafter Modernization Plan (2026)

## Target Stack
- **PyTorch**: 2.9.1
- **CUDA**: 13.0
- **Flash Attention**: 2.8.3
- **Diffusers**: 0.32+ (latest)
- **GPU**: RTX PRO 6000 Blackwell / B200

---

## Phase 1: Dependency Updates

### 1.1 Core Dependencies
```toml
# pyproject.toml
[project]
dependencies = [
    "torch>=2.9.0",
    "torchvision>=0.20.0",
    "diffusers>=0.32.0",
    "transformers>=4.48.0",
    "accelerate>=1.2.0",
    "flash-attn>=2.8.0",
    "decord>=0.6.0",
    "opencv-python>=4.10.0",
    "fire>=0.5.0",
]
```

### 1.2 Remove Deprecated Dependencies
- ❌ `xformers` - replaced by native SDPA + Flash Attention
- ❌ `Forward_Warp` - rewrite with pure PyTorch for compatibility

---

## Phase 2: Attention Modernization

### 2.1 Current State (Broken)
```python
# depth_splatting_inference.py (line 95)
self.pipe.enable_attention_slicing()  # Slow fallback
# xformers disabled due to "Blackwell incompatibility"
```

### 2.2 Target State
```python
# Option A: Native PyTorch SDPA (automatic Flash Attention)
# PyTorch 2.9 auto-selects best backend (Flash, Memory-Efficient, Math)
# No code changes needed - just remove enable_attention_slicing()

# Option B: Explicit Flash Attention 2
from diffusers import AttnProcessor2_0
pipe.unet.set_attn_processor(AttnProcessor2_0())

# Option C: Direct flash-attn integration
from flash_attn import flash_attn_func
# Requires custom attention processor
```

### 2.3 Implementation
```python
# New initialization in depth_splatting_inference.py
class DepthCrafterDemo:
    def __init__(self, ...):
        # ... load models ...

        # Enable Flash Attention via SDPA (PyTorch 2.9+)
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # SDPA will auto-select Flash Attention on compatible hardware
            print(f"Using PyTorch SDPA (Flash: {torch.backends.cuda.flash_sdp_enabled()})")

        # Optional: Force Flash Attention processor
        if torch.cuda.get_device_capability()[0] >= 8:  # Ampere+
            from diffusers import AttnProcessor2_0
            self.pipe.unet.set_attn_processor(AttnProcessor2_0())
            print("Using AttnProcessor2_0 (Flash Attention)")

        # Remove slow fallback
        # self.pipe.enable_attention_slicing()  # DELETE THIS
```

---

## Phase 3: Memory Optimization

### 3.1 Fix 32-bit Index Limit
```python
# Problem: Loading full video exceeds 2^31 elements
# Solution: Chunked video loading

def read_video_frames_chunked(video_path, chunk_size=500, ...):
    """Load video in chunks to avoid 32-bit index overflow."""
    vid = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vid)

    for start in range(0, total_frames, chunk_size):
        end = min(start + chunk_size, total_frames)
        chunk = vid.get_batch(list(range(start, end))).asnumpy()
        yield chunk, start, end
```

### 3.2 Gradient Checkpointing
```python
# Reduce memory during inference
pipe.unet.enable_gradient_checkpointing()
```

### 3.3 VAE Slicing
```python
# Process VAE in slices for high-res
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
```

### 3.4 Model Offloading (Optional)
```python
# For limited VRAM scenarios
pipe.enable_sequential_cpu_offload()
# Or
pipe.enable_model_cpu_offload()
```

---

## Phase 4: torch.compile() Integration

### 4.1 Compile UNet for Speed
```python
# 20-40% speedup on Blackwell
import torch

# Compile with max-autotune for best performance
pipe.unet = torch.compile(
    pipe.unet,
    mode="max-autotune",
    fullgraph=True,
)

# Or reduce-overhead for faster compilation
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
```

### 4.2 Compile VAE
```python
pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune")
```

---

## Phase 5: Rewrite Forward_Warp

### 5.1 Current Issue
```
Forward_Warp-0.0.1 has SyntaxWarnings and potential compatibility issues
```

### 5.2 Pure PyTorch Replacement
```python
import torch
import torch.nn.functional as F

def forward_warp_pytorch(img, flow):
    """
    Pure PyTorch forward warping.

    Args:
        img: [B, C, H, W] source image
        flow: [B, H, W, 2] optical flow (dx, dy)

    Returns:
        warped: [B, C, H, W] warped image
    """
    B, C, H, W = img.shape

    # Create base grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=img.device),
        torch.arange(W, device=img.device),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=-1).float()
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)

    # Add flow to grid
    new_grid = grid + flow

    # Normalize to [-1, 1]
    new_grid[..., 0] = 2 * new_grid[..., 0] / (W - 1) - 1
    new_grid[..., 1] = 2 * new_grid[..., 1] / (H - 1) - 1

    # Warp using grid_sample
    warped = F.grid_sample(
        img, new_grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )

    return warped


class ForwardWarpStereo(torch.nn.Module):
    """Pure PyTorch stereo warping."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, img, disp):
        """
        Warp image using disparity (horizontal shift only).

        Args:
            img: [B, C, H, W]
            disp: [B, 1, H, W] disparity map

        Returns:
            warped: [B, C, H, W]
            occlusion: [B, 1, H, W] occlusion mask
        """
        B, C, H, W = img.shape

        # Create horizontal flow from disparity
        flow = torch.zeros(B, H, W, 2, device=img.device)
        flow[..., 0] = -disp.squeeze(1)  # Horizontal shift

        # Forward warp with splatting
        warped = forward_warp_pytorch(img, flow)

        # Compute occlusion mask (areas that weren't filled)
        ones = torch.ones_like(disp)
        coverage = forward_warp_pytorch(ones, flow)
        occlusion = (coverage < self.eps).float()

        return warped, occlusion
```

---

## Phase 6: Color Consistency Fix

### 6.1 Add L/R Color Matching
```python
def match_colors_lab(source, reference):
    """Match source colors to reference in LAB space."""
    import cv2

    # Convert to LAB
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(float)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(float)

    # Match mean and std per channel
    for i in range(3):
        src_mean, src_std = src_lab[:,:,i].mean(), src_lab[:,:,i].std()
        ref_mean, ref_std = ref_lab[:,:,i].mean(), ref_lab[:,:,i].std()

        src_lab[:,:,i] = (src_lab[:,:,i] - src_mean) * (ref_std / (src_std + 1e-6)) + ref_mean

    # Convert back
    src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(src_lab, cv2.COLOR_LAB2BGR)


# Add to inpainting_inference.py after line 278:
# frames_output = match_colors_batch(frames_output, frames_left)
```

---

## Phase 7: Implementation Checklist

### Stage 1 (depth_splatting_inference.py)
- [ ] Remove `enable_attention_slicing()`
- [ ] Add `AttnProcessor2_0` for Flash Attention
- [ ] Add chunked video loading
- [ ] Replace `Forward_Warp` with pure PyTorch
- [ ] Add `torch.compile()` for UNet
- [ ] Update diffusers API calls

### Stage 2 (inpainting_inference.py)
- [ ] Add `AttnProcessor2_0` for Flash Attention
- [ ] Add color matching post-process
- [ ] Add `torch.compile()` for UNet/VAE
- [ ] Update diffusers API calls

### Infrastructure
- [ ] Update requirements.txt / pyproject.toml
- [ ] Test on Blackwell (RTX PRO 6000)
- [ ] Test on B200 (RunPod)
- [ ] Benchmark speedup vs original
- [ ] Document new parameters

---

## Expected Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Attention | Slicing (slow) | Flash Attention 2 | ~2-3x faster |
| Compile | None | torch.compile | ~1.3-1.5x faster |
| Max frames | ~1100 | Unlimited (chunked) | No limit |
| Color match | None | LAB matching | Fixed artifacts |
| VRAM (1080p) | ~24GB | ~16GB | ~33% reduction |

---

## Timeline Estimate

| Phase | Tasks | Duration |
|-------|-------|----------|
| 1 | Dependency updates | 1 day |
| 2 | Attention modernization | 1 day |
| 3 | Memory optimization | 2 days |
| 4 | torch.compile | 1 day |
| 5 | Forward_Warp rewrite | 2 days |
| 6 | Color consistency | 1 day |
| 7 | Testing & docs | 2 days |
| **Total** | | **~10 days** |
