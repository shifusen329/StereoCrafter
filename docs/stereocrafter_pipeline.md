# StereoCrafter Pipeline Documentation

## Overview

StereoCrafter converts monocular (2D) video to stereoscopic (3D) side-by-side video through a two-stage pipeline:

1. **Stage 1: Depth Splatting** - Estimate depth and warp left eye to create initial right eye
2. **Stage 2: Stereo Inpainting** - Fill in disoccluded regions using diffusion model

---

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           run_inference.sh                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  INPUT: video.mp4                                                            │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STAGE 1: depth_splatting_inference.py                               │    │
│  │                                                                      │    │
│  │  1. Load video frames                                                │    │
│  │  2. DepthCrafter: estimate per-frame depth maps                     │    │
│  │  3. ForwardWarpStereo: warp left→right using depth                  │    │
│  │  4. Generate occlusion mask                                          │    │
│  │  5. Output: 2x2 grid video (temp_splatting.mp4)                     │    │
│  │     ┌─────────────┬─────────────┐                                    │    │
│  │     │ Left Eye    │ Depth Vis   │                                    │    │
│  │     ├─────────────┼─────────────┤                                    │    │
│  │     │ Occlu Mask  │ Warped Right│                                    │    │
│  │     └─────────────┴─────────────┘                                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STAGE 2: inpainting_inference.py                                    │    │
│  │                                                                      │    │
│  │  1. Parse 2x2 grid → extract left, mask, warped                     │    │
│  │  2. Process in temporal chunks (frames_chunk=23, overlap=3)         │    │
│  │  3. For each chunk:                                                  │    │
│  │     a. Spatial tiled processing (tile_num controls memory)          │    │
│  │     b. StableVideoDiffusionInpaintingPipeline                       │    │
│  │     c. Blend overlapping regions                                     │    │
│  │  4. Decode latents → RGB frames                                      │    │
│  │  5. Concatenate left + inpainted_right → SBS                        │    │
│  │  6. Output: video_180_sbs.mp4                                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  OUTPUT: video_180_sbs.mp4 (Side-by-Side stereoscopic)                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Depth Splatting (`depth_splatting_inference.py`)

### 1.1 Video Loading
```python
# Read video, resize to nearest multiple of 64, cap at max_res (1024)
frames, fps = read_video_frames(input_video_path, process_length, target_fps, max_res)
# frames shape: [T, H, W, 3] normalized to [0, 1]
```

### 1.2 Depth Estimation (DepthCrafter)
```python
# Uses DepthCrafterPipeline (modified Stable Video Diffusion)
# Model: weights/DepthCrafter (fine-tuned UNet)
# Base: weights/stable-video-diffusion-img2vid-xt-1-1

res = self.pipe(
    frames,
    guidance_scale=1.2,          # CFG scale
    num_inference_steps=8,       # Denoising steps
    window_size=70,              # Temporal window
    overlap=25,                  # Temporal overlap
)
# Output: depth maps [T, H, W] normalized to [0, 1]
```

### 1.3 Forward Warping (Left → Right Eye)
```python
# ForwardWarpStereo: depth-based image warping
# Disparity = depth * max_disp (default max_disp=20.0)

disp_map = depth * 2.0 - 1.0  # Normalize to [-1, 1]
disp_map = disp_map * max_disp  # Scale by max disparity

right_video, occlusion_mask = stereo_projector(left_video, disp_map)
# right_video: warped right eye (has holes where occluded)
# occlusion_mask: 1.0 where pixels are missing (need inpainting)
```

### 1.4 Output Format
```python
# Writes 2x2 grid video at 2x resolution:
# ┌─────────────┬─────────────┐
# │ Left Eye    │ Depth Vis   │  (for debugging)
# ├─────────────┼─────────────┤
# │ Occlu Mask  │ Warped Right│  (used by Stage 2)
# └─────────────┴─────────────┘
```

---

## Stage 2: Stereo Inpainting (`inpainting_inference.py`)

### 2.1 Parse Input Grid
```python
# Read 2x2 grid from Stage 1
frames_left = frames[:, :, :height, :width]      # Top-left
frames_mask = frames[:, :, height:, :width]      # Bottom-left (occlusion)
frames_warpped = frames[:, :, height:, width:]   # Bottom-right (warped)
```

### 2.2 Temporal Chunking
```python
# Process video in overlapping chunks for memory efficiency
frames_chunk = 23  # Frames per chunk
overlap = 3        # Overlap between chunks

for i in range(0, num_frames, frames_chunk - overlap):
    # Process chunk with overlap blending
```

### 2.3 Spatial Tiled Processing
```python
# tile_num controls memory usage (default=1, increase for high-res)
# tile_overlap = (128, 128) pixels

spatial_tiled_process(
    cond_frames,      # Warped right eye (condition)
    mask_frames,      # Occlusion mask
    pipeline,         # StableVideoDiffusionInpaintingPipeline
    tile_num,         # 1=no tiling, 2=2x2 tiles, etc.
)
```

### 2.4 Diffusion Inpainting
```python
# StableVideoDiffusionInpaintingPipeline
# Model: weights/StereoCrafter (fine-tuned for stereo inpainting)
# Base: weights/stable-video-diffusion-img2vid-xt-1-1

pipeline(
    frames=warped_right,
    frames_mask=occlusion_mask,
    min_guidance_scale=1.01,
    max_guidance_scale=1.01,
    num_inference_steps=8,
    fps=7,
    motion_bucket_id=127,
    noise_aug_strength=0.0,
)
```

### 2.5 Output Assembly
```python
# Concatenate left (original) + right (inpainted) → SBS
frames_sbs = torch.cat([frames_left, frames_output], dim=3)
# Also generates anaglyph (red/cyan) for quick preview
```

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_disp` | 20.0 | Maximum stereo disparity (depth → shift) |
| `max_res` | 1024 | Max resolution (larger dimension) |
| `frames_chunk` | 23 | Frames per temporal chunk |
| `overlap` | 3 | Temporal overlap between chunks |
| `tile_num` | 2 | Spatial tiling (1=none, 2=2x2) |
| `num_inference_steps` | 8 | Diffusion denoising steps |
| `guidance_scale` | 1.2 | CFG scale for depth estimation |

---

## Memory Bottlenecks

### 1. Video Loading (32-bit Index Limit)
```python
# FAILS when: T × H × W × C > 2^31 (~2.1 billion)
# Example: 1372 × 640 × 960 × 3 = 2.5B ❌
frames = vid.get_batch(frames_idx).asnumpy()
```

### 2. Depth Estimation
```python
# Full video tensor loaded for depth estimation
video_224 = _resize_with_antialiasing(video.float(), (224, 224))
```

### 3. Solutions
- Split long videos into chunks < 1000 frames
- Reduce resolution
- Increase `tile_num` for spatial tiling

---

## Known Issues

### 1. Color Mismatch (Left vs Right Eye)
- Inpainting model generates colors independently
- No explicit color consistency loss
- **Fix**: Post-process histogram matching

### 2. Inpainting Artifacts
- Sky/uniform regions prone to artifacts
- Edge bleeding near occlusion boundaries
- **Fix**: Increase `num_inference_steps`, adjust `max_disp`

### 3. Temporal Flickering
- Chunk boundaries may have discontinuities
- **Fix**: Increase `overlap`, post-process temporal smoothing

---

## File Locations

```
StereoCrafter/
├── run_inference.sh                 # Entry point
├── depth_splatting_inference.py     # Stage 1
├── inpainting_inference.py          # Stage 2
├── pipelines/
│   └── stereo_video_inpainting.py   # Custom pipeline
├── dependency/
│   └── DepthCrafter/                # Depth estimation
└── weights/
    ├── stable-video-diffusion-img2vid-xt-1-1/  # Base model
    ├── DepthCrafter/                # Depth UNet
    └── StereoCrafter/               # Inpainting UNet
```
