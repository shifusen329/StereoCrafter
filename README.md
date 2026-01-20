
<div align="center">
<h2>StereoCrafter: Diffusion-based Generation of Long and High-fidelity Stereoscopic 3D from Monocular Videos</h2>

Sijie Zhao*&emsp;
Wenbo Hu*&emsp;
Xiaodong Cun*&emsp;
Yong Zhang&dagger;&emsp;
Xiaoyu Li&dagger;&emsp;<br>
Zhe Kong&emsp;
Xiangjun Gao&emsp;
Muyao Niu&emsp;
Ying Shan

&emsp;* equal contribution &emsp; &dagger; corresponding author

<h3>Tencent AI Lab&emsp;&emsp;ARC Lab, Tencent PCG</h3>

<a href='https://arxiv.org/abs/2409.07447'><img src='https://img.shields.io/badge/arXiv-PDF-a92225'></a> &emsp;
<a href='https://stereocrafter.github.io/'><img src='https://img.shields.io/badge/Project_Page-Page-64fefe' alt='Project Page'></a> &emsp;
<a href='https://huggingface.co/TencentARC/StereoCrafter'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights-yellow'></a>
</div>

## 💡 Abstract

We propose a novel framework to convert any 2D videos to immersive stereoscopic 3D ones that can be viewed on different display devices, like 3D Glasses, Apple Vision Pro and 3D Display. It can be applied to various video sources, such as movies, vlogs, 3D cartoons, and AIGC videos.

![teaser](assets/teaser.jpg)

## 📣 News
- `2025/01` Modernized for PyTorch 2.9+, Flash Attention 2, and Blackwell GPUs. Pure PyTorch forward warping (no CUDA compilation needed).
- `2024/12/27` We released our inference code and model weights.
- `2024/09/11` We submitted our technical report on arXiv and released our project page.

## 🎞️ Showcases
Here we show some examples of input videos and their corresponding stereo outputs in Anaglyph 3D format.
<div align="center">
    <img src="assets/demo.gif">
</div>


## 🛠️ Installation

#### 1. Requirements
- Python 3.11+
- CUDA 12.x
- PyTorch 2.9+
- Flash Attention 2.8+ (optional, for faster inference)

#### 2. Clone the repo
```bash
git clone --recursive https://github.com/TencentARC/StereoCrafter
cd StereoCrafter
```

#### 3. Install dependencies

**Using uv (recommended):**
```bash
uv sync
```

**Using pip:**
```bash
pip install -e .
```

**Install Flash Attention (optional, requires PyTorch first):**
```bash
pip install flash-attn --no-build-isolation
```


## 📦 Model Weights

#### 1. Download the [SVD img2vid model](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1) for the image encoder and VAE.

```bash
# in StereoCrafter project root directory
mkdir weights
cd ./weights
git lfs install
git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1
```

#### 2. Download the [DepthCrafter model](https://huggingface.co/tencent/DepthCrafter) for the video depth estimation.
```bash
git clone https://huggingface.co/tencent/DepthCrafter
```

#### 3. Download the [StereoCrafter model](https://huggingface.co/TencentARC/StereoCrafter) for the stereo video generation.
```bash
git clone https://huggingface.co/TencentARC/StereoCrafter
```


## 🔄 Inference

**Quick start:**
```bash
./run_inference.sh ./source_video/your_video.mp4
```

Or process a directory of videos:
```bash
./run_inference.sh ./source_video/
```

### Manual Two-Stage Process

#### 1. Depth-Based Video Splatting Using the Video Depth from DepthCrafter
```bash
python depth_splatting_inference.py \
    --pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1 \
    --unet_path ./weights/DepthCrafter \
    --input_video_path ./source_video/camel.mp4 \
    --output_video_path ./outputs/camel_splatting_results.mp4
```

Arguments:
- `--pre_trained_path`: Path to the SVD img2vid model weights.
- `--unet_path`: Path to the DepthCrafter model weights.
- `--input_video_path`: Path to the input video.
- `--output_video_path`: Path to the output video.
- `--max_disp`: Maximum disparity between left/right video. Default: `20` pixels.

The first step generates a video grid with input video, visualized depth map, occlusion mask, and splatting right video:

<img src="assets/camel_splatting_results.jpg" alt="camel_splatting_results" width="800"/>

#### 2. Stereo Video Inpainting of the Splatting Video
```bash
python inpainting_inference.py \
    --pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1 \
    --unet_path ./weights/StereoCrafter \
    --input_video_path ./outputs/camel_splatting_results.mp4 \
    --save_dir ./outputs
```

Arguments:
- `--pre_trained_path`: Path to the SVD img2vid model weights.
- `--unet_path`: Path to the StereoCrafter model weights.
- `--input_video_path`: Path to the splatting video from stage 1.
- `--save_dir`: Directory for the output stereo video.
- `--tile_num`: Tiles for memory-efficient processing. Default: `1`. Use `2` for 2K+ resolution.

The stereo video inpainting generates the stereo video result in side-by-side format and anaglyph 3D format:

<img src="assets/camel_sbs.jpg" alt="camel_sbs" width="800"/>

<img src="assets/camel_anaglyph.jpg" alt="camel_anaglyph" width="400"/>

## 🤝 Acknowledgements

We would like to express our gratitude to the following open-source projects:
- [Stable Video Diffusion](https://github.com/Stability-AI/generative-models): A latent diffusion model trained to generate video clips from an image or text conditioning.
- [DepthCrafter](https://github.com/Tencent/DepthCrafter): A novel method to generate temporally consistent depth sequences from videos.


## 📚 Citation

```bibtex
@article{zhao2024stereocrafter,
  title={Stereocrafter: Diffusion-based generation of long and high-fidelity stereoscopic 3d from monocular videos},
  author={Zhao, Sijie and Hu, Wenbo and Cun, Xiaodong and Zhang, Yong and Li, Xiaoyu and Kong, Zhe and Gao, Xiangjun and Niu, Muyao and Shan, Ying},
  journal={arXiv preprint arXiv:2409.07447},
  year={2024}
}
```
