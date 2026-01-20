import sys
# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

import gc
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import write_video

from diffusers.training_utils import set_seed
from fire import Fire
from decord import VideoReader, cpu

from dependency.DepthCrafter.depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from dependency.DepthCrafter.depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from dependency.DepthCrafter.depthcrafter.utils import vis_sequence_depth

def read_video_frames_chunked(vid, frames_idx, chunk_size=500):
    """
    Load video frames in chunks to avoid 32-bit index overflow.
    For videos with T*H*W*C > 2^31 elements, loading all at once fails.
    """
    chunks = []
    for start in range(0, len(frames_idx), chunk_size):
        end = min(start + chunk_size, len(frames_idx))
        chunk_indices = frames_idx[start:end]
        chunk = vid.get_batch(chunk_indices).asnumpy().astype("float32") / 255.0
        chunks.append(chunk)
    return np.concatenate(chunks, axis=0)


def read_video_frames(video_path, process_length, target_fps, max_res, dataset="open"):
    if dataset == "open":
        print("==> processing video: ", video_path)
        vid = VideoReader(video_path, ctx=cpu(0))
        print("==> original video shape: ", (len(vid), *vid.get_batch([0]).shape[1:]))
        original_height, original_width = vid.get_batch([0]).shape[1:3]
        height = round(original_height / 64) * 64
        width = round(original_width / 64) * 64
        if max(height, width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = round(original_height * scale / 64) * 64
            width = round(original_width * scale / 64) * 64
    else:
        height = dataset_res_dict[dataset][0]
        width = dataset_res_dict[dataset][1]

    vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)

    fps = vid.get_avg_fps() if target_fps == -1 else target_fps
    stride = round(vid.get_avg_fps() / fps)
    stride = max(stride, 1)
    frames_idx = list(range(0, len(vid), stride))
    print(
        f"==> downsampled shape: {len(frames_idx), *vid.get_batch([0]).shape[1:]}, with stride: {stride}"
    )
    if process_length != -1 and process_length < len(frames_idx):
        frames_idx = frames_idx[:process_length]
    print(
        f"==> final processing shape: {len(frames_idx), *vid.get_batch([0]).shape[1:]}"
    )

    # Use chunked loading to avoid 32-bit index overflow on long/high-res videos
    frames = read_video_frames_chunked(vid, frames_idx, chunk_size=500)

    return frames, fps, original_height, original_width


class DepthCrafterDemo:
    def __init__(
        self,
        unet_path: str,
        pre_trained_path: str,
        cpu_offload: str = "model",
    ):
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        # load weights of other components from the provided checkpoint
        self.pipe = DepthCrafterPipeline.from_pretrained(
            pre_trained_path,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        # for saving memory, we can offload the model to CPU, or even run the model sequentially to save more memory
        if cpu_offload is not None:
            if cpu_offload == "sequential":
                # This will slow, but save more memory
                self.pipe.enable_sequential_cpu_offload()
            elif cpu_offload == "model":
                self.pipe.enable_model_cpu_offload()
            else:
                raise ValueError(f"Unknown cpu offload option: {cpu_offload}")
        else:
            self.pipe.to("cuda")
        # Enable Flash Attention via PyTorch SDPA (2.9+) and AttnProcessor2_0
        # This replaces the slow attention slicing fallback
        try:
            from diffusers.models.attention_processor import AttnProcessor2_0
            self.pipe.unet.set_attn_processor(AttnProcessor2_0())
            print(f"Flash Attention enabled via AttnProcessor2_0 (SDPA backend)")
        except Exception as e:
            print(f"AttnProcessor2_0 not available: {e}, falling back to default attention")

        # Enable VAE optimizations for memory efficiency (if available)
        if hasattr(self.pipe, 'enable_vae_slicing'):
            self.pipe.enable_vae_slicing()
        if hasattr(self.pipe, 'enable_vae_tiling'):
            self.pipe.enable_vae_tiling()

        # NOTE: torch.compile disabled for DepthCrafter - custom UNet causes CUDA graph conflicts
        # The inpainting stage still uses torch.compile for speedup

    def infer(
        self,
        input_video_path: str,
        output_video_path: str,
        process_length: int = -1,
        num_denoising_steps: int = 8,
        guidance_scale: float = 1.2,
        window_size: int = 70,
        overlap: int = 25,
        max_res: int = 1024,
        dataset: str = "open",
        target_fps: int = -1,
        seed: int = 42,
        track_time: bool = False,
        save_depth: bool = False,
    ):
        set_seed(seed)

        frames, target_fps, original_height, original_width = read_video_frames(
            input_video_path,
            process_length,
            target_fps,
            max_res,
            dataset,
        )

        # inference the depth map using the DepthCrafter pipeline
        print("==> Starting depth inference...")
        with torch.inference_mode():
            res = self.pipe(
                frames,
                height=frames.shape[1],
                width=frames.shape[2],
                output_type="np",
                guidance_scale=guidance_scale,
                num_inference_steps=num_denoising_steps,
                window_size=window_size,
                overlap=overlap,
                track_time=track_time,
            ).frames[0]
        print("==> Depth inference done, post-processing...")

        # convert the three-channel output to a single channel depth map
        res = res.sum(-1) / res.shape[-1]
        print("==> Converted to single channel")

        # resize the depth to the original size
        tensor_res = torch.tensor(res).unsqueeze(1).float().contiguous().cuda()
        res = F.interpolate(tensor_res, size=(original_height, original_width), mode='bilinear', align_corners=False)
        res = res.cpu().numpy()[:,0,:,:]
        print("==> Resized to original size")

        # normalize the depth map to [0, 1] across the whole video
        res = (res - res.min()) / (res.max() - res.min())
        print("==> Normalized depth map")
        # visualize the depth map and save the results
        vis = vis_sequence_depth(res)
        print("==> Visualization complete")
        # save the depth map and visualization with the target FPS
        save_path = os.path.join(
            os.path.dirname(output_video_path), os.path.splitext(os.path.basename(output_video_path))[0]
        )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_depth:
            np.savez_compressed(save_path + ".npz", depth=res)
            write_video(save_path + "_depth_vis.mp4", vis*255.0, fps=target_fps, video_codec="h264", options={"crf": "16"})

        return res, vis
    

class ForwardWarpStereo(nn.Module):
    """
    Pure PyTorch stereo warping using forward splatting.
    Replaces the CUDA Forward_Warp dependency for better compatibility.
    """
    def __init__(self, eps=1e-6, occlu_map=False):
        super(ForwardWarpStereo, self).__init__()
        self.eps = eps
        self.occlu_map = occlu_map

    def forward_warp_splat(self, img, flow):
        """
        Forward warp using splatting (scatter-add operation).

        Args:
            img: [B, C, H, W] source image
            flow: [B, H, W, 2] optical flow (dx, dy)

        Returns:
            warped: [B, C, H, W] forward-warped image
        """
        B, C, H, W = img.shape
        device = img.device
        dtype = img.dtype

        # Create base coordinate grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing='ij'
        )
        grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
        grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]

        # Compute target coordinates
        target_x = grid_x + flow[..., 0]  # [B, H, W]
        target_y = grid_y + flow[..., 1]  # [B, H, W]

        # Bilinear splatting - distribute each source pixel to 4 neighbors
        x0 = torch.floor(target_x).long()
        x1 = x0 + 1
        y0 = torch.floor(target_y).long()
        y1 = y0 + 1

        # Compute bilinear weights
        wa = (x1.float() - target_x) * (y1.float() - target_y)
        wb = (x1.float() - target_x) * (target_y - y0.float())
        wc = (target_x - x0.float()) * (y1.float() - target_y)
        wd = (target_x - x0.float()) * (target_y - y0.float())

        # Initialize output
        output = torch.zeros_like(img)
        weight_sum = torch.zeros(B, 1, H, W, device=device, dtype=dtype)

        # Flatten for scatter_add
        img_flat = img.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, H*W, C]

        for dx, dy, w in [(x0, y0, wa), (x0, y1, wb), (x1, y0, wc), (x1, y1, wd)]:
            # Clamp coordinates and create valid mask
            valid = (dx >= 0) & (dx < W) & (dy >= 0) & (dy < H)
            dx_clamped = dx.clamp(0, W - 1)
            dy_clamped = dy.clamp(0, H - 1)

            # Compute linear indices
            indices = (dy_clamped * W + dx_clamped).view(B, H * W)  # [B, H*W]

            # Apply validity mask to weights
            w_masked = w.view(B, H * W, 1) * valid.view(B, H * W, 1).float()  # [B, H*W, 1]

            # Scatter-add weighted pixels
            weighted_img = img_flat * w_masked  # [B, H*W, C]

            for b in range(B):
                output[b] = output[b].view(C, H * W).scatter_add(
                    1, indices[b:b+1].expand(C, -1), weighted_img[b].t()
                ).view(C, H, W)
                weight_sum[b, 0] = weight_sum[b, 0].view(H * W).scatter_add(
                    0, indices[b], w_masked[b, :, 0]
                ).view(H, W)

        return output, weight_sum

    def forward(self, im, disp):
        """
        :param im: BCHW
        :param disp: B1HW disparity map
        :return: BCHW warped image, optionally B1HW occlusion mask
        """
        im = im.contiguous()
        disp = disp.contiguous()

        # Compute depth-based weights (closer = higher weight)
        weights_map = disp - disp.min()
        weights_map = (1.414) ** weights_map  # Avoid exp overflow

        # Create flow from disparity (horizontal shift only)
        flow_x = -disp.squeeze(1)  # [B, H, W]
        flow_y = torch.zeros_like(flow_x)
        flow = torch.stack((flow_x, flow_y), dim=-1)  # [B, H, W, 2]

        # Forward warp weighted image and weights
        res_accum, _ = self.forward_warp_splat(im * weights_map, flow)
        mask, _ = self.forward_warp_splat(weights_map, flow)

        # Normalize by accumulated weights
        mask = mask.clamp(min=self.eps)
        res = res_accum / mask

        if not self.occlu_map:
            return res
        else:
            # Compute occlusion map (areas with no coverage)
            ones = torch.ones_like(disp)
            coverage, _ = self.forward_warp_splat(ones, flow)
            occlu_map = 1.0 - coverage.clamp(0.0, 1.0)
            return res, occlu_map
        

def DepthSplatting(
        input_video_path, 
        output_video_path, 
        video_depth, 
        depth_vis, 
        max_disp, 
        process_length, 
        batch_size):
    '''
    Depth-Based Video Splatting Using the Video Depth.
    Args:
        input_video_path: Path to the input video.
        output_video_path: Path to the output video.
        video_depth: Video depth with shape of [T, H, W] in [0, 1].
        depth_vis: Visualized video depth with shape of [T, H, W, 3] in [0, 1].
        process_length: The length of video to process.
        batch_size: The batch size for splatting to save GPU memory. 
    '''
    vid_reader = VideoReader(input_video_path, ctx=cpu(0))
    original_fps = vid_reader.get_avg_fps()
    # Use chunked loading to avoid 32-bit index overflow
    frame_indices = list(range(len(vid_reader)))
    input_frames = read_video_frames_chunked(vid_reader, frame_indices, chunk_size=500)

    if process_length != -1 and process_length < len(input_frames):
        input_frames = input_frames[:process_length]
        video_depth = video_depth[:process_length]
        depth_vis = depth_vis[:process_length]

    stereo_projector = ForwardWarpStereo(occlu_map=True).cuda()

    num_frames = len(input_frames)
    height, width, _ = input_frames[0].shape

    # Initialize OpenCV VideoWriter
    out = cv2.VideoWriter(
        output_video_path, 
        cv2.VideoWriter_fourcc(*"mp4v"),
        original_fps, 
        (width * 2, height * 2)
    )

    total_batches = (num_frames + batch_size - 1) // batch_size
    for batch_idx, i in enumerate(range(0, num_frames, batch_size)):
        print(f"    Processing batch {batch_idx + 1}/{total_batches}...", end='\r')
        batch_frames = input_frames[i:i+batch_size]
        batch_depth = video_depth[i:i+batch_size]
        batch_depth_vis = depth_vis[i:i+batch_size]

        left_video = torch.from_numpy(batch_frames).permute(0, 3, 1, 2).float().cuda()
        disp_map = torch.from_numpy(batch_depth).unsqueeze(1).float().cuda()

        disp_map = disp_map * 2.0 - 1.0
        disp_map = disp_map * max_disp

        with torch.no_grad():
            right_video, occlusion_mask = stereo_projector(left_video, disp_map)

        right_video = right_video.cpu().permute(0, 2, 3, 1).numpy()
        occlusion_mask = occlusion_mask.cpu().permute(0, 2, 3, 1).numpy().repeat(3, axis=-1)

        for j in range(len(batch_frames)):
            video_grid_top = np.concatenate([batch_frames[j], batch_depth_vis[j]], axis=1)
            video_grid_bottom = np.concatenate([occlusion_mask[j], right_video[j]], axis=1)
            video_grid = np.concatenate([video_grid_top, video_grid_bottom], axis=0)

            video_grid_uint8 = np.clip(video_grid * 255.0, 0, 255).astype(np.uint8)
            video_grid_bgr = cv2.cvtColor(video_grid_uint8, cv2.COLOR_RGB2BGR)
            out.write(video_grid_bgr)

        # Free up GPU memory
        del left_video, disp_map, right_video, occlusion_mask
        torch.cuda.empty_cache()
        gc.collect()

    print()  # newline after progress
    out.release()
    print(f"    Video written: {output_video_path}")


def main(
    input_video_path: str,
    output_video_path: str,
    unet_path: str,
    pre_trained_path: str,
    max_disp: float = 20.0,
    process_length = -1,
    batch_size = 10
):
    depthcrafter_demo = DepthCrafterDemo(
        unet_path=unet_path,
        pre_trained_path=pre_trained_path
    )

    video_depth, depth_vis = depthcrafter_demo.infer(
        input_video_path,
        output_video_path,
        process_length
    )
    print(f"==> Depth estimation complete. Shape: {video_depth.shape}")

    print("==> Starting stereo splatting...")
    DepthSplatting(
        input_video_path,
        output_video_path,
        video_depth,
        depth_vis,
        max_disp,
        process_length,
        batch_size
    )
    print(f"==> Splatting complete. Output saved to: {output_video_path}")


if __name__ == "__main__":
    Fire(main)
