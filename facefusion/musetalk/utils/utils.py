import os
from typing import Union, List

import cv2
import torch
import torch.nn.functional as F
from einops import rearrange

from facefusion.filesystem import resolve_relative_path
from facefusion.musetalk.models.unet import UNet, PositionalEncoding
from facefusion.musetalk.models.vae import VAE


def load_all_model(
        unet_model_path=None,
        vae_type="sd-vae",
        unet_config=None,
        device=None,
):
    if unet_model_path is None:
        unet_model_path = resolve_relative_path('../.assets/models/musetalk_v15/unet.pth')
    if unet_config is None:
        unet_config = resolve_relative_path('../.assets/models/musetalk_v15/musetalk.json')

    vae_model_path = resolve_relative_path('../.assets/models/musetalk_v15/sd-vae-ft-mse/vae-ft-mse-840000-ema-pruned.safetensors')

    vae = VAE(
        model_path=vae_model_path,
    )
    print(f"load unet model from {unet_model_path}")
    unet = UNet(
        unet_config=unet_config,
        model_path=unet_model_path,
        device=device
    )
    pe = PositionalEncoding(d_model=384)
    return vae, unet, pe


def get_file_type(video_path):
    _, ext = os.path.splitext(video_path)

    if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
        return 'image'
    elif ext.lower() in ['.avi', '.mp4', '.mov', '.flv', '.mkv']:
        return 'video'
    else:
        return 'unsupported'


def get_video_fps(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps


def datagen(
        whisper_chunks,
        vae_encode_latents,
        batch_size=8,
        delay_frame=0,
        device="cuda:0",
):
    whisper_batch, latent_batch = [], []
    for i, w in enumerate(whisper_chunks):
        idx = (i + delay_frame) % len(vae_encode_latents)
        latent = vae_encode_latents[idx]
        whisper_batch.append(w)
        latent_batch.append(latent)

        if len(latent_batch) >= batch_size:
            whisper_batch = torch.stack(whisper_batch)
            latent_batch = torch.cat(latent_batch, dim=0)
            yield whisper_batch, latent_batch
            whisper_batch, latent_batch = [], []

    # the last batch may smaller than batch size
    if len(latent_batch) > 0:
        whisper_batch = torch.stack(whisper_batch)
        latent_batch = torch.cat(latent_batch, dim=0)

        yield whisper_batch.to(device), latent_batch.to(device)


def cast_training_params(
        model: Union[torch.nn.Module, List[torch.nn.Module]],
        dtype=torch.float32,
):
    if not isinstance(model, list):
        model = [model]
    for m in model:
        for param in m.parameters():
            # only upcast trainable parameters into fp32
            if param.requires_grad:
                param.data = param.to(dtype)


def rand_log_normal(
        shape,
        loc=0.,
        scale=1.,
        device='cpu',
        dtype=torch.float32,
        generator=None
):
    """Draws samples from an lognormal distribution."""
    rnd_normal = torch.randn(
        shape, device=device, dtype=dtype, generator=generator)  # N(0, I)
    sigma = (rnd_normal * scale + loc).exp()
    return sigma


def get_mouth_region(frames, image_pred, pixel_values_face_mask):
    # Initialize lists to store the results for each image in the batch
    mouth_real_list = []
    mouth_generated_list = []

    # Process each image in the batch
    for b in range(frames.shape[0]):
        # Find the non-zero area in the face mask
        non_zero_indices = torch.nonzero(pixel_values_face_mask[b])
        # If there are no non-zero indices, skip this image
        if non_zero_indices.numel() == 0:
            continue

        min_y, max_y = torch.min(non_zero_indices[:, 1]), torch.max(
            non_zero_indices[:, 1])
        min_x, max_x = torch.min(non_zero_indices[:, 2]), torch.max(
            non_zero_indices[:, 2])

        # Crop the frames and image_pred according to the non-zero area
        frames_cropped = frames[b, :, min_y:max_y, min_x:max_x]
        image_pred_cropped = image_pred[b, :, min_y:max_y, min_x:max_x]
        # Resize the cropped images to 256*256
        frames_resized = F.interpolate(frames_cropped.unsqueeze(
            0), size=(256, 256), mode='bilinear', align_corners=False)
        image_pred_resized = F.interpolate(image_pred_cropped.unsqueeze(
            0), size=(256, 256), mode='bilinear', align_corners=False)

        # Append the resized images to the result lists
        mouth_real_list.append(frames_resized)
        mouth_generated_list.append(image_pred_resized)

    # Convert the lists to tensors if they are not empty
    mouth_real = torch.cat(mouth_real_list, dim=0) if mouth_real_list else None
    mouth_generated = torch.cat(
        mouth_generated_list, dim=0) if mouth_generated_list else None

    return mouth_real, mouth_generated


def get_image_pred(pixel_values,
                   ref_pixel_values,
                   audio_prompts,
                   vae,
                   net,
                   weight_dtype):
    with torch.no_grad():
        bsz, num_frames, c, h, w = pixel_values.shape

        masked_pixel_values = pixel_values.clone()
        masked_pixel_values[:, :, :, h // 2:, :] = -1

        masked_frames = rearrange(
            masked_pixel_values, 'b f c h w -> (b f) c h w')
        masked_latents = vae.encode(masked_frames).latent_dist.mode()
        masked_latents = masked_latents * vae.config.scaling_factor
        masked_latents = masked_latents.float()

        ref_frames = rearrange(ref_pixel_values, 'b f c h w-> (b f) c h w')
        ref_latents = vae.encode(ref_frames).latent_dist.mode()
        ref_latents = ref_latents * vae.config.scaling_factor
        ref_latents = ref_latents.float()

        input_latents = torch.cat([masked_latents, ref_latents], dim=1)
        input_latents = input_latents.to(weight_dtype)
        timesteps = torch.tensor([0], device=input_latents.device)
        latents_pred = net(
            input_latents,
            timesteps,
            audio_prompts,
        )
        latents_pred = (1 / vae.config.scaling_factor) * latents_pred
        image_pred = vae.decode(latents_pred).sample
        image_pred = image_pred.float()

    return image_pred
