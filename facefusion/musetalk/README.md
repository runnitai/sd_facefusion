# MuseTalk Integration for FaceFusion

This directory contains the MuseTalk integration for FaceFusion, providing advanced lip-syncing capabilities using the MuseTalk v1.5 model.

## Overview

MuseTalk is a state-of-the-art lip-syncing model that provides more natural and accurate lip movements compared to traditional methods like Wav2Lip. This integration allows FaceFusion users to leverage MuseTalk's capabilities for high-quality lip synchronization.

## Components

### Models
- `models/vae.py` - Variational Autoencoder for image encoding/decoding
- `models/unet.py` - U-Net model with positional encoding for lip sync generation

### Utilities
- `utils/audio_processor.py` - Audio processing using Whisper features
- `utils/blending.py` - Advanced blending techniques for seamless integration
- `utils/face_parsing/` - Face parsing models for precise mask generation
- `utils/utils.py` - Core utility functions for model loading and processing

## Required Models

To use MuseTalk, you need to download the following models:

1. **MuseTalk UNet Model** (`musetalkV15/unet.pth`)
2. **MuseTalk Config** (`musetalkV15/musetalk.json`)
3. **Stable Diffusion VAE** (`sd-vae-ft-mse/`)
4. **Whisper Model** (`whisper/`)
5. **Face Parsing Models** (`face-parse-bisent/`)

## Usage

To use MuseTalk in FaceFusion, set the lip syncer model to `musetalk_v15`:

```bash
python facefusion.py --lip-syncer-model musetalk_v15 --source audio.wav --target video.mp4 --output result.mp4
```

## Features

- **High-quality lip sync**: More natural and accurate lip movements
- **Multiple reference faces**: Support for processing multiple faces in a single video
- **Advanced blending**: Seamless integration with original video content
- **Audio processing**: Sophisticated audio feature extraction using Whisper

## Technical Details

The MuseTalk integration works by:

1. **Face Detection**: Detecting and tracking faces in the target video
2. **Audio Processing**: Converting audio to Whisper features for conditioning
3. **Latent Generation**: Using the U-Net model to generate lip-synced latents
4. **Image Decoding**: Converting latents back to images using the VAE
5. **Blending**: Seamlessly blending the generated lips with the original face

## Compatibility

This integration is designed to work seamlessly with the existing FaceFusion pipeline, supporting:

- All existing face selector modes (one, many, reference)
- Multiple audio sources
- Various video formats and resolutions
- GPU acceleration when available

## Performance

MuseTalk provides superior quality compared to Wav2Lip but may require more computational resources. GPU acceleration is recommended for optimal performance. 