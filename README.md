🌟 **SD FaceFusion, by RunDiffusion**
==========

![image](https://github.com/runnitai/sd_facefusion/assets/1633844/bbfa6b69-c8db-4089-81df-048dd6fe89a5)

> 🚀 Next generation face swapper and enhancer extension for Automatic1111, based on the original FaceFusion project.

[![Build Status](https://img.shields.io/github/actions/workflow/status/facefusion/facefusion/ci.yml.svg?branch=master)](https://github.com/facefusion/facefusion/actions?query=workflow:ci)
![License](https://img.shields.io/badge/license-MIT-brightgreen)


👀 **Preview**
-------

<img width="1199" alt="image" src="https://github.com/runnitai/sd_facefusion/assets/1633844/7534bc81-1305-427e-b6e8-1b6e0617397c">
🌐 For the most optimal cloud experience for Automatic, SD FaceFusion, and everything else, check out [Rundiffusion](https://rundiffusion.com/).


🔥 **Features**
--------
- Job Queue to allow pre-staging multiple jobs and letting it run.
- Multiple reference face selection for better face consistency.
- Optional target video loading from URL or direct path. Works for Youtube (see below) and other video files where the type is directly detectable from the URL.
- Automatic threading for optimal job performance regardless of GPU.
- Auto-downloading of models, and integration with Automatic1111 for outputs and model storage.
- Live job updates with preview images while processing.



🛠 **Installation**
------------

- Install from the URL for this repo. All models will be downloaded on first run.
- For youtube downloading, you need to manually patch the pytube library in the venv to fix some age restricted error.
  [Pytube Patch](https://github.com/pytube/pytube/pull/1790/files)

⚠️ **Disclaimer**
----------

We acknowledge the unethical potential of FaceFusion and are resolutely dedicated to establishing safeguards against such misuse. This program has been engineered to abstain from processing inappropriate content such as nudity, graphic content, and sensitive material.

It is important to note that we maintain a strong stance against any type of pornographic nature and do not collaborate with any websites promoting the unauthorized use of our software.

Users who seek to engage in such activities will face consequences, including being banned from our community. We reserve the right to report developers on GitHub who distribute unlocked forks of our software at any time.


📚 **Documentation**
-------------

Read the [documentation](https://docs.facefusion.io) for a deep dive.

## YOLO-based Custom Masking

The extension now supports custom object detection masks using YOLO models. This allows you to selectively mask regions of the image based on object detection, providing more precise control over face swapping.

### Features

- **Custom YOLO Model Selection**: Choose from any compatible YOLO `.pt` model file (from [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolo/), adetailer, or other sources)
- **Detection Confidence Threshold**: Control detection precision with a configurable confidence threshold
- **Mask Radius Control**: Adjust the blur radius of the mask edges for smoother blending
- **Face Proximity Prioritization**: Objects detected closer to the face will be given higher priority
- **Integration with existing mask types**: Works alongside box, occlusion, and region masks

### Using Custom YOLO Masks

1. Install the required dependency: `pip install ultralytics`
2. Place YOLO model files (`.pt`) in one of these locations:
   - For SD WebUI: `models/adetailer/` or `models/facefusion/yolo/`
   - For standalone: `.assets/models/yolo/`
3. In the UI, select the "custom" mask type along with any other desired mask types
4. Choose a YOLO model from the dropdown menu
5. Adjust confidence threshold and mask radius as needed

### Command Line Options

```bash
--face-mask-types custom
--custom-yolo-model "/path/to/your/model.pt"
--custom-yolo-confidence 0.5
--custom-yolo-radius 10
```

### Supported YOLO Models

The feature supports various YOLO detection models, including:
- Face detection models (face_yolov8n.pt, face_yolov8s.pt, etc.)
- Hand detection models (hand_yolov8n.pt, hand_yolov8s.pt, etc.)
- Person detection models (person_yolov8n-seg.pt, etc.)
- Clothing detection models (deepfashion2_yolov8s-seg.pt)
- Any other custom YOLO model trained for specific object detection

Models can be obtained from [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer) or trained using Ultralytics.
