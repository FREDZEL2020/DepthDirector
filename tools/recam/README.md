# Preprocessing Tools for DepthDirector

This directory provides the data preprocessing pipeline for DepthDirector. It handles depth estimation, 3D geometry construction, and camera-trajectory-based warping to produce the View Stream conditions used during inference and training.

## 📋 Prerequisites

- Python 3.11
- PyTorch >= 2.7.0
- CUDA-compatible GPU

## 🚀 Installation

1. Clone the repository with submodules:
```bash
git clone --recursive https://github.com/FREDZEL2020/DepthDirector.git
```

2. Install dependencies:
```bash
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r tools/recam/requirements.txt
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git
```

## 📁 Checkpoint Setup

Download the required third-party checkpoints:

```bash
# Default depth model (recommended)
huggingface-cli download yyfz233/Pi3X --local-dir checkpoints/yyfz233/Pi3X

# Optional: additional depth estimators
huggingface-cli download Rain729/Prior-Depth-Anything --local-dir checkpoints/Rain729/Prior-Depth-Anything
huggingface-cli download Ruicheng/moge-vitl --local-dir checkpoints/Ruicheng/moge-vitl
huggingface-cli download depth-anything/DA3NESTED-GIANT-LARGE --local-dir checkpoints/depth-anything/DA3NESTED-GIANT-LARGE
huggingface-cli download tencent/DepthCrafter --local-dir checkpoints/tencent/DepthCrafter

# Optional: geometry-based alternative
huggingface-cli download TencentARC/GeometryCrafter --local-dir checkpoints/TencentARC/GeometryCrafter
huggingface-cli download stabilityai/stable-video-diffusion-img2vid --local-dir checkpoints/stabilityai/stable-video-diffusion-img2vid
huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt --local-dir checkpoints/stabilityai/stable-video-diffusion-img2vid-xt

# Required for captioning
huggingface-cli download Salesforce/blip2-opt-2.7b --local-dir checkpoints/Salesforce/blip2-opt-2.7b
```

Expected checkpoint layout:

```
checkpoints/
├── Rain729/
│   └── Prior-Depth-Anything
├── Ruicheng/
│   └── moge-vitl
├── Salesforce/
│   └── blip2-opt-2.7b
├── TencentARC/
│   └── GeometryCrafter
├── depth-anything/
│   └── DA3NESTED-GIANT-LARGE
├── stabilityai/
│   ├── stable-video-diffusion-img2vid
│   └── stable-video-diffusion-img2vid-xt
├── tencent/
│   └── DepthCrafter
└── yyfz233/
    └── Pi3X
```

## 💻 Inference

### Dynamic Video Processing

Takes a dynamic video as input, reconstructs a Dynamic3DMesh, and produces warped depth conditions under the target camera trajectory.

```bash
python tools/recam/inference.py \
    --video_path assets/examples/anne.mp4 \
    --out_dir assets/examples/preprocess/anne \
    --video_length 81 \
    --depth_model Pi3X \
    --task process \
    --traj_txt Rotate,0,30,0,0,0
```

### Camera Trajectory Options

The `--traj_txt` parameter follows the format: `(mode, dtheta, dphi, dr, dx, dy)`

- **Modes**: `Rotate`, `Static`, `Orbit`, `Helix_In`, `S_Curve_Dolly`, `Orbit_Ascend`, `Zoom-out`, `Parabola_Dive`
- **Parameters**: rotation up/down (`dtheta`); rotation left/right (`dphi`); zoom in/out (`dr`); translate x (`dx`); translate y (`dy`)

### Alternative Geometry: GeometryCrafter

For better geometry in complex scenes, use GeometryCrafter instead of Dynamic3DMesh:

```bash
pip install mediapy fire piqp moderngl h5py

python tools/recam/inference.py \
    --video_path assets/examples/anne.mp4 \
    --out_dir assets/examples/preprocess/anne \
    --video_length 81 \
    --geometry GeometryCrafter \
    --task process \
    --traj_txt Rotate,0,30,0,0,0
```

## 🗃️ Training Data Generation

### Unreal Engine Dataset (EXR Format)

Preprocess a raw Unreal Engine rendering dataset (EXR format) into the training format:

```bash
python tools/recam/run.py \
    --method UE_Dynamic \
    --task process \
    --video_path [path to cameras.json] \
    --out_dir [path to output directory] \
    --raw_dir [path to raw EXR images directory] \
    --device cuda:0
```

The `cameras.json` file defines the multi-camera setup exported from Unreal Engine. The expected input layout is:

```
dataset/
├── OrbitSequence_scene_0/
│   ├── cam_name1/
│   │   └── camera_params.json
│   └── cam_name2/
│       └── camera_params.json
└── raw/
    └── OrbitSequence/
        ├── OrbitSequence.0000.exr
        ├── OrbitSequence.0001.exr
        ├── OrbitSequence.0002.exr
        └── ...
```
