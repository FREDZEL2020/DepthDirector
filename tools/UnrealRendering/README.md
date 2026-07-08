# Unreal Engine Rendering Toolkit

A toolkit for generating **multi-camera synchronized datasets** inside Unreal Engine, designed for training camera-controlled video generation models. It drives UE headlessly via a Redis task queue, produces raw EXR/JPG frame sequences, and post-processes them with [recam](../recam/README.md).

The rendered dataset (**MultiCamWarp**) is publicly available on HuggingFace: **[Fred24/MultiCamWarp](https://huggingface.co/datasets/Fred24/MultiCamWarp)**. We provide the training video sequences along with an example subset of the Unreal Engine scene and character assets required to run this pipeline. Additional scenes and characters can be downloaded from [fab.com](https://www.fab.com/).

<p align="center">
  <img width="100%" alt="Dataset preview: input video, warped depth, and target view side by side" src="assets/dataset_preview.gif">
</p>

---

## ⚙️ How It Works

The rendering pipeline runs entirely inside Unreal Engine's Python interpreter and is orchestrated by `run.py` on the host. Each job goes through two decoupled stages — **build** and **render** — so navmesh baking and GPU rendering can be pipelined across workers independently.

### Scene Setup

Before rendering, the pipeline clears any pre-existing cameras, skeletal meshes, and trigger volumes from the level, then rebuilds all `StaticMesh` collision bodies as `BlockAllDynamic`. This ensures clean, consistent collision geometry for the placement and validation steps that follow.

### Random Character Placement with Collision Validation

A key design goal is **diversity**: each rendered sequence places characters in different, physically plausible locations across the scene. Placement uses a two-stage strategy:

1. **NavMesh sampling** — UE's `get_random_reachable_point_in_radius` samples a candidate point anywhere on the navigable mesh, using the `PlayerStart` actor (or a random scene actor) as the origin. This guarantees the point is on a walkable surface without needing explicit ground detection.
2. **Collision filtering** — the candidate is then validated through a pipeline of sphere-trace checks: roof clearance, surrounding obstacle distance, and map-edge detection (at least 9 of 12 radial directions must hit scene geometry). Failed candidates are discarded and a new NavMesh point is sampled.
3. **Direction sampling** — once a location passes, a valid facing direction is found via a weighted angular grid (15° bins, ±180° yaw), penalizing directions that fail a forward-path clearance check with a 0.5× penalty factor.

### Camera Trajectory Generation

Once a character is placed, `OrbitCamera` generates synchronized multi-camera trajectories around it. Eight trajectory types are supported (`orbit`, `helix_in`, `s_curve_dolly`, `zoom_out`, `orbit_ascend`, `parabola_dive`, `pan_left_right`, `rotate_left_right`), each parameterized with randomized scale and direction. All camera keyframes are validated against the same collision checks before being accepted.

Each trajectory prepends a **warmup segment** (configurable frames, camera held still) so Lumen GI and TAA history buffers stabilize before the usable frames begin. Warmup frames are stripped during post-processing.

### Sequencer & Rendering

Validated character transforms, animation tracks, and camera trajectories are all written into a single UE `LevelSequence`. Multiple characters are stacked sequentially on the same timeline, keeping the number of render jobs per scene minimal. The sequence is then submitted to Movie Pipeline for offline rendering, with Lumen cvars injected per-job to ensure hardware ray tracing (Vulkan SM6) is active.

After rendering, `camera_params.json` is exported for each trajectory by interpolating the Sequencer transform channels (with slerp for rotations), providing the ground-truth camera poses consumed by the recam post-processing pipeline.

---

## 🚀 Installation

### Prerequisites

- **Unreal Engine 5.6** (Linux source build or mounted from host)
- **NVIDIA GPU** with ray tracing support (RTX series recommended)
- **NVIDIA driver 580** (strongly recommended)
  - Driver 580 enables **Vulkan SM6** on Linux, which unlocks **hardware ray tracing** and fully utilizes UE5.6's Lumen global illumination.
  - Drivers above 580 have known compatibility issues with UE5.6 Lumen and are not recommended.
  - Drivers below 550 only support **Vulkan SM5**, limiting ray tracing to the software path.
- **Redis** server accessible from the container

### 1. Install Python Packages

```bash
pip install -r tools/UnrealRendering/requirements.txt
```

### 2. Prepare Assets

Assets should be organized as follows and made available to the storage backend configured in `configs/storage/local.json`:

```
/path/to/assets/
├── Scene/          ← UE scene projects (ASSETS_PATH)
└── Actor/          ← character assets (AVATAR_ASSETS_PATH)
```

The UE project template is included in the repo at `UE_Template/`. In `configs/storage/local.json`, set `SRC_PROJECT_PATH` to its absolute path inside the container (e.g. `/home/ue_user/code/UE_Template`).

### UE_Template

`UE_Template/` is the base Unreal Engine project used for all rendering jobs. It pre-configures two key systems:

- **Lumen Global Illumination** — project-level settings for Lumen GI and hardware ray tracing (requires Vulkan SM6, see [Prerequisites](#prerequisites)), ensuring physically accurate lighting across all rendered scenes without per-scene setup.
- **Derived Data Cache (DDC)** — the project is configured to store UE's shader compilation artifacts at `/data/ddc`. Pre-cooking this cache once (via `bash run/render.sh cook`) avoids redundant recompilation across render workers and significantly reduces per-job startup time.

Each scene asset is staged by copying `UE_Template/` as a base and injecting the scene content on top, so all jobs inherit the same lighting and DDC configuration.

### 3. Generate Asset Lists

> **Note:** The scripts below must be executed inside the Unreal Editor's built-in Python interpreter, not from a regular shell. The commands shown are for reference only. We plan to replace this step with a fully automated asset curation pipeline in a future release.

```python
# Collect available levels → info/levels.json
# Run inside UE Editor Python console:
import scripts.collect_levels

# Generate actor and animation manifests → info/actor.json
import src.libs.assets
```

---

## 💻 Usage

The scripts under `src/` run inside Unreal Engine's embedded Python interpreter (`UnrealEditor-Cmd`). We recommend using Windows or macOS for interactive debugging in the UE Editor, and Linux or Docker for headless batch rendering.

### Editor Mode (Interactive Debugging)

Open the UE Editor and run the render script directly. See `run_scripts/render.bat` for the reference invocation.

### Batch Rendering Mode

#### Step 0 — Pre-cook Shaders

Pre-compile UE shaders and save the derived data cache (DDC) to `UE_Template/ddc`. This avoids recompilation on every render job.

```bash
bash run/render.sh cook
```

#### Step 1 — Submit Scenes to the Queue

Push all levels listed in `levels_json` into the Redis task queue:

```bash
bash run/render.sh submit
```

#### Step 2 — Start Render Workers

Launch the continuous build → render loop. Each worker processes one scene at a time:

```bash
bash run/render.sh full_loop
```

### Runtime File Structure

After the pipeline runs, the following directory structure is automatically created under `/data`:

```
/data
├── UnrealProject
│   └── QA_Office                        ← staged UE project (copied from UE_Template)
│       ├── Blank.uproject
│       ├── Config
│       ├── Content
│       ├── DerivedDataCache
│       ├── Intermediate
│       └── Saved
├── ddc                                  ← shared DDC cache (pre-cooked via `cook` step)
│   ├── Buckets
│   ├── Content
│   └── TestData
└── renders_raw_output
    └── QA_Office_QA_Office_DemoM        ← one directory per rendered sequence
        ├── QA_Office_QA_Office_DemoM..0000.exr          ← depth (EXR)
        ├── QA_Office_QA_Office_DemoM..0001.exr
        ├── ...
        ├── QA_Office_QA_Office_DemoM.FinalImage.0000.jpeg   ← RGB (JPEG)
        ├── QA_Office_QA_Office_DemoM.FinalImage.0001.jpeg
        └── ...
```

The `renders_raw_output` directory contains the raw EXR depth frames and JPEG RGB frames produced by Movie Pipeline. These are consumed by the recam post-processing step to generate the final dataset.

### Docker Environment

Build the image from the project root:

```bash
docker build -f envs/dockerfile -t ue_render .
```

Launch one worker per GPU using `envs/run.sh`:

```bash
GPU=0 UE_PATH=/path/to/UnrealEngine ./envs/run.sh full_loop
```

---

### Available Tasks

| Task | Description |
|------|-------------|
| `submit` | Push levels from `levels_json` into the Redis queue |
| `cook` | Pre-cook shaders and DDC for the template project |
| `build` | Bake navmesh and build Sequencer tracks for one scene |
| `render` | Render a pre-built scene and run recam post-processing |
| `full_loop` | Continuous `build → render` loop (default) |
| `build_loop` | Continuous build-only loop |
| `render_loop` | Continuous render-only loop |

---

## 🛠️ Configuration

### `configs/ue_config/recam.json`

Core paths and Redis connection used by `run.py`:

| Key | Description |
|-----|-------------|
| `unreal_engine_path` | Path to `UnrealEditor-Cmd` inside the container |
| `project_path` | Directory where UE projects are staged |
| `render_code` | Path to `src/` inside the container |
| `output_path` | Final processed dataset output directory |
| `raw_output_path` | Raw EXR/JPG frames output from UE |
| `logs_path` | UE and post-processing logs directory |
| `levels_json` | JSON list of `.umap` paths to render |
| `redis_url` | Redis connection string |
| `redis_passwd` | Redis password |
| `depth_downsample` | Downsample ratios for depth maps (e.g. `"4 2 2"`) |

### `configs/storage/local.json`

Asset source paths for local disk storage:

```json
{
    "storage_type": "local",
    "SRC_PROJECT_PATH": "/path/to/UE_Template",
    "ASSETS_PATH": "/path/to/Scene",
    "AVATAR_ASSETS_PATH": "/path/to/Actor"
}
```

### `configs/render_setting_recam.json`

Controls the render pipeline per job:

| Section | Key fields |
|---------|-----------|
| `renderer` | `"v2v"` (video-to-video) or `"i2v"` (image-to-video) |
| `render` | `render_depth`, `render_jpg`, `render_exr`, `enable_dof` |
| `camera` | `lens` (focal length in mm), `aperture`, `auto_exposure` |
| `navigator` | `frame_num`, `camera_type`, `indoor`, distance/height ranges |
| `charactor` | `static` (freeze character motion) |

Available camera types: `center`, `rand`, `double`, `orbit`, `zoom_out`, `pan_left_right`, `rotate_left_right`


---

## 🔮 Future Plans

We are working on a fully automated **world traversal data construction pipeline** for training interactive world models. This pipeline will enable large-scale generation of continuous, navigable scene traversals with long-horizon spatiotemporal consistency — going beyond fixed camera trajectories toward free, user-driven exploration of persistent 3D worlds. Stay tuned.
