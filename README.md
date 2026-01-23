# Real-Time Vision Pipeline

A real-time scene understanding application that combines multiple AI models to analyze webcam feeds. Features object detection, segmentation, action recognition, and natural language scene descriptions.

## Features

- **Multi-Model Pipeline**: Combines DINOv2/v3, SAM3, V-JEPA2, and BLIP for comprehensive scene analysis
- **Real-Time Processing**: Configurable inference rate with per-stage cadence control
- **Object Tracking**: Persistent track IDs with Kalman filter motion prediction, Hungarian optimal assignment, and ReID using DINO embeddings
- **Natural Language Narration**: Synthesizes scene descriptions from detected objects, actions, and interactions
- **Interactive UI**: Quad-pane view, timeline navigation, freeze/resume functionality

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          REAL-TIME VISION PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────┐
                              │  CAMERA  │
                              └────┬─────┘
                                   │ frames
                                   ▼
                    ┌──────────────────────────────┐
                    │         Frame Buffer         │
                    └──────────────┬───────────────┘
                                   │
                 ┌─────────────────┼─────────────────┐
                 │                 │                 │
                 ▼                 ▼                 ▼
          ┌────────────┐   ┌────────────┐   ┌────────────────┐
          │   DINO     │   │    SAM3    │   │  Clip Buffer   │
          │  Saliency  │   │Segmentation│   │  (N frames)    │
          └─────┬──────┘   └─────┬──────┘   └───────┬────────┘
                │                │                  │
                │ heatmap        │                  │
                │ + proposals    │                  ▼
                │    ┌───────────┘           ┌────────────┐
                │    │ boxes (optional)      │  V-JEPA2   │
                │    │                       │  Actions   │
                │    ▼                       └─────┬──────┘
                │  ┌───────────────┐               │
                │  │ masks, boxes, │               │ action
                │  │ labels        │               │ predictions
                │  └───────┬───────┘               │
                │          │                       │
                ▼          ▼                       │
          ┌─────────────────────┐                  │
          │      TRACKER        │◄─────────────────┘
          │  (persistent IDs,   │
          │   ReID embeddings)  │
          └──────────┬──────────┘
                     │ tracks + velocities
                     ▼
          ┌─────────────────────┐         ┌────────────┐
          │  NARRATION ENGINE   │◄────────│    BLIP    │
          │                     │         │  Captions  │
          │ • object counts     │         └────────────┘
          │ • interactions      │
          │ • action context    │
          │ • scene description │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │     UI OUTPUT       │
          │                     │
          │ ┌─────┐ ┌─────┐    │
          │ │ Raw │ │DINO │    │
          │ └─────┘ └─────┘    │
          │ ┌─────┐ ┌─────┐    │
          │ │ SAM │ │VJEPA│    │
          │ └─────┘ └─────┘    │
          │                     │
          │ [Timeline] [Stats]  │
          │ [Narration Panel]   │
          └─────────────────────┘
```

## Requirements

- Python 3.10+
- macOS (MPS), Linux/Windows (CUDA), or CPU
- Webcam
- 8GB+ RAM (16GB recommended)
- GPU with 6GB+ VRAM recommended

## Installation

```bash
# Clone the repository
git clone https://github.com/FaigleLabs/dino-sam-vjepa-blip-test
cd dino-sam-vjepa-blip-test

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Basic run (uses webcam 0)
python dino_sam_vjepa_blip_test.py

# With options
python dino_sam_vjepa_blip_test.py --camera 0 --infer-fps 2.0 --fp16

# Disable specific stages for faster performance
python dino_sam_vjepa_blip_test.py --no-vjepa
```

## Hugging Face Authentication

Some models require authentication. Provide your token via:
- Environment variable: `HF_TOKEN`
- Command line: `--hf-token YOUR_TOKEN`
- UI: Paste into the token field before clicking Start

## Pipeline Stages

### 1. DINO (Saliency & Features)
Generates attention-based saliency heatmaps to identify interesting regions. Three saliency modes available:
- **inverse**: Highlights patches dissimilar to image average (unusual = salient)
- **magnitude**: Scores by feature activation strength
- **attention**: Uses transformer attention weights

### 2. SAM3 (Segmentation) - Default
Segments objects using text prompts (e.g., "person, phone, keyboard") or DINO-proposed bounding boxes.
- Provides accurate segmentation masks
- Slower but more detailed output

### 2b. YOLO Detection (Alternative)
Fast object detection using YOLO11 or YOLO26 models from Ultralytics.
- **80 COCO classes** including person, car, phone, keyboard, cup, etc.
- **Much faster** than SAM - good for real-time applications
- **No segmentation masks** - bounding boxes only
- Select via Settings > Detection tab

| Model | Size | Speed | Use Case |
|-------|------|-------|----------|
| yolo11n/yolo26n | 6MB | Fastest | Real-time, edge devices |
| yolo11s/yolo26s | 21MB | Fast | Balanced |
| yolo11m/yolo26m | 39MB | Medium | Default recommendation |
| yolo11l/yolo26l | 49MB | Slower | Accuracy-focused |
| yolo11x/yolo26x | 109MB | Slowest | Maximum accuracy |

### 3. V-JEPA2 (Action Recognition)
Classifies activities from a rolling video clip buffer. Recognizes actions like typing, waving, picking up objects.

### 4. BLIP (Captioning)
Generates natural language descriptions of the scene.

### 5. Object Tracking

Two tracking algorithms are available:

#### Simple Tracker
- Greedy IoU-based assignment
- Fast and lightweight
- Best for slow-moving objects with consistent detections

#### Kalman Tracker (Default)
- **Kalman filter** for motion prediction - handles fast-moving objects
- **Hungarian algorithm** for optimal track-detection assignment
- **Re-ID graveyard** - resurrects tracks after occlusion using DINO embeddings
- Smooth trajectories even with noisy detections

| Feature | Simple | Kalman |
|---------|--------|--------|
| Assignment | Greedy | Hungarian (optimal) |
| Motion prediction | None | Kalman filter |
| Fast motion | Poor | Good |
| Occlusion recovery | Max missed only | Graveyard + ReID |

## UI Overview

### Views
- **Raw**: Original camera feed
- **DINO**: Saliency heatmap overlay with proposal boxes
- **SAM**: Segmentation masks with labels
- **V-JEPA**: Video clip strip with action predictions
- **Composite**: All overlays combined

### Controls
- **Start/Stop**: Begin or end inference
- **Freeze/Resume**: Pause on current frame or timeline selection
- **Timeline**: Click any row to inspect that tick's results
- **Settings**: Configure all parameters (Ctrl+,)

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Toggle freeze/resume |
| Escape | Resume live |
| Q | Quad view |
| 1-4 | Single pane view |
| 5 | Composite view |
| Up/Down | Navigate timeline |
| Ctrl+, | Open settings |

## Configuration

### Key Settings (Settings Dialog)

| Setting | Description |
|---------|-------------|
| Inference FPS | Target processing rate (default: 2.0) |
| Prompts | Comma-separated object prompts for SAM |
| DINO saliency mode | inverse / magnitude / attention |
| DINO proposal count | Max proposal boxes (1-20) |
| DINO min box area | Minimum box size as image fraction |
| SAM threshold | Confidence threshold for detections |
| Detection source | sam (default) / yolo11 / yolo26 |
| YOLO model | Model variant: n/s/m/l/x (default: yolo11m) |
| YOLO confidence | Detection confidence threshold (default: 0.25) |
| Enable Tracking | Toggle persistent object IDs |
| Tracker algorithm | kalman (default) / simple |
| Use Hungarian | Optimal assignment (requires scipy) |
| Track graveyard time | Seconds to keep dead tracks for resurrection |
| FP16 | Half-precision inference (default: enabled) |

### Command Line Options

```
--camera INDEX       Camera device index (default: 0)
--infer-fps RATE     Inference FPS (default: 2.0)
--fp16               Use FP16 precision (faster, less VRAM)
--max-width WIDTH    Max frame width for processing
--no-dino            Disable DINO stage
--no-sam             Disable SAM stage
--no-vjepa           Disable V-JEPA stage
--auto-boxes         Use DINO proposals for SAM instead of text prompts
--hf-token TOKEN     Hugging Face API token
```

## Supported Models

All models can be changed in Settings > Models tab.

### DINO (Feature Extraction / Saliency)

| Model ID | Size | Parameters | Notes |
|----------|------|------------|-------|
| `facebook/dinov2-small` | Small | 22M | Fastest, lowest memory |
| `facebook/dinov2-base` | Base | 86M | Good balance (default) |
| `facebook/dinov2-large` | Large | 300M | Better features |
| `facebook/dinov2-giant` | Giant | 1B | Best quality, slow |
| `facebook/dinov2-with-registers-small` | Small | 22M | Cleaner attention maps |
| `facebook/dinov2-with-registers-base` | Base | 86M | Cleaner attention maps |
| `facebook/dinov2-with-registers-large` | Large | 300M | Cleaner attention maps |
| `facebook/dinov2-with-registers-giant` | Giant | 1B | Cleaner attention maps |

### SAM (Segmentation)

| Model ID | Type | Notes |
|----------|------|-------|
| `facebook/sam3` | SAM 3 | Text + geometry + exemplar prompts, 848M params (default) |
| `facebook/sam2-hiera-tiny` | SAM 2 | Fastest, for real-time |
| `facebook/sam2-hiera-small` | SAM 2 | Good balance |
| `facebook/sam2-hiera-base-plus` | SAM 2 | Better accuracy |
| `facebook/sam2-hiera-large` | SAM 2 | Best accuracy |
| `facebook/sam2.1-hiera-tiny` | SAM 2.1 | Updated checkpoints |
| `facebook/sam2.1-hiera-small` | SAM 2.1 | Updated checkpoints |
| `facebook/sam2.1-hiera-base-plus` | SAM 2.1 | Updated checkpoints |
| `facebook/sam2.1-hiera-large` | SAM 2.1 | Updated checkpoints |

### YOLO (Object Detection - Alternative to SAM)

| Model ID | Version | Parameters | Notes |
|----------|---------|------------|-------|
| `yolo11n` | YOLO11 | 2.5M | Nano - fastest, edge devices |
| `yolo11s` | YOLO11 | 9.4M | Small - fast |
| `yolo11m` | YOLO11 | 20M | Medium (default) |
| `yolo11l` | YOLO11 | 25M | Large - more accurate |
| `yolo11x` | YOLO11 | 57M | Extra large - most accurate |
| `yolo26n` | YOLO26 | ~3M | Nano - 2025 architecture |
| `yolo26s` | YOLO26 | ~10M | Small - 2025 architecture |
| `yolo26m` | YOLO26 | ~22M | Medium - 2025 architecture |
| `yolo26l` | YOLO26 | ~27M | Large - 2025 architecture |
| `yolo26x` | YOLO26 | ~60M | Extra large - 2025 architecture |

**Note:** YOLO models detect 80 COCO classes (person, car, phone, keyboard, cup, etc.). No segmentation masks - bounding boxes only.

### V-JEPA (Video Action Classification)

| Model ID | Size | Resolution | Notes |
|----------|------|------------|-------|
| `facebook/vjepa2-vitl-fpc16-256-ssv2` | ViT-L | 256px | SSv2 fine-tuned (default) |
| `facebook/vjepa2-vitl-fpc64-256-ssv2` | ViT-L | 256px | Longer temporal context |
| `facebook/vjepa2-vith-fpc16-256-ssv2` | ViT-H | 256px | Larger model |
| `facebook/vjepa2-vith-fpc64-256-ssv2` | ViT-H | 256px | Larger + longer context |
| `facebook/vjepa2-vitg-fpc16-384-ssv2` | ViT-G | 384px | Giant model, best accuracy |
| `facebook/vjepa2-vitg-fpc64-384-ssv2` | ViT-G | 384px | Giant + longer context |

### Caption (Image Captioning)

| Model ID | Type | Parameters | Notes |
|----------|------|------------|-------|
| `Salesforce/blip-image-captioning-base` | BLIP | 247M | Fast, good quality (default) |
| `Salesforce/blip-image-captioning-large` | BLIP | 470M | Better captions |
| `Salesforce/blip2-opt-2.7b` | BLIP-2 | 3.7B | Much better, needs more VRAM |
| `Salesforce/blip2-opt-6.7b` | BLIP-2 | 7.5B | Even better quality |
| `Salesforce/blip2-flan-t5-xl` | BLIP-2 | 4.1B | Good for detailed captions |
| `Salesforce/blip2-flan-t5-xxl` | BLIP-2 | 12B | Best quality, very heavy |

### Recommended Configurations

| Use Case | DINO | SAM | V-JEPA | Caption |
|----------|------|-----|--------|---------|
| **Fast / Low VRAM** | dinov2-small | sam2-hiera-tiny | vjepa2-vitl-fpc16-256-ssv2 | blip-image-captioning-base |
| **Balanced** | dinov2-base | sam3 | vjepa2-vitl-fpc16-256-ssv2 | blip-image-captioning-large |
| **Best Quality** | dinov2-giant | sam3 | vjepa2-vitg-fpc64-384-ssv2 | blip2-flan-t5-xl |

## Performance Tips

1. **Lower inference FPS** for smoother UI (1-2 FPS is usually sufficient)
2. **Disable V-JEPA** if you don't need action recognition (biggest speedup)
3. **Use YOLO instead of SAM** for much faster detection (no masks, boxes only)
4. **Use --fp16** for ~2x faster inference on compatible GPUs
5. **Reduce max-width** to process smaller frames
6. **Increase stage cadence** (run DINO/SAM every N ticks instead of every tick)

## Troubleshooting

### "No camera found"
- Check camera index (try --camera 1, 2, etc.)
- Ensure camera permissions are granted

### Out of memory
- Use smaller models in Settings > Models
- Enable --fp16
- Reduce --max-width
- Disable unused stages

### Slow performance
- Lower --infer-fps
- Disable V-JEPA (most expensive stage)
- Use --fp16

## License

Unlicense: https://unlicense.org/

## Acknowledgments

This project uses models from:
- [Meta AI (DINO, SAM, V-JEPA)](https://ai.meta.com/)
- [Salesforce (BLIP)](https://github.com/salesforce/BLIP)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
