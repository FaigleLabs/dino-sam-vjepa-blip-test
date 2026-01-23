"""
Qt + OpenCV camera viewer with staged vision pipeline + structured timeline (Mac MPS optimized knobs).

Stages:
- DINOv3: dense features -> patch-norm "saliency" heatmap (visual stage)
- SAM3: concept segmentation from text prompts OR DINO-proposed boxes (visual stage)
- V-JEPA2: rolling-clip video classification (text stage + clip strip)

Timeline:
- Each inference tick becomes a timeline row showing stage durations, SAM counts, VJ top-1
- Clicking a row freezes panes to that tick
- "Resume Live" returns to live display
- Optional "Pause inference when frozen" to stop GPU work + stop timeline growth

MPS optimization controls:
- FP16 on MPS: loads models in float16 and wraps forwards in autocast where supported
- Auto-fallback to FP32 if FP16 hits unsupported ops (keeps running)
- Per-stage cadence: run each stage every N ticks

HF token support:
- Provide token via env (HF_TOKEN / HUGGINGFACE_HUB_TOKEN / HUGGINGFACEHUB_API_TOKEN),
  or CLI --hf-token, or paste into the UI before pressing Start.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import os
import pathlib
import re
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
from contextlib import nullcontext

import cv2
import numpy as np
from PIL import Image

import torch
from PySide6 import QtCore, QtGui, QtWidgets


# ----------------------------
# HF token helpers
# ----------------------------

def get_hf_token_from_env() -> Optional[str]:
    for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        v = os.environ.get(k)
        if v and v.strip():
            return v.strip()
    return None


def hf_from_pretrained(cls, model_id: str, hf_token: Optional[str] = None, **kwargs):
    """
    Call cls.from_pretrained(model_id, token=...) with compatibility fallback to use_auth_token=...
    """
    if hf_token and hf_token.strip():
        tok = hf_token.strip()
        try:
            return cls.from_pretrained(model_id, token=tok, **kwargs)
        except TypeError:
            return cls.from_pretrained(model_id, use_auth_token=tok, **kwargs)
    return cls.from_pretrained(model_id, **kwargs)


# ----------------------------
# Config
# ----------------------------

@dataclass
class ModelIds:
    dino: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    sam3: str = "facebook/sam3"
    vjepa2_cls: str = "facebook/vjepa2-vitl-fpc16-256-ssv2"

    caption: str = "Salesforce/blip-image-captioning-base"


@dataclass
class RuntimeConfig:
    # Camera
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps_hint: int = 30

    # Scheduling
    inference_fps: float = 2.0

    # V-JEPA2 clip settings
    vjepa_samples: int = 16  # Number of frames to sample for V-JEPA (buffer auto-manages)

    # Prompts
    default_prompts: str = "person, face, hand, phone, keyboard, cup"

    # Enable stages
    enable_dino: bool = True
    enable_sam: bool = True
    enable_vjepa: bool = True

    # Captioning (true scene description via BLIP)
    enable_caption: bool = True
    caption_every_n: int = 5          # run caption every N ticks
    caption_on_events: bool = True    # also refresh caption on narration events
    caption_min_tick_gap: int = 2     # min ticks between event-triggered captions
    caption_max_new_tokens: int = 24
    caption_num_beams: int = 3

    # Tracking / narration
    enable_tracking: bool = True
    tracker_iou_thresh: float = 0.30
    tracker_max_missed: int = 6
    tracker_label_ema: float = 0.70


    # Phase 3: optional ReID embeddings (DINO) + better matching
    enable_reid: bool = True
    reid_every_n: int = 2          # compute per-detection DINO embeddings every N ticks
    reid_max_dets: int = 6         # cap embeddings per tick (largest masks/boxes)

    tracker_use_embedding: bool = True
    tracker_iou_weight: float = 0.65
    tracker_emb_weight: float = 0.35
    tracker_min_sim: float = 0.15  # if IoU is low, require sim >= this to allow a match
    tracker_emb_ema: float = 0.40  # how fast to update track embedding

    # Tracker algorithm selection
    tracker_algorithm: str = "kalman"  # "simple" or "kalman" (with motion prediction)
    tracker_use_hungarian: bool = True  # Use optimal Hungarian assignment (requires scipy)
    tracker_graveyard_seconds: float = 10.0  # How long to keep dead tracks for resurrection

    # Phase 3: simple interaction heuristics (normalized by frame diagonal)
    evt_contact_dist_frac: float = 0.18
    evt_typing_dist_frac: float = 0.25
    evt_typing_motion_frac: float = 0.004
    evt_hold_motion_frac: float = 0.012
    evt_settle_motion_frac: float = 0.003

    # DINO saliency threshold: raw_std must exceed this to generate proposals
    # Lower = more sensitive (more boxes), Higher = stricter (fewer false positives)
    # Typical range: 0.01 (very sensitive) to 0.05 (strict)
    dino_saliency_threshold: float = 0.02

    # DINO saliency mode: how to compute the heatmap
    # - "attention": CLS token attention weights (where model looks)
    # - "inverse": patches DISSIMILAR to image average (unusual = salient)
    # - "magnitude": patch feature vector magnitude (high activation = interesting)
    dino_saliency_mode: str = "inverse"  # Default to inverse as it often works better

    # DINO proposal box settings
    dino_proposal_count: int = 5  # Max number of proposal boxes to generate
    dino_proposal_min_area: float = 0.005  # Min box area as fraction of image (0.5% = small objects ok)
    dino_proposal_max_area: float = 0.80  # Max box area as fraction of image

    # Legacy / compatibility names (kept; synced in __post_init__)
    use_dino_box_proposals_for_sam: bool = False
    dino_heatmap_strength: float = 0.45
    dino_heatmap_blur: int = 3
    max_display_width: int = 640

    # Canonical UI / pipeline controls
    use_dino_boxes_for_sam: bool = False
    sam_threshold: float = 0.50
    sam_mask_threshold: float = 0.50

    # Detection source: "sam" (default), "yolo11", or "yolo26"
    detection_source: str = "sam"
    yolo_model: str = "yolo11m"  # Specific YOLO variant (n/s/m/l/x)
    yolo_conf_threshold: float = 0.25  # YOLO confidence threshold

    dino_overlay_strength: float = 0.45
    dino_blur_ksize: int = 3
    display_max_width: int = 640
    pause_on_freeze: bool = True

    # Per-stage cadence
    run_dino_every_n: int = 1
    run_sam_every_n: int = 1
    run_vjepa_every_n: int = 2

    # Hugging Face
    hf_token: Optional[str] = None

    # timeline storage
    timeline_max_items: int = 250

    # MPS acceleration knobs
    fp16_on_mps: bool = True  # Default to FP16 for faster inference

    def __post_init__(self):
        # --- width alias ---
        # Prefer an explicitly changed value if one differs from the defaults.
        default_w = 640
        w_max = int(getattr(self, 'max_display_width', default_w))
        w_disp = int(getattr(self, 'display_max_width', default_w))
        if w_disp == default_w and w_max != default_w:
            w = w_max
        else:
            w = w_disp
        self.max_display_width = int(w)
        self.display_max_width = int(w)

        # --- DINO strength alias ---
        default_s = 0.45
        s_heat = float(getattr(self, 'dino_heatmap_strength', default_s))
        s_olay = float(getattr(self, 'dino_overlay_strength', default_s))
        if abs(s_olay - default_s) < 1e-9 < abs(s_heat - default_s):
            s = s_heat
        else:
            s = s_olay
        self.dino_heatmap_strength = float(s)
        self.dino_overlay_strength = float(s)

        # --- DINO blur alias ---
        default_b = 3
        b_heat = int(getattr(self, 'dino_heatmap_blur', default_b))
        b_ksz = int(getattr(self, 'dino_blur_ksize', default_b))
        b = b_heat if (b_ksz == default_b and b_heat != default_b) else b_ksz
        # keep odd blur for Gaussian/median
        if b % 2 == 0:
            b += 1
        self.dino_heatmap_blur = int(b)
        self.dino_blur_ksize = int(b)

        # --- DINO boxes alias ---
        ub = bool(getattr(self, 'use_dino_boxes_for_sam', False))
        ul = bool(getattr(self, 'use_dino_box_proposals_for_sam', False))
        use_boxes = ub or ul
        self.use_dino_boxes_for_sam = bool(use_boxes)
        self.use_dino_box_proposals_for_sam = bool(use_boxes)



# ----------------------------
# Small parsing helpers
# ----------------------------

def split_prompts(s: str) -> List[str]:
    """Split a user-entered prompt string into a clean list.

    Accepts comma/semicolon/newline separated input.
    """
    if not s:
        return []
    # Normalize separators
    s = s.replace(";", ",").replace("\n", ",")
    return [p.strip() for p in s.split(",") if p.strip()]


def humanize_vjepa_label(label: str) -> str:
    """Heuristic cleanup for Something-Something-style labels."""
    s = str(label).strip()
    s = re.sub(r"\s*\(\s*\d+(?:\.\d+)?%?\s*\)\s*$", "", s)
    s = re.sub(r"\[\s*something else\s*\]", "another object", s, flags=re.I)
    s = re.sub(r"\[\s*something\s*\]", "object", s, flags=re.I)
    s = re.sub(r"\[\s*somewhere\s*\]", "somewhere", s, flags=re.I)
    s = s.replace("[", "").replace("]", "")
    s = re.sub(r"\s+", " ", s).strip()
    low = s.lower()
    if low.startswith("pretending or trying and failing to "):
        s = "Trying to " + s[len("Pretending or trying and failing to "):]
    elif low.startswith("pretending to "):
        s = "Pretending to " + s[len("Pretending to "):]
    s = re.sub(r"\bobject object\b", "object", s, flags=re.I)
    if not s:
        return str(label)
    return s[:1].upper() + s[1:]

def shorten_text(s: str, n: int = 90) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[:max(0, n-1)].rstrip() + "…"

def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_float_tensor(x: Any) -> bool:
    return torch.is_tensor(x) and x.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16)


def autocast_ctx(device: torch.device, dtype: torch.dtype, enabled: bool):
    if not enabled:
        return nullcontext()
    try:
        return torch.autocast(device_type=device.type, dtype=dtype)
    except Exception:
        # Older builds may not support autocast on some device types
        return nullcontext()


# ----------------------------
# Helpers: rendering overlays / thumbnails
# ----------------------------

def bgr_to_qimage(bgr: np.ndarray) -> QtGui.QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888).copy()


def fit_to_width(bgr: np.ndarray, width: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    if w <= width:
        return bgr
    scale = width / float(w)
    nh = max(1, int(round(h * scale)))
    return cv2.resize(bgr, (width, nh), interpolation=cv2.INTER_AREA)


def overlay_heatmap(bgr: np.ndarray, heat: np.ndarray, strength: float) -> np.ndarray:
    """Overlay a scalar heatmap on a BGR image.

    Robust to NaN/Inf, and uses percentile contrast stretching for a readable overlay even when the
    heat distribution is very peaky or very flat.
    """
    heat = np.asarray(heat, dtype=np.float32)
    heat = np.nan_to_num(heat, nan=0.0, posinf=1.0, neginf=0.0)
    heat = np.clip(heat, 0.0, 1.0)

    # Percentile contrast stretch (for visualization only)
    flat = heat.reshape(-1)
    if flat.size > 0:
        lo = float(np.quantile(flat, 0.05))
        hi = float(np.quantile(flat, 0.95))
        if hi > lo + 1e-6:
            heat_vis = (heat - lo) / (hi - lo)
        else:
            heat_vis = heat
    else:
        heat_vis = heat
    heat_vis = np.clip(heat_vis, 0.0, 1.0)

    heat_u8 = (heat_vis * 255.0).astype(np.uint8)
    cm = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
    return cv2.addWeighted(bgr, 1.0 - strength, cm, strength, 0.0)


def draw_boxes(
    bgr: np.ndarray,
    boxes_xyxy: List[List[float]],
    labels: Optional[List[str]] = None,
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw XYXY boxes in-place on a BGR image."""
    if not boxes_xyxy:
        return bgr
    out = bgr.copy()
    h, w = out.shape[:2]
    for i, box in enumerate(boxes_xyxy):
        if box is None or len(box) < 4:
            continue
        x1, y1, x2, y2 = box[:4]
        x1 = int(max(0, min(w - 1, round(x1))))
        y1 = int(max(0, min(h - 1, round(y1))))
        x2 = int(max(0, min(w - 1, round(x2))))
        y2 = int(max(0, min(h - 1, round(y2))))
        cv2.rectangle(out, (x1, y1), (x2, y2), color=color, thickness=thickness)
        if labels and i < len(labels) and labels[i]:
            cv2.putText(
                out,
                labels[i],
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
    return out


def overlay_masks(
    bgr: np.ndarray,
    masks: List[np.ndarray],
    boxes: Optional[List[List[float]]] = None,
    labels: Optional[List[str]] = None,
    alpha: float = 0.45,
) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]

    rng = np.random.default_rng(12345)
    colors = rng.integers(40, 255, size=(max(1, len(masks)), 3), dtype=np.uint8)

    for i, mask in enumerate(masks):
        if mask.dtype != np.bool_:
            mask_bin = mask > 0.5
        else:
            mask_bin = mask

        if mask_bin.shape[:2] != (h, w):
            mask_bin = cv2.resize(mask_bin.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

        color = colors[i % len(colors)].tolist()
        overlay = np.zeros_like(out, dtype=np.uint8)
        overlay[mask_bin] = color
        out = cv2.addWeighted(out, 1.0, overlay, alpha, 0.0)

        if boxes and i < len(boxes) and boxes[i] is not None:
            x1, y1, x2, y2 = boxes[i]
            x1 = int(max(0, min(w - 1, round(x1))))
            y1 = int(max(0, min(h - 1, round(y1))))
            x2 = int(max(0, min(w - 1, round(x2))))
            y2 = int(max(0, min(h - 1, round(y2))))
            cv2.rectangle(out, (x1, y1), (x2, y2), color=tuple(int(c) for c in color), thickness=2)

            if labels and i < len(labels) and labels[i]:
                cv2.putText(
                    out, labels[i], (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, tuple(int(c) for c in color), 2, cv2.LINE_AA
                )

    return out


def downsample_masks(
    masks: List[np.ndarray],
    target_h: int = 120,
    target_w: int = 160,
) -> List[np.ndarray]:
    """Downsample masks for memory-efficient storage in StageOutputs.

    overlay_masks() resizes masks to display size anyway, so we don't need
    to store full-resolution masks. This reduces memory ~16x for typical frames.

    Args:
        masks: List of full-resolution binary/float masks
        target_h: Target height (default 120, ~1/4 of 480)
        target_w: Target width (default 160, ~1/4 of 640)

    Returns:
        List of downsampled masks
    """
    if not masks:
        return []

    downsampled = []
    for m in masks:
        if m is None:
            downsampled.append(None)
            continue
        # Squeeze extra dimensions
        m2 = np.squeeze(m) if m.ndim > 2 else m
        if m2.ndim != 2:
            # Can't handle 3D+ masks, keep as-is
            downsampled.append(m)
            continue
        # Resize using INTER_AREA for downsampling (averages pixels, good for masks)
        small = cv2.resize(m2.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_AREA)
        # Store as uint8 to save memory (0 or 255)
        downsampled.append((small > 0.5).astype(np.uint8))
    return downsampled


def make_clip_strip(frames_bgr: List[np.ndarray], height: int = 100) -> np.ndarray:
    if not frames_bgr:
        return np.zeros((height, height, 3), dtype=np.uint8)
    thumbs = []
    for f in frames_bgr:
        h, w = f.shape[:2]
        scale = height / float(h)
        tw = max(1, int(round(w * scale)))
        thumbs.append(cv2.resize(f, (tw, height), interpolation=cv2.INTER_AREA))
    return cv2.hconcat(thumbs)


def make_clip_grid(frames_bgr: List[np.ndarray], target_size: int = 480) -> np.ndarray:
    """Create a grid of frames (4x4 for 16 frames, 8x8 for 64 frames).

    Args:
        frames_bgr: List of BGR frames
        target_size: Target width/height of the output grid image

    Returns:
        BGR image with frames arranged in a square grid
    """
    if not frames_bgr:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)

    n = len(frames_bgr)
    # Determine grid size: 4x4 for <=16, 8x8 for <=64
    if n <= 16:
        cols = 4
    elif n <= 64:
        cols = 8
    else:
        cols = int(np.ceil(np.sqrt(n)))

    rows = int(np.ceil(n / cols))

    # Calculate tile size to fill target
    tile_w = target_size // cols
    tile_h = target_size // rows

    # Create output grid
    grid_h = tile_h * rows
    grid_w = tile_w * cols
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for i, f in enumerate(frames_bgr):
        row = i // cols
        col = i % cols

        # Resize frame to fit tile (maintain aspect ratio, center)
        fh, fw = f.shape[:2]
        scale = min(tile_w / fw, tile_h / fh)
        new_w = max(1, int(fw * scale))
        new_h = max(1, int(fh * scale))
        resized = cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Center in tile
        y_off = (tile_h - new_h) // 2
        x_off = (tile_w - new_w) // 2

        y_start = row * tile_h + y_off
        x_start = col * tile_w + x_off

        grid[y_start:y_start + new_h, x_start:x_start + new_w] = resized

    return grid


def make_composite_view(
    raw_bgr: np.ndarray,
    dino_heat: Optional[np.ndarray],
    sam_masks: List[np.ndarray],
    sam_boxes: List[List[float]],
    sam_labels: List[str],
    tracks: List['Track'],
    heatmap_strength: float = 0.35,
    mask_alpha: float = 0.30,
) -> np.ndarray:
    """
    Create a unified composite view showing all stage outputs overlaid on a single image.
    Layers (bottom to top): DINO heatmap -> SAM masks -> Tracker boxes
    """
    result = raw_bgr.copy()

    # Layer 1: DINO heatmap (subtle, in background)
    if dino_heat is not None:
        # Scale strength based on heatmap variance to avoid flat tinting
        heat_std = float(np.std(dino_heat))
        strength_eff = heatmap_strength
        if heat_std < 0.03:
            strength_eff *= max(0.0, min(1.0, heat_std / 0.03))
        result = overlay_heatmap(result, dino_heat, strength_eff)

    # Layer 2: SAM masks (semi-transparent, no labels to reduce clutter)
    if sam_masks:
        result = overlay_masks(result, sam_masks, sam_boxes, labels=None, alpha=mask_alpha)

    # Layer 3: Tracker boxes (on top, with IDs and labels)
    if tracks:
        result = tracks_to_overlay(result, tracks)

    return result


def summarize_counts(counts: Dict[str, int], max_len: int = 64) -> str:
    parts = [f"{k}:{v}" for k, v in counts.items()]
    s = ", ".join(parts)
    if len(s) <= max_len:
        return s
    nonzero = [f"{k}:{v}" for k, v in counts.items() if v > 0]
    s2 = ", ".join(nonzero) if nonzero else s
    return (s2[: max_len - 1] + "…") if len(s2) > max_len else s2


# ----------------------------
# Tracking + narration helpers
# ----------------------------

@dataclass
class Detection:
    label: str
    bbox_xyxy: List[float]
    area_frac: float
    score: float = 1.0
    emb: Optional[np.ndarray] = None  # optional DINO embedding for ReID


@dataclass
class Track:
    id: int
    bbox_xyxy: List[float]
    area_frac: float
    label_probs: Dict[str, float]
    cx: float
    cy: float
    vx: float = 0.0
    vy: float = 0.0
    age: int = 0
    hits: int = 0
    last_seen_tick: int = 0
    emb: Optional[np.ndarray] = None
    emb_tick: int = 0
    pos_hist: Deque[Tuple[float, float]] = dataclasses.field(default_factory=lambda: deque(maxlen=8))
    speed: float = 0.0

    def top_label(self) -> Tuple[str, float]:
        if not self.label_probs:
            return ("unknown", 0.0)
        k = max(self.label_probs.items(), key=lambda kv: kv[1])
        return (str(k[0]), float(k[1]))


def _bbox_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a_area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    b_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    denom = a_area + b_area - inter
    return float(inter / denom) if denom > 1e-9 else 0.0


def _bbox_center(bb: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bb[:4]
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))



def _norm_vec(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = float(np.linalg.norm(x))
    if n < 1e-9:
        return x
    return (x / n).astype(np.float32)


def _cos_sim01(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    """Cosine similarity mapped to [0,1]. Returns None if missing."""
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return None
    sim = float(np.dot(a, b) / (na * nb))
    sim = max(-1.0, min(1.0, sim))
    return 0.5 * (sim + 1.0)

def detections_from_sam(
    masks: List[np.ndarray],
    boxes: List[List[float]],
    labels: List[str],
    hw: Tuple[int, int],
) -> Tuple[List[Detection], Dict[str, float]]:
    """Build detections from SAM masks/boxes/labels. Returns (detections, union_area_frac_by_label)."""
    h, w = hw
    dets: List[Detection] = []
    unions: Dict[str, np.ndarray] = {}

    for i, mask in enumerate(masks):
        if i >= len(boxes) or i >= len(labels):
            break
        lab = str(labels[i]) if labels[i] else "object"
        bb = boxes[i] if boxes[i] is not None else [0.0, 0.0, float(w), float(h)]

        if mask.dtype != np.bool_:
            mb = (mask > 0.5)
        else:
            mb = mask
        if mb.shape[:2] != (h, w):
            mb = cv2.resize(mb.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

        area = float(mb.sum())
        area_frac = area / float(h * w) if h > 0 and w > 0 else 0.0
        dets.append(Detection(label=lab, bbox_xyxy=[float(x) for x in bb[:4]], area_frac=area_frac, score=1.0))

        if lab not in unions:
            unions[lab] = mb.copy()
        else:
            unions[lab] |= mb

    union_fracs: Dict[str, float] = {}
    for lab, um in unions.items():
        union_fracs[lab] = float(um.sum()) / float(h * w) if h > 0 and w > 0 else 0.0

    return dets, union_fracs


def detections_from_yolo(
    boxes: List[List[float]],
    labels: List[str],
    scores: List[float],
    hw: Tuple[int, int],
) -> Tuple[List[Detection], Dict[str, float]]:
    """Build detections from YOLO boxes/labels/scores. Returns (detections, union_area_frac_by_label)."""
    h, w = hw
    dets: List[Detection] = []
    label_areas: Dict[str, float] = {}

    for i, box in enumerate(boxes):
        if i >= len(labels) or i >= len(scores):
            break
        lab = str(labels[i]) if labels[i] else "object"
        x1, y1, x2, y2 = box[:4]
        box_area = max(0.0, (x2 - x1) * (y2 - y1))
        area_frac = box_area / float(h * w) if h > 0 and w > 0 else 0.0

        dets.append(Detection(
            label=lab,
            bbox_xyxy=[float(x) for x in box[:4]],
            area_frac=area_frac,
            score=float(scores[i])
        ))

        # Accumulate area per label (simple sum, not union since no masks)
        label_areas[lab] = label_areas.get(lab, 0.0) + area_frac

    return dets, label_areas


class SimpleTracker:
    """A lightweight IoU-based tracker (greedy assignment) suitable for small detection counts.

    - Maintains persistent IDs
    - Smooths label confidence with EMA
    """

    def __init__(
        self,
        iou_thresh: float = 0.30,
        max_missed: int = 6,
        label_ema: float = 0.70,
        use_embedding: bool = False,
        iou_weight: float = 0.65,
        emb_weight: float = 0.35,
        min_sim: float = 0.15,
        emb_ema: float = 0.40,
    ):
        self.iou_thresh = float(iou_thresh)
        self.max_missed = int(max_missed)
        self.label_ema = float(label_ema)

        # Phase 3: optional ReID support
        self.use_embedding = bool(use_embedding)
        self.iou_weight = float(iou_weight)
        self.emb_weight = float(emb_weight)
        self.min_sim = float(min_sim)
        self.emb_ema = float(emb_ema)

        # normalize weights to sum to 1
        self.iou_weight = max(0.0, self.iou_weight)
        self.emb_weight = max(0.0, self.emb_weight)
        tot = self.iou_weight + self.emb_weight
        if tot <= 1e-9:
            self.iou_weight, self.emb_weight = 1.0, 0.0
        else:
            self.iou_weight /= tot
            self.emb_weight /= tot

        self._next_id = 1
        self.tracks: Dict[int, Track] = {}


    def _new_track(self, det: Detection, tick_id: int) -> Track:
        cx, cy = _bbox_center(det.bbox_xyxy)
        t = Track(
            id=self._next_id,
            bbox_xyxy=list(det.bbox_xyxy),
            area_frac=float(det.area_frac),
            label_probs={det.label: float(det.score)},
            cx=cx,
            cy=cy,
            vx=0.0,
            vy=0.0,
            age=1,
            hits=1,
            last_seen_tick=tick_id,
            emb=_norm_vec(det.emb) if det.emb is not None else None,
            emb_tick=tick_id if det.emb is not None else 0,
        )
        t.pos_hist.append((t.cx, t.cy))
        self._next_id += 1
        return t

    def update(self, dets: List[Detection], tick_id: int) -> Tuple[List[Track], List[Track], List[Track]]:
        """Update tracker with detections.

        Returns (active_tracks, created_tracks, removed_tracks).
        """
        # Age existing tracks
        for t in self.tracks.values():
            t.age += 1

        track_ids = list(self.tracks.keys())
        created: List[Track] = []
        removed: List[Track] = []

        assigned_det = set()
        assigned_track = set()

        # Build all candidate matches (greedy). Phase 3: optionally blend IoU + embedding similarity.
        cand = []
        for tid in track_ids:
            tr0 = self.tracks[tid]
            tb = tr0.bbox_xyxy
            for j, d in enumerate(dets):
                iou = _bbox_iou(tb, d.bbox_xyxy)
                sim01 = _cos_sim01(tr0.emb, d.emb) if self.use_embedding else None

                # Gate: normally require IoU >= threshold; if IoU is low, allow match only when embedding is strong.
                if iou < self.iou_thresh:
                    if sim01 is None or sim01 < self.min_sim:
                        continue

                cost = self.iou_weight * (1.0 - iou)
                if sim01 is not None:
                    cost += self.emb_weight * (1.0 - sim01)

                cand.append((cost, iou, (sim01 if sim01 is not None else -1.0), tid, j))

        cand.sort(key=lambda x: x[0])

        for cost, iou, sim01, tid, j in cand:
            if tid in assigned_track or j in assigned_det:
                continue
            assigned_track.add(tid)
            assigned_det.add(j)

            tr = self.tracks[tid]
            det = dets[j]

            # velocity from center delta
            ncx, ncy = _bbox_center(det.bbox_xyxy)
            tr.vx = float(ncx - tr.cx)
            tr.vy = float(ncy - tr.cy)
            tr.cx, tr.cy = ncx, ncy
            tr.speed = float(math.hypot(tr.vx, tr.vy))
            tr.pos_hist.append((tr.cx, tr.cy))

            tr.bbox_xyxy = list(det.bbox_xyxy)
            tr.area_frac = float(det.area_frac)
            tr.last_seen_tick = tick_id
            tr.hits += 1

            # label EMA update
            ema = self.label_ema
            for k in list(tr.label_probs.keys()):
                tr.label_probs[k] *= ema
            tr.label_probs[det.label] = tr.label_probs.get(det.label, 0.0) + (1.0 - ema) * float(det.score)


            # embedding EMA update (optional)
            if det.emb is not None:
                if tr.emb is None:
                    tr.emb = _norm_vec(det.emb)
                else:
                    a = float(self.emb_ema)
                    tr.emb = _norm_vec((1.0 - a) * tr.emb + a * det.emb)
                tr.emb_tick = tick_id

        # Unmatched detections -> new tracks
        for j, det in enumerate(dets):
            if j in assigned_det:
                continue
            nt = self._new_track(det, tick_id)
            self.tracks[nt.id] = nt
            created.append(nt)

        # Retire tracks not seen recently
        for tid in list(self.tracks.keys()):
            tr = self.tracks[tid]
            if (tick_id - tr.last_seen_tick) > self.max_missed:
                removed.append(tr)
                del self.tracks[tid]

        active = sorted(self.tracks.values(), key=lambda t: t.id)
        return active, created, removed


class KalmanBoxTracker:
    """Kalman filter for tracking bounding boxes in image space.

    State vector: [x_center, y_center, area, aspect_ratio, vx, vy, va, var]
    where v* are velocities.
    """

    def __init__(self, bbox: List[float]):
        """Initialize tracker with bounding box [x1, y1, x2, y2]."""
        # Convert bbox to [x_center, y_center, area, aspect_ratio]
        x1, y1, x2, y2 = bbox[:4]
        w = x2 - x1
        h = y2 - y1
        x_c = x1 + w / 2
        y_c = y1 + h / 2
        area = w * h
        aspect = w / max(h, 1e-6)

        # State: [x, y, a, r, vx, vy, va, vr]
        self.state = np.array([x_c, y_c, area, aspect, 0, 0, 0, 0], dtype=np.float64)

        # State transition matrix (constant velocity model)
        self.F = np.eye(8, dtype=np.float64)
        self.F[0, 4] = 1  # x += vx
        self.F[1, 5] = 1  # y += vy
        self.F[2, 6] = 1  # a += va
        self.F[3, 7] = 1  # r += vr

        # Measurement matrix (we observe [x, y, a, r])
        self.H = np.zeros((4, 8), dtype=np.float64)
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.H[3, 3] = 1

        # Covariance matrix
        self.P = np.eye(8, dtype=np.float64) * 10
        self.P[4:, 4:] *= 100  # Higher uncertainty for velocities

        # Process noise
        self.Q = np.eye(8, dtype=np.float64)
        self.Q[0:4, 0:4] *= 1.0
        self.Q[4:8, 4:8] *= 0.01

        # Measurement noise
        self.R = np.eye(4, dtype=np.float64) * 1.0

        self.time_since_update = 0
        self.hit_streak = 0
        self.age = 0

    def predict(self) -> List[float]:
        """Advance state and return predicted bbox [x1, y1, x2, y2]."""
        # Predict state
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Ensure area stays positive
        if self.state[2] < 1:
            self.state[2] = 1

        self.age += 1
        self.time_since_update += 1

        return self._state_to_bbox()

    def update(self, bbox: List[float]):
        """Update state with observed bbox [x1, y1, x2, y2]."""
        # Convert bbox to measurement
        x1, y1, x2, y2 = bbox[:4]
        w = x2 - x1
        h = y2 - y1
        x_c = x1 + w / 2
        y_c = y1 + h / 2
        area = w * h
        aspect = w / max(h, 1e-6)
        z = np.array([x_c, y_c, area, aspect], dtype=np.float64)

        # Kalman update
        y = z - self.H @ self.state  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.state = self.state + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

        self.time_since_update = 0
        self.hit_streak += 1

    def _state_to_bbox(self) -> List[float]:
        """Convert state [x, y, a, r, ...] to bbox [x1, y1, x2, y2]."""
        x_c, y_c, area, aspect = self.state[:4]
        area = max(1, area)
        aspect = max(0.1, min(10, aspect))

        h = np.sqrt(area / aspect)
        w = aspect * h

        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2

        return [float(x1), float(y1), float(x2), float(y2)]

    def get_state(self) -> List[float]:
        """Get current bbox estimate [x1, y1, x2, y2]."""
        return self._state_to_bbox()


class KalmanTracker:
    """Tracker using Kalman filter for motion prediction and Hungarian algorithm for assignment.

    This is a Deep SORT-style tracker with:
    - Kalman filter for motion prediction
    - Hungarian (optimal) assignment
    - Optional embedding-based ReID
    - Graveyard for track resurrection
    """

    def __init__(
        self,
        iou_thresh: float = 0.30,
        max_missed: int = 6,
        label_ema: float = 0.70,
        use_embedding: bool = False,
        iou_weight: float = 0.65,
        emb_weight: float = 0.35,
        min_sim: float = 0.15,
        emb_ema: float = 0.40,
        graveyard_seconds: float = 10.0,
        use_hungarian: bool = True,
    ):
        self.iou_thresh = float(iou_thresh)
        self.max_missed = int(max_missed)
        self.label_ema = float(label_ema)
        self.use_embedding = bool(use_embedding)
        self.emb_ema = float(emb_ema)
        self.min_sim = float(min_sim)
        self.graveyard_seconds = float(graveyard_seconds)
        self.use_hungarian = bool(use_hungarian)

        # Normalize weights
        self.iou_weight = max(0.0, float(iou_weight))
        self.emb_weight = max(0.0, float(emb_weight))
        tot = self.iou_weight + self.emb_weight
        if tot <= 1e-9:
            self.iou_weight, self.emb_weight = 1.0, 0.0
        else:
            self.iou_weight /= tot
            self.emb_weight /= tot

        self._next_id = 1
        self.tracks: Dict[int, Track] = {}
        self._kalman: Dict[int, KalmanBoxTracker] = {}  # Kalman filter per track
        self._graveyard: Dict[int, Tuple[Track, float]] = {}  # (track, death_time)

    def _new_track(self, det: Detection, tick_id: int) -> Track:
        cx, cy = _bbox_center(det.bbox_xyxy)
        t = Track(
            id=self._next_id,
            bbox_xyxy=list(det.bbox_xyxy),
            area_frac=float(det.area_frac),
            label_probs={det.label: float(det.score)},
            cx=cx,
            cy=cy,
            vx=0.0,
            vy=0.0,
            age=1,
            hits=1,
            last_seen_tick=tick_id,
            emb=_norm_vec(det.emb) if det.emb is not None else None,
            emb_tick=tick_id if det.emb is not None else 0,
        )
        t.pos_hist.append((t.cx, t.cy))

        # Create Kalman filter for this track
        self._kalman[self._next_id] = KalmanBoxTracker(det.bbox_xyxy)

        self._next_id += 1
        return t

    def _try_resurrect(self, det: Detection, tick_id: int, now: float) -> Optional[Track]:
        """Try to match detection to a graveyard track for resurrection."""
        if not self.use_embedding or det.emb is None:
            return None

        # Prune old graveyard entries
        cutoff = now - self.graveyard_seconds
        self._graveyard = {k: v for k, v in self._graveyard.items() if v[1] > cutoff}

        best_match = None
        best_sim = self.min_sim

        for tid, (tr, death_time) in self._graveyard.items():
            if tr.emb is None:
                continue
            sim = _cos_sim01(tr.emb, det.emb)
            if sim is not None and sim > best_sim:
                best_sim = sim
                best_match = tid

        if best_match is not None:
            tr, _ = self._graveyard.pop(best_match)
            # Resurrect with new detection
            cx, cy = _bbox_center(det.bbox_xyxy)
            tr.bbox_xyxy = list(det.bbox_xyxy)
            tr.area_frac = float(det.area_frac)
            tr.cx, tr.cy = cx, cy
            tr.last_seen_tick = tick_id
            tr.hits += 1
            tr.age += 1
            tr.pos_hist.append((cx, cy))

            # Update embedding
            if det.emb is not None:
                if tr.emb is None:
                    tr.emb = _norm_vec(det.emb)
                else:
                    tr.emb = _norm_vec((1 - self.emb_ema) * tr.emb + self.emb_ema * det.emb)
                tr.emb_tick = tick_id

            # Reinitialize Kalman filter
            self._kalman[tr.id] = KalmanBoxTracker(det.bbox_xyxy)

            return tr

        return None

    def _hungarian_assign(
        self, cost_matrix: np.ndarray, threshold: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Optimal assignment using Hungarian algorithm.

        Returns: (matches, unmatched_tracks, unmatched_dets)
        """
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            # Fallback to greedy if scipy not available
            return self._greedy_assign(cost_matrix, threshold)

        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_tracks = list(range(cost_matrix.shape[0]))
        unmatched_dets = list(range(cost_matrix.shape[1]))

        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] < threshold:
                matches.append((r, c))
                unmatched_tracks.remove(r)
                unmatched_dets.remove(c)

        return matches, unmatched_tracks, unmatched_dets

    def _greedy_assign(
        self, cost_matrix: np.ndarray, threshold: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Greedy assignment fallback."""
        matches = []
        unmatched_tracks = set(range(cost_matrix.shape[0]))
        unmatched_dets = set(range(cost_matrix.shape[1]))

        # Get all valid candidates
        candidates = []
        for i in range(cost_matrix.shape[0]):
            for j in range(cost_matrix.shape[1]):
                if cost_matrix[i, j] < threshold:
                    candidates.append((cost_matrix[i, j], i, j))

        candidates.sort()

        for cost, i, j in candidates:
            if i in unmatched_tracks and j in unmatched_dets:
                matches.append((i, j))
                unmatched_tracks.remove(i)
                unmatched_dets.remove(j)

        return matches, list(unmatched_tracks), list(unmatched_dets)

    def update(self, dets: List[Detection], tick_id: int) -> Tuple[List[Track], List[Track], List[Track]]:
        """Update tracker with detections.

        Returns (active_tracks, created_tracks, removed_tracks).
        """
        now = time.time()

        # Age existing tracks
        for t in self.tracks.values():
            t.age += 1

        track_ids = list(self.tracks.keys())
        created: List[Track] = []
        removed: List[Track] = []

        # Predict next positions using Kalman filters
        predicted_boxes: Dict[int, List[float]] = {}
        for tid in track_ids:
            if tid in self._kalman:
                predicted_boxes[tid] = self._kalman[tid].predict()
            else:
                predicted_boxes[tid] = self.tracks[tid].bbox_xyxy

        # Build cost matrix
        n_tracks = len(track_ids)
        n_dets = len(dets)

        if n_tracks > 0 and n_dets > 0:
            cost_matrix = np.ones((n_tracks, n_dets), dtype=np.float64)

            for i, tid in enumerate(track_ids):
                tr = self.tracks[tid]
                pred_box = predicted_boxes[tid]

                for j, det in enumerate(dets):
                    iou = _bbox_iou(pred_box, det.bbox_xyxy)
                    sim01 = _cos_sim01(tr.emb, det.emb) if self.use_embedding else None

                    # Gate: require IoU >= threshold OR strong embedding match
                    if iou < self.iou_thresh:
                        if sim01 is None or sim01 < self.min_sim:
                            cost_matrix[i, j] = 1.0  # High cost = no match
                            continue

                    cost = self.iou_weight * (1.0 - iou)
                    if sim01 is not None:
                        cost += self.emb_weight * (1.0 - sim01)

                    cost_matrix[i, j] = cost

            # Assignment
            if self.use_hungarian:
                matches, unmatched_track_idx, unmatched_det_idx = self._hungarian_assign(
                    cost_matrix, threshold=1.0 - self.iou_thresh
                )
            else:
                matches, unmatched_track_idx, unmatched_det_idx = self._greedy_assign(
                    cost_matrix, threshold=1.0 - self.iou_thresh
                )
        else:
            matches = []
            unmatched_track_idx = list(range(n_tracks))
            unmatched_det_idx = list(range(n_dets))

        # Update matched tracks
        for track_idx, det_idx in matches:
            tid = track_ids[track_idx]
            det = dets[det_idx]
            tr = self.tracks[tid]

            # Update Kalman filter
            if tid in self._kalman:
                self._kalman[tid].update(det.bbox_xyxy)
                smoothed_box = self._kalman[tid].get_state()
            else:
                smoothed_box = det.bbox_xyxy

            # Velocity from center delta
            ncx, ncy = _bbox_center(smoothed_box)
            tr.vx = float(ncx - tr.cx)
            tr.vy = float(ncy - tr.cy)
            tr.cx, tr.cy = ncx, ncy
            tr.speed = float(math.hypot(tr.vx, tr.vy))
            tr.pos_hist.append((tr.cx, tr.cy))

            tr.bbox_xyxy = list(smoothed_box)
            tr.area_frac = float(det.area_frac)
            tr.last_seen_tick = tick_id
            tr.hits += 1

            # Label EMA update
            ema = self.label_ema
            for k in list(tr.label_probs.keys()):
                tr.label_probs[k] *= ema
            tr.label_probs[det.label] = tr.label_probs.get(det.label, 0.0) + (1.0 - ema) * float(det.score)

            # Embedding EMA update
            if det.emb is not None:
                if tr.emb is None:
                    tr.emb = _norm_vec(det.emb)
                else:
                    tr.emb = _norm_vec((1.0 - self.emb_ema) * tr.emb + self.emb_ema * det.emb)
                tr.emb_tick = tick_id

        # Unmatched detections -> try resurrection or create new tracks
        for det_idx in unmatched_det_idx:
            det = dets[det_idx]

            # Try to resurrect from graveyard
            resurrected = self._try_resurrect(det, tick_id, now)
            if resurrected is not None:
                self.tracks[resurrected.id] = resurrected
                created.append(resurrected)
            else:
                # Create new track
                nt = self._new_track(det, tick_id)
                self.tracks[nt.id] = nt
                created.append(nt)

        # Retire tracks not seen recently
        for track_idx in unmatched_track_idx:
            tid = track_ids[track_idx]
            tr = self.tracks[tid]
            if (tick_id - tr.last_seen_tick) > self.max_missed:
                removed.append(tr)
                # Move to graveyard for potential resurrection
                if self.use_embedding and tr.emb is not None:
                    self._graveyard[tid] = (tr, now)
                del self.tracks[tid]
                if tid in self._kalman:
                    del self._kalman[tid]

        active = sorted(self.tracks.values(), key=lambda t: t.id)
        return active, created, removed


def _get_track_color(tr: Track, max_missed: int = 6) -> Tuple[int, int, int]:
    """Get color based on track health status.

    Returns BGR color:
    - Green (0, 255, 0): Healthy, recently seen
    - Yellow (0, 255, 255): Moderate, seen a few frames ago
    - Orange (0, 165, 255): Warning, about to be dropped
    - Red (0, 0, 255): Critical, almost dropped
    - Purple (255, 0, 255): Resurrected from graveyard (high hits after gap)
    """
    # Check for resurrection pattern: high hits but track is young relative to hits
    if tr.hits > 3 and tr.age > tr.hits * 2:
        return (255, 0, 255)  # Purple - resurrected

    # frames_missed approximated by age - hits (not perfect but gives indication)
    frames_since_update = tr.age - tr.hits
    miss_ratio = frames_since_update / max(1, max_missed)

    if miss_ratio < 0.3:
        return (0, 255, 0)  # Green - healthy
    elif miss_ratio < 0.5:
        return (0, 255, 255)  # Yellow - moderate
    elif miss_ratio < 0.75:
        return (0, 165, 255)  # Orange - warning
    else:
        return (0, 0, 255)  # Red - critical


def tracks_to_overlay(
    bgr: np.ndarray,
    tracks: List[Track],
    color: Tuple[int, int, int] = (255, 255, 0),
    show_trails: bool = True,
    color_by_status: bool = True,
    max_missed: int = 6,
) -> np.ndarray:
    """Draw track boxes with optional trails and status-based coloring."""
    if not tracks:
        return bgr
    out = bgr.copy()
    h, w = out.shape[:2]

    for tr in tracks:
        # Determine color
        if color_by_status:
            box_color = _get_track_color(tr, max_missed)
        else:
            box_color = color

        # Draw trajectory trail (fading polyline)
        if show_trails and len(tr.pos_hist) > 1:
            pts = list(tr.pos_hist)
            for i in range(len(pts) - 1):
                # Fade from dim (old) to bright (recent)
                alpha = (i + 1) / len(pts)
                trail_color = tuple(int(c * alpha) for c in box_color)
                p1 = (int(pts[i][0]), int(pts[i][1]))
                p2 = (int(pts[i + 1][0]), int(pts[i + 1][1]))
                cv2.line(out, p1, p2, trail_color, thickness=2, lineType=cv2.LINE_AA)

        # Draw bounding box
        x1, y1, x2, y2 = tr.bbox_xyxy[:4]
        x1 = int(max(0, min(w - 1, round(x1))))
        y1 = int(max(0, min(h - 1, round(y1))))
        x2 = int(max(0, min(w - 1, round(x2))))
        y2 = int(max(0, min(h - 1, round(y2))))
        cv2.rectangle(out, (x1, y1), (x2, y2), color=box_color, thickness=2)

        # Draw label
        lab, conf = tr.top_label()
        txt = f"{tr.id}:{lab} {conf*100:.0f}%"
        cv2.putText(out, txt, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2, cv2.LINE_AA)

    return out


def make_tracking_view(
    bgr: np.ndarray,
    tracks: List[Track],
    max_missed: int = 6,
    show_velocity: bool = True,
    show_trails: bool = True,
) -> np.ndarray:
    """Create dedicated tracking visualization with all track info.

    Shows:
    - Color-coded boxes by track health
    - Trajectory trails
    - Velocity vectors
    - Track statistics
    """
    if bgr is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    out = bgr.copy()
    h, w = out.shape[:2]

    for tr in tracks:
        box_color = _get_track_color(tr, max_missed)

        # Draw trajectory trail
        if show_trails and len(tr.pos_hist) > 1:
            pts = list(tr.pos_hist)
            for i in range(len(pts) - 1):
                alpha = (i + 1) / len(pts)
                trail_color = tuple(int(c * alpha) for c in box_color)
                p1 = (int(pts[i][0]), int(pts[i][1]))
                p2 = (int(pts[i + 1][0]), int(pts[i + 1][1]))
                cv2.line(out, p1, p2, trail_color, thickness=2, lineType=cv2.LINE_AA)

        # Draw bounding box
        x1, y1, x2, y2 = tr.bbox_xyxy[:4]
        x1 = int(max(0, min(w - 1, round(x1))))
        y1 = int(max(0, min(h - 1, round(y1))))
        x2 = int(max(0, min(w - 1, round(x2))))
        y2 = int(max(0, min(h - 1, round(y2))))
        cv2.rectangle(out, (x1, y1), (x2, y2), color=box_color, thickness=2)

        # Draw velocity vector
        if show_velocity and (abs(tr.vx) > 1 or abs(tr.vy) > 1):
            cx, cy = int(tr.cx), int(tr.cy)
            # Scale velocity for visibility (multiply by 5)
            vx_scaled = int(tr.vx * 5)
            vy_scaled = int(tr.vy * 5)
            end_x = max(0, min(w - 1, cx + vx_scaled))
            end_y = max(0, min(h - 1, cy + vy_scaled))
            cv2.arrowedLine(out, (cx, cy), (end_x, end_y), (255, 255, 255), 2, tipLength=0.3)

        # Draw label with more info
        lab, conf = tr.top_label()
        txt = f"{tr.id}:{lab} {conf*100:.0f}%"
        cv2.putText(out, txt, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2, cv2.LINE_AA)

        # Draw track stats below box
        stats_txt = f"age:{tr.age} hits:{tr.hits}"
        cv2.putText(out, stats_txt, (x1, min(h - 4, y2 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1, cv2.LINE_AA)

    # Draw legend in top-left corner
    legend_y = 20
    cv2.putText(out, "Track Status:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    legend_items = [
        ((0, 255, 0), "Healthy"),
        ((0, 255, 255), "Moderate"),
        ((0, 165, 255), "Warning"),
        ((0, 0, 255), "Critical"),
        ((255, 0, 255), "Resurrected"),
    ]
    for i, (col, label) in enumerate(legend_items):
        y = legend_y + 18 + i * 16
        cv2.rectangle(out, (10, y - 10), (22, y + 2), col, -1)
        cv2.putText(out, label, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Draw track count
    cv2.putText(out, f"Tracks: {len(tracks)}", (w - 100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return out


def detect_relationships(tracks: List[Track], frame_hw: Tuple[int, int]) -> List[Tuple[str, str, str]]:
    """Detect semantic relationships between tracked objects.

    Returns list of (subject_label, relationship, object_label) tuples.
    Relationships: NEAR, HOLDING, ABOVE, BELOW, LEFT_OF, RIGHT_OF, MOVING_TOWARD
    """
    if not tracks or len(tracks) < 2:
        return []

    h, w = frame_hw
    diag = math.hypot(w, h) if w > 0 and h > 0 else 1.0

    # Filter stable tracks with sufficient confidence
    stable = [t for t in tracks if t.hits >= 2]
    if len(stable) < 2:
        return []

    relationships: List[Tuple[str, str, str]] = []

    # Distance thresholds (fraction of diagonal)
    near_thresh = 0.15 * diag
    contact_thresh = 0.08 * diag

    for i, t1 in enumerate(stable):
        lab1, conf1 = t1.top_label()
        if conf1 < 0.30:
            continue

        for t2 in stable[i + 1:]:
            lab2, conf2 = t2.top_label()
            if conf2 < 0.30:
                continue

            # Calculate distance and relative position
            dx = t2.cx - t1.cx
            dy = t2.cy - t1.cy
            dist = math.hypot(dx, dy)

            # NEAR relationship
            if dist < near_thresh:
                relationships.append((lab1, "NEAR", lab2))

            # HOLDING: hand + object in contact
            if dist < contact_thresh:
                if lab1 == "hand" and lab2 not in ("hand", "person"):
                    relationships.append((lab1, "HOLDING", lab2))
                elif lab2 == "hand" and lab1 not in ("hand", "person"):
                    relationships.append((lab2, "HOLDING", lab1))

            # Spatial relationships (only for nearby objects)
            if dist < near_thresh * 2:
                # Vertical relationships
                if abs(dy) > abs(dx) * 1.5:  # Primarily vertical
                    if dy > 0:
                        relationships.append((lab1, "ABOVE", lab2))
                    else:
                        relationships.append((lab1, "BELOW", lab2))
                # Horizontal relationships
                elif abs(dx) > abs(dy) * 1.5:  # Primarily horizontal
                    if dx > 0:
                        relationships.append((lab1, "LEFT_OF", lab2))
                    else:
                        relationships.append((lab1, "RIGHT_OF", lab2))

            # MOVING_TOWARD: check velocity direction
            if t1.speed > 0.01 * diag and dist < near_thresh * 3:
                # Normalize velocity and direction to target
                v_mag = math.hypot(t1.vx, t1.vy)
                if v_mag > 1e-6:
                    v_norm = (t1.vx / v_mag, t1.vy / v_mag)
                    d_norm = (dx / dist, dy / dist) if dist > 1e-6 else (0, 0)
                    # Dot product: positive means moving toward
                    dot = v_norm[0] * d_norm[0] + v_norm[1] * d_norm[1]
                    if dot > 0.7:  # Moving mostly toward
                        relationships.append((lab1, "MOVING_TOWARD", lab2))

    return relationships


# Narration attribution for color-coded display
@dataclass
class NarrationSegment:
    """A segment of narration text with its source attribution."""
    text: str
    source: str  # "dino", "sam", "vjepa", "relationship", "context", "caption"


@dataclass
class AttributedNarration:
    """Narration with source attribution for each segment."""
    segments: List[NarrationSegment]
    plain_text: str  # For backward compatibility

    def to_html(self) -> str:
        """Convert to HTML with color-coded spans."""
        SOURCE_COLORS = {
            "dino": "#FF8C00",      # Orange - attention/heatmap
            "sam": "#32CD32",       # Green - detected objects
            "vjepa": "#1E90FF",     # Blue - actions
            "relationship": "#9370DB",  # Purple - relationships
            "context": "#808080",   # Gray - temporal context
            "caption": "#FFD700",   # Gold - caption/scene
        }
        html_parts = []
        for seg in self.segments:
            color = SOURCE_COLORS.get(seg.source, "#ddd")
            escaped = seg.text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html_parts.append(f'<span style="color:{color}">{escaped}</span>')
        return "".join(html_parts)


class NarrationEngine:
    """Enhanced narration engine with temporal context and natural language output."""

    def __init__(self, inference_fps: float = 2.0):
        self.inference_fps = max(0.1, inference_fps)
        # Temporal tracking
        self._current_action: Optional[str] = None
        self._action_start_tick: int = 0
        self._current_objects: Set[str] = set()
        self._objects_start_tick: int = 0
        self._last_caption: str = ""
        self._scene_state: str = "idle"  # idle, active, transitioning
        self._last_tick: int = 0

    def _ticks_to_seconds(self, ticks: int) -> float:
        return ticks / self.inference_fps

    def _format_duration(self, seconds: float) -> str:
        if seconds < 2:
            return "just now"
        elif seconds < 60:
            return f"for {int(seconds)}s"
        else:
            mins = int(seconds // 60)
            return f"for {mins}m"

    def compose(
        self,
        tracks: List[Track],
        action_top: Optional[Tuple[str, float]],
        frame_hw: Tuple[int, int],
        flags: Optional[Dict[str, Any]] = None,
        caption: Optional[str] = None,
        tick_id: int = 0,
    ) -> AttributedNarration:
        """Produce a natural language scene description with temporal context and source attribution."""
        flags = flags or {}

        # Filter stable tracks
        stable = [t for t in tracks if t.hits >= 2 and t.last_seen_tick > 0]

        # Group by label
        by_label: Dict[str, List[Track]] = {}
        for t in stable:
            lab, conf = t.top_label()
            if conf < 0.30:
                continue
            by_label.setdefault(lab, []).append(t)

        def present(label_name: str) -> bool:
            return label_name in by_label and len(by_label[label_name]) > 0

        # Track object set changes
        current_objects = set(by_label.keys())
        if current_objects != self._current_objects:
            self._current_objects = current_objects
            self._objects_start_tick = tick_id

        # Track action changes
        current_action = action_top[0] if action_top and action_top[1] > 0.40 else None
        if current_action != self._current_action:
            self._current_action = current_action
            self._action_start_tick = tick_id

        # Determine scene state
        has_person = present('person')
        has_activity = current_action is not None or flags.get('typing') or flags.get('phone_held')

        if has_person and has_activity:
            self._scene_state = "active"
        elif has_person:
            self._scene_state = "idle"
        elif stable:
            self._scene_state = "observing"
        else:
            self._scene_state = "empty"

        # Build natural language sentences with source attribution
        segments: List[NarrationSegment] = []

        # Subject sentence (from SAM object detection)
        if has_person:
            person_count = len(by_label.get('person', []))
            if person_count > 1:
                segments.append(NarrationSegment(f"{person_count} people in view", "sam"))
            else:
                segments.append(NarrationSegment("Person in view", "sam"))

        # Activity sentence with duration
        activity_parts = []
        if flags.get('typing'):
            activity_parts.append("typing")
        if flags.get('phone_held'):
            activity_parts.append("using phone")
        if flags.get('cup_held'):
            activity_parts.append("holding cup")

        if activity_parts:
            # Interaction flags are from SAM object relationships
            segments.append(NarrationSegment(", ".join(activity_parts), "relationship"))
        elif current_action and action_top:
            # V-JEPA action detection
            action_duration = self._ticks_to_seconds(tick_id - self._action_start_tick)
            action_name = humanize_vjepa_label(current_action)
            conf_str = f"{action_top[1]*100:.0f}%"
            if action_duration >= 2:
                segments.append(NarrationSegment(f"{action_name} ({conf_str}, {self._format_duration(action_duration)})", "vjepa"))
            else:
                segments.append(NarrationSegment(f"{action_name} ({conf_str})", "vjepa"))

        # Objects sentence (excluding person/hand) - from SAM
        salient_objects = []
        for lab, ts in by_label.items():
            if lab in ('person', 'hand'):
                continue
            count = len(ts)
            if count > 1:
                salient_objects.append(f"{count}× {lab}")
            else:
                salient_objects.append(lab)

        if salient_objects:
            segments.append(NarrationSegment("Objects: " + ", ".join(salient_objects[:5]), "sam"))

        # Semantic relationships between objects
        relationships = detect_relationships(stable, frame_hw)
        if relationships:
            # Format: "phone NEAR hand" -> "phone near hand"
            rel_strs = [f"{s} {r.lower().replace('_', ' ')} {o}" for s, r, o in relationships[:3]]
            segments.append(NarrationSegment("Relations: " + ", ".join(rel_strs), "relationship"))

        # Integrate caption (extract novel info)
        if caption and caption.strip():
            cap_lower = caption.lower()
            # Only add caption info if it provides new context not in objects
            novel = True
            for obj in current_objects:
                if obj.lower() in cap_lower:
                    novel = False
                    break
            if novel or len(segments) < 2:
                # Extract scene context from caption (first clause typically most useful)
                cap_short = shorten_text(caption, 60)
                segments.append(NarrationSegment(f"Scene: {cap_short}", "caption"))

        self._last_tick = tick_id
        self._last_caption = caption or ""

        # Build plain text for backward compatibility
        plain_parts = [seg.text for seg in segments]
        plain_text = ". ".join(plain_parts) if plain_parts else "—"

        return AttributedNarration(segments=segments, plain_text=plain_text)


# Keep the old function for backward compatibility but delegate to NarrationEngine
_default_narration_engine: Optional[NarrationEngine] = None


def compose_narration_attributed(
    tracks: List[Track],
    action_top: Optional[Tuple[str, float]],
    frame_hw: Tuple[int, int],
    flags: Optional[Dict[str, Any]] = None,
    caption: Optional[str] = None,
    tick_id: int = 0,
) -> AttributedNarration:
    """Produce a natural language scene description with source attribution."""
    global _default_narration_engine
    if _default_narration_engine is None:
        _default_narration_engine = NarrationEngine()
    return _default_narration_engine.compose(
        tracks=tracks,
        action_top=action_top,
        frame_hw=frame_hw,
        flags=flags,
        caption=caption,
        tick_id=tick_id,
    )


def compose_narration(
    tracks: List[Track],
    action_top: Optional[Tuple[str, float]],
    frame_hw: Tuple[int, int],
    flags: Optional[Dict[str, Any]] = None,
    caption: Optional[str] = None,
    tick_id: int = 0,
) -> str:
    """Produce a natural language scene description (backward-compatible string)."""
    return compose_narration_attributed(
        tracks=tracks,
        action_top=action_top,
        frame_hw=frame_hw,
        flags=flags,
        caption=caption,
        tick_id=tick_id,
    ).plain_text

def fmt_ms(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{x:.1f}"


# ----------------------------
# Stage base: dtype + autocast
# ----------------------------

@dataclass
class StageRuntime:
    device: torch.device
    dtype: torch.dtype
    amp_enabled: bool


def pick_runtime(device: torch.device, fp16_on_mps: bool) -> StageRuntime:
    if device.type == "mps" and fp16_on_mps:
        return StageRuntime(device=device, dtype=torch.float16, amp_enabled=True)
    if device.type == "cuda":
        return StageRuntime(device=device, dtype=torch.float16, amp_enabled=True)
    return StageRuntime(device=device, dtype=torch.float32, amp_enabled=False)


def to_device_dtype(batch: Dict[str, Any], runtime: StageRuntime) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            v2 = v.to(runtime.device)
            if is_float_tensor(v2) and v2.dtype != runtime.dtype:
                v2 = v2.to(dtype=runtime.dtype)
            out[k] = v2
        else:
            out[k] = v
    return out


# ----------------------------
# DINOv3 stage
# ----------------------------

class DinoV3Stage:
    def __init__(self, model_id: str, runtime: StageRuntime, hf_token: Optional[str]):
        from transformers import AutoImageProcessor, AutoModel
        self.proc = hf_from_pretrained(AutoImageProcessor, model_id, hf_token=hf_token, use_fast=True)
        self.model = hf_from_pretrained(AutoModel, model_id, hf_token=hf_token)
        self.model = self.model.to(device=runtime.device, dtype=runtime.dtype)
        self.model.eval()
        self.rt = runtime

    @torch.inference_mode()
    def saliency_heatmap(
        self,
        rgb: np.ndarray,
        saliency_threshold: float = 0.02,
        mode: str = "inverse",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute saliency heatmap from DINO using one of three methods.

        Args:
            rgb: Input image as RGB numpy array
            saliency_threshold: Minimum raw_std to consider saliency "confident"
            mode: Saliency computation method:
                - "attention": CLS token attention weights (where model looks)
                - "inverse": Patches DISSIMILAR to image average (unusual = salient)
                - "magnitude": Patch feature vector magnitude (high activation = interesting)
        """
        t0 = time.perf_counter()
        pil = Image.fromarray(rgb)

        inputs = self.proc(images=pil, return_tensors="pt")
        inputs = to_device_dtype(inputs, self.rt)
        t1 = time.perf_counter()

        # Forward pass - only request attentions if needed
        need_attentions = (mode == "attention")
        with autocast_ctx(self.rt.device, self.rt.dtype, self.rt.amp_enabled):
            outputs = self.model(**inputs, output_attentions=need_attentions)
        t2 = time.perf_counter()

        last = outputs.last_hidden_state  # [B, seq, C]
        B, seq, c = last.shape

        # Compute expected patch grid from the processor output size.
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            raise RuntimeError('DINO inputs missing pixel_values')
        _, _, ih, iw = pixel_values.shape
        patch = int(getattr(self.model.config, "patch_size", 16) or 16)
        ph, pw = max(1, ih // patch), max(1, iw // patch)
        num_patches = ph * pw

        # Register tokens: prefer config, otherwise infer from sequence length.
        num_regs = int(getattr(self.model.config, "num_register_tokens", 0) or 0)
        auto_regs = max(0, int(seq) - 1 - int(num_patches))
        if num_regs <= 0 and 0 < auto_regs < 64:
            num_regs = auto_regs

        patch_start = 1 + num_regs
        patch_end = min(int(seq), patch_start + int(num_patches))
        p = patch_end - patch_start

        # Clean hidden states for feature-based modes
        last_f = torch.nan_to_num(last.float(), nan=0.0, posinf=0.0, neginf=0.0)
        patches = last_f[0, patch_start:patch_end, :]  # [p, C]

        actual_mode = mode

        if mode == "attention":
            # Use CLS token's attention weights to patches
            attentions = outputs.attentions
            if attentions is not None and len(attentions) > 0:
                # Use last 4 layers averaged for more stable saliency
                n_layers_to_use = min(4, len(attentions))
                attn_layers = attentions[-n_layers_to_use:]

                # Stack and average: [n_layers, B, heads, seq, seq] -> [B, heads, seq, seq]
                attn_stack = torch.stack(attn_layers, dim=0).mean(dim=0)

                # Get CLS attention to patches: [heads, patch_tokens]
                cls_attn = attn_stack[0, :, 0, patch_start:patch_end]

                # Average across heads
                scores = cls_attn.mean(dim=0)  # [patch_tokens]
                scores_np = scores.detach().cpu().float().numpy()
            else:
                # Fallback to inverse if no attentions
                actual_mode = "inverse_fallback"
                mode = "inverse"

        if mode == "inverse":
            # Score patches by how DIFFERENT they are from the mean (unusual = salient)
            # Normalize patches first
            patches_norm = torch.nn.functional.normalize(patches, dim=-1)
            mean_patch = patches_norm.mean(dim=0, keepdim=True)  # [1, C]
            mean_patch_norm = torch.nn.functional.normalize(mean_patch, dim=-1)

            # Cosine similarity to mean (will be high for "average" patches)
            similarity = (patches_norm * mean_patch_norm).sum(dim=-1)  # [p]

            # INVERT: low similarity to mean = high saliency (unusual patches)
            # Transform from [-1, 1] similarity to [0, 1] saliency
            scores = 1.0 - (similarity + 1.0) / 2.0  # Now high = unusual
            scores_np = scores.detach().cpu().float().numpy()
            if actual_mode != "inverse_fallback":
                actual_mode = "inverse"

        elif mode == "magnitude":
            # Score patches by feature vector magnitude (high activation = interesting)
            # Use L2 norm of each patch's feature vector
            magnitudes = torch.norm(patches, dim=-1)  # [p]
            scores_np = magnitudes.detach().cpu().float().numpy()
            actual_mode = "magnitude"

        scores_np = np.nan_to_num(scores_np, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute raw score statistics BEFORE normalization to assess true saliency
        raw_std = float(np.std(scores_np)) if scores_np.size else 0.0
        raw_range = float(np.ptp(scores_np)) if scores_np.size else 0.0
        raw_max = float(np.max(scores_np)) if scores_np.size else 0.0
        raw_mean = float(np.mean(scores_np)) if scores_np.size else 0.0

        # Normalize using quantiles (robust to outliers)
        if scores_np.size:
            lo = float(np.quantile(scores_np, 0.05))
            hi = float(np.quantile(scores_np, 0.95))
            if hi > lo + 1e-6:
                scores_np = (scores_np - lo) / (hi - lo)
            else:
                # fallback min/max
                lo2 = float(scores_np.min())
                hi2 = float(scores_np.max())
                scores_np = (scores_np - lo2) / ((hi2 - lo2) + 1e-6)
        scores_np = np.clip(scores_np, 0.0, 1.0)

        # Reshape into a grid. If counts don't match, infer a grid from sqrt(P).
        if p != ph * pw:
            side = int(round((p) ** 0.5))
            ph, pw = max(1, side), max(1, p // max(1, side))
            scores_np = scores_np[: ph * pw]

        heat = scores_np.reshape(ph, pw)
        heat = cv2.resize(heat, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
        heat = np.nan_to_num(heat, nan=0.0, posinf=1.0, neginf=0.0)
        heat = np.clip(heat, 0.0, 1.0).astype(np.float32)

        flat = heat.reshape(-1)
        p50 = float(np.quantile(flat, 0.50)) if flat.size else 0.0
        p90 = float(np.quantile(flat, 0.90)) if flat.size else 0.0
        p99 = float(np.quantile(flat, 0.99)) if flat.size else 0.0

        info = {
            "pre_ms": (t1 - t0) * 1000.0,
            "fwd_ms": (t2 - t1) * 1000.0,
            "total_ms": (t2 - t0) * 1000.0,
            "mode": actual_mode,
            "tokens": int(seq),
            "grid": (int(ph), int(pw)),
            "hidden": int(c),
            "dtype": str(self.rt.dtype).replace("torch.", ""),
            "num_regs": int(num_regs),
            "patch_tokens": int(p),
            "heat_std": float(np.std(heat)),
            "heat_p50": p50,
            "heat_p90": p90,
            "heat_p99": p99,
            # Raw (pre-normalization) statistics
            "raw_std": raw_std,
            "raw_range": raw_range,
            "raw_max": raw_max,
            "raw_mean": raw_mean,
            "saliency_threshold": saliency_threshold,
            "saliency_confident": raw_std > saliency_threshold,
        }
        return heat, info


    @torch.inference_mode()
    def embed_boxes(self, rgb: np.ndarray, boxes_xyxy: List[List[float]], out_size: int = 224) -> np.ndarray:
        """Compute DINO embeddings (CLS token) for each box crop.

        Used for Phase 3 ReID + interaction reasoning. Returns float32 unit vectors.
        """
        if not boxes_xyxy:
            return np.zeros((0, 1), dtype=np.float32)

        H, W = rgb.shape[:2]
        crops = []
        for bb in boxes_xyxy:
            if bb is None or len(bb) < 4:
                x1, y1, x2, y2 = 0, 0, W, H
            else:
                x1, y1, x2, y2 = bb[:4]
                x1 = int(max(0, min(W - 1, round(x1))))
                y1 = int(max(0, min(H - 1, round(y1))))
                x2 = int(max(1, min(W, round(x2))))
                y2 = int(max(1, min(H, round(y2))))
                if x2 <= x1 + 1 or y2 <= y1 + 1:
                    x1, y1, x2, y2 = 0, 0, W, H

            crop = rgb[y1:y2, x1:x2]
            if crop.size == 0:
                crop = rgb
            pil = Image.fromarray(crop)
            crops.append(pil)

        inputs = self.proc(images=crops, return_tensors="pt")
        inputs = to_device_dtype(inputs, self.rt)

        with autocast_ctx(self.rt.device, self.rt.dtype, self.rt.amp_enabled):
            outputs = self.model(**inputs)

        last = torch.nan_to_num(outputs.last_hidden_state.float(), nan=0.0, posinf=0.0, neginf=0.0)
        cls = last[:, 0, :]  # [B,C]
        cls = cls / (cls.norm(dim=-1, keepdim=True) + 1e-6)
        return cls.detach().cpu().numpy().astype(np.float32)


    @staticmethod
    def heatmap_to_boxes(
        heat: np.ndarray,
        top_k: int = 5,
        min_area_frac: float = 0.005,
        max_area_frac: float = 0.80,
    ) -> List[List[float]]:
        """Convert a heatmap into proposal boxes.

        Args:
            heat: Saliency heatmap [H, W] with values in [0, 1]
            top_k: Maximum number of boxes to return
            min_area_frac: Minimum box area as fraction of image (default 0.5%)
            max_area_frac: Maximum box area as fraction of image (default 80%)

        Returns:
            List of [x1, y1, x2, y2] boxes, sorted by area (largest first)
        """
        heat = np.asarray(heat, dtype=np.float32)
        heat = np.nan_to_num(heat, nan=0.0, posinf=1.0, neginf=0.0)
        heat = np.clip(heat, 0.0, 1.0)

        h, w = heat.shape[:2]
        if h <= 2 or w <= 2:
            return []

        std = float(np.std(heat))
        if std < 1e-3:
            return []

        img_area = float(h * w)
        min_area = min_area_frac * img_area
        max_area = max_area_frac * img_area

        def _boxes_for_threshold(th: float, kernel_size: int = 3) -> List[List[float]]:
            bw = (heat >= float(th)).astype(np.uint8) * 255
            # Use smaller kernel to preserve more distinct regions
            if kernel_size >= 3:
                bw = cv2.medianBlur(bw, kernel_size)
                bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((kernel_size, kernel_size), np.uint8))
            cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes2 = []
            for c in cnts:
                x, y, ww, hh = cv2.boundingRect(c)
                area = float(ww * hh)
                if area < min_area:
                    continue
                if area > max_area:
                    continue
                boxes2.append([float(x), float(y), float(x + ww), float(y + hh), area])
            boxes2.sort(key=lambda b: b[4], reverse=True)
            return boxes2

        def _iou(b1: List[float], b2: List[float]) -> float:
            """Compute IoU between two boxes."""
            x1 = max(b1[0], b2[0])
            y1 = max(b1[1], b2[1])
            x2 = min(b1[2], b2[2])
            y2 = min(b1[3], b2[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            union = a1 + a2 - inter
            return inter / union if union > 0 else 0.0

        # Accumulate boxes across multiple thresholds to find more proposals
        all_boxes: List[List[float]] = []
        seen_boxes: List[List[float]] = []  # Track to avoid duplicates

        # Try quantiles from strict -> relaxed, accumulating unique boxes
        for q in (0.98, 0.95, 0.92, 0.88, 0.85, 0.80, 0.75):
            th = float(np.quantile(heat, q))
            # Use smaller kernel at stricter thresholds to find small objects
            kernel = 3 if q >= 0.90 else 5
            boxes = _boxes_for_threshold(th, kernel_size=kernel)

            for b in boxes:
                # Check if this box overlaps significantly with any we've already found
                is_duplicate = False
                for sb in seen_boxes:
                    if _iou(b, sb) > 0.5:  # >50% overlap = duplicate
                        is_duplicate = True
                        break
                if not is_duplicate:
                    all_boxes.append(b)
                    seen_boxes.append(b)

                if len(all_boxes) >= top_k:
                    break

            if len(all_boxes) >= top_k:
                break

        # Sort by area (largest first) and return
        all_boxes.sort(key=lambda b: b[4], reverse=True)
        return [b[:4] for b in all_boxes[:top_k]]


# ----------------------------
# SAM3 stage
# ----------------------------

class Sam3Stage:
    def __init__(self, model_id: str, runtime: StageRuntime, hf_token: Optional[str]):
        try:

            from transformers import Sam3Processor, Sam3Model

        except Exception as e:

            raise ImportError(

                "SAM3 support is missing from your installed 'transformers'. "

                "Install a version that includes SAM3 (recommended: 'pip install --upgrade --pre transformers'). "

                "Alternatively, set Detection source to YOLO (yolo11/yolo26) to avoid SAM3."

            ) from e
        self.proc = hf_from_pretrained(Sam3Processor, model_id, hf_token=hf_token, use_fast=True)
        self.model = hf_from_pretrained(Sam3Model, model_id, hf_token=hf_token)
        self.model = self.model.to(device=runtime.device, dtype=runtime.dtype)
        self.model.eval()
        self.rt = runtime

        # Vision embedding cache: stores (vision_embeds, img_inputs, target_sizes) keyed by frame hash
        # This avoids recomputing vision embeddings when the same frame is processed multiple times
        self._vision_cache: Dict[int, Tuple[torch.Tensor, Any, Optional[List]]] = {}
        self._vision_cache_max = 4  # Keep last N frames' embeddings

    def _compute_frame_hash(self, rgb: np.ndarray) -> int:
        """Compute a fast hash of the RGB frame for caching."""
        # Subsample for speed: every 16th pixel in each dimension
        subsampled = rgb[::16, ::16, :].tobytes()
        return hash(subsampled)

    @torch.inference_mode()
    def get_vision_features(
        self, rgb: np.ndarray
    ) -> Tuple[torch.Tensor, Any, Optional[List]]:
        """Pre-compute vision features for reuse across multiple prompt queries.

        Returns:
            vision_embeds: Pre-computed vision embeddings tensor
            img_inputs: Processor outputs (for pixel_values reference)
            target_sizes: Original sizes for post-processing
        """
        image = Image.fromarray(rgb)
        img_inputs = self.proc(images=image, return_tensors="pt")
        pixel_values = img_inputs["pixel_values"].to(self.rt.device)
        if is_float_tensor(pixel_values) and pixel_values.dtype != self.rt.dtype:
            pixel_values = pixel_values.to(dtype=self.rt.dtype)

        orig = img_inputs.get("original_sizes", None)
        if torch.is_tensor(orig):
            target_sizes = orig.detach().cpu().tolist()
        elif hasattr(orig, "tolist"):
            target_sizes = orig.tolist()
        else:
            target_sizes = None

        with autocast_ctx(self.rt.device, self.rt.dtype, self.rt.amp_enabled):
            vision_embeds = self.model.get_vision_features(pixel_values=pixel_values)

        return vision_embeds, img_inputs, target_sizes

    @torch.inference_mode()
    def segment_prompts(
        self,
        rgb: np.ndarray,
        prompts: List[str],
        threshold: float,
        mask_threshold: float,
    ) -> Tuple[List[np.ndarray], List[List[float]], List[str], Dict[str, Any]]:
        t0 = time.perf_counter()

        # Check vision embedding cache first
        frame_hash = self._compute_frame_hash(rgb)
        cache_hit = frame_hash in self._vision_cache

        if cache_hit:
            vision_embeds, img_inputs, target_sizes = self._vision_cache[frame_hash]
            vision_t0 = vision_t1 = time.perf_counter()  # No vision compute time
        else:
            image = Image.fromarray(rgb)

            # Keep processor metadata (original_sizes) on CPU; only move pixel_values to device.
            img_inputs = self.proc(images=image, return_tensors="pt")
            pixel_values = img_inputs["pixel_values"].to(self.rt.device)
            if is_float_tensor(pixel_values) and pixel_values.dtype != self.rt.dtype:
                pixel_values = pixel_values.to(dtype=self.rt.dtype)

            orig = img_inputs.get("original_sizes", None)
            if torch.is_tensor(orig):
                target_sizes = orig.detach().cpu().tolist()
            elif hasattr(orig, "tolist"):
                target_sizes = orig.tolist()
            else:
                target_sizes = None

            vision_t0 = time.perf_counter()
            with autocast_ctx(self.rt.device, self.rt.dtype, self.rt.amp_enabled):
                vision_embeds = self.model.get_vision_features(pixel_values=pixel_values)
            vision_t1 = time.perf_counter()

            # Cache the vision embeddings for potential reuse
            if len(self._vision_cache) >= self._vision_cache_max:
                # Evict oldest entry (FIFO)
                oldest_key = next(iter(self._vision_cache))
                del self._vision_cache[oldest_key]
            self._vision_cache[frame_hash] = (vision_embeds, img_inputs, target_sizes)

        all_masks: List[np.ndarray] = []
        all_boxes: List[List[float]] = []
        all_labels: List[str] = []
        per_prompt_counts: Dict[str, int] = {}
        per_prompt_areas: Dict[str, float] = {}
        H, W = rgb.shape[:2]

        for p in prompts:
            p = p.strip()
            if not p:
                continue

            try:
                text_inputs = self.proc(text=p, return_tensors="pt")
                # Move token tensors to device; do NOT cast token dtypes.
                text_inputs = {k: (v.to(self.rt.device) if torch.is_tensor(v) else v) for k, v in text_inputs.items()}

                with autocast_ctx(self.rt.device, self.rt.dtype, self.rt.amp_enabled):
                    out = self.model(vision_embeds=vision_embeds, **text_inputs)

                results = self.proc.post_process_instance_segmentation(
                    out,
                    threshold=float(threshold),
                    mask_threshold=float(mask_threshold),
                    target_sizes=target_sizes,
                )[0]

                masks = results.get("masks", None)
                boxes = results.get("boxes", None)

                if masks is None:
                    per_prompt_counts[p] = 0
                    per_prompt_areas[p] = 0.0
                    continue

                masks_np = masks.detach().float().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
                if masks_np.ndim == 2:
                    masks_np = masks_np[None, ...]

                if boxes is None:
                    boxes_np = [None] * masks_np.shape[0]
                else:
                    boxes_np = boxes.detach().float().cpu().numpy() if torch.is_tensor(boxes) else np.asarray(boxes)
                    if boxes_np.ndim == 1:
                        boxes_np = boxes_np[None, ...]
                    boxes_np = boxes_np.tolist()

                per_prompt_counts[p] = int(masks_np.shape[0])

                # Union area fraction per prompt (helps make the "scene" summary more meaningful than counts)
                union: Optional[np.ndarray] = None
                for i in range(masks_np.shape[0]):
                    m = masks_np[i]
                    if m.ndim != 2:
                        m = np.squeeze(m)
                    # Resize to the current frame size if needed
                    if m.shape[:2] != (H, W):
                        m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
                    mb = m > 0.5
                    union = mb if union is None else (union | mb)
                per_prompt_areas[p] = float(union.mean()) if union is not None else 0.0

                for i in range(masks_np.shape[0]):
                    all_masks.append(masks_np[i])
                    if boxes_np and i < len(boxes_np) and boxes_np[i] is not None:
                        all_boxes.append([float(x) for x in boxes_np[i]])
                    else:
                        all_boxes.append([0.0, 0.0, 0.0, 0.0])
                    all_labels.append(p)

            except Exception as e:
                # Log error but continue with other prompts
                print(f"[SAM] Error processing prompt '{p}': {e}")
                per_prompt_counts[p] = 0
                per_prompt_areas[p] = 0.0
                continue

        t1 = time.perf_counter()
        info = {
            "vision_ms": (vision_t1 - vision_t0) * 1000.0,
            "vision_cached": cache_hit,
            "total_ms": (t1 - t0) * 1000.0,
            "counts": per_prompt_counts,
            "areas": per_prompt_areas,
            "dtype": str(self.rt.dtype).replace("torch.", ""),
        }
        return all_masks, all_boxes, all_labels, info

    @torch.inference_mode()
    def segment_boxes(
        self,
        rgb: np.ndarray,
        boxes_xyxy: List[List[float]],
        threshold: float,
        mask_threshold: float,
    ) -> Tuple[List[np.ndarray], List[List[float]], List[str], Dict[str, Any]]:
        t0 = time.perf_counter()
        image = Image.fromarray(rgb)

        inputs = self.proc(
            images=image,
            input_boxes=[boxes_xyxy],
            input_boxes_labels=[[1] * len(boxes_xyxy)],
            return_tensors="pt",
        )

        orig = inputs.get("original_sizes", None)
        if torch.is_tensor(orig):
            target_sizes = orig.detach().cpu().tolist()
        elif hasattr(orig, "tolist"):
            target_sizes = orig.tolist()
        else:
            target_sizes = None

        # Move/cast only what the model needs
        model_inputs = to_device_dtype(inputs, self.rt)

        with autocast_ctx(self.rt.device, self.rt.dtype, self.rt.amp_enabled):
            out = self.model(**model_inputs)

        results = self.proc.post_process_instance_segmentation(
            out,
            threshold=float(threshold),
            mask_threshold=float(mask_threshold),
            target_sizes=target_sizes,
        )[0]

        masks = results.get("masks", None)
        out_boxes = results.get("boxes", None)

        if masks is None:
            return [], [], [], {
                "total_ms": (time.perf_counter() - t0) * 1000.0,
                "counts": {"box": 0},
                "areas": {"box": 0.0},
                "dtype": str(self.rt.dtype).replace("torch.", ""),
            }

        masks_np = masks.detach().float().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
        if masks_np.ndim == 2:
            masks_np = masks_np[None, ...]

        if out_boxes is None:
            boxes_np = [[0.0, 0.0, 0.0, 0.0] for _ in range(masks_np.shape[0])]
        else:
            boxes_np = out_boxes.detach().float().cpu().numpy().tolist() if torch.is_tensor(out_boxes) else np.asarray(out_boxes).tolist()
            if isinstance(boxes_np[0], (float, int)):
                boxes_np = [boxes_np]

        labels = ["box" for _ in range(masks_np.shape[0])]

        # Area fraction of the union of all masks returned for these boxes.
        H, W = rgb.shape[:2]
        union: Optional[np.ndarray] = None
        for i in range(masks_np.shape[0]):
            m = masks_np[i]
            if m.ndim != 2:
                m = np.squeeze(m)
            if m.shape[:2] != (H, W):
                m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
            mb = m > 0.5
            union = mb if union is None else (union | mb)
        area_frac = float(union.mean()) if union is not None else 0.0
        info = {
            "total_ms": (time.perf_counter() - t0) * 1000.0,
            "counts": {"box": int(masks_np.shape[0])},
            "areas": {"box": area_frac},
            "dtype": str(self.rt.dtype).replace("torch.", ""),
        }
        return [masks_np[i] for i in range(masks_np.shape[0])], boxes_np, labels, info

    @torch.inference_mode()
    def segment_prompts_in_boxes(
        self,
        rgb: np.ndarray,
        boxes_xyxy: List[List[float]],
        prompts: List[str],
        threshold: float,
        mask_threshold: float,
        pad: int = 8,
        max_boxes: int = 2,
    ) -> Tuple[List[np.ndarray], List[List[float]], List[str], Dict[str, Any]]:
        """Run prompt-conditioned segmentation, but only inside a few coarse proposal boxes.

        This is the main way DINO can materially help the "scene" description: it focuses SAM
        on salient regions so prompt masks don't get diluted by the whole frame.

        Notes:
        - This is slower than a single full-frame SAM call (it reruns SAM per box).
        - We cap the number of boxes for responsiveness.
        """
        H, W = rgb.shape[:2]
        if not boxes_xyxy:
            return self.segment_prompts(rgb, prompts, threshold=threshold, mask_threshold=mask_threshold)

        # Sort by area (largest first) and keep a few non-trivial boxes.
        boxes_s = []
        for b in boxes_xyxy:
            if not b or len(b) < 4:
                continue
            x1, y1, x2, y2 = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
            x1 = max(0.0, min(float(W - 1), x1))
            x2 = max(0.0, min(float(W), x2))
            y1 = max(0.0, min(float(H - 1), y1))
            y2 = max(0.0, min(float(H), y2))
            if x2 <= x1 + 2 or y2 <= y1 + 2:
                continue
            area = (x2 - x1) * (y2 - y1)
            # Skip "almost whole image" boxes; they don't help.
            if area > 0.85 * (H * W):
                continue
            boxes_s.append([x1, y1, x2, y2, area])
        boxes_s.sort(key=lambda bb: bb[4], reverse=True)
        boxes_s = boxes_s[: max(1, int(max_boxes))]

        if not boxes_s:
            return self.segment_prompts(rgb, prompts, threshold=threshold, mask_threshold=mask_threshold)

        t0 = time.perf_counter()
        all_masks: List[np.ndarray] = []
        all_boxes: List[List[float]] = []
        all_labels: List[str] = []

        counts: Dict[str, int] = {p.strip(): 0 for p in prompts if str(p).strip()}
        unions: Dict[str, np.ndarray] = {p: np.zeros((H, W), dtype=np.bool_) for p in counts.keys()}
        vision_ms = 0.0

        for (x1f, y1f, x2f, y2f, _) in boxes_s:
            x1 = int(max(0, min(W - 1, math.floor(x1f) - pad)))
            y1 = int(max(0, min(H - 1, math.floor(y1f) - pad)))
            x2 = int(max(0, min(W, math.ceil(x2f) + pad)))
            y2 = int(max(0, min(H, math.ceil(y2f) + pad)))
            if x2 <= x1 + 2 or y2 <= y1 + 2:
                continue

            crop = rgb[y1:y2, x1:x2]
            masks_c, boxes_c, labels_c, info_c = self.segment_prompts(
                crop,
                prompts,
                threshold=threshold,
                mask_threshold=mask_threshold,
            )
            vision_ms += float(info_c.get("vision_ms", 0.0))

            # Paste crop masks into full-frame masks, offsetting boxes.
            ch, cw = crop.shape[:2]
            for i, m in enumerate(masks_c):
                m2 = m
                if m2.ndim != 2:
                    m2 = np.squeeze(m2)
                if m2.shape[:2] != (ch, cw):
                    m2 = cv2.resize(m2.astype(np.float32), (cw, ch), interpolation=cv2.INTER_NEAREST)

                full = np.zeros((H, W), dtype=np.float32)
                full[y1:y2, x1:x2] = m2
                all_masks.append(full)

                # box
                if boxes_c and i < len(boxes_c) and boxes_c[i] is not None and len(boxes_c[i]) >= 4:
                    bx1, by1, bx2, by2 = boxes_c[i][:4]
                    all_boxes.append([float(bx1 + x1), float(by1 + y1), float(bx2 + x1), float(by2 + y1)])
                else:
                    all_boxes.append([float(x1), float(y1), float(x2), float(y2)])

                # label / counts / unions
                lbl = labels_c[i] if labels_c and i < len(labels_c) else ""
                lbl = str(lbl).strip()
                all_labels.append(lbl)
                if lbl:
                    counts[lbl] = int(counts.get(lbl, 0) + 1)
                    unions.setdefault(lbl, np.zeros((H, W), dtype=np.bool_))
                    unions[lbl] |= (full > 0.5)

        areas = {k: float(v.mean()) for k, v in unions.items()}
        t1 = time.perf_counter()
        info = {
            "vision_ms": vision_ms,
            "total_ms": (t1 - t0) * 1000.0,
            "counts": counts,
            "areas": areas,
            "dtype": str(self.rt.dtype).replace("torch.", ""),
            "note": f"prompt-in-boxes (boxes={len(boxes_s)})",
        }
        return all_masks, all_boxes, all_labels, info

# ----------------------------
# V-JEPA2 stage
# ----------------------------

class VJepa2Stage:
    def __init__(self, model_id: str, runtime: StageRuntime, hf_token: Optional[str]):
        from transformers import AutoVideoProcessor, AutoModelForVideoClassification
        self.proc = hf_from_pretrained(AutoVideoProcessor, model_id, hf_token=hf_token, use_fast=True)
        self.model = hf_from_pretrained(AutoModelForVideoClassification, model_id, hf_token=hf_token)
        self.model = self.model.to(device=runtime.device, dtype=runtime.dtype)
        self.model.eval()
        self.rt = runtime
        self.id2label = getattr(self.model.config, "id2label", None)

        # Cache for recent clip classifications (avoid re-processing identical/similar clips)
        self._cache: Dict[int, Tuple[List[Tuple[str, float]], Dict[str, Any]]] = {}
        self._cache_max_size = 8

    def _compute_clip_hash(self, clip_rgb_tchw: np.ndarray) -> int:
        """Compute a fast hash of the clip for caching using subsampled pixels."""
        # Subsample spatially and temporally for fast hashing
        subsampled = clip_rgb_tchw[::2, ::8, ::16, ::16].tobytes()
        return hash(subsampled)

    @torch.inference_mode()
    def classify_clip(self, clip_rgb_tchw: np.ndarray) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
        # Check cache first
        clip_hash = self._compute_clip_hash(clip_rgb_tchw)
        if clip_hash in self._cache:
            cached_pairs, cached_info = self._cache[clip_hash]
            # Return a copy of info with "cached" flag
            cached_info_copy = dict(cached_info)
            cached_info_copy["cached"] = True
            return cached_pairs, cached_info_copy

        t0 = time.perf_counter()
        inputs = self.proc(clip_rgb_tchw, return_tensors="pt")
        inputs = to_device_dtype(inputs, self.rt)
        t1 = time.perf_counter()

        with autocast_ctx(self.rt.device, self.rt.dtype, self.rt.amp_enabled):
            out = self.model(**inputs)
        t2 = time.perf_counter()

        logits = out.logits[0].float()
        probs = torch.softmax(logits, dim=-1)
        topk = torch.topk(probs, k=min(5, probs.shape[-1]))

        pairs: List[Tuple[str, float]] = []
        for idx, p in zip(topk.indices.tolist(), topk.values.tolist()):
            if isinstance(self.id2label, dict):
                label = self.id2label.get(int(idx), str(idx))
            else:
                label = str(idx)
            pairs.append((label, float(p)))

        info = {
            "pre_ms": (t1 - t0) * 1000.0,
            "fwd_ms": (t2 - t1) * 1000.0,
            "total_ms": (t2 - t0) * 1000.0,
            "T": int(clip_rgb_tchw.shape[0]),
            "H": int(clip_rgb_tchw.shape[2]),
            "W": int(clip_rgb_tchw.shape[3]),
            "dtype": str(self.rt.dtype).replace("torch.", ""),
            "cached": False,
        }

        # Update cache
        self._cache[clip_hash] = (pairs, info)
        if len(self._cache) > self._cache_max_size:
            # Remove oldest entry (first key in dict)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        return pairs, info


class CaptionStage:
    """Frame-level image captioning (BLIP) for true semantic scene description."""

    def __init__(self, model_id: str, runtime: StageRuntime, hf_token: Optional[str]):
        from transformers import BlipProcessor, BlipForConditionalGeneration
        self.proc = hf_from_pretrained(BlipProcessor, model_id, hf_token=hf_token, use_fast=True)
        self.model = hf_from_pretrained(BlipForConditionalGeneration, model_id, hf_token=hf_token)
        self.device = runtime.device
        # Use bfloat16 on MPS (more stable than float16), float16 on CUDA, float32 on CPU
        if self.device.type == "mps":
            self.dtype = torch.bfloat16
            self.amp_enabled = True
        elif self.device.type == "cuda":
            self.dtype = torch.float16
            self.amp_enabled = True
        else:
            self.dtype = torch.float32
            self.amp_enabled = False
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()
        self._fallback_to_fp32 = False  # Track if we needed to fall back

    @torch.inference_mode()
    def caption(self, rgb: np.ndarray, max_new_tokens: int = 24, num_beams: int = 3) -> Tuple[str, Dict[str, Any]]:
        t0 = time.perf_counter()
        pil = Image.fromarray(rgb)
        inputs = self.proc(images=pil, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        t1 = time.perf_counter()

        try:
            with autocast_ctx(self.device, self.dtype, self.amp_enabled):
                out_ids = self.model.generate(
                    pixel_values=pixel_values,
                    max_new_tokens=int(max_new_tokens),
                    num_beams=int(num_beams),
                    do_sample=False,
                )
        except RuntimeError as e:
            # Fallback to FP32 if mixed precision fails (some ops unsupported)
            if not self._fallback_to_fp32 and ("not implemented" in str(e).lower() or "unsupported" in str(e).lower()):
                self._fallback_to_fp32 = True
                self.dtype = torch.float32
                self.amp_enabled = False
                self.model = self.model.to(dtype=self.dtype)
                pixel_values = pixel_values.to(dtype=self.dtype)
                out_ids = self.model.generate(
                    pixel_values=pixel_values,
                    max_new_tokens=int(max_new_tokens),
                    num_beams=int(num_beams),
                    do_sample=False,
                )
            else:
                raise

        t2 = time.perf_counter()
        cap = self.proc.decode(out_ids[0], skip_special_tokens=True)
        cap = (cap or "").strip()
        info = {
            "pre_ms": (t1 - t0) * 1000.0,
            "gen_ms": (t2 - t1) * 1000.0,
            "ms": (t2 - t0) * 1000.0,
            "dtype": str(self.dtype).replace("torch.", ""),
        }
        return cap, info


# ----------------------------
# YOLO Detection Stage
# ----------------------------

class YOLOStage:
    """Object detection using YOLO models from Ultralytics.

    Supports YOLO11 (2024) and YOLO26 (2025) with all size variants.
    Returns detections in same format as Sam3Stage for pipeline compatibility.
    """

    YOLO11_MODELS = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]
    YOLO26_MODELS = ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"]

    # All 80 COCO classes
    COCO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    def __init__(self, model_id: str, runtime: StageRuntime, conf_threshold: float = 0.25):
        """Initialize YOLO stage.

        Args:
            model_id: Model name like "yolo11m" or "yolo26s"
            runtime: StageRuntime with device info
            conf_threshold: Confidence threshold for detections (0.0-1.0)
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package required for YOLO models. "
                "Install with: pip install ultralytics>=8.3.0"
            )

        self.model_id = model_id
        self.conf_threshold = conf_threshold
        self.device = runtime.device

        # Load model (downloads automatically if not cached)
        self.model = YOLO(f"{model_id}.pt")

        # Move to device
        if self.device.type == "cuda":
            self.model.to("cuda")
        elif self.device.type == "mps":
            self.model.to("mps")
        # CPU is default

    @torch.inference_mode()
    def __call__(
        self,
        bgr: np.ndarray,
        class_filter: Optional[List[str]] = None,
    ) -> Tuple[List[Optional[np.ndarray]], List[List[float]], List[str], Dict[str, Any]]:
        """Run YOLO detection.

        Args:
            bgr: Input image in BGR format (OpenCV)
            class_filter: Optional list of class names to detect (None = all 80 classes)

        Returns:
            Tuple of (masks, boxes, labels, info) matching Sam3Stage format:
            - masks: List of None (YOLO doesn't provide segmentation masks)
            - boxes: List of [x1, y1, x2, y2] bounding boxes
            - labels: List of class name strings
            - info: Dict with scores, timings, etc.
        """
        t0 = time.perf_counter()

        # Run inference
        results = self.model(bgr, conf=self.conf_threshold, verbose=False)

        t1 = time.perf_counter()

        boxes = []
        labels = []
        scores = []

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls.item())
                cls_name = r.names[cls_id]

                # Apply class filter if specified
                if class_filter and cls_name not in class_filter:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                boxes.append([float(x1), float(y1), float(x2), float(y2)])
                labels.append(cls_name)
                scores.append(float(box.conf.item()))

        # YOLO doesn't provide masks - return None for each detection
        masks = [None] * len(boxes)

        info = {
            "scores": scores,
            "model": self.model_id,
            "num_detections": len(boxes),
            "ms": (t1 - t0) * 1000.0,
            "conf_threshold": self.conf_threshold,
        }

        return masks, boxes, labels, info

    @classmethod
    def get_all_models(cls) -> List[str]:
        """Return list of all supported YOLO models."""
        return cls.YOLO11_MODELS + cls.YOLO26_MODELS


# ----------------------------
# Model container
# ----------------------------

@dataclass
class LoadedModels:
    dino: Optional[DinoV3Stage] = None
    sam: Optional[Sam3Stage] = None
    vjepa: Optional[VJepa2Stage] = None
    caption: Optional[CaptionStage] = None
    yolo: Optional[YOLOStage] = None


# ----------------------------
# Threads
# ----------------------------

class CameraWorker(QtCore.QThread):
    frame_signal = QtCore.Signal(np.ndarray)  # BGR frame
    error_signal = QtCore.Signal(str)

    def __init__(self, cfg: RuntimeConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._running = False
        self.cap: Optional[cv2.VideoCapture] = None

    def run(self):
        self._running = True
        try:
            self.cap = cv2.VideoCapture(self.cfg.camera_index, cv2.CAP_ANY)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera index {self.cfg.camera_index}")

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.cfg.camera_width))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.cfg.camera_height))
            self.cap.set(cv2.CAP_PROP_FPS, float(self.cfg.camera_fps_hint))

            while self._running:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue
                self.frame_signal.emit(frame)
        except Exception:
            self.error_signal.emit(traceback.format_exc())
        finally:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.cap = None

    def stop(self):
        self._running = False


@dataclass
class StageOutputs:
    tick_id: int = 0
    ts_wall: float = 0.0

    # store DISPLAY-SIZED frames to keep timeline memory sane
    raw_bgr: Optional[np.ndarray] = None
    dino_bgr: Optional[np.ndarray] = None
    sam_bgr: Optional[np.ndarray] = None
    clip_strip_bgr: Optional[np.ndarray] = None
    composite_bgr: Optional[np.ndarray] = None  # Unified view with all overlays
    tracking_bgr: Optional[np.ndarray] = None  # Dedicated tracking visualization

    # lightweight, numeric artifacts for the selected tick
    dino_boxes: List[List[float]] = dataclasses.field(default_factory=list)

    # Intermediate data for composite view (stored for regeneration)
    dino_heat: Optional[np.ndarray] = None  # Raw DINO heatmap
    sam_masks: List[np.ndarray] = dataclasses.field(default_factory=list)
    sam_boxes: List[List[float]] = dataclasses.field(default_factory=list)
    sam_labels: List[str] = dataclasses.field(default_factory=list)
    tracks_raw: List['Track'] = dataclasses.field(default_factory=list)  # Raw Track objects

    dino_info: Optional[Dict[str, Any]] = None
    sam_info: Optional[Dict[str, Any]] = None
    vjepa_info: Optional[Dict[str, Any]] = None
    caption: str = ""
    caption_info: Dict[str, Any] = dataclasses.field(default_factory=dict)

    sam_counts: Dict[str, int] = dataclasses.field(default_factory=dict)
    vjepa_pairs: List[Tuple[str, float]] = dataclasses.field(default_factory=list)

    # Stateful tracking / narration (lightweight summary only)
    tracks: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    narration: str = ""
    narration_attributed: Optional['AttributedNarration'] = None  # Color-coded narration
    narration_events: List[str] = dataclasses.field(default_factory=list)
    stable_action: Optional[Tuple[str, float]] = None



class InferenceWorker(QtCore.QThread):
    outputs_signal = QtCore.Signal(object)  # StageOutputs
    event_signal = QtCore.Signal(str)

    def __init__(self, cfg: RuntimeConfig, ids: ModelIds, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.ids = ids
        self.device = choose_device()

        self._running = False
        self.models = LoadedModels()

        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = QtCore.QMutex()

        self._clip_lock = QtCore.QMutex()
        self._clip_frames: Deque[Tuple[float, np.ndarray]] = deque()  # (timestamp, RGB frame)

        self._tick = 0

        # Tracking + narration state
        tracker_kwargs = dict(
            iou_thresh=float(self.cfg.tracker_iou_thresh),
            max_missed=int(self.cfg.tracker_max_missed),
            label_ema=float(self.cfg.tracker_label_ema),
            use_embedding=bool(getattr(self.cfg, 'enable_reid', False) and getattr(self.cfg, 'tracker_use_embedding', True)),
            iou_weight=float(getattr(self.cfg, 'tracker_iou_weight', 0.65)),
            emb_weight=float(getattr(self.cfg, 'tracker_emb_weight', 0.35)),
            min_sim=float(getattr(self.cfg, 'tracker_min_sim', 0.15)),
            emb_ema=float(getattr(self.cfg, 'tracker_emb_ema', 0.40)),
        )
        if getattr(self.cfg, 'tracker_algorithm', 'simple') == 'kalman':
            self._tracker = KalmanTracker(
                **tracker_kwargs,
                graveyard_seconds=float(getattr(self.cfg, 'tracker_graveyard_seconds', 10.0)),
                use_hungarian=bool(getattr(self.cfg, 'tracker_use_hungarian', True)),
            )
        else:
            self._tracker = SimpleTracker(**tracker_kwargs)
        self._vjepa_ema: Dict[str, float] = {}
        self._stable_action: Optional[Tuple[str, float]] = None
        self._last_action_label: Optional[str] = None
        self._last_event_cooldown: Dict[str, int] = {}
        self._last_sam_dets: List[Detection] = []  # Cache for tracker when SAM skips ticks
        self._last_sam_tick: int = -1  # Tick when SAM last ran


        # Phase 3: interaction state (debounced)
        self._phone_contact = 0
        self._phone_held = False
        self._cup_contact = 0
        self._cup_held = False
        self._typing_score = 0
        self._typing_on = False


        self._ctrl_lock = QtCore.QMutex()
        self._prompts: List[str] = [p.strip() for p in self.cfg.default_prompts.split(",") if p.strip()]
        self._use_boxes = self.cfg.use_dino_box_proposals_for_sam
        self._sam_thr = self.cfg.sam_threshold
        self._sam_mask_thr = self.cfg.sam_mask_threshold
        self._enable_dino = self.cfg.enable_dino
        self._enable_sam = self.cfg.enable_sam
        self._enable_vjepa = self.cfg.enable_vjepa
        self._dino_strength = self.cfg.dino_heatmap_strength
        self._dino_blur = self.cfg.dino_heatmap_blur
        self._max_w = self.cfg.max_display_width

        self._run_dino_n = max(1, int(self.cfg.run_dino_every_n))
        self._run_sam_n = max(1, int(self.cfg.run_sam_every_n))
        self._run_vjepa_n = max(1, int(self.cfg.run_vjepa_every_n))
        # Caption controls (phase 4)
        self._enable_caption = bool(getattr(self.cfg, 'enable_caption', True))
        self._caption_every_n = max(1, int(getattr(self.cfg, 'caption_every_n', 5)))
        self._caption_on_events = bool(getattr(self.cfg, 'caption_on_events', True))
        self._caption_min_tick_gap = max(0, int(getattr(self.cfg, 'caption_min_tick_gap', 2)))
        self._caption_max_new_tokens = max(8, int(getattr(self.cfg, 'caption_max_new_tokens', 24)))
        self._caption_num_beams = max(1, int(getattr(self.cfg, 'caption_num_beams', 3)))
        self._last_caption_text = ''
        self._last_caption_tick = -10**9

        self._hf_token = self.cfg.hf_token or get_hf_token_from_env()
        self._fp16_on_mps = bool(self.cfg.fp16_on_mps)

        # YOLO detection settings (alternative to SAM)
        self._detection_source = getattr(self.cfg, 'detection_source', 'sam')
        self._yolo_model = getattr(self.cfg, 'yolo_model', 'yolo11m')
        self._yolo_conf_threshold = float(getattr(self.cfg, 'yolo_conf_threshold', 0.25))

        # If FP16 hits an unsupported op, we auto-reload as FP32 once.
        self._did_fp16_fallback = False

    def _get_vjepa_frame_count(self) -> int:
        """Get the appropriate frame count for V-JEPA based on model type.

        fpc64 models need 64 frames, fpc16 models need 16 frames.
        """
        model_id = self.ids.vjepa2_cls.lower()
        if "fpc64" in model_id:
            return 64
        elif "fpc16" in model_id:
            return 16
        # Fall back to configured value
        return max(2, int(self.cfg.vjepa_samples))

    def _ensure_yolo_stage(self) -> Optional[YOLOStage]:
        """Lazy-load YOLO stage when first needed. Returns None if loading fails."""
        if self.models.yolo is not None:
            return self.models.yolo

        try:
            rt = StageRuntime(device=self.device, dtype=torch.float16, amp_enabled=self._enable_dino)
            self.models.yolo = YOLOStage(
                model_id=self._yolo_model,
                runtime=rt,
                conf_threshold=self._yolo_conf_threshold,
            )
            self._event(f"[yolo] loaded {self._yolo_model}")
            return self.models.yolo
        except ImportError as e:
            self._event(f"[yolo] ultralytics not installed: {e}")
            return None
        except Exception as e:
            self._event(f"[yolo] failed to load {self._yolo_model}: {e}")
            return None

    def set_latest_frame(self, bgr: np.ndarray):
        with QtCore.QMutexLocker(self._frame_lock):
            self._latest_frame = bgr

        # Store BGR frames directly (avoid per-frame BGR->RGB conversion)
        # Only convert to RGB when sampling for V-JEPA
        # Buffer management: keep only what V-JEPA needs, with uniform temporal subsampling
        now = time.time()
        with QtCore.QMutexLocker(self._clip_lock):
            self._clip_frames.append((now, bgr))
            # Target = number of frames V-JEPA processes (auto-detected from model)
            target = self._get_vjepa_frame_count()
            # When buffer exceeds 2x target, subsample to maintain temporal coverage
            if len(self._clip_frames) > target * 2:
                frames = list(self._clip_frames)
                # Keep uniformly spaced frames to maintain temporal coverage
                idxs = np.linspace(0, len(frames) - 1, num=target).round().astype(int).tolist()
                self._clip_frames = deque([frames[i] for i in idxs])



    def _ensure_sam_stage(self) -> Optional['Sam3Stage']:
        """Lazy-load SAM3 stage only when needed.

        Avoids hard-failing startup when using YOLO detection or when 'transformers'
        is installed without SAM3 support.
        """
        if self.models.sam is not None:
            return self.models.sam
        try:
            rt = self._make_runtime()
            self._event(f"[models] lazy-loading SAM3: {self.ids.sam3}")
            self.models.sam = Sam3Stage(self.ids.sam3, rt, hf_token=self._hf_token)
            self._event("[models] SAM3 loaded")
            return self.models.sam
        except ImportError as e:
            self.models.sam = None
            self._event(
                "[sam] SAM3 unavailable in this environment. "
                "Fix: pip install --upgrade --pre transformers. "
                "Or set Detection source to YOLO (yolo11/yolo26).\n"
                f"Details: {e}"
            )
            return None
        except Exception as e:
            self.models.sam = None
            tb = traceback.format_exc()
            self._event(f"[sam] failed to load SAM3 ({self.ids.sam3}): {e}\n{tb}")
            return None


    def update_controls(
        self,
        prompts_csv: Optional[str] = None,
        use_boxes: Optional[bool] = None,
        sam_thr: Optional[float] = None,
        sam_mask_thr: Optional[float] = None,
        enable_dino: Optional[bool] = None,
        enable_sam: Optional[bool] = None,
        enable_vjepa: Optional[bool] = None,
        dino_strength: Optional[float] = None,
        dino_blur: Optional[int] = None,
        max_w: Optional[int] = None,
        run_dino_n: Optional[int] = None,
        run_sam_n: Optional[int] = None,
        run_vjepa_n: Optional[int] = None,
        enable_caption: Optional[bool] = None,
        caption_every_n: Optional[int] = None,
        caption_on_events: Optional[bool] = None,
        caption_min_tick_gap: Optional[int] = None,
        caption_max_new_tokens: Optional[int] = None,
        caption_num_beams: Optional[int] = None,
        detection_source: Optional[str] = None,
        yolo_model: Optional[str] = None,
        yolo_conf_threshold: Optional[float] = None,
        **kwargs,
    ):
        # Back-compat mapping for the newer UI keyword names.
        if prompts_csv is None and 'prompts' in kwargs:
            p = kwargs.get('prompts')
            if isinstance(p, (list, tuple)):
                prompts_csv = ', '.join([str(x) for x in p if str(x).strip()])
            else:
                prompts_csv = str(p)

        if use_boxes is None and 'use_boxes_for_sam' in kwargs:
            use_boxes = kwargs.get('use_boxes_for_sam')

        if sam_thr is None and 'sam_threshold' in kwargs:
            sam_thr = kwargs.get('sam_threshold')

        if sam_mask_thr is None and 'sam_mask_threshold' in kwargs:
            sam_mask_thr = kwargs.get('sam_mask_threshold')

        if dino_strength is None and 'dino_strength' in kwargs:
            dino_strength = kwargs.get('dino_strength')

        if dino_blur is None and 'dino_blur' in kwargs:
            dino_blur = kwargs.get('dino_blur')

        if max_w is None and 'display_max_width' in kwargs:
            max_w = kwargs.get('display_max_width')

        if run_dino_n is None and 'run_dino_every_n' in kwargs:
            run_dino_n = kwargs.get('run_dino_every_n')

        if run_sam_n is None and 'run_sam_every_n' in kwargs:
            run_sam_n = kwargs.get('run_sam_every_n')

        if run_vjepa_n is None and 'run_vjepa_every_n' in kwargs:
            run_vjepa_n = kwargs.get('run_vjepa_every_n')

        with QtCore.QMutexLocker(self._ctrl_lock):
            if prompts_csv is not None:
                s = str(prompts_csv)
                self._prompts = [p.strip() for p in s.split(',') if p.strip()]

            if use_boxes is not None:
                self._use_boxes = bool(use_boxes)

            if sam_thr is not None:
                self._sam_thr = float(sam_thr)

            if sam_mask_thr is not None:
                self._sam_mask_thr = float(sam_mask_thr)

            if enable_dino is not None:
                self._enable_dino = bool(enable_dino)

            if enable_sam is not None:
                self._enable_sam = bool(enable_sam)

            if enable_vjepa is not None:
                self._enable_vjepa = bool(enable_vjepa)

            if dino_strength is not None:
                self._dino_strength = float(dino_strength)

            if dino_blur is not None:
                self._dino_blur = int(dino_blur)

            if max_w is not None:
                self._max_w = int(max_w)

            if run_dino_n is not None:
                self._run_dino_n = max(1, int(run_dino_n))

            if run_sam_n is not None:
                self._run_sam_n = max(1, int(run_sam_n))

            if run_vjepa_n is not None:
                self._run_vjepa_n = max(1, int(run_vjepa_n))


            if enable_caption is not None:
                self._enable_caption = bool(enable_caption)

            if caption_every_n is not None:
                self._caption_every_n = max(1, int(caption_every_n))

            if caption_on_events is not None:
                self._caption_on_events = bool(caption_on_events)

            if caption_min_tick_gap is not None:
                self._caption_min_tick_gap = max(0, int(caption_min_tick_gap))

            if caption_max_new_tokens is not None:
                self._caption_max_new_tokens = max(8, int(caption_max_new_tokens))

            if caption_num_beams is not None:
                self._caption_num_beams = max(1, int(caption_num_beams))

            # YOLO detection settings
            if detection_source is not None:
                if detection_source in ("sam", "yolo11", "yolo26"):
                    old_source = self._detection_source
                    self._detection_source = detection_source
                    # If the user switches YOLO family (yolo11 vs yolo26), keep the size suffix but
                    # align the model name prefix so the selection does what it says.
                    if detection_source.startswith("yolo"):
                        cur = str(getattr(self, "_yolo_model", "yolo11m"))
                        if not cur.startswith(detection_source):
                            size = cur[-1] if (len(cur) >= 6 and cur[-1] in "nsmxl") else "m"
                            self._yolo_model = f"{detection_source}{size}"
                            self.models.yolo = None
                    # Clear cached YOLO stage if model family changed
                    if old_source != detection_source and detection_source.startswith("yolo"):
                        self.models.yolo = None

            if yolo_model is not None:
                old_model = self._yolo_model
                self._yolo_model = str(yolo_model)
                # Clear cached YOLO stage if model changed
                if old_model != self._yolo_model:
                    self.models.yolo = None

            if yolo_conf_threshold is not None:
                self._yolo_conf_threshold = max(0.01, min(0.99, float(yolo_conf_threshold)))

    def _event(self, s: str):
        self.event_signal.emit(s)

    def _make_runtime(self) -> StageRuntime:
        return pick_runtime(self.device, fp16_on_mps=self._fp16_on_mps)

    def _load_models(self):
        rt = self._make_runtime()
        self._event(f"[models] device={self.device} dtype={rt.dtype} amp={rt.amp_enabled}")

        try:
            if self.cfg.enable_dino:
                self._event(f"[models] loading DINOv3: {self.ids.dino}")
                self.models.dino = DinoV3Stage(self.ids.dino, rt, hf_token=self._hf_token)
                self._event("[models] DINOv3 loaded")

            # SAM3 is only needed when detection_source == 'sam'.
            if self.cfg.enable_sam and getattr(self.cfg, 'detection_source', 'sam') == 'sam':
                self._event(f"[models] loading SAM3: {self.ids.sam3}")
                self.models.sam = Sam3Stage(self.ids.sam3, rt, hf_token=self._hf_token)
                self._event("[models] SAM3 loaded")
            elif self.cfg.enable_sam:
                # Keep startup fast and avoid hard-failing when using YOLO as detection source.
                self.models.sam = None
                self._event("[models] SAM3 skipped (Detection source != 'sam'); will lazy-load if selected.")

            if self.cfg.enable_vjepa:
                self._event(f"[models] loading V-JEPA2: {self.ids.vjepa2_cls}")
                self.models.vjepa = VJepa2Stage(self.ids.vjepa2_cls, rt, hf_token=self._hf_token)
                self._event("[models] V-JEPA2 loaded")

            if getattr(self.cfg, "enable_caption", True):
                self._event(f"[models] loading Caption: {self.ids.caption}")
                self.models.caption = CaptionStage(self.ids.caption, rt, hf_token=self._hf_token)
                self._event("[models] Caption loaded")
        except Exception:
            self._event("[models] FAILED to load:\n" + traceback.format_exc())

    def _reload_models_fp32(self):
        self._event("[mps] FP16 appears unsupported for an op; reloading models as FP32 (once).")
        self._fp16_on_mps = False
        self._did_fp16_fallback = True

        # Clear old models and free GPU memory
        old_models = self.models
        self.models = LoadedModels()
        del old_models
        gc.collect()

        # Release GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass  # MPS empty_cache may not be available in all PyTorch versions

        self._load_models()

    def _sample_clip_tchw(self) -> Tuple[Optional[np.ndarray], List[np.ndarray]]:
        """Sample frames from clip buffer for V-JEPA classification.

        Optimized: frames stored as BGR, only sampled frames converted to RGB.
        """
        with QtCore.QMutexLocker(self._clip_lock):
            items = list(self._clip_frames)
        if not items:
            return None, []

        T = len(items)
        samples = self._get_vjepa_frame_count()

        # Calculate sample indices
        if T < samples:
            idxs = list(range(T))
        else:
            idxs = np.linspace(0, T - 1, num=samples).round().astype(int).tolist()

        # Extract only sampled frames (stored as BGR)
        sampled_bgr = [items[i][1] for i in idxs]

        # Convert only sampled frames to RGB (not all frames in buffer)
        sampled_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in sampled_bgr]

        arr = np.stack(sampled_rgb, axis=0)      # T,H,W,C
        arr = np.transpose(arr, (0, 3, 1, 2))    # T,C,H,W
        return arr.astype(np.uint8), sampled_bgr

    def _should_fp16_fallback(self, tb: str) -> bool:
        low = tb.lower()
        if "attributeerror" in low or "keyerror" in low or "typeerror" in low:
            return False
        return (
            "not currently supported on the mps backend" in low
            or "not implemented" in low
            or "unsupported" in low
            or "float16" in low
            or "half" in low
        )

    def run(self):
        self._running = True
        self._load_models()

        next_t = time.perf_counter()

        while self._running:
            # Recompute each iteration so UI can change the inference cadence live.
            period = 1.0 / max(0.1, float(self.cfg.inference_fps))
            now = time.perf_counter()
            if now < next_t:
                time.sleep(min(0.01, next_t - now))
                continue
            next_t = now + period

            with QtCore.QMutexLocker(self._frame_lock):
                bgr_full = None if self._latest_frame is None else self._latest_frame.copy()
            if bgr_full is None:
                continue

            self._tick += 1
            tick_id = self._tick
            ts_wall = time.time()

            with QtCore.QMutexLocker(self._ctrl_lock):
                prompts = list(self._prompts)
                use_boxes = bool(self._use_boxes)
                sam_thr = float(self._sam_thr)
                sam_mask_thr = float(self._sam_mask_thr)
                enable_dino = bool(self._enable_dino)
                enable_sam = bool(self._enable_sam)
                enable_vjepa = bool(self._enable_vjepa)
                dino_strength = float(self._dino_strength)
                dino_blur = int(self._dino_blur)
                max_w = int(self._max_w)
                run_dino_n = int(self._run_dino_n)
                run_sam_n = int(self._run_sam_n)
                run_vjepa_n = int(self._run_vjepa_n)

            # Resize once to reduce compute + memory
            bgr = fit_to_width(bgr_full, max_w)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            outs = StageOutputs(tick_id=tick_id, ts_wall=ts_wall, raw_bgr=bgr)

            # DINO
            heat = None
            dino_info = {}
            saliency_confident = False
            dino_sal_thr = float(self.cfg.dino_saliency_threshold)
            dino_sal_mode = str(self.cfg.dino_saliency_mode)
            do_dino = (tick_id % max(1, run_dino_n) == 0)
            if self.models.dino and enable_dino and do_dino:
                try:
                    heat, dino_info = self.models.dino.saliency_heatmap(
                        rgb, saliency_threshold=dino_sal_thr, mode=dino_sal_mode
                    )
                    if dino_blur >= 3 and dino_blur % 2 == 1:
                        heat = cv2.GaussianBlur(heat, (dino_blur, dino_blur), 0)

                    # Check if saliency is confident (raw score variance is meaningful)
                    saliency_confident = dino_info.get("saliency_confident", False)

                    # Only generate proposal boxes if saliency is confident
                    prop_boxes = DinoV3Stage.heatmap_to_boxes(
                        heat,
                        top_k=int(self.cfg.dino_proposal_count),
                        min_area_frac=float(self.cfg.dino_proposal_min_area),
                        max_area_frac=float(self.cfg.dino_proposal_max_area),
                    ) if saliency_confident else []
                    outs.dino_boxes = prop_boxes

                    # If the map is almost flat, a full-strength overlay just tints the whole frame.
                    # Scale the overlay strength down in that case.
                    heat_std = float(np.std(heat))
                    strength_eff = float(dino_strength)
                    if heat_std < 0.03:
                        strength_eff *= max(0.0, min(1.0, heat_std / 0.03))
                    vis = overlay_heatmap(bgr, heat, strength_eff)
                    if prop_boxes and saliency_confident:
                        vis = draw_boxes(
                            vis,
                            prop_boxes,
                            labels=[f"p{i+1}" for i in range(len(prop_boxes))],
                            color=(0, 255, 255),
                            thickness=2,
                        )
                    outs.dino_bgr = vis
                    outs.dino_info = dino_info
                    outs.dino_heat = heat  # Store for composite view
                except Exception:
                    # If FP16 is enabled on MPS, try a one-time fallback
                    tb = traceback.format_exc()
                    if (self.device.type == "mps") and self._fp16_on_mps and (
                    not self._did_fp16_fallback) and self._should_fp16_fallback(tb):
                        self._event("[dino] error (fp16/mps fallback):\n" + tb)
                        self._reload_models_fp32()
                    else:
                        self._event("[dino] error:\n" + tb)
            # Detection (SAM or YOLO based on detection_source setting)
            sam_dets: List[Detection] = []
            union_areas: Dict[str, float] = {}
            do_detection = (tick_id % max(1, run_sam_n) == 0)
            detection_source = self._detection_source

            # YOLO detection path
            if detection_source in ("yolo11", "yolo26") and do_detection:
                try:
                    yolo_stage = self._ensure_yolo_stage()
                    if yolo_stage is not None:
                        masks, boxes, labels, yolo_info = yolo_stage(bgr)
                        scores = yolo_info.get("scores", [1.0] * len(boxes))

                        # Create visualization (boxes only, no masks from YOLO)
                        outs.sam_bgr = draw_boxes(bgr.copy(), boxes, labels, color=(0, 255, 0), thickness=2)

                        # Convert to detections
                        try:
                            sam_dets, union_areas = detections_from_yolo(boxes, labels, scores, hw=rgb.shape[:2])
                            yolo_info['areas'] = union_areas
                        except Exception:
                            sam_dets, union_areas = [], {}

                        # Phase 3: compute per-detection DINO embeddings for ReID
                        try:
                            if (
                                sam_dets
                                and getattr(self.cfg, 'enable_reid', False)
                                and getattr(self.cfg, 'tracker_use_embedding', True)
                                and self.models.dino is not None
                                and enable_dino
                                and (tick_id % max(1, int(getattr(self.cfg, 'reid_every_n', 2))) == 0)
                            ):
                                max_dets = max(1, int(getattr(self.cfg, 'reid_max_dets', 6)))
                                idxs = sorted(range(len(sam_dets)), key=lambda i: float(sam_dets[i].area_frac), reverse=True)[:max_dets]
                                boxes_sel = [sam_dets[i].bbox_xyxy for i in idxs]
                                embs = self.models.dino.embed_boxes(rgb, boxes_sel)
                                for k, i in enumerate(idxs):
                                    if k < len(embs):
                                        sam_dets[i].emb = embs[k]
                        except Exception:
                            pass

                        outs.sam_info = yolo_info
                        outs.sam_counts = {lab: labels.count(lab) for lab in set(labels)}
                        outs.sam_masks = []  # YOLO doesn't provide masks
                        outs.sam_boxes = boxes
                        outs.sam_labels = labels
                        self._last_sam_dets = sam_dets
                        self._last_sam_tick = tick_id
                except Exception:
                    tb = traceback.format_exc()
                    self._event(f"[yolo] error:\n" + tb)

            # SAM detection path (default)
            elif detection_source == "sam" and enable_sam and do_detection and (self._ensure_sam_stage() is not None):
                try:
                    masks: List[np.ndarray] = []
                    boxes: List[List[float]] = []
                    labels: List[str] = []
                    sam_info: Dict[str, Any] = {}
                    sam_counts: Dict[str, int] = {}

                    if use_boxes and heat is not None and saliency_confident:
                        # Use DINO proposals to focus SAM, but only if saliency is confident.
                        # Reuse prop_boxes from DINO stage (already stored in outs.dino_boxes).
                        prop_boxes = outs.dino_boxes
                        if prop_boxes:
                            if prompts:
                                masks, boxes, labels, sam_info = self.models.sam.segment_prompts_in_boxes(
                                    rgb,
                                    prop_boxes,
                                    prompts=prompts,
                                    threshold=sam_thr,
                                    mask_threshold=sam_mask_thr,
                                )
                            else:
                                masks, boxes, labels, sam_info = self.models.sam.segment_boxes(
                                    rgb, prop_boxes, threshold=sam_thr, mask_threshold=sam_mask_thr
                                )
                            sam_counts = sam_info.get("counts", {}) if isinstance(sam_info.get("counts"), dict) else {"box": len(masks)}
                        else:
                            masks, boxes, labels, sam_info = self.models.sam.segment_prompts(
                                rgb, prompts, threshold=sam_thr, mask_threshold=sam_mask_thr
                            )
                            sam_counts = sam_info.get("counts", {}) if isinstance(sam_info.get("counts"), dict) else {}
                    else:
                        masks, boxes, labels, sam_info = self.models.sam.segment_prompts(
                            rgb, prompts, threshold=sam_thr, mask_threshold=sam_mask_thr
                        )
                        sam_counts = sam_info.get("counts", {}) if isinstance(sam_info.get("counts"), dict) else {}

                    outs.sam_bgr = overlay_masks(bgr, masks=masks, boxes=boxes, labels=labels, alpha=0.45) if masks else bgr.copy()
                    # Derive detections + union-area fractions for tracking and better summaries
                    try:
                        sam_dets, union_areas = detections_from_sam(masks, boxes, labels, hw=rgb.shape[:2])
                        if isinstance(sam_info, dict):
                            sam_info = dict(sam_info)
                            sam_info.setdefault('areas', union_areas)
                    except Exception:
                        sam_dets, union_areas = [], {}

                    # Phase 3: compute per-detection DINO embeddings for ReID (capped + cadenced)
                    try:
                        if (
                            sam_dets
                            and getattr(self.cfg, 'enable_reid', False)
                            and getattr(self.cfg, 'tracker_use_embedding', True)
                            and self.models.dino is not None
                            and enable_dino
                            and (tick_id % max(1, int(getattr(self.cfg, 'reid_every_n', 2))) == 0)
                        ):
                            max_dets = max(1, int(getattr(self.cfg, 'reid_max_dets', 6)))
                            idxs = sorted(range(len(sam_dets)), key=lambda i: float(sam_dets[i].area_frac), reverse=True)[:max_dets]
                            boxes_sel = [sam_dets[i].bbox_xyxy for i in idxs]
                            embs = self.models.dino.embed_boxes(rgb, boxes_sel)
                            for k, i in enumerate(idxs):
                                if k < len(embs):
                                    sam_dets[i].emb = embs[k]
                    except Exception:
                        # ReID is best-effort; never break the pipeline
                        pass

                    outs.sam_info = sam_info
                    outs.sam_counts = sam_counts
                    # Store intermediate data for composite view (downsample for memory)
                    outs.sam_masks = downsample_masks(masks) if masks else []
                    outs.sam_boxes = boxes
                    outs.sam_labels = labels
                    # Cache detections for tracker on ticks when SAM is skipped
                    self._last_sam_dets = sam_dets
                    self._last_sam_tick = tick_id
                except Exception:
                    tb = traceback.format_exc()
                    if (self.device.type == "mps") and self._fp16_on_mps and (
                    not self._did_fp16_fallback) and self._should_fp16_fallback(tb):
                        self._event("[sam] error (fp16/mps fallback):\n" + tb)
                        self._reload_models_fp32()
                    else:
                        self._event("[sam] error:\n" + tb)
            # V-JEPA
            do_vjepa = (tick_id % max(1, run_vjepa_n) == 0)
            if self.models.vjepa and enable_vjepa and do_vjepa:
                try:
                    clip_tchw, sampled_bgr = self._sample_clip_tchw()
                    if clip_tchw is not None:
                        pairs, vinfo = self.models.vjepa.classify_clip(clip_tchw)
                        outs.vjepa_pairs = pairs
                        outs.vjepa_info = vinfo
                        if sampled_bgr:
                            # Create grid view: 4x4 for 16 frames, 8x8 for 64 frames
                            outs.clip_strip_bgr = make_clip_grid(sampled_bgr, target_size=480)
                        # Clear buffer after V-JEPA processes so next run gets fresh frames
                        with QtCore.QMutexLocker(self._clip_lock):
                            self._clip_frames.clear()
                except Exception:
                    tb = traceback.format_exc()
                    if (self.device.type == "mps") and self._fp16_on_mps and (
                    not self._did_fp16_fallback) and self._should_fp16_fallback(tb):
                        self._event("[vjepa] error (fp16/mps fallback):\n" + tb)
                        self._reload_models_fp32()
                    else:
                        self._event("[vjepa] error:\n" + tb)
            # ----------------------------
            # Tracking + narration (stateful)
            # ----------------------------
            if bool(self.cfg.enable_tracking):
                # Use cached detections if SAM was skipped (not failed) this tick
                # This prevents tracker from thinking objects disappeared just because SAM didn't run
                tracker_dets = sam_dets
                if not sam_dets and (not do_detection) and self._last_sam_dets:
                    # SAM was skipped (cadence), reuse last detections
                    tracker_dets = self._last_sam_dets

                try:
                    tracks, created, removed = self._tracker.update(tracker_dets, tick_id=tick_id)
                except Exception:
                    tracks, created, removed = [], [], []

                # Smooth V-JEPA predictions (EMA) so "activity" feels continuous
                # Decay ALWAYS happens (even when V-JEPA is skipped) to prevent stale labels
                for k in list(self._vjepa_ema.keys()):
                    self._vjepa_ema[k] *= 0.80
                # Prune near-zero entries to prevent unbounded growth
                self._vjepa_ema = {k: v for k, v in self._vjepa_ema.items() if v > 0.01}

                if outs.vjepa_pairs:
                    # update from current top-k
                    for lbl, p in outs.vjepa_pairs[:10]:
                        self._vjepa_ema[str(lbl)] = self._vjepa_ema.get(str(lbl), 0.0) + 0.20 * float(p)

                # Update stable action from EMA (even when V-JEPA didn't run this tick)
                if self._vjepa_ema:
                    top_lbl, top_p = max(self._vjepa_ema.items(), key=lambda kv: kv[1])
                    if top_p > 0.10:  # Only report if confidence is meaningful
                        self._stable_action = (str(top_lbl), float(top_p))
                    else:
                        self._stable_action = None  # Clear stale action
                else:
                    self._stable_action = None

                outs.stable_action = self._stable_action

                # Build narration events (debounced)
                events: List[str] = []

                def _cooldown(key: str, cd: int = 6) -> bool:
                    last = self._last_event_cooldown.get(key, -10**9)
                    if (tick_id - last) < cd:
                        return False
                    self._last_event_cooldown[key] = tick_id
                    return True

                for tr in created:
                    lab, conf = tr.top_label()
                    if lab in ('person', 'hand', 'phone', 'keyboard', 'mouse', 'laptop', 'cup') and conf >= 0.30:
                        if _cooldown(f"new:{lab}", cd=12):
                            events.append(f"Detected {lab}")

                for tr in removed:
                    lab, conf = tr.top_label()
                    if lab in ('person', 'hand', 'phone', 'keyboard', 'mouse', 'laptop', 'cup') and conf >= 0.30:
                        if _cooldown(f"gone:{lab}", cd=12):
                            events.append(f"{lab} left view")

                if outs.stable_action and outs.stable_action[1] > 0.45:
                    cur = outs.stable_action[0]
                    if self._last_action_label is None:
                        self._last_action_label = cur
                    elif cur != self._last_action_label:
                        if _cooldown(f"act:{cur}", cd=10):
                            events.append(f"Activity changed: {cur}")
                            self._last_action_label = cur

                # Compose a stable-ish one-line state description
                # Phase 3: infer simple interaction flags and emit debounced events
                diag = math.hypot(bgr.shape[1], bgr.shape[0]) if bgr is not None else 1.0
                diag = diag if diag > 1e-6 else 1.0

                def _stable_by_label(label: str):
                    cands = []
                    for t in tracks:
                        lab, conf = t.top_label()
                        if lab == label and conf >= 0.30 and t.hits >= 2:
                            cands.append((conf * (0.5 + t.area_frac), t))
                    if not cands:
                        return None
                    cands.sort(reverse=True, key=lambda x: x[0])
                    return cands[0][1]

                def _all_by_label(label: str):
                    out = []
                    for t in tracks:
                        lab, conf = t.top_label()
                        if lab == label and conf >= 0.30 and t.hits >= 2:
                            out.append(t)
                    return out

                hands = _all_by_label('hand')
                phone = _stable_by_label('phone')
                keyboard = _stable_by_label('keyboard')
                cup = _stable_by_label('cup')

                contact_d = float(getattr(self.cfg, 'evt_contact_dist_frac', 0.18)) * diag
                typing_d = float(getattr(self.cfg, 'evt_typing_dist_frac', 0.25)) * diag
                typing_motion = float(getattr(self.cfg, 'evt_typing_motion_frac', 0.004))
                hold_motion = float(getattr(self.cfg, 'evt_hold_motion_frac', 0.012))
                settle_motion = float(getattr(self.cfg, 'evt_settle_motion_frac', 0.003))

                def _min_dist(tr, hs):
                    if tr is None or not hs:
                        return 1e9
                    best = 1e9
                    for htr in hs:
                        best = min(best, _dist((tr.cx, tr.cy), (htr.cx, htr.cy)))
                    return best

                # Phone held state
                if phone is not None and hands:
                    dph = _min_dist(phone, hands)
                    if dph < contact_d:
                        self._phone_contact = min(self._phone_contact + 1, 10)
                    else:
                        self._phone_contact = max(self._phone_contact - 1, 0)
                    pm = (phone.speed / diag)
                    if (not self._phone_held) and self._phone_contact >= 2 and pm > hold_motion:
                        if _cooldown('evt:phone_pick', cd=12):
                            events.append('Phone picked up')
                        self._phone_held = True
                    if self._phone_held and self._phone_contact == 0 and pm < settle_motion:
                        if _cooldown('evt:phone_down', cd=12):
                            events.append('Phone put down')
                        self._phone_held = False
                else:
                    self._phone_contact = max(self._phone_contact - 1, 0)
                    if phone is None:
                        self._phone_held = False

                # Cup held state
                if cup is not None and hands:
                    dch = _min_dist(cup, hands)
                    if dch < contact_d:
                        self._cup_contact = min(self._cup_contact + 1, 10)
                    else:
                        self._cup_contact = max(self._cup_contact - 1, 0)
                    cm = (cup.speed / diag)
                    if (not self._cup_held) and self._cup_contact >= 2 and cm > hold_motion:
                        if _cooldown('evt:cup_pick', cd=18):
                            events.append('Cup picked up')
                        self._cup_held = True
                    if self._cup_held and self._cup_contact == 0 and cm < settle_motion:
                        if _cooldown('evt:cup_down', cd=18):
                            events.append('Cup put down')
                        self._cup_held = False
                else:
                    self._cup_contact = max(self._cup_contact - 1, 0)
                    if cup is None:
                        self._cup_held = False

                # Typing (debounced) = hands near keyboard + hands moving a bit
                typing_now = False
                if keyboard is not None and hands:
                    dk = _min_dist(keyboard, hands)
                    hm = 0.0
                    for htr in hands[:3]:
                        hm = max(hm, (htr.speed / diag))
                    typing_now = (dk < typing_d) and (hm > typing_motion)

                if typing_now:
                    self._typing_score = min(self._typing_score + 1, 10)
                else:
                    self._typing_score = max(self._typing_score - 1, 0)

                if (not self._typing_on) and self._typing_score >= 3:
                    if _cooldown('evt:typing_on', cd=12):
                        events.append('Started typing')
                    self._typing_on = True
                if self._typing_on and self._typing_score == 0:
                    if _cooldown('evt:typing_off', cd=12):
                        events.append('Stopped typing')
                    self._typing_on = False

                flags = {
                    'typing': bool(self._typing_on),
                    'phone_held': bool(self._phone_held),
                    'cup_held': bool(self._cup_held),
                }

                # Captioning: periodic + optional on-events
                fresh_caption = False
                cap_txt = ""
                if self.models.caption is not None and bool(getattr(self, "_enable_caption", True)):
                    do_cap = (tick_id % max(1, int(getattr(self, "_caption_every_n", 5))) == 0)
                    if bool(getattr(self, "_caption_on_events", True)) and events:
                        if (tick_id - int(getattr(self, "_last_caption_tick", -10**9))) >= int(getattr(self, "_caption_min_tick_gap", 2)):
                            do_cap = True
                    if do_cap:
                        try:
                            cap_txt, cap_info = self.models.caption.caption(
                                rgb,
                                max_new_tokens=int(getattr(self, "_caption_max_new_tokens", 24)),
                                num_beams=int(getattr(self, "_caption_num_beams", 3)),
                            )
                            if cap_txt:
                                self._last_caption_text = cap_txt
                                self._last_caption_tick = tick_id
                                outs.caption = cap_txt
                                outs.caption_info = dict(cap_info)
                                outs.caption_info["fresh"] = True
                                fresh_caption = True
                        except Exception:
                            self._event("[caption] error:\n" + traceback.format_exc())
                if not getattr(outs, "caption", "") and getattr(self, "_last_caption_text", ""):
                    outs.caption = self._last_caption_text
                    outs.caption_info = {"fresh": False, "age_ticks": tick_id - int(getattr(self, "_last_caption_tick", tick_id))}

                # Compose a stable-ish one-line state description with source attribution
                narration_attr = compose_narration_attributed(tracks, outs.stable_action, frame_hw=bgr.shape[:2], flags=flags, caption=(outs.caption if fresh_caption else None), tick_id=tick_id)
                narration = narration_attr.plain_text
                outs.narration_attributed = narration_attr

                # Attach lightweight track summaries for UI/Timeline
                outs.tracks = []
                for tr in tracks:
                    lab, conf = tr.top_label()
                    outs.tracks.append({
                        'id': tr.id,
                        'label': lab,
                        'conf': conf,
                        'bbox': [float(x) for x in tr.bbox_xyxy[:4]],
                        'area': float(tr.area_frac),
                        'cx': float(tr.cx),
                        'cy': float(tr.cy),
                        'vx': float(tr.vx),
                        'vy': float(tr.vy),
                        'hits': int(tr.hits),
                    })

                outs.narration = narration
                outs.narration_events = events
                outs.tracks_raw = tracks  # Store raw Track objects for composite view

                # Overlay tracks on SAM visualization for immediate feedback
                if outs.sam_bgr is not None and tracks:
                    outs.sam_bgr = tracks_to_overlay(outs.sam_bgr, tracks)

            # Generate composite view (all layers combined)
            if outs.raw_bgr is not None:
                outs.composite_bgr = make_composite_view(
                    raw_bgr=outs.raw_bgr,
                    dino_heat=outs.dino_heat,
                    sam_masks=outs.sam_masks,
                    sam_boxes=outs.sam_boxes,
                    sam_labels=outs.sam_labels,
                    tracks=outs.tracks_raw,
                )

            # Generate dedicated tracking view
            if outs.raw_bgr is not None:
                outs.tracking_bgr = make_tracking_view(
                    bgr=outs.raw_bgr,
                    tracks=outs.tracks_raw,
                    max_missed=int(self.cfg.tracker_max_missed),
                )

            self.outputs_signal.emit(outs)


    def stop(self):
        self._running = False


# ----------------------------
# UI widgets
# ----------------------------

class ScaledPixmapLabel(QtWidgets.QLabel):
    """QLabel that keeps the original pixmap and scales it on resize."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._orig: Optional[QtGui.QPixmap] = None
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(200, 150)
        self.setStyleSheet("background: #111; border: 2px solid #222;")

    def setPixmap(self, pm: QtGui.QPixmap) -> None:  # type: ignore[override]
        self._orig = pm
        self._rescale()

    def clearPixmap(self) -> None:
        self._orig = None
        super().clear()

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
        super().resizeEvent(e)
        self._rescale()

    def _rescale(self) -> None:
        if self._orig is None or self.width() <= 1 or self.height() <= 1:
            return
        scaled = self._orig.scaled(
            self.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        super().setPixmap(scaled)


class ImagePane(QtWidgets.QGroupBox):
    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self.label = ScaledPixmapLabel()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.label)

    def set_image_bgr(self, bgr: Optional[np.ndarray]):
        if bgr is None:
            return
        qi = bgr_to_qimage(bgr)
        self.label.setPixmap(QtGui.QPixmap.fromImage(qi))


class PipelineStatusWidget(QtWidgets.QWidget):
    """Compact pipeline flow indicator showing live data flow with counts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(28)
        self.setMinimumWidth(400)
        self._stats = {
            "fps": 0.0,
            "dino": 0,
            "sam": 0,
            "vjepa": 0,
            "tracks": 0,
        }
        # Stage colors
        self._colors = {
            "cam": QtGui.QColor("#888888"),
            "dino": QtGui.QColor("#FF8C00"),
            "sam": QtGui.QColor("#32CD32"),
            "vjepa": QtGui.QColor("#1E90FF"),
            "caption": QtGui.QColor("#FFD700"),
        }

    def update_stats(self, fps: float = 0.0, dino_count: int = 0, sam_count: int = 0,
                     vjepa_count: int = 0, track_count: int = 0):
        self._stats = {
            "fps": fps,
            "dino": dino_count,
            "sam": sam_count,
            "vjepa": vjepa_count,
            "tracks": track_count,
        }
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # Font setup
        font = painter.font()
        font.setPointSize(9)
        font.setFamily("monospace")
        painter.setFont(font)

        # Layout: [CAM] -> [DINO:n] -> [SAM:n] -> [VJEPA:n] -> [Caption]
        stages = [
            ("CAM", f"{self._stats['fps']:.1f}", self._colors["cam"]),
            ("DINO", str(self._stats['dino']), self._colors["dino"]),
            ("SAM", str(self._stats['sam']), self._colors["sam"]),
            ("VJEPA", str(self._stats['vjepa']), self._colors["vjepa"]),
            ("TRK", str(self._stats['tracks']), self._colors["caption"]),
        ]

        x = 5
        y = 4
        box_h = 20
        arrow_w = 12

        for i, (name, count, color) in enumerate(stages):
            # Calculate box width based on text
            text = f"{name}:{count}"
            text_width = painter.fontMetrics().horizontalAdvance(text) + 12
            box_w = max(45, text_width)

            # Draw box
            rect = QtCore.QRectF(x, y, box_w, box_h)
            painter.setPen(QtGui.QPen(color, 1.5))
            painter.setBrush(QtGui.QBrush(color.darker(400)))
            painter.drawRoundedRect(rect, 4, 4)

            # Draw text
            painter.setPen(color.lighter(150))
            painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, text)

            x += box_w

            # Draw arrow (except after last)
            if i < len(stages) - 1:
                arrow_y = y + box_h // 2
                painter.setPen(QtGui.QPen(QtGui.QColor("#666"), 1.5))
                painter.drawLine(int(x + 2), int(arrow_y), int(x + arrow_w - 4), int(arrow_y))
                # Arrow head
                painter.drawLine(int(x + arrow_w - 6), int(arrow_y - 3), int(x + arrow_w - 4), int(arrow_y))
                painter.drawLine(int(x + arrow_w - 6), int(arrow_y + 3), int(x + arrow_w - 4), int(arrow_y))
                x += arrow_w

        painter.end()


class TimelineTable(QtWidgets.QTableWidget):
    def __init__(self, parent=None):
        super().__init__(0, 8, parent)
        self.setHorizontalHeaderLabels(["Tick", "Time", "DINO", "SAM", "VJ", "Events", "Counts", "Activity"])
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.setStyleSheet("""
            QTableWidget { background:#0d0d0d; color:#ddd; gridline-color:#222; }
            QHeaderView::section { background:#111; color:#ddd; padding:6px; border:1px solid #222; }
            QTableWidget::item:selected { background:#2a3a52; }
        """)



class SettingsDialog(QtWidgets.QDialog):
    """Modal settings dialog. Updates RuntimeConfig + ModelIds without losing functionality."""
    def __init__(self, cfg: RuntimeConfig, ids: ModelIds, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pipeline Settings")
        self.setModal(True)
        self._cfg = dataclasses.replace(cfg)
        self._ids = dataclasses.replace(ids)

        self.tabs = QtWidgets.QTabWidget()

        # --- Pipeline tab
        w_pipeline = QtWidgets.QWidget()
        f_pipeline = QtWidgets.QFormLayout(w_pipeline)
        f_pipeline.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        self.camera_index = QtWidgets.QSpinBox()
        self.camera_index.setRange(0, 16)
        self.camera_index.setValue(self._cfg.camera_index)

        self.camera_width = QtWidgets.QSpinBox()
        self.camera_width.setRange(160, 3840)
        self.camera_width.setSingleStep(160)
        self.camera_width.setValue(self._cfg.camera_width)

        self.camera_height = QtWidgets.QSpinBox()
        self.camera_height.setRange(120, 2160)
        self.camera_height.setSingleStep(120)
        self.camera_height.setValue(self._cfg.camera_height)

        self.inference_fps = QtWidgets.QDoubleSpinBox()
        self.inference_fps.setRange(0.1, 30.0)
        self.inference_fps.setDecimals(2)
        self.inference_fps.setSingleStep(0.25)
        self.inference_fps.setValue(float(self._cfg.inference_fps))

        self.prompts = QtWidgets.QLineEdit(self._cfg.default_prompts)
        self.prompts.setPlaceholderText("comma-separated prompts e.g. person, phone, keyboard")

        self.use_boxes = QtWidgets.QCheckBox("Use DINO proposal boxes as SAM prompts when DINO enabled")
        self.use_boxes.setChecked(self._cfg.use_dino_boxes_for_sam)

        self.enable_dino = QtWidgets.QCheckBox("Enable DINOv3")
        self.enable_dino.setChecked(self._cfg.enable_dino)

        self.dino_saliency_mode = QtWidgets.QComboBox()
        self.dino_saliency_mode.addItems(["inverse", "magnitude", "attention"])
        current_mode = str(getattr(self._cfg, "dino_saliency_mode", "inverse"))
        idx = self.dino_saliency_mode.findText(current_mode)
        if idx >= 0:
            self.dino_saliency_mode.setCurrentIndex(idx)

        self.dino_proposal_count = QtWidgets.QSpinBox()
        self.dino_proposal_count.setRange(1, 20)
        self.dino_proposal_count.setValue(int(getattr(self._cfg, "dino_proposal_count", 5)))

        self.dino_proposal_min_area = QtWidgets.QDoubleSpinBox()
        self.dino_proposal_min_area.setRange(0.001, 0.20)
        self.dino_proposal_min_area.setDecimals(3)
        self.dino_proposal_min_area.setSingleStep(0.005)
        self.dino_proposal_min_area.setValue(float(getattr(self._cfg, "dino_proposal_min_area", 0.005)))
        self.dino_proposal_min_area.setSuffix(" (frac)")

        self.enable_sam = QtWidgets.QCheckBox("Enable SAM3")
        self.enable_sam.setChecked(self._cfg.enable_sam)
        self.enable_vjepa = QtWidgets.QCheckBox("Enable V-JEPA2")
        self.enable_vjepa.setChecked(self._cfg.enable_vjepa)

        self.enable_caption = QtWidgets.QCheckBox("Enable Captioning (BLIP scene description)")
        self.enable_caption.setChecked(bool(getattr(self._cfg, "enable_caption", True)))

        self.pause_on_freeze = QtWidgets.QCheckBox("Pause inference when frozen")
        self.pause_on_freeze.setChecked(self._cfg.pause_on_freeze)

        f_pipeline.addRow("Camera index", self.camera_index)

        hw = QtWidgets.QHBoxLayout()
        hw.setContentsMargins(0, 0, 0, 0)
        hw.addWidget(self.camera_width, 1)
        hw.addWidget(QtWidgets.QLabel("x"))
        hw.addWidget(self.camera_height, 1)
        w_hw = QtWidgets.QWidget()
        w_hw.setLayout(hw)
        f_pipeline.addRow("Capture size", w_hw)

        f_pipeline.addRow("Inference FPS", self.inference_fps)
        f_pipeline.addRow("Prompts", self.prompts)
        f_pipeline.addRow("", self.use_boxes)
        f_pipeline.addRow("", self.enable_dino)
        f_pipeline.addRow("DINO saliency mode", self.dino_saliency_mode)
        f_pipeline.addRow("DINO proposal count", self.dino_proposal_count)
        f_pipeline.addRow("DINO min box area", self.dino_proposal_min_area)
        f_pipeline.addRow("", self.enable_sam)
        f_pipeline.addRow("", self.enable_vjepa)
        f_pipeline.addRow("", self.enable_caption)
        f_pipeline.addRow("", self.pause_on_freeze)

        self.tabs.addTab(w_pipeline, "Pipeline")

        # --- Thresholds/visuals tab
        w_vis = QtWidgets.QWidget()
        f_vis = QtWidgets.QFormLayout(w_vis)
        f_vis.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        self.max_w = QtWidgets.QSpinBox()
        self.max_w.setRange(320, 1920)
        self.max_w.setSingleStep(64)
        self.max_w.setValue(int(getattr(self._cfg, "display_max_width", getattr(self._cfg, "max_display_width", 640))))

        self.dino_strength = QtWidgets.QDoubleSpinBox()
        self.dino_strength.setRange(0.0, 1.0)
        self.dino_strength.setSingleStep(0.05)
        self.dino_strength.setValue(self._cfg.dino_overlay_strength)

        self.dino_blur = QtWidgets.QSpinBox()
        self.dino_blur.setRange(0, 51)
        self.dino_blur.setSingleStep(2)
        self.dino_blur.setValue(self._cfg.dino_blur_ksize)

        self.sam_thr = QtWidgets.QDoubleSpinBox()
        self.sam_thr.setRange(0.05, 0.95)
        self.sam_thr.setSingleStep(0.05)
        self.sam_thr.setValue(self._cfg.sam_threshold)

        self.sam_mask_thr = QtWidgets.QDoubleSpinBox()
        self.sam_mask_thr.setRange(0.0, 1.0)
        self.sam_mask_thr.setSingleStep(0.05)
        self.sam_mask_thr.setValue(self._cfg.sam_mask_threshold)

        f_vis.addRow("Display max width", self.max_w)
        f_vis.addRow("DINO overlay strength", self.dino_strength)
        f_vis.addRow("DINO blur (odd)", self.dino_blur)
        f_vis.addRow("SAM threshold", self.sam_thr)
        f_vis.addRow("SAM mask threshold", self.sam_mask_thr)

        self.tabs.addTab(w_vis, "Visuals")

        # --- Detection tab (SAM vs YOLO)
        w_detect = QtWidgets.QWidget()
        f_detect = QtWidgets.QFormLayout(w_detect)
        f_detect.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        self.detection_source = QtWidgets.QComboBox()
        self.detection_source.addItems(["sam", "yolo11", "yolo26"])
        current_source = str(getattr(self._cfg, "detection_source", "sam"))
        src_idx = self.detection_source.findText(current_source)
        if src_idx >= 0:
            self.detection_source.setCurrentIndex(src_idx)

        self.yolo_model = QtWidgets.QComboBox()
        self.yolo_model.addItems([
            "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
            "yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x",
        ])
        current_yolo = str(getattr(self._cfg, "yolo_model", "yolo11m"))
        yolo_idx = self.yolo_model.findText(current_yolo)
        if yolo_idx >= 0:
            self.yolo_model.setCurrentIndex(yolo_idx)

        self.yolo_conf = QtWidgets.QDoubleSpinBox()
        self.yolo_conf.setRange(0.05, 0.95)
        self.yolo_conf.setSingleStep(0.05)
        self.yolo_conf.setValue(float(getattr(self._cfg, "yolo_conf_threshold", 0.25)))

        # Note about YOLO
        yolo_note = QtWidgets.QLabel(
            "YOLO provides fast object detection with 80 COCO classes.\n"
            "No segmentation masks - boxes only. Requires 'ultralytics' package.\n"
            "Models: n=nano(fast), s=small, m=medium, l=large, x=xlarge(accurate)"
        )
        yolo_note.setWordWrap(True)
        yolo_note.setStyleSheet("color:#888;")

        f_detect.addRow("Detection source", self.detection_source)
        f_detect.addRow("YOLO model", self.yolo_model)
        f_detect.addRow("YOLO confidence", self.yolo_conf)
        f_detect.addRow("", yolo_note)

        self.tabs.addTab(w_detect, "Detection")

        # --- Cadence / performance tab
        w_perf = QtWidgets.QWidget()
        f_perf = QtWidgets.QFormLayout(w_perf)
        f_perf.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        self.run_dino_n = QtWidgets.QSpinBox()
        self.run_dino_n.setRange(1, 60)
        self.run_dino_n.setValue(self._cfg.run_dino_every_n)

        self.run_sam_n = QtWidgets.QSpinBox()
        self.run_sam_n.setRange(1, 60)
        self.run_sam_n.setValue(self._cfg.run_sam_every_n)

        self.run_vjepa_n = QtWidgets.QSpinBox()
        self.run_vjepa_n.setRange(1, 60)
        self.run_vjepa_n.setValue(self._cfg.run_vjepa_every_n)

        self.fp16_on_mps = QtWidgets.QCheckBox("FP16 on MPS (faster, may fall back to FP32)")
        self.fp16_on_mps.setChecked(self._cfg.fp16_on_mps)

        # Phase 3 tracking controls
        self.enable_tracking = QtWidgets.QCheckBox("Enable tracking + narration")
        self.enable_tracking.setChecked(bool(getattr(self._cfg, 'enable_tracking', True)))

        self.enable_reid = QtWidgets.QCheckBox("Enable ReID embeddings (DINO crops)")
        self.enable_reid.setChecked(bool(getattr(self._cfg, 'enable_reid', True)))

        self.reid_every_n = QtWidgets.QSpinBox()
        self.reid_every_n.setRange(1, 60)
        self.reid_every_n.setValue(int(getattr(self._cfg, 'reid_every_n', 2)))

        self.reid_max_dets = QtWidgets.QSpinBox()
        self.reid_max_dets.setRange(1, 40)
        self.reid_max_dets.setValue(int(getattr(self._cfg, 'reid_max_dets', 6)))

        self.tracker_use_embedding = QtWidgets.QCheckBox("Use embeddings for tracking association")
        self.tracker_use_embedding.setChecked(bool(getattr(self._cfg, 'tracker_use_embedding', True)))

        self.tracker_emb_weight = QtWidgets.QDoubleSpinBox()
        self.tracker_emb_weight.setRange(0.0, 1.0)
        self.tracker_emb_weight.setSingleStep(0.05)
        self.tracker_emb_weight.setValue(float(getattr(self._cfg, 'tracker_emb_weight', 0.35)))

        self.tracker_algorithm = QtWidgets.QComboBox()
        self.tracker_algorithm.addItems(["kalman", "simple"])
        current_algo = str(getattr(self._cfg, "tracker_algorithm", "kalman"))
        algo_idx = self.tracker_algorithm.findText(current_algo)
        if algo_idx >= 0:
            self.tracker_algorithm.setCurrentIndex(algo_idx)

        self.tracker_use_hungarian = QtWidgets.QCheckBox("Use Hungarian (optimal) assignment")
        self.tracker_use_hungarian.setChecked(bool(getattr(self._cfg, 'tracker_use_hungarian', True)))

        self.tracker_graveyard = QtWidgets.QDoubleSpinBox()
        self.tracker_graveyard.setRange(0.0, 60.0)
        self.tracker_graveyard.setSingleStep(1.0)
        self.tracker_graveyard.setValue(float(getattr(self._cfg, 'tracker_graveyard_seconds', 10.0)))
        self.tracker_graveyard.setSuffix(" sec")

        self.caption_every_n = QtWidgets.QSpinBox()
        self.caption_every_n.setRange(1, 60)
        self.caption_every_n.setValue(int(getattr(self._cfg, "caption_every_n", 5)))

        self.caption_on_events = QtWidgets.QCheckBox("Refresh caption on narration events")
        self.caption_on_events.setChecked(bool(getattr(self._cfg, "caption_on_events", True)))

        self.caption_gap = QtWidgets.QSpinBox()
        self.caption_gap.setRange(1, 60)
        self.caption_gap.setValue(int(getattr(self._cfg, "caption_min_tick_gap", 2)))

        self.caption_tokens = QtWidgets.QSpinBox()
        self.caption_tokens.setRange(8, 128)
        self.caption_tokens.setValue(int(getattr(self._cfg, "caption_max_new_tokens", 24)))

        self.caption_beams = QtWidgets.QSpinBox()
        self.caption_beams.setRange(1, 8)
        self.caption_beams.setValue(int(getattr(self._cfg, "caption_num_beams", 3)))

        f_perf.addRow("Run DINO every N ticks", self.run_dino_n)
        f_perf.addRow("Run SAM every N ticks", self.run_sam_n)
        f_perf.addRow("Run V-JEPA every N ticks", self.run_vjepa_n)
        f_perf.addRow("Run Caption every N ticks", self.caption_every_n)
        f_perf.addRow("Caption min gap (ticks)", self.caption_gap)
        f_perf.addRow("Caption max new tokens", self.caption_tokens)
        f_perf.addRow("Caption beams", self.caption_beams)
        f_perf.addRow("", self.caption_on_events)
        f_perf.addRow("", self.fp16_on_mps)
        f_perf.addRow("", self.enable_tracking)
        f_perf.addRow("", self.enable_reid)
        f_perf.addRow("ReID every N ticks", self.reid_every_n)
        f_perf.addRow("ReID max detections", self.reid_max_dets)
        f_perf.addRow("", self.tracker_use_embedding)
        f_perf.addRow("Tracker embedding weight", self.tracker_emb_weight)
        f_perf.addRow("Tracker algorithm", self.tracker_algorithm)
        f_perf.addRow("", self.tracker_use_hungarian)
        f_perf.addRow("Track graveyard time", self.tracker_graveyard)

        self.tabs.addTab(w_perf, "Performance")

        # --- Models / HF tab
        w_models = QtWidgets.QWidget()
        f_models = QtWidgets.QFormLayout(w_models)
        f_models.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        self.hf_token = QtWidgets.QLineEdit(self._cfg.hf_token or "")
        self.hf_token.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.hf_token.setPlaceholderText("Optional Hugging Face token (for gated / faster downloads)")

        self.dino_id = QtWidgets.QLineEdit(self._ids.dino)
        self.sam_id = QtWidgets.QLineEdit(self._ids.sam3)
        self.vjepa_id = QtWidgets.QLineEdit(self._ids.vjepa2_cls)

        self.caption_id = QtWidgets.QLineEdit(getattr(self._ids, "caption", "Salesforce/blip-image-captioning-base"))

        note = QtWidgets.QLabel(
            "Note: Changing model IDs / HF token / FP16 requires a model reload.\n"
            "This app will automatically restart the inference thread after you Apply."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#bbb;")

        f_models.addRow("HF token", self.hf_token)
        f_models.addRow("DINOv3 model", self.dino_id)
        f_models.addRow("SAM3 model", self.sam_id)
        f_models.addRow("V-JEPA2 model", self.vjepa_id)
        f_models.addRow("Caption model", self.caption_id)
        f_models.addRow("", note)

        self.tabs.addTab(w_models, "Models")

        # Buttons
        self.btn_reset = QtWidgets.QPushButton("Reset to Defaults")
        self.btn_save = QtWidgets.QPushButton("Save Settings")
        self.btn_load = QtWidgets.QPushButton("Load Settings")
        self.btn_apply = QtWidgets.QPushButton("Apply")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_reset.clicked.connect(self._reset_defaults)
        self.btn_save.clicked.connect(self._save_settings)
        self.btn_load.clicked.connect(self._load_settings)
        self.btn_apply.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self.btn_reset)
        btns.addWidget(self.btn_save)
        btns.addWidget(self.btn_load)
        btns.addStretch(1)
        btns.addWidget(self.btn_apply)
        btns.addWidget(self.btn_cancel)

        # Scroll for small screens
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        wrap = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(wrap)
        v.addWidget(self.tabs)
        v.addLayout(btns)
        v.setContentsMargins(10, 10, 10, 10)
        scroll.setWidget(wrap)

        outer = QtWidgets.QVBoxLayout(self)
        outer.addWidget(scroll)

        self.resize(700, 700)

    def _reset_defaults(self):
        """Reset all settings to their default values."""
        default_cfg = RuntimeConfig()
        default_ids = ModelIds()

        # Pipeline tab
        self.camera_index.setValue(default_cfg.camera_index)
        self.camera_width.setValue(default_cfg.camera_width)
        self.camera_height.setValue(default_cfg.camera_height)
        self.inference_fps.setValue(default_cfg.inference_fps)
        self.prompts.setText(default_cfg.default_prompts)
        self.use_boxes.setChecked(default_cfg.use_dino_boxes_for_sam)
        self.enable_dino.setChecked(default_cfg.enable_dino)
        idx = self.dino_saliency_mode.findText(default_cfg.dino_saliency_mode)
        if idx >= 0:
            self.dino_saliency_mode.setCurrentIndex(idx)
        self.dino_proposal_count.setValue(default_cfg.dino_proposal_count)
        self.dino_proposal_min_area.setValue(default_cfg.dino_proposal_min_area)
        self.enable_sam.setChecked(default_cfg.enable_sam)
        self.enable_vjepa.setChecked(default_cfg.enable_vjepa)
        self.enable_caption.setChecked(default_cfg.enable_caption)
        self.pause_on_freeze.setChecked(default_cfg.pause_on_freeze)

        # Visuals tab
        self.max_w.setValue(default_cfg.display_max_width)
        self.dino_strength.setValue(default_cfg.dino_overlay_strength)
        self.dino_blur.setValue(default_cfg.dino_blur_ksize)
        self.sam_thr.setValue(default_cfg.sam_threshold)
        self.sam_mask_thr.setValue(default_cfg.sam_mask_threshold)

        # Detection tab
        src_idx = self.detection_source.findText(default_cfg.detection_source)
        if src_idx >= 0:
            self.detection_source.setCurrentIndex(src_idx)
        yolo_idx = self.yolo_model.findText(default_cfg.yolo_model)
        if yolo_idx >= 0:
            self.yolo_model.setCurrentIndex(yolo_idx)
        self.yolo_conf.setValue(default_cfg.yolo_conf_threshold)

        # Performance tab
        self.run_dino_n.setValue(default_cfg.run_dino_every_n)
        self.run_sam_n.setValue(default_cfg.run_sam_every_n)
        self.run_vjepa_n.setValue(default_cfg.run_vjepa_every_n)
        self.caption_every_n.setValue(default_cfg.caption_every_n)
        self.caption_gap.setValue(default_cfg.caption_min_tick_gap)
        self.caption_tokens.setValue(default_cfg.caption_max_new_tokens)
        self.caption_beams.setValue(default_cfg.caption_num_beams)
        self.caption_on_events.setChecked(default_cfg.caption_on_events)
        self.fp16_on_mps.setChecked(default_cfg.fp16_on_mps)
        self.enable_tracking.setChecked(default_cfg.enable_tracking)
        self.enable_reid.setChecked(default_cfg.enable_reid)
        self.reid_every_n.setValue(default_cfg.reid_every_n)
        self.reid_max_dets.setValue(default_cfg.reid_max_dets)
        self.tracker_use_embedding.setChecked(default_cfg.tracker_use_embedding)
        self.tracker_emb_weight.setValue(default_cfg.tracker_emb_weight)
        algo_idx = self.tracker_algorithm.findText(default_cfg.tracker_algorithm)
        if algo_idx >= 0:
            self.tracker_algorithm.setCurrentIndex(algo_idx)
        self.tracker_use_hungarian.setChecked(default_cfg.tracker_use_hungarian)
        self.tracker_graveyard.setValue(default_cfg.tracker_graveyard_seconds)

        # Models tab
        self.hf_token.setText("")
        self.dino_id.setText(default_ids.dino)
        self.sam_id.setText(default_ids.sam3)
        self.vjepa_id.setText(default_ids.vjepa2_cls)
        self.caption_id.setText(default_ids.caption)

    def _get_config_path(self) -> pathlib.Path:
        """Get path for settings JSON file."""
        config_dir = pathlib.Path.home() / ".config" / "vision_pipeline"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "settings.json"

    def _save_settings(self):
        """Save current settings to JSON file."""
        try:
            cfg, ids = self.collect()

            data = {
                "version": 1,
                "runtime_config": {
                    "camera_index": cfg.camera_index,
                    "camera_width": cfg.camera_width,
                    "camera_height": cfg.camera_height,
                    "inference_fps": cfg.inference_fps,
                    "default_prompts": cfg.default_prompts,
                    "enable_dino": cfg.enable_dino,
                    "dino_saliency_mode": cfg.dino_saliency_mode,
                    "dino_proposal_count": cfg.dino_proposal_count,
                    "dino_proposal_min_area": cfg.dino_proposal_min_area,
                    "enable_sam": cfg.enable_sam,
                    "enable_vjepa": cfg.enable_vjepa,
                    "enable_caption": cfg.enable_caption,
                    "pause_on_freeze": cfg.pause_on_freeze,
                    "use_dino_boxes_for_sam": cfg.use_dino_boxes_for_sam,
                    "display_max_width": cfg.display_max_width,
                    "dino_overlay_strength": cfg.dino_overlay_strength,
                    "dino_blur_ksize": cfg.dino_blur_ksize,
                    "sam_threshold": cfg.sam_threshold,
                    "sam_mask_threshold": cfg.sam_mask_threshold,
                    "run_dino_every_n": cfg.run_dino_every_n,
                    "run_sam_every_n": cfg.run_sam_every_n,
                    "run_vjepa_every_n": cfg.run_vjepa_every_n,
                    "caption_every_n": cfg.caption_every_n,
                    "caption_on_events": cfg.caption_on_events,
                    "caption_min_tick_gap": cfg.caption_min_tick_gap,
                    "caption_max_new_tokens": cfg.caption_max_new_tokens,
                    "caption_num_beams": cfg.caption_num_beams,
                    "fp16_on_mps": cfg.fp16_on_mps,
                    "enable_tracking": cfg.enable_tracking,
                    "enable_reid": cfg.enable_reid,
                    "reid_every_n": cfg.reid_every_n,
                    "reid_max_dets": cfg.reid_max_dets,
                    "tracker_use_embedding": cfg.tracker_use_embedding,
                    "tracker_emb_weight": cfg.tracker_emb_weight,
                    "tracker_algorithm": cfg.tracker_algorithm,
                    "tracker_use_hungarian": cfg.tracker_use_hungarian,
                    "tracker_graveyard_seconds": cfg.tracker_graveyard_seconds,
                    "timeline_max_items": cfg.timeline_max_items,
                    "detection_source": cfg.detection_source,
                    "yolo_model": cfg.yolo_model,
                    "yolo_conf_threshold": cfg.yolo_conf_threshold,
                },
                "model_ids": {
                    "dino": ids.dino,
                    "sam3": ids.sam3,
                    "vjepa2_cls": ids.vjepa2_cls,
                    "caption": ids.caption,
                }
            }

            path = self._get_config_path()
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

            QtWidgets.QMessageBox.information(
                self, "Settings Saved", f"Settings saved to:\n{path}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Save Failed", f"Could not save settings:\n{str(e)}"
            )

    def _load_settings(self):
        """Load settings from JSON file."""
        try:
            path = self._get_config_path()
            if not path.exists():
                QtWidgets.QMessageBox.information(
                    self, "No Saved Settings", "No saved settings file found."
                )
                return

            with open(path, 'r') as f:
                data = json.load(f)

            if data.get("version") != 1:
                QtWidgets.QMessageBox.warning(
                    self, "Version Mismatch", "Settings file version not supported."
                )
                return

            rc = data.get("runtime_config", {})
            mi = data.get("model_ids", {})

            # Pipeline tab
            self.camera_index.setValue(rc.get("camera_index", 0))
            self.camera_width.setValue(rc.get("camera_width", 1280))
            self.camera_height.setValue(rc.get("camera_height", 720))
            self.inference_fps.setValue(rc.get("inference_fps", 2.0))
            self.prompts.setText(rc.get("default_prompts", ""))
            self.enable_dino.setChecked(rc.get("enable_dino", True))
            sal_mode = rc.get("dino_saliency_mode", "inverse")
            idx = self.dino_saliency_mode.findText(sal_mode)
            if idx >= 0:
                self.dino_saliency_mode.setCurrentIndex(idx)
            self.dino_proposal_count.setValue(rc.get("dino_proposal_count", 5))
            self.dino_proposal_min_area.setValue(rc.get("dino_proposal_min_area", 0.005))
            self.enable_sam.setChecked(rc.get("enable_sam", True))
            self.enable_vjepa.setChecked(rc.get("enable_vjepa", True))
            self.enable_caption.setChecked(rc.get("enable_caption", True))
            self.pause_on_freeze.setChecked(rc.get("pause_on_freeze", True))
            self.use_boxes.setChecked(rc.get("use_dino_boxes_for_sam", False))

            # Visuals tab
            self.max_w.setValue(rc.get("display_max_width", 640))
            self.dino_strength.setValue(rc.get("dino_overlay_strength", 0.45))
            self.dino_blur.setValue(rc.get("dino_blur_ksize", 3))
            self.sam_thr.setValue(rc.get("sam_threshold", 0.5))
            self.sam_mask_thr.setValue(rc.get("sam_mask_threshold", 0.5))

            # Detection tab (YOLO)
            det_src_idx = self.detection_source.findText(rc.get("detection_source", "sam"))
            if det_src_idx >= 0:
                self.detection_source.setCurrentIndex(det_src_idx)
            yolo_model_idx = self.yolo_model.findText(rc.get("yolo_model", "yolo11m"))
            if yolo_model_idx >= 0:
                self.yolo_model.setCurrentIndex(yolo_model_idx)
            self.yolo_conf.setValue(rc.get("yolo_conf_threshold", 0.25))

            # Performance tab
            self.run_dino_n.setValue(rc.get("run_dino_every_n", 1))
            self.run_sam_n.setValue(rc.get("run_sam_every_n", 1))
            self.run_vjepa_n.setValue(rc.get("run_vjepa_every_n", 2))
            self.caption_every_n.setValue(rc.get("caption_every_n", 5))
            self.caption_on_events.setChecked(rc.get("caption_on_events", True))
            self.caption_gap.setValue(rc.get("caption_min_tick_gap", 2))
            self.caption_tokens.setValue(rc.get("caption_max_new_tokens", 24))
            self.caption_beams.setValue(rc.get("caption_num_beams", 3))
            self.fp16_on_mps.setChecked(rc.get("fp16_on_mps", False))
            self.enable_tracking.setChecked(rc.get("enable_tracking", True))
            self.enable_reid.setChecked(rc.get("enable_reid", True))
            self.reid_every_n.setValue(rc.get("reid_every_n", 2))
            self.reid_max_dets.setValue(rc.get("reid_max_dets", 6))
            self.tracker_use_embedding.setChecked(rc.get("tracker_use_embedding", True))
            self.tracker_emb_weight.setValue(rc.get("tracker_emb_weight", 0.35))
            algo_idx = self.tracker_algorithm.findText(rc.get("tracker_algorithm", "kalman"))
            if algo_idx >= 0:
                self.tracker_algorithm.setCurrentIndex(algo_idx)
            self.tracker_use_hungarian.setChecked(rc.get("tracker_use_hungarian", True))
            self.tracker_graveyard.setValue(rc.get("tracker_graveyard_seconds", 10.0))

            # Models tab
            self.dino_id.setText(mi.get("dino", ""))
            self.sam_id.setText(mi.get("sam3", ""))
            self.vjepa_id.setText(mi.get("vjepa2_cls", ""))
            self.caption_id.setText(mi.get("caption", ""))

            QtWidgets.QMessageBox.information(
                self, "Settings Loaded", f"Settings loaded from:\n{path}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Load Failed", f"Could not load settings:\n{str(e)}"
            )

    def collect(self) -> Tuple[RuntimeConfig, ModelIds]:
        cfg = dataclasses.replace(self._cfg)
        ids = dataclasses.replace(self._ids)

        cfg.camera_index = int(self.camera_index.value())
        cfg.camera_width = int(self.camera_width.value())
        cfg.camera_height = int(self.camera_height.value())
        cfg.inference_fps = float(self.inference_fps.value())

        cfg.default_prompts = self.prompts.text().strip() or cfg.default_prompts
        cfg.use_dino_boxes_for_sam = bool(self.use_boxes.isChecked())

        cfg.enable_dino = bool(self.enable_dino.isChecked())
        cfg.dino_saliency_mode = str(self.dino_saliency_mode.currentText())
        cfg.dino_proposal_count = int(self.dino_proposal_count.value())
        cfg.dino_proposal_min_area = float(self.dino_proposal_min_area.value())
        cfg.enable_sam = bool(self.enable_sam.isChecked())
        cfg.enable_vjepa = bool(self.enable_vjepa.isChecked())

        cfg.enable_caption = bool(self.enable_caption.isChecked())

        cfg.pause_on_freeze = bool(self.pause_on_freeze.isChecked())

        cfg.display_max_width = int(self.max_w.value())
        cfg.max_display_width = int(cfg.display_max_width)
        cfg.dino_overlay_strength = float(self.dino_strength.value())
        cfg.dino_blur_ksize = int(self.dino_blur.value())
        if cfg.dino_blur_ksize and cfg.dino_blur_ksize % 2 == 0:
            cfg.dino_blur_ksize += 1  # enforce odd
        cfg.sam_threshold = float(self.sam_thr.value())
        cfg.sam_mask_threshold = float(self.sam_mask_thr.value())

        # Detection source (SAM vs YOLO)
        cfg.detection_source = str(self.detection_source.currentText())
        cfg.yolo_model = str(self.yolo_model.currentText())
        cfg.yolo_conf_threshold = float(self.yolo_conf.value())

        cfg.run_dino_every_n = int(self.run_dino_n.value())
        cfg.run_sam_every_n = int(self.run_sam_n.value())
        cfg.run_vjepa_every_n = int(self.run_vjepa_n.value())
        cfg.caption_every_n = int(self.caption_every_n.value())
        cfg.caption_on_events = bool(self.caption_on_events.isChecked())
        cfg.caption_min_tick_gap = int(self.caption_gap.value())
        cfg.caption_max_new_tokens = int(self.caption_tokens.value())
        cfg.caption_num_beams = int(self.caption_beams.value())
        cfg.fp16_on_mps = bool(self.fp16_on_mps.isChecked())

        cfg.enable_tracking = bool(self.enable_tracking.isChecked())
        cfg.enable_reid = bool(self.enable_reid.isChecked())
        cfg.reid_every_n = int(self.reid_every_n.value())
        cfg.reid_max_dets = int(self.reid_max_dets.value())
        cfg.tracker_use_embedding = bool(self.tracker_use_embedding.isChecked())
        cfg.tracker_emb_weight = float(self.tracker_emb_weight.value())
        cfg.tracker_algorithm = str(self.tracker_algorithm.currentText())
        cfg.tracker_use_hungarian = bool(self.tracker_use_hungarian.isChecked())
        cfg.tracker_graveyard_seconds = float(self.tracker_graveyard.value())

        tok = self.hf_token.text().strip()
        cfg.hf_token = tok if tok else None

        ids.dino = self.dino_id.text().strip() or ids.dino
        ids.sam3 = self.sam_id.text().strip() or ids.sam3
        ids.vjepa2_cls = self.vjepa_id.text().strip() or ids.vjepa2_cls

        ids.caption = self.caption_id.text().strip() or getattr(ids, "caption", "Salesforce/blip-image-captioning-base")

        return cfg, ids


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, cfg: RuntimeConfig, ids: ModelIds):
        super().__init__()
        self.setWindowTitle("Camera → DINOv3 → SAM3 → V-JEPA2 (Quad + Tabs, Timeline)")
        self.cfg = cfg
        self.ids = ids

        self.frozen_tick: Optional[int] = None
        self.snapshots: Dict[int, StageOutputs] = {}

        self.cam: Optional[CameraWorker] = None
        self.inf: Optional[InferenceWorker] = None

        # --- Top toolbar
        tb = QtWidgets.QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        self.act_start = QtGui.QAction("Start", self)
        self.act_stop = QtGui.QAction("Stop", self)
        self.act_resume = QtGui.QAction("Resume Live", self)
        self.act_settings = QtGui.QAction("Settings…", self)
        self.act_export = QtGui.QAction("Export Timeline…", self)
        self.act_help = QtGui.QAction("Help", self)
        self.act_help.setShortcut(QtGui.QKeySequence("F1"))

        self.act_stop.setEnabled(False)
        self.act_resume.setEnabled(False)

        self.act_start.triggered.connect(self.start_all)
        self.act_stop.triggered.connect(self.stop_all)
        self.act_resume.triggered.connect(self.resume_live)
        self.act_settings.triggered.connect(self.open_settings)
        self.act_export.triggered.connect(self._export_timeline)
        self.act_help.triggered.connect(self._show_help)

        tb.addAction(self.act_start)
        tb.addAction(self.act_stop)
        tb.addSeparator()
        tb.addAction(self.act_resume)
        tb.addSeparator()
        tb.addAction(self.act_settings)
        tb.addAction(self.act_export)
        tb.addAction(self.act_help)

        tb.addSeparator()
        self.badge = QtWidgets.QLabel("LIVE")
        self.badge.setStyleSheet("padding:3px 8px; border-radius:10px; background:#1b5e20; color:#e8ffe8;")
        tb.addWidget(self.badge)

        # FPS and latency display
        tb.addSeparator()
        self.fps_label = QtWidgets.QLabel("-- FPS")
        self.fps_label.setStyleSheet("color:#8f8; font-family:monospace; padding:0 4px;")
        self.fps_label.setToolTip("Inference frames per second")
        tb.addWidget(self.fps_label)

        self.latency_label = QtWidgets.QLabel("D:-- S:-- V:-- C:--")
        self.latency_label.setStyleSheet("color:#aaa; font-family:monospace; font-size:11px; padding:0 4px;")
        self.latency_label.setToolTip("Latency (ms): DINO / SAM / V-JEPA / Caption")
        tb.addWidget(self.latency_label)

        self.gpu_mem_label = QtWidgets.QLabel("GPU: --")
        self.gpu_mem_label.setStyleSheet("color:#aaf; font-family:monospace; font-size:11px; padding:0 4px;")
        self.gpu_mem_label.setToolTip("GPU memory usage (allocated/reserved)")
        tb.addWidget(self.gpu_mem_label)

        self._fps_times: Deque[float] = deque(maxlen=20)  # Track recent output times for FPS calc

        # GPU memory monitor timer
        self._gpu_timer = QtCore.QTimer(self)
        self._gpu_timer.timeout.connect(self._update_gpu_memory)
        self._gpu_timer.start(2000)  # Update every 2 seconds

        tb.addSeparator()

        # Pipeline status indicator (live data flow visualization)
        self.pipeline_status = PipelineStatusWidget()
        self.pipeline_status.setToolTip("Pipeline data flow: CAM → DINO → SAM → V-JEPA → Tracking")
        tb.addWidget(self.pipeline_status)

        tb.addSeparator()
        self.scene_label = QtWidgets.QLabel("")
        self.scene_label.setStyleSheet("color:#ddd; padding-left:6px;")
        self.scene_label.setMinimumWidth(320)
        self.scene_label.setWordWrap(False)
        tb.addWidget(self.scene_label)

        # --- Stage panes (single set; we re-layout them when switching tabs)
        self.pane_raw = ImagePane("Raw (OpenCV)")
        self.pane_dino = ImagePane("DINOv3 (heatmap + proposals)")
        self.pane_sam = ImagePane("SAM3 (masks)")
        self.pane_clip = ImagePane("V-JEPA2 clip samples")
        self.pane_composite = ImagePane("Composite (all layers)")
        self.pane_tracking = ImagePane("Tracking (trails + velocity)")

        # Left "tabs" as a TabBar that controls visibility/layout (no widget duplication)
        self.stage_tabs = QtWidgets.QTabBar()
        self.stage_tabs.addTab("Quad")
        self.stage_tabs.addTab("Raw")
        self.stage_tabs.addTab("DINO")
        self.stage_tabs.addTab("SAM")
        self.stage_tabs.addTab("V-JEPA2")
        self.stage_tabs.addTab("Composite")
        self.stage_tabs.addTab("Tracking")
        self.stage_tabs.setExpanding(False)
        self.stage_tabs.currentChanged.connect(self._on_stage_tab_changed)

        self.stage_grid_host = QtWidgets.QWidget()
        self.stage_grid = QtWidgets.QGridLayout(self.stage_grid_host)
        self.stage_grid.setContentsMargins(0, 0, 0, 0)
        self.stage_grid.setSpacing(8)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        left_layout.addWidget(self.stage_tabs)
        left_layout.addWidget(self.stage_grid_host, 1)

        # Right tabs: Timeline / Details / Events
        self.right_tabs = QtWidgets.QTabWidget()

        # Timeline page
        tl_page = QtWidgets.QWidget()
        tl_layout = QtWidgets.QVBoxLayout(tl_page)
        tl_layout.setContentsMargins(8, 8, 8, 8)
        tl_layout.setSpacing(8)

        self.timeline = TimelineTable()
        self.timeline.itemSelectionChanged.connect(self.on_timeline_selection_changed)
        tl_layout.addWidget(self.timeline, 1)

        self.right_tabs.addTab(tl_page, "Timeline")

        # Details page
        details_page = QtWidgets.QWidget()
        details_layout = QtWidgets.QVBoxLayout(details_page)
        details_layout.setContentsMargins(8, 8, 8, 8)
        details_layout.setSpacing(8)

        self.details = QtWidgets.QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setStyleSheet("background:#0d0d0d; color:#ddd; border:1px solid #222;")
        self.details.setMinimumWidth(360)
        details_layout.addWidget(self.details, 1)

        self.right_tabs.addTab(details_page, "Details")

        # Narration / Captions page
        narr_page = QtWidgets.QWidget()
        narr_layout = QtWidgets.QVBoxLayout(narr_page)
        narr_layout.setContentsMargins(8, 8, 8, 8)
        narr_layout.setSpacing(8)

        self.narr_current = QtWidgets.QTextEdit()  # QTextEdit for HTML support
        self.narr_current.setReadOnly(True)
        self.narr_current.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.WidgetWidth)
        self.narr_current.setStyleSheet("background:#0d0d0d; color:#ddd; border:1px solid #222;")
        self.narr_current.setPlaceholderText("Selected tick narration will appear here...")

        # Color legend for narration sources
        legend = QtWidgets.QLabel(
            '<span style="color:#32CD32">Objects</span> · '
            '<span style="color:#1E90FF">Actions</span> · '
            '<span style="color:#9370DB">Relations</span> · '
            '<span style="color:#FFD700">Caption</span>'
        )
        legend.setStyleSheet("font-size:10px; color:#888; padding:2px;")

        self.narr_history = QtWidgets.QPlainTextEdit()
        self.narr_history.setReadOnly(True)
        self.narr_history.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.narr_history.setStyleSheet("background:#0d0d0d; color:#ddd; border:1px solid #222;")
        self.narr_history.document().setMaximumBlockCount(800)
        self.narr_history.setPlaceholderText("Live narration history...")

        narr_layout.addWidget(QtWidgets.QLabel("Selected Tick"), 0)
        narr_layout.addWidget(self.narr_current, 1)
        narr_layout.addWidget(legend, 0)
        narr_layout.addWidget(QtWidgets.QLabel("Live History"), 0)
        narr_layout.addWidget(self.narr_history, 2)

        self.right_tabs.addTab(narr_page, "Narration")

        # Events page
        events_page = QtWidgets.QWidget()
        events_layout = QtWidgets.QVBoxLayout(events_page)
        events_layout.setContentsMargins(8, 8, 8, 8)
        events_layout.setSpacing(8)

        self.events = QtWidgets.QPlainTextEdit()
        self.events.setReadOnly(True)
        self.events.setStyleSheet("background:#0d0d0d; color:#ddd; border:1px solid #222;")
        events_layout.addWidget(self.events, 1)

        btn_clear = QtWidgets.QPushButton("Clear")
        btn_clear.clicked.connect(lambda: self.events.setPlainText(""))
        events_layout.addWidget(btn_clear, 0, QtCore.Qt.AlignmentFlag.AlignRight)

        self.right_tabs.addTab(events_page, "Events")

        # Splitter
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter.addWidget(left)
        self.splitter.addWidget(self.right_tabs)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setChildrenCollapsible(False)

        cw = QtWidgets.QWidget()
        cwl = QtWidgets.QVBoxLayout(cw)
        cwl.setContentsMargins(10, 10, 10, 10)
        cwl.addWidget(self.splitter, 1)
        self.setCentralWidget(cw)

        # initial quad layout
        self._layout_quad()

        # size to available screen
        scr = QtWidgets.QApplication.primaryScreen()
        if scr is not None:
            g = scr.availableGeometry()
            w = int(g.width() * 0.92)
            h = int(g.height() * 0.90)
            self.resize(w, h)

        # Keyboard shortcuts
        QtGui.QShortcut(QtGui.QKeySequence("Space"), self, self._toggle_freeze)
        QtGui.QShortcut(QtGui.QKeySequence("Escape"), self, self.resume_live)
        QtGui.QShortcut(QtGui.QKeySequence("Q"), self, lambda: self.stage_tabs.setCurrentIndex(0))  # Quad
        QtGui.QShortcut(QtGui.QKeySequence("1"), self, lambda: self.stage_tabs.setCurrentIndex(1))  # Raw
        QtGui.QShortcut(QtGui.QKeySequence("2"), self, lambda: self.stage_tabs.setCurrentIndex(2))  # DINO
        QtGui.QShortcut(QtGui.QKeySequence("3"), self, lambda: self.stage_tabs.setCurrentIndex(3))  # SAM
        QtGui.QShortcut(QtGui.QKeySequence("4"), self, lambda: self.stage_tabs.setCurrentIndex(4))  # V-JEPA
        QtGui.QShortcut(QtGui.QKeySequence("5"), self, lambda: self.stage_tabs.setCurrentIndex(5))  # Composite
        QtGui.QShortcut(QtGui.QKeySequence("6"), self, lambda: self.stage_tabs.setCurrentIndex(6))  # Tracking
        QtGui.QShortcut(QtGui.QKeySequence("T"), self, lambda: self.stage_tabs.setCurrentIndex(6))  # Tracking (alt)
        QtGui.QShortcut(QtGui.QKeySequence("Up"), self, self._select_prev_tick)
        QtGui.QShortcut(QtGui.QKeySequence("Down"), self, self._select_next_tick)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+,"), self, self.open_settings)

        self.statusBar().showMessage("Ready (Space=freeze, Q/1-6/T=views, ↑↓=timeline)")

    def _append_event(self, msg: str):
        self.events.appendPlainText(msg)

    def _update_gpu_memory(self):
        """Update GPU memory display."""
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
                reserved = torch.cuda.memory_reserved() / (1024**2)
                self.gpu_mem_label.setText(f"GPU: {allocated:.0f}/{reserved:.0f}MB")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # MPS doesn't have detailed memory tracking in most PyTorch versions
                self.gpu_mem_label.setText("GPU: MPS")
            else:
                self.gpu_mem_label.setText("GPU: CPU")
        except Exception:
            self.gpu_mem_label.setText("GPU: --")

    def _show_help(self):
        """Show help dialog with keyboard shortcuts."""
        help_text = """
<h2>Vision Pipeline - Keyboard Shortcuts</h2>

<h3>Playback Control</h3>
<ul>
<li><b>Space</b> - Toggle freeze/live mode</li>
<li><b>Escape</b> - Resume live mode</li>
</ul>

<h3>View Switching</h3>
<ul>
<li><b>Q</b> - Quad view (all panes)</li>
<li><b>1</b> - Raw camera view</li>
<li><b>2</b> - DINO saliency view</li>
<li><b>3</b> - SAM segmentation view</li>
<li><b>4</b> - V-JEPA clip strip view</li>
</ul>

<h3>Timeline Navigation</h3>
<ul>
<li><b>Up Arrow</b> - Select previous tick</li>
<li><b>Down Arrow</b> - Select next tick</li>
</ul>

<h3>Application</h3>
<ul>
<li><b>Ctrl+,</b> - Open settings</li>
<li><b>F1</b> - Show this help</li>
</ul>

<h3>Pipeline Stages</h3>
<ul>
<li><b>DINOv3</b> - Dense features for saliency heatmap</li>
<li><b>SAM3</b> - Text/box-prompted segmentation</li>
<li><b>V-JEPA2</b> - Video clip action classification</li>
<li><b>BLIP</b> - Image captioning</li>
</ul>
"""

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Help - Vision Pipeline")
        dialog.setMinimumSize(450, 500)

        layout = QtWidgets.QVBoxLayout(dialog)

        text = QtWidgets.QTextBrowser()
        text.setHtml(help_text)
        text.setOpenExternalLinks(True)
        layout.addWidget(text)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.exec()

    def open_settings(self):
        dlg = SettingsDialog(self.cfg, self.ids, parent=self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        new_cfg, new_ids = dlg.collect()

        # determine if we need to restart models
        restart_needed = (
            (new_cfg.hf_token != self.cfg.hf_token)
            or (new_cfg.fp16_on_mps != self.cfg.fp16_on_mps)
            or (new_ids.dino != self.ids.dino)
            or (new_ids.sam3 != self.ids.sam3)
            or (new_ids.vjepa2_cls != self.ids.vjepa2_cls)
            or (getattr(new_ids, 'caption', None) != getattr(self.ids, 'caption', None))
            or (getattr(new_cfg, 'enable_caption', True) != getattr(self.cfg, 'enable_caption', True))
            or (new_cfg.camera_index != self.cfg.camera_index)
            or (new_cfg.camera_width != self.cfg.camera_width)
            or (new_cfg.camera_height != self.cfg.camera_height)
        )

        self.cfg = new_cfg
        self.ids = new_ids

        if self.inf is not None:
            # apply live-updatable controls
            self.inf.cfg = self.cfg
            w = int(getattr(self.cfg, "display_max_width", getattr(self.cfg, "max_display_width", 640)))
            self.inf.update_controls(
                prompts=split_prompts(self.cfg.default_prompts),
                use_boxes_for_sam=bool(getattr(self.cfg, 'use_dino_boxes_for_sam', getattr(self.cfg, 'use_dino_box_proposals_for_sam', False))),
                sam_threshold=self.cfg.sam_threshold,
                sam_mask_threshold=self.cfg.sam_mask_threshold,
                dino_strength=float(getattr(self.cfg, 'dino_overlay_strength', getattr(self.cfg, 'dino_heatmap_strength', 0.45))),
                dino_blur=int(getattr(self.cfg, 'dino_blur_ksize', getattr(self.cfg, 'dino_heatmap_blur', 3))),
                enable_dino=self.cfg.enable_dino,
                enable_sam=self.cfg.enable_sam,
                enable_vjepa=self.cfg.enable_vjepa,
                run_dino_every_n=self.cfg.run_dino_every_n,
                run_sam_every_n=self.cfg.run_sam_every_n,
                run_vjepa_every_n=self.cfg.run_vjepa_every_n,
                enable_caption=self.cfg.enable_caption,
                caption_every_n=self.cfg.caption_every_n,
                caption_on_events=self.cfg.caption_on_events,
                caption_min_tick_gap=self.cfg.caption_min_tick_gap,
                caption_max_new_tokens=self.cfg.caption_max_new_tokens,
                caption_num_beams=self.cfg.caption_num_beams,
                display_max_width=w,
            )

        if restart_needed and (self.cam is not None or self.inf is not None):
            self._append_event("[ui] Settings changed; restarting pipeline to apply.")
            self.stop_all()
            self.start_all()

    def _export_timeline(self):
        """Export timeline data to JSON or CSV."""
        if not self.snapshots:
            QtWidgets.QMessageBox.information(
                self, "No Data", "No timeline data to export."
            )
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Timeline",
            "timeline_export",
            "JSON Files (*.json);;CSV Files (*.csv);;All Files (*)"
        )

        if not path:
            return

        try:
            if path.endswith('.csv'):
                self._export_timeline_csv(path)
            else:
                if not path.endswith('.json'):
                    path += '.json'
                self._export_timeline_json(path)

            QtWidgets.QMessageBox.information(
                self, "Export Complete", f"Timeline exported to:\n{path}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Export Failed", f"Could not export timeline:\n{str(e)}"
            )

    def _export_timeline_json(self, path: str):
        """Export timeline to JSON format."""
        data = {
            "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tick_count": len(self.snapshots),
            "ticks": []
        }

        for tick_id in sorted(self.snapshots.keys()):
            outs = self.snapshots[tick_id]
            tick_data = {
                "tick_id": outs.tick_id,
                "timestamp": outs.ts_wall,
                "timestamp_str": time.strftime("%H:%M:%S", time.localtime(outs.ts_wall)),
                "dino_ms": (outs.dino_info or {}).get("total_ms", 0),
                "sam_ms": (outs.sam_info or {}).get("total_ms", 0),
                "vjepa_ms": (outs.vjepa_info or {}).get("total_ms", 0),
                "caption_ms": (outs.caption_info or {}).get("ms", 0),
                "sam_counts": outs.sam_counts,
                "vjepa_predictions": [
                    {"label": lbl, "probability": p}
                    for lbl, p in (outs.vjepa_pairs or [])
                ],
                "caption": getattr(outs, "caption", ""),
                "narration": getattr(outs, "narration", ""),
                "events": getattr(outs, "narration_events", []),
            }
            data["ticks"].append(tick_data)

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def _export_timeline_csv(self, path: str):
        """Export timeline to CSV format."""
        import csv

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "tick_id", "timestamp", "time_str",
                "dino_ms", "sam_ms", "vjepa_ms", "caption_ms",
                "sam_counts", "vjepa_top1", "vjepa_top1_prob",
                "caption", "narration", "events"
            ])

            # Data rows
            for tick_id in sorted(self.snapshots.keys()):
                outs = self.snapshots[tick_id]

                sam_counts_str = "; ".join(
                    f"{k}:{v}" for k, v in (outs.sam_counts or {}).items()
                )

                vjepa_top1 = ""
                vjepa_prob = ""
                if outs.vjepa_pairs:
                    vjepa_top1 = humanize_vjepa_label(outs.vjepa_pairs[0][0])
                    vjepa_prob = f"{outs.vjepa_pairs[0][1] * 100:.1f}%"

                events_str = "; ".join(getattr(outs, "narration_events", []))

                writer.writerow([
                    outs.tick_id,
                    outs.ts_wall,
                    time.strftime("%H:%M:%S", time.localtime(outs.ts_wall)),
                    (outs.dino_info or {}).get("total_ms", 0),
                    (outs.sam_info or {}).get("total_ms", 0),
                    (outs.vjepa_info or {}).get("total_ms", 0),
                    (outs.caption_info or {}).get("ms", 0),
                    sam_counts_str,
                    vjepa_top1,
                    vjepa_prob,
                    getattr(outs, "caption", ""),
                    getattr(outs, "narration", ""),
                    events_str,
                ])

    def start_all(self):
        if self.cam is not None or self.inf is not None:
            return

        self.snapshots.clear()
        self.timeline.setRowCount(0)
        self.frozen_tick = None
        self.act_resume.setEnabled(False)
        self.badge.setText("LIVE")
        self.badge.setStyleSheet("padding:3px 8px; border-radius:10px; background:#1b5e20; color:#e8ffe8;")

        self.cam = CameraWorker(self.cfg)
        self.inf = InferenceWorker(self.cfg, self.ids)

        self.cam.frame_signal.connect(self.on_frame)
        self.inf.outputs_signal.connect(self.on_outputs)
        self.inf.event_signal.connect(self._append_event)

        self.cam.start()
        self.inf.start()

        self.act_start.setEnabled(False)
        self.act_stop.setEnabled(True)
        self.statusBar().showMessage("Running")

    def stop_all(self):
        if self.cam:
            self.cam.stop()
            self.cam.wait(1000)
            self.cam = None
        if self.inf:
            self.inf.stop()
            self.inf.wait(2000)
            self.inf = None

        self.act_start.setEnabled(True)
        self.act_stop.setEnabled(False)
        self.statusBar().showMessage("Stopped")

    def closeEvent(self, e: QtGui.QCloseEvent):
        self.stop_all()
        super().closeEvent(e)

    def resume_live(self):
        self.frozen_tick = None
        self.act_resume.setEnabled(False)
        self.badge.setText("LIVE")
        self.badge.setStyleSheet("padding:3px 8px; border-radius:10px; background:#1b5e20; color:#e8ffe8;")
        self.statusBar().showMessage("Live")
        # If inference was paused on freeze, it will naturally resume next tick.

    def freeze_to_tick(self, tick: int):
        self.frozen_tick = tick
        self.act_resume.setEnabled(True)
        self.badge.setText("FROZEN")
        self.badge.setStyleSheet("padding:3px 8px; border-radius:10px; background:#6a1b9a; color:#fff;")
        self.statusBar().showMessage(f"Frozen to tick #{tick}")

        outs = self.snapshots.get(tick)
        if outs:
            self.render_snapshot(outs)

        # Optional: pause inference by not feeding frames
        # (we implement this by simply skipping set_latest_frame when frozen if pause_on_freeze enabled)

    def _toggle_freeze(self):
        """Toggle between frozen and live mode (Space key)."""
        if self.frozen_tick is not None:
            self.resume_live()
        else:
            # Freeze to the most recent tick
            if self.snapshots:
                latest_tick = max(self.snapshots.keys())
                self.freeze_to_tick(latest_tick)

    def _select_prev_tick(self):
        """Select previous tick in timeline (Up arrow)."""
        if not self.snapshots:
            return
        ticks = sorted(self.snapshots.keys())
        if self.frozen_tick is None:
            # Not frozen, freeze to latest
            self.freeze_to_tick(ticks[-1])
        else:
            # Move to previous tick
            idx = ticks.index(self.frozen_tick) if self.frozen_tick in ticks else len(ticks) - 1
            if idx > 0:
                self.freeze_to_tick(ticks[idx - 1])

    def _select_next_tick(self):
        """Select next tick in timeline (Down arrow)."""
        if not self.snapshots:
            return
        ticks = sorted(self.snapshots.keys())
        if self.frozen_tick is None:
            return  # Already live
        idx = ticks.index(self.frozen_tick) if self.frozen_tick in ticks else 0
        if idx < len(ticks) - 1:
            self.freeze_to_tick(ticks[idx + 1])
        else:
            # At the end, resume live
            self.resume_live()

    def _on_stage_tab_changed(self, idx: int):
        if idx == 0:
            self._layout_quad()
        elif idx == 1:
            self._layout_single(self.pane_raw)
        elif idx == 2:
            self._layout_single(self.pane_dino)
        elif idx == 3:
            self._layout_single(self.pane_sam)
        elif idx == 4:
            self._layout_single(self.pane_clip)
        elif idx == 5:
            self._layout_single(self.pane_composite)
        elif idx == 6:
            self._layout_single(self.pane_tracking)

    def _clear_stage_grid(self):
        while self.stage_grid.count():
            item = self.stage_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

    def _layout_quad(self):
        self._clear_stage_grid()
        self.stage_grid.addWidget(self.pane_raw, 0, 0)
        self.stage_grid.addWidget(self.pane_dino, 0, 1)
        self.stage_grid.addWidget(self.pane_sam, 1, 0)
        self.stage_grid.addWidget(self.pane_clip, 1, 1)

        self.stage_grid.setRowStretch(0, 1)
        self.stage_grid.setRowStretch(1, 1)
        self.stage_grid.setColumnStretch(0, 1)
        self.stage_grid.setColumnStretch(1, 1)

    def _layout_single(self, pane: QtWidgets.QWidget):
        self._clear_stage_grid()
        self.stage_grid.addWidget(pane, 0, 0, 2, 2)
        self.stage_grid.setRowStretch(0, 1)
        self.stage_grid.setRowStretch(1, 1)
        self.stage_grid.setColumnStretch(0, 1)
        self.stage_grid.setColumnStretch(1, 1)

    def on_frame(self, bgr: np.ndarray):
        # If frozen, keep panes pinned to the selected snapshot.
        if self.frozen_tick is not None:
            if getattr(self.cfg, 'pause_on_freeze', True):
                return  # pause inference + UI updates
            # else allow inference to continue, but keep UI frozen
            if self.inf:
                self.inf.set_latest_frame(bgr)
            return

        # Live mode: show the raw frame immediately and feed inference
        # (Be tolerant of older configs that used max_display_width.)
        w = int(getattr(self.cfg, "display_max_width", getattr(self.cfg, "max_display_width", 640)))
        disp = fit_to_width(bgr, w)
        self.pane_raw.set_image_bgr(disp)

        if self.inf:
            # Keep worker controls in sync (cheap, avoids missing updates)
            self.inf.cfg = self.cfg
            self.inf.set_latest_frame(bgr)
            self.inf.update_controls(
                prompts=split_prompts(self.cfg.default_prompts),
                use_boxes_for_sam=bool(getattr(self.cfg, 'use_dino_boxes_for_sam', getattr(self.cfg, 'use_dino_box_proposals_for_sam', False))),
                sam_threshold=self.cfg.sam_threshold,
                sam_mask_threshold=self.cfg.sam_mask_threshold,
                dino_strength=float(getattr(self.cfg, 'dino_overlay_strength', getattr(self.cfg, 'dino_heatmap_strength', 0.45))),
                dino_blur=int(getattr(self.cfg, 'dino_blur_ksize', getattr(self.cfg, 'dino_heatmap_blur', 3))),
                enable_dino=self.cfg.enable_dino,
                enable_sam=self.cfg.enable_sam,
                enable_vjepa=self.cfg.enable_vjepa,
                run_dino_every_n=self.cfg.run_dino_every_n,
                run_sam_every_n=self.cfg.run_sam_every_n,
                run_vjepa_every_n=self.cfg.run_vjepa_every_n,
                display_max_width=w,
            )

    def on_outputs(self, outs: StageOutputs):
        # Update FPS display
        now = time.time()
        self._fps_times.append(now)
        if len(self._fps_times) >= 2:
            elapsed = self._fps_times[-1] - self._fps_times[0]
            if elapsed > 0:
                fps = (len(self._fps_times) - 1) / elapsed
                self.fps_label.setText(f"{fps:.1f} FPS")

        # Update latency display
        d_ms = (outs.dino_info or {}).get("total_ms", 0)
        s_ms = (outs.sam_info or {}).get("total_ms", 0)
        v_ms = (outs.vjepa_info or {}).get("total_ms", 0)
        c_ms = (outs.caption_info or {}).get("ms", 0) if outs.caption_info else 0
        self.latency_label.setText(f"D:{d_ms:.0f} S:{s_ms:.0f} V:{v_ms:.0f} C:{c_ms:.0f}")

        # Update pipeline status indicator
        fps = 0.0
        if len(self._fps_times) >= 2:
            elapsed = self._fps_times[-1] - self._fps_times[0]
            if elapsed > 0:
                fps = (len(self._fps_times) - 1) / elapsed
        dino_count = len(outs.dino_boxes) if outs.dino_boxes else 0
        sam_count = len(outs.sam_masks) if outs.sam_masks else 0
        vjepa_count = len(outs.vjepa_pairs) if outs.vjepa_pairs else 0
        track_count = len(outs.tracks) if outs.tracks else 0
        self.pipeline_status.update_stats(
            fps=fps,
            dino_count=dino_count,
            sam_count=sam_count,
            vjepa_count=vjepa_count,
            track_count=track_count,
        )

        # Store
        self._store_snapshot(outs)
        # Add to timeline
        self._add_timeline_row(outs)
        if self.frozen_tick is None and hasattr(self, "narr_history"):
            t_local = time.strftime("%H:%M:%S", time.localtime(outs.ts_wall))
            line = f"[{t_local}] #{outs.tick_id}  {self._scene_summary(outs)}"
            self.narr_history.appendPlainText(line)
            if getattr(outs, "narration_events", None):
                for e in outs.narration_events[-2:]:
                    self.narr_history.appendPlainText(f"    EVENT: {e}")

        # If live, render latest
        if self.frozen_tick is None:
            self.render_snapshot(outs)

    def _store_snapshot(self, outs: StageOutputs):
        # Hard cap at 150 for memory safety (each snapshot ~1-3MB with 4 BGR images)
        max_snapshots = min(self.cfg.timeline_max_items, 150)
        self.snapshots[outs.tick_id] = outs

        # Enforce max and help GC by nulling large arrays
        while len(self.snapshots) > max_snapshots:
            oldest = min(self.snapshots.keys())
            removed = self.snapshots.pop(oldest, None)
            if removed is not None:
                # Clear large BGR arrays to help garbage collection
                removed.raw_bgr = None
                removed.dino_bgr = None
                removed.sam_bgr = None
                removed.clip_strip_bgr = None
                removed.composite_bgr = None
                removed.tracking_bgr = None
                removed.dino_heat = None
                removed.sam_masks = []
                removed.tracks_raw = []

    def _add_timeline_row(self, outs: StageOutputs):
        # Keep table capped
        if self.timeline.rowCount() >= self.cfg.timeline_max_items:
            self.timeline.removeRow(0)

        r = self.timeline.rowCount()
        self.timeline.insertRow(r)

        def it(txt: str) -> QtWidgets.QTableWidgetItem:
            item = QtWidgets.QTableWidgetItem(txt)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, outs.tick_id)
            return item

        t_local = time.strftime("%H:%M:%S", time.localtime(outs.ts_wall))
        d_ms = (outs.dino_info or {}).get("total_ms", 0.0)
        s_ms = (outs.sam_info or {}).get("total_ms", 0.0)
        v_ms = (outs.vjepa_info or {}).get("total_ms", 0.0)
        c_ms = (outs.caption_info or {}).get("ms", 0.0) if outs.caption_info else 0.0

        counts = "—"
        if outs.sam_counts:
            nz = [(k, v) for k, v in outs.sam_counts.items() if v > 0]
            nz.sort(key=lambda kv: kv[1], reverse=True)
            counts = ", ".join([f"{k}:{v}" for k, v in nz[:4]]) if nz else "0"

        top = "—"
        if outs.vjepa_pairs:
            lbl, p = outs.vjepa_pairs[0]
            top = f"{humanize_vjepa_label(lbl)} {p * 100:.0f}%"

        # Events column with visual indicator
        event_count = len(outs.narration_events) if getattr(outs, "narration_events", None) else 0
        if event_count > 0:
            events_txt = f"●×{event_count}"
        else:
            events_txt = ""

        self.timeline.setItem(r, 0, it(str(outs.tick_id)))
        self.timeline.setItem(r, 1, it(t_local))
        self.timeline.setItem(r, 2, it(f"{d_ms:.0f}"))
        self.timeline.setItem(r, 3, it(f"{s_ms:.0f}"))
        self.timeline.setItem(r, 4, it(f"{v_ms:.0f}"))

        # Events column (index 5) with orange color if events present
        event_item = it(events_txt)
        if event_count > 0:
            event_item.setForeground(QtGui.QColor("#ff9800"))  # Orange for events
        self.timeline.setItem(r, 5, event_item)

        self.timeline.setItem(r, 6, it(counts))
        self.timeline.setItem(r, 7, it(top))

        # Tooltip: narration + caption + events + mask counts + vjepa top-k
        tip_lines = []
        if getattr(outs, "narration", ""):
            tip_lines.append(str(outs.narration))

        if getattr(outs, "caption", ""):
            if c_ms:
                tip_lines.append(f"CAPTION ({c_ms:.1f} ms): {outs.caption}")
            else:
                tip_lines.append(f"CAPTION: {outs.caption}")

        if getattr(outs, "narration_events", None):
            for e in outs.narration_events[-3:]:
                tip_lines.append("EVENT: " + str(e))

        if outs.sam_counts:
            for k, v in outs.sam_counts.items():
                tip_lines.append(f"SAM {k}: {v}")

        if outs.vjepa_pairs:
            tip_lines.append("V-JEPA2 top-k:")
            for lbl, p in outs.vjepa_pairs[:5]:
                tip_lines.append(f"  {humanize_vjepa_label(lbl)}: {p * 100:.1f}%")

        tip = "\n".join(tip_lines).strip()
        if tip:
            for c in range(self.timeline.columnCount()):
                self.timeline.item(r, c).setToolTip(tip)

        self.timeline.scrollToBottom()

    def on_timeline_selection_changed(self):
        items = self.timeline.selectedItems()
        if not items:
            return
        tick = items[0].data(QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(tick, int):
            self.freeze_to_tick(tick)

    def _scene_summary(self, outs: StageOutputs) -> str:
        # Prefer stateful narration if available.
        if getattr(outs, 'narration', ''):
            evs = getattr(outs, 'narration_events', None)
            if evs:
                return f"{outs.narration} | Event: {evs[-1]}"
            return str(outs.narration)

        parts: List[str] = []
        # SAM: prefer union-area fractions if available (more meaningful than raw counts).
        sam_areas = outs.sam_info.get("areas") if isinstance(outs.sam_info, dict) else None
        if isinstance(sam_areas, dict) and sam_areas:
            nz_a = [(str(k), float(v)) for k, v in sam_areas.items() if float(v) > 0.005]
            nz_a.sort(key=lambda kv: kv[1], reverse=True)
            if nz_a:
                parts.append("Objects: " + ", ".join([f"{k} {v*100:.0f}%" for k, v in nz_a[:4]]))
        elif outs.sam_counts:
            nz = [(k, v) for k, v in outs.sam_counts.items() if v > 0]
            nz.sort(key=lambda kv: kv[1], reverse=True)
            if nz:
                parts.append("Segments: " + ", ".join([f"{k}×{v}" for k, v in nz[:4]]))
        if outs.vjepa_pairs:
            topk = outs.vjepa_pairs[:3]
            parts.append("Activity: " + ", ".join([f"{humanize_vjepa_label(lbl)} {p*100:.0f}%" for lbl, p in topk]))
        if getattr(outs, "caption", ""):
            parts.append("CAP: " + shorten_text(outs.caption, 60))
        if not parts:
            return "—"
        return " | ".join(parts)


    def render_snapshot(self, outs: StageOutputs):
        if outs.raw_bgr is not None:
            self.pane_raw.set_image_bgr(outs.raw_bgr)
        if outs.dino_bgr is not None:
            self.pane_dino.set_image_bgr(outs.dino_bgr)
        if outs.sam_bgr is not None:
            self.pane_sam.set_image_bgr(outs.sam_bgr)
        if outs.clip_strip_bgr is not None:
            self.pane_clip.set_image_bgr(outs.clip_strip_bgr)
        if outs.composite_bgr is not None:
            self.pane_composite.set_image_bgr(outs.composite_bgr)
        if outs.tracking_bgr is not None:
            self.pane_tracking.set_image_bgr(outs.tracking_bgr)

        summary = self._scene_summary(outs)
        self.scene_label.setText(summary)
        self.details.setPlainText(self._format_details(outs, summary))

        # Narration tab (selected tick) - with color-coded HTML
        html_parts = []
        if getattr(outs, "narration_attributed", None):
            # Use color-coded HTML from attributed narration
            html_parts.append("<b>NARRATION:</b><br>" + outs.narration_attributed.to_html())
        elif getattr(outs, "narration", ""):
            html_parts.append("<b>NARRATION:</b><br>" + outs.narration)

        if getattr(outs, "caption", ""):
            cap_escaped = outs.caption.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html_parts.append(f'<br><br><b>CAPTION:</b><br><span style="color:#FFD700">{cap_escaped}</span>')

        if getattr(outs, "narration_events", None):
            if outs.narration_events:
                html_parts.append("<br><br><b>EVENTS:</b>")
                for e in outs.narration_events[-8:]:
                    e_escaped = str(e).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    html_parts.append(f'<br>  • <span style="color:#aaa">{e_escaped}</span>')

        html = "".join(html_parts) or "—"
        if hasattr(self, "narr_current"):
            self.narr_current.setHtml(html)

    def _format_details(self, outs: StageOutputs, summary: str) -> str:
        tstr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(outs.ts_wall))
        lines = [f"Scene: {summary}", "", f"Tick #{outs.tick_id}", f"Time: {tstr}", ""]

        # Narration events
        if getattr(outs, 'narration_events', None):
            lines.append('Events')
            for e in outs.narration_events[-6:]:
                lines.append(f'  - {e}')
            lines.append('')

        # Tracks (stateful)
        if getattr(outs, 'tracks', None):
            lines.append('Tracks')
            for t in outs.tracks[:12]:
                tid = t.get('id')
                lbl = t.get('label')
                conf = float(t.get('conf', 0.0))
                area = float(t.get('area', 0.0))
                vx = float(t.get('vx', 0.0))
                vy = float(t.get('vy', 0.0))
                lines.append(f"  #{tid} {lbl} {conf*100:.0f}%  area {area*100:.1f}%  v=({vx:.1f},{vy:.1f})")
            lines.append('')

        if outs.dino_info:
            lines.append("DINOv3")
            lines.append(f"  total: {outs.dino_info.get('total_ms', 0):.1f} ms (pre {outs.dino_info.get('pre_ms', 0):.1f} / fwd {outs.dino_info.get('fwd_ms', 0):.1f})")
            lines.append(f"  grid: {outs.dino_info.get('grid')}  tokens: {outs.dino_info.get('tokens')}  hidden: {outs.dino_info.get('hidden')}  dtype: {outs.dino_info.get('dtype')}")
            lines.append(
                f"  mode: {outs.dino_info.get('mode')}  heat std: {outs.dino_info.get('heat_std', 0):.3f}  p50/p90/p99: {outs.dino_info.get('heat_p50', 0):.3f}/{outs.dino_info.get('heat_p90', 0):.3f}/{outs.dino_info.get('heat_p99', 0):.3f}"
            )
            if outs.dino_boxes:
                lines.append(f"  proposals: {len(outs.dino_boxes)}")
            lines.append("")

        if getattr(outs, "caption", ""):
            lines.append("Caption")
            ci = outs.caption_info if isinstance(outs.caption_info, dict) else {}
            if ci:
                lines.append(f"  total: {ci.get('ms', 0):.1f} ms  dtype: {ci.get('dtype')}  fresh: {ci.get('fresh')}")
            lines.append(f"  {outs.caption}")
            lines.append("")

        if outs.sam_info:
            lines.append("SAM3")
            lines.append(f"  total: {outs.sam_info.get('total_ms', 0):.1f} ms (vision {outs.sam_info.get('vision_ms', 0):.1f} ms)  dtype: {outs.sam_info.get('dtype')}")
            if outs.sam_counts:
                for k, v in outs.sam_counts.items():
                    lines.append(f"  {k}: {v}")
            else:
                lines.append("  (no counts)")
            lines.append("")

        if outs.vjepa_info:
            lines.append("V-JEPA2")
            lines.append(f"  total: {outs.vjepa_info.get('total_ms', 0):.1f} ms (pre {outs.vjepa_info.get('pre_ms', 0):.1f} / fwd {outs.vjepa_info.get('fwd_ms', 0):.1f})  dtype: {outs.vjepa_info.get('dtype')}")
            lines.append(f"  T={outs.vjepa_info.get('T')}  H={outs.vjepa_info.get('H')}  W={outs.vjepa_info.get('W')}")
            if outs.vjepa_pairs:
                lines.append("  top-k:")
                for lbl, p in outs.vjepa_pairs:
                    lines.append(f"    {humanize_vjepa_label(lbl)}: {p*100:.1f}%")
            else:
                lines.append("  (no predictions)")
            lines.append("")
        return "\n".join(lines)


# ----------------------------
# Main
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--infer-fps", type=float, default=2.0)

    p.add_argument("--dino", type=str, default=ModelIds().dino)
    p.add_argument("--sam3", type=str, default=ModelIds().sam3)
    p.add_argument("--vjepa2", type=str, default=ModelIds().vjepa2_cls)

    p.add_argument("--hf-token", type=str, default=None)

    p.add_argument("--no-dino", action="store_true")
    p.add_argument("--no-sam", action="store_true")
    p.add_argument("--no-vjepa", action="store_true")

    p.add_argument("--auto-boxes", action="store_true", help="Use DINO->boxes->SAM (ignore text prompts)")

    # MPS knobs
    p.add_argument("--fp16", action="store_true", help="Enable FP16 autocast on MPS (and CUDA).")

    # cadence / size
    p.add_argument("--max-width", type=int, default=640)
    p.add_argument("--dino-every", type=int, default=1)
    p.add_argument("--sam-every", type=int, default=1)
    p.add_argument("--vj-every", type=int, default=2)

    return p.parse_args()


def main():
    args = parse_args()
    env_tok = get_hf_token_from_env()
    tok = args.hf_token or env_tok

    cfg = RuntimeConfig(
        camera_index=int(args.camera),
        inference_fps=float(args.infer_fps),
        enable_dino=not bool(args.no_dino),
        enable_sam=not bool(args.no_sam),
        enable_vjepa=not bool(args.no_vjepa),
        use_dino_box_proposals_for_sam=bool(args.auto_boxes),
        hf_token=tok,
        fp16_on_mps=bool(args.fp16),
        max_display_width=int(args.max_width),
        run_dino_every_n=max(1, int(args.dino_every)),
        run_sam_every_n=max(1, int(args.sam_every)),
        run_vjepa_every_n=max(1, int(args.vj_every)),
    )

    ids = ModelIds(dino=args.dino, sam3=args.sam3, vjepa2_cls=args.vjepa2)

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(cfg, ids)
    win.resize(1650, 980)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
