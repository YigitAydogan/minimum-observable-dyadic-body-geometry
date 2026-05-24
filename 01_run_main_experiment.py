# ======================================================================================
# CalMS21 Minimum Observable Body Geometry
# ======================================================================================

import os
import sys
import time
import json
import math
import random
import zipfile
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

warnings.filterwarnings("ignore")


# ======================================================================================
# Dependency setup
# ======================================================================================

def pip_install(package: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


try:
    import ijson
except ImportError:
    print("[setup] Installing ijson...")
    pip_install("ijson")
    import ijson

try:
    import requests
except ImportError:
    print("[setup] Installing requests...")
    pip_install("requests")
    import requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import scipy
    from scipy import stats
except ImportError:
    print("[setup] Installing scipy...")
    pip_install("scipy")
    import scipy
    from scipy import stats

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("[setup] Installing torch...")
    pip_install("torch")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader


# ======================================================================================
# Configuration
# ======================================================================================

CONFIG: Dict[str, Any] = {
    "project_dir": "/content/calms21_minimum_body_paper_grade",

    "previous_pilot_dirs": [
        "/content/calms21_minimum_body_multiseed_l4",
        "/content/calms21_minimum_body_l4",
        "/content/calms21_minimum_body_30min",
        "/content/calms21_minimum_body_true_30min",
        "/content/calms21_minimum_body_pilot",
    ],

    "zip_name": "task1_classic_classification.zip",
    "train_json_name": "calms21_task1_train.json",

    "download_urls": [
        "https://data.caltech.edu/records/s0vdx-0k302/files/task1_classic_classification.zip/content",
        "https://data.caltech.edu/records/s0vdx-0k302/files/task1_classic_classification.zip?download=1",
    ],

    # Windowing.
    # CalMS21 videos are 30 Hz; 45 frames is 1.5 s.
    "window_size": 45,
    "window_stride": 2,
    "max_windows_per_class_for_cache": 12000,

    # Multi-seed experiment.
    "seeds": [11, 22, 33, 44, 55],

    # Video-level split constraints.
    "validation_fraction": 0.25,
    "max_split_attempts": 700,
    "minimum_train_per_class": {
        0: 1000,
        1: 1000,
        2: 1000,
        3: 1000,
    },
    "minimum_val_per_class": {
        0: 300,
        1: 300,
        2: 300,
        3: 300,
    },

    # Class-balanced caps after valid video-level split.
    "max_train_per_class": 8000,
    "max_val_per_class": 2000,

    # Temporal convolutional model.
    "epochs": 12,
    "batch_size": 512,
    "hidden_channels": 96,
    "num_tcn_blocks": 4,
    "dropout": 0.20,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "gradient_clip_norm": 5.0,

    # Bootstrap intervals.
    "bootstrap_iterations": 1000,

    # Degradation robustness.
    "run_degradation_experiment": True,
    "degradation_regions": [
        "full_body_both_mice",
        "head_neck_both_mice",
        "no_tail_both_mice",
        "trunk_tail_both_mice",
        "hips_tail_both_mice",
        "resident_only_full_body",
        "intruder_only_full_body",
    ],
    "degradation_grid": [
        {"mode": "clean", "severity": 0.00},
        {"mode": "jitter", "severity": 0.02},
        {"mode": "jitter", "severity": 0.05},
        {"mode": "jitter", "severity": 0.10},
        {"mode": "dropout", "severity": 0.05},
        {"mode": "dropout", "severity": 0.15},
        {"mode": "dropout", "severity": 0.30},
        {"mode": "temporal_subsample", "severity": 2.00},
        {"mode": "temporal_subsample", "severity": 3.00},
        {"mode": "temporal_subsample", "severity": 5.00},
    ],

    # DataLoader.
    "num_workers": 2,

    # Cache.
    "force_rebuild_window_cache": False,

    # Plot settings.
    "figure_dpi": 300,

    # Runtime guard.
    "global_budget_minutes": 90.0,
}


LABEL_ID_TO_NAME: Dict[int, str] = {
    0: "attack",
    1: "investigation",
    2: "mount",
    3: "other",
}

KEYPOINT_ID_TO_NAME: Dict[int, str] = {
    0: "nose",
    1: "left_ear",
    2: "right_ear",
    3: "neck",
    4: "left_hip",
    5: "right_hip",
    6: "tail_base",
}


BODY_REGION_CONFIGS: Dict[str, Dict[str, List[int]]] = {
    "full_body_both_mice": {
        "mice": [0, 1],
        "keypoints": [0, 1, 2, 3, 4, 5, 6],
    },
    "head_neck_both_mice": {
        "mice": [0, 1],
        "keypoints": [0, 1, 2, 3],
    },
    "head_only_both_mice": {
        "mice": [0, 1],
        "keypoints": [0, 1, 2],
    },
    "no_tail_both_mice": {
        "mice": [0, 1],
        "keypoints": [0, 1, 2, 3, 4, 5],
    },
    "trunk_tail_both_mice": {
        "mice": [0, 1],
        "keypoints": [3, 4, 5, 6],
    },
    "hips_tail_both_mice": {
        "mice": [0, 1],
        "keypoints": [4, 5, 6],
    },
    "hips_only_both_mice": {
        "mice": [0, 1],
        "keypoints": [4, 5],
    },
    "neck_only_both_mice": {
        "mice": [0, 1],
        "keypoints": [3],
    },
    "tail_base_only_both_mice": {
        "mice": [0, 1],
        "keypoints": [6],
    },
    "resident_only_full_body": {
        "mice": [0],
        "keypoints": [0, 1, 2, 3, 4, 5, 6],
    },
    "intruder_only_full_body": {
        "mice": [1],
        "keypoints": [0, 1, 2, 3, 4, 5, 6],
    },
}


# ======================================================================================
# Paths and device
# ======================================================================================

PROJECT_DIR = Path(CONFIG["project_dir"])
DATA_DIR = PROJECT_DIR / "data"
CACHE_DIR = PROJECT_DIR / "cache"
RESULTS_DIR = PROJECT_DIR / "results"
FIG_DIR = PROJECT_DIR / "figures"
TABLE_DIR = PROJECT_DIR / "tables"
MODEL_DIR = PROJECT_DIR / "models"

for directory in [PROJECT_DIR, DATA_DIR, CACHE_DIR, RESULTS_DIR, FIG_DIR, TABLE_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

ZIP_PATH = DATA_DIR / CONFIG["zip_name"]
TRAIN_JSON_PATH = DATA_DIR / CONFIG["train_json_name"]

WINDOW_CACHE_PATH = CACHE_DIR / (
    f"calms21_windows_T{CONFIG['window_size']}_S{CONFIG['window_stride']}"
    f"_C{CONFIG['max_windows_per_class_for_cache']}.npz"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ======================================================================================
# General utilities
# ======================================================================================

def print_banner(title: str) -> None:
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def minutes_elapsed(start_time: float) -> float:
    return (time.time() - start_time) / 60.0


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    def convert(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        return value

    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert(obj), f, indent=2)


def require_l4_or_cuda() -> None:
    print(f"[device] Device: {DEVICE}")

    if DEVICE.type != "cuda":
        raise RuntimeError(
            "CUDA GPU was not detected. Switch Colab runtime to L4 GPU before running this paper-grade script."
        )

    gpu_name = torch.cuda.get_device_name(0)
    print(f"[device] GPU name: {gpu_name}")


# ======================================================================================
# Dataset download/extraction
# ======================================================================================

def copy_existing_file_if_available(filename: str, target_path: Path) -> bool:
    target_path = Path(target_path)

    if target_path.exists() and target_path.stat().st_size > 10 * 1024 * 1024:
        return True

    for previous_dir in CONFIG["previous_pilot_dirs"]:
        candidate = Path(previous_dir) / "data" / filename

        if candidate.exists() and candidate.stat().st_size > 10 * 1024 * 1024:
            print(f"[reuse] Copying {filename} from: {candidate}")
            shutil.copy2(candidate, target_path)
            return True

    return False


def copy_existing_cache_if_available() -> bool:
    if WINDOW_CACHE_PATH.exists() and WINDOW_CACHE_PATH.stat().st_size > 1024:
        return True

    for previous_dir in CONFIG["previous_pilot_dirs"]:
        candidate = Path(previous_dir) / "cache" / WINDOW_CACHE_PATH.name

        if candidate.exists() and candidate.stat().st_size > 1024:
            print(f"[reuse] Copying window cache from: {candidate}")
            shutil.copy2(candidate, WINDOW_CACHE_PATH)
            return True

    return False


def download_file_with_fallback(urls: List[str], out_path: Path, chunk_size: int = 1024 * 1024) -> Path:
    out_path = Path(out_path)

    if out_path.exists() and out_path.stat().st_size > 10 * 1024 * 1024:
        print(f"[download] Existing file found: {out_path}")
        print(f"[download] Size: {out_path.stat().st_size / (1024 ** 2):.1f} MB")
        return out_path

    last_error = None

    for url in urls:
        print("[download] Trying URL:")
        print(f"           {url}")

        tmp_path = out_path.with_suffix(out_path.suffix + ".part")

        try:
            with requests.get(url, stream=True, timeout=60, allow_redirects=True) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                downloaded = 0
                t0 = time.time()

                with open(tmp_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue

                        file.write(chunk)
                        downloaded += len(chunk)

                        if total > 0:
                            pct = 100.0 * downloaded / total
                            elapsed = max(time.time() - t0, 1e-6)
                            speed = downloaded / elapsed / (1024 ** 2)

                            print(
                                f"\r[download] {pct:6.2f}% | "
                                f"{downloaded / (1024 ** 2):.1f}/"
                                f"{total / (1024 ** 2):.1f} MB | "
                                f"{speed:.1f} MB/s",
                                end="",
                            )

            print()

            if tmp_path.stat().st_size < 10 * 1024 * 1024:
                raise RuntimeError(
                    f"Downloaded file is unexpectedly small: {tmp_path.stat().st_size} bytes"
                )

            tmp_path.rename(out_path)
            print(f"[download] Saved to: {out_path}")
            print(f"[download] Final size: {out_path.stat().st_size / (1024 ** 2):.1f} MB")
            return out_path

        except Exception as error:
            last_error = error
            print(f"[download] Failed with this URL: {error}")

            if tmp_path.exists():
                tmp_path.unlink()

    raise RuntimeError(f"All download URLs failed. Last error: {last_error}")


def extract_train_json(zip_path: Path, train_json_path: Path, target_json_name: str) -> Path:
    zip_path = Path(zip_path)
    train_json_path = Path(train_json_path)

    if train_json_path.exists() and train_json_path.stat().st_size > 10 * 1024 * 1024:
        print(f"[extract] Existing train JSON found: {train_json_path}")
        print(f"[extract] Size: {train_json_path.stat().st_size / (1024 ** 2):.1f} MB")
        return train_json_path

    print(f"[extract] Opening zip: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

        candidate_members = [
            name for name in names
            if Path(name).name == target_json_name
        ]

        if len(candidate_members) == 0:
            print("[extract] Zip members:")
            for name in names:
                print(f"          {name}")

            raise FileNotFoundError(f"Could not find {target_json_name} in zip.")

        member = candidate_members[0]
        print(f"[extract] Extracting only: {member}")

        t0 = time.time()

        with zf.open(member, "r") as src, open(train_json_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

        print(f"[extract] Saved to: {train_json_path}")
        print(f"[extract] Size: {train_json_path.stat().st_size / (1024 ** 2):.1f} MB")
        print(f"[extract] Time: {format_seconds(time.time() - t0)}")

    return train_json_path


def stream_calms21_task1_videos(json_path: Path):
    with open(json_path, "rb") as file:
        for video_id, video_obj in ijson.kvitems(file, "annotator-id_0"):
            yield video_id, video_obj


# ======================================================================================
# Window cache
# ======================================================================================

def sample_or_load_windows() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    copy_existing_cache_if_available()

    if WINDOW_CACHE_PATH.exists() and not CONFIG["force_rebuild_window_cache"]:
        print(f"[cache] Loading existing window cache: {WINDOW_CACHE_PATH}")

        data = np.load(WINDOW_CACHE_PATH, allow_pickle=True)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        groups = data["groups"].astype(str)

        print(f"[cache] X shape: {X.shape}")
        print(f"[cache] y shape: {y.shape}")
        print(f"[cache] groups shape: {groups.shape}")

        return X, y, groups

    print_banner("Building window cache")

    rng = np.random.default_rng(42)

    X_windows = []
    y_labels = []
    groups = []

    per_class_counts = {class_id: 0 for class_id in LABEL_ID_TO_NAME.keys()}
    video_rows = []

    t0 = time.time()
    n_videos = 0

    for video_id, video_obj in stream_calms21_task1_videos(TRAIN_JSON_PATH):
        n_videos += 1

        keypoints = np.asarray(video_obj["keypoints"], dtype=np.float32)
        annotations = np.asarray(video_obj["annotations"], dtype=np.int64)

        if keypoints.ndim != 4:
            print(f"[stream] Skipping {video_id}: unexpected keypoint shape {keypoints.shape}")
            continue

        n_frames = keypoints.shape[0]

        if n_frames < CONFIG["window_size"] or annotations.shape[0] != n_frames:
            print(f"[stream] Skipping {video_id}: invalid frame/annotation count")
            continue

        sampled_this_video = 0
        offset = int(rng.integers(0, max(CONFIG["window_stride"], 1)))

        for start in range(offset, n_frames - CONFIG["window_size"] + 1, CONFIG["window_stride"]):
            end = start + CONFIG["window_size"]
            center_idx = start + CONFIG["window_size"] // 2
            label = int(annotations[center_idx])

            if label not in per_class_counts:
                continue

            if per_class_counts[label] >= CONFIG["max_windows_per_class_for_cache"]:
                continue

            window = keypoints[start:end]

            if not np.isfinite(window).all():
                continue

            X_windows.append(window)
            y_labels.append(label)
            groups.append(video_id)

            per_class_counts[label] += 1
            sampled_this_video += 1

        video_rows.append({
            "video_id": video_id,
            "n_frames": n_frames,
            "sampled_windows": sampled_this_video,
            **{
                f"frames_{LABEL_ID_TO_NAME[class_id]}": int(np.sum(annotations == class_id))
                for class_id in LABEL_ID_TO_NAME.keys()
            },
        })

        if n_videos % 5 == 0:
            class_text = ", ".join(
                f"{LABEL_ID_TO_NAME[class_id]}={count}"
                for class_id, count in per_class_counts.items()
            )

            print(
                f"[stream] videos={n_videos:03d} | "
                f"windows={len(X_windows):06d} | "
                f"{class_text} | "
                f"elapsed={format_seconds(time.time() - t0)}"
            )

        if all(count >= CONFIG["max_windows_per_class_for_cache"] for count in per_class_counts.values()):
            print("[stream] Reached maximum windows for all classes.")
            break

    if len(X_windows) == 0:
        raise RuntimeError("No windows sampled. Check dataset paths and JSON format.")

    X = np.stack(X_windows, axis=0).astype(np.float32)
    y = np.asarray(y_labels, dtype=np.int64)
    groups = np.asarray(groups).astype(str)

    print("[stream] Finished.")
    print(f"[stream] Videos read: {n_videos}")
    print(f"[stream] X shape: {X.shape}")
    print(f"[stream] y shape: {y.shape}")

    print("[stream] Class counts:")
    for class_id, class_name in LABEL_ID_TO_NAME.items():
        print(f"          {class_name:14s}: {int(np.sum(y == class_id))}")

    video_summary_df = pd.DataFrame(video_rows)
    video_summary_path = RESULTS_DIR / "video_sampling_summary.csv"
    video_summary_df.to_csv(video_summary_path, index=False)
    print(f"[save] Video summary: {video_summary_path}")

    print(f"[cache] Saving window cache: {WINDOW_CACHE_PATH}")
    np.savez_compressed(WINDOW_CACHE_PATH, X=X, y=y, groups=groups)
    print(f"[cache] Saved cache size: {WINDOW_CACHE_PATH.stat().st_size / (1024 ** 2):.1f} MB")

    return X, y, groups


# ======================================================================================
# Split construction
# ======================================================================================

def get_class_counts(y: np.ndarray, indices: np.ndarray) -> Dict[int, int]:
    return {
        class_id: int(np.sum(y[indices] == class_id))
        for class_id in sorted(LABEL_ID_TO_NAME.keys())
    }


def split_satisfies_constraints(y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray) -> bool:
    train_counts = get_class_counts(y, train_idx)
    val_counts = get_class_counts(y, val_idx)

    for class_id, minimum_count in CONFIG["minimum_train_per_class"].items():
        if train_counts[int(class_id)] < minimum_count:
            return False

    for class_id, minimum_count in CONFIG["minimum_val_per_class"].items():
        if val_counts[int(class_id)] < minimum_count:
            return False

    return True


def make_valid_video_level_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    for attempt in range(CONFIG["max_split_attempts"]):
        split_seed = seed + attempt * 997

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=CONFIG["validation_fraction"],
            random_state=split_seed,
        )

        train_idx_raw, val_idx_raw = next(splitter.split(X, y, groups=groups))

        if split_satisfies_constraints(y, train_idx_raw, val_idx_raw):
            print(f"[split] Found valid video-level split for seed {seed} at attempt {attempt + 1}.")
            print(f"[split] Raw train counts: {get_class_counts(y, train_idx_raw)}")
            print(f"[split] Raw val counts  : {get_class_counts(y, val_idx_raw)}")
            return train_idx_raw, val_idx_raw, split_seed

    raise RuntimeError(
        f"Could not find a valid split for seed {seed} after "
        f"{CONFIG['max_split_attempts']} attempts. "
        f"Lower minimum_val_per_class or adjust validation_fraction."
    )


def class_balanced_cap(
    indices: np.ndarray,
    y: np.ndarray,
    max_per_class: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []

    for class_id in sorted(LABEL_ID_TO_NAME.keys()):
        class_indices = indices[y[indices] == class_id]

        if len(class_indices) > max_per_class:
            class_indices = rng.choice(
                class_indices,
                size=max_per_class,
                replace=False,
            )

        selected.extend(class_indices.tolist())

    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)
    return selected


def make_split_for_seed(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    train_idx_raw, val_idx_raw, split_seed = make_valid_video_level_split(X, y, groups, seed)

    train_idx = class_balanced_cap(
        indices=train_idx_raw,
        y=y,
        max_per_class=CONFIG["max_train_per_class"],
        seed=seed + 100,
    )

    val_idx = class_balanced_cap(
        indices=val_idx_raw,
        y=y,
        max_per_class=CONFIG["max_val_per_class"],
        seed=seed + 200,
    )

    split_rows = []

    for idx in train_idx:
        split_rows.append({
            "seed": seed,
            "index": int(idx),
            "group": groups[idx],
            "label_id": int(y[idx]),
            "label_name": LABEL_ID_TO_NAME[int(y[idx])],
            "split": "train",
            "split_seed": split_seed,
        })

    for idx in val_idx:
        split_rows.append({
            "seed": seed,
            "index": int(idx),
            "group": groups[idx],
            "label_id": int(y[idx]),
            "label_name": LABEL_ID_TO_NAME[int(y[idx])],
            "split": "val",
            "split_seed": split_seed,
        })

    split_df = pd.DataFrame(split_rows)

    split_path = RESULTS_DIR / f"split_seed_{seed}.csv"
    split_df.to_csv(split_path, index=False)

    print(f"[split] Used train counts: {get_class_counts(y, train_idx)}")
    print(f"[split] Used val counts  : {get_class_counts(y, val_idx)}")
    print(f"[save] Split table: {split_path}")

    return train_idx, val_idx, split_df


# ======================================================================================
# Feature construction and degradation
# ======================================================================================

def impute_nan_keypoints(X: np.ndarray) -> np.ndarray:
    X = X.copy().astype(np.float32)

    if np.isfinite(X).all():
        return X

    # X shape: [N, T, mouse, coord, keypoint]
    N, T, M, C, K = X.shape

    for n in range(N):
        for m in range(M):
            for k in range(K):
                for c in range(C):
                    values = X[n, :, m, c, k]
                    finite_mask = np.isfinite(values)

                    if finite_mask.all():
                        continue

                    if finite_mask.sum() == 0:
                        X[n, :, m, c, k] = 0.0
                        continue

                    finite_indices = np.where(finite_mask)[0]
                    finite_values = values[finite_mask]
                    all_indices = np.arange(T)
                    X[n, :, m, c, k] = np.interp(all_indices, finite_indices, finite_values)

    return X


def make_temporal_features(
    X_windows: np.ndarray,
    mice: List[int],
    keypoints: List[int],
) -> np.ndarray:
    X = X_windows[:, :, mice, :, :]
    X = X[:, :, :, :, keypoints]
    X = impute_nan_keypoints(X)

    # [N, T, mouse, coord, keypoint] -> [N, T, mouse, keypoint, coord]
    X = np.transpose(X, (0, 1, 2, 4, 3)).astype(np.float32)

    centroid = np.mean(X, axis=(1, 2, 3), keepdims=True)
    X_centered = X - centroid

    scale = np.std(X_centered, axis=(1, 2, 3, 4), keepdims=True)
    scale = np.maximum(scale, 1e-6)
    X_norm = X_centered / scale

    velocity = np.diff(X_norm, axis=1)
    velocity = np.concatenate([velocity[:, :1], velocity], axis=1)

    N, T = X_norm.shape[0], X_norm.shape[1]

    pos_features = X_norm.reshape(N, T, -1)
    vel_features = velocity.reshape(N, T, -1)

    extra_features = []

    if len(mice) == 2:
        mouse_centroids = np.mean(X_norm, axis=3)
        centroid_delta = mouse_centroids[:, :, 0, :] - mouse_centroids[:, :, 1, :]
        centroid_dist = np.sqrt(np.sum(centroid_delta ** 2, axis=-1, keepdims=True))
        extra_features.append(centroid_dist.astype(np.float32))

        mouse0 = X_norm[:, :, 0, :, :]
        mouse1 = X_norm[:, :, 1, :, :]
        keypoint_delta = mouse0 - mouse1
        keypoint_dist = np.sqrt(np.sum(keypoint_delta ** 2, axis=-1))
        extra_features.append(keypoint_dist.astype(np.float32))

        selected_names = [KEYPOINT_ID_TO_NAME[keypoint_id] for keypoint_id in keypoints]

        if "neck" in selected_names and "tail_base" in selected_names:
            neck_local = selected_names.index("neck")
            tail_local = selected_names.index("tail_base")

            vec0 = mouse0[:, :, neck_local, :] - mouse0[:, :, tail_local, :]
            vec1 = mouse1[:, :, neck_local, :] - mouse1[:, :, tail_local, :]

            dot = np.sum(vec0 * vec1, axis=-1, keepdims=True)
            norm0 = np.sqrt(np.sum(vec0 ** 2, axis=-1, keepdims=True))
            norm1 = np.sqrt(np.sum(vec1 ** 2, axis=-1, keepdims=True))
            cos_angle = dot / np.maximum(norm0 * norm1, 1e-6)

            extra_features.append(cos_angle.astype(np.float32))

    if len(extra_features) > 0:
        features = np.concatenate([pos_features, vel_features] + extra_features, axis=-1)
    else:
        features = np.concatenate([pos_features, vel_features], axis=-1)

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features.astype(np.float32)


def apply_degradation(
    X: np.ndarray,
    mode: str,
    severity: float,
    seed: int,
) -> np.ndarray:
    X_deg = X.copy().astype(np.float32)
    rng = np.random.default_rng(seed)

    if mode == "clean" or severity == 0:
        return X_deg

    if mode == "jitter":
        # Add Gaussian noise scaled to within-window pose scale.
        # Severity is a fraction of each window's coordinate standard deviation.
        scale = np.std(X_deg, axis=(1, 2, 3, 4), keepdims=True)
        scale = np.maximum(scale, 1e-6)
        noise = rng.normal(loc=0.0, scale=severity, size=X_deg.shape).astype(np.float32)
        X_deg = X_deg + noise * scale
        return X_deg

    if mode == "dropout":
        # Randomly remove keypoint coordinates, then impute by temporal interpolation.
        # Severity is missing-coordinate probability.
        mask = rng.random(size=X_deg.shape) < severity
        X_deg[mask] = np.nan
        X_deg = impute_nan_keypoints(X_deg)
        return X_deg

    if mode == "temporal_subsample":
        # Simulate lower temporal resolution by keeping every k-th frame and repeating.
        # Severity is integer frame step.
        step = int(round(severity))
        step = max(step, 1)

        if step == 1:
            return X_deg

        N, T, M, C, K = X_deg.shape
        keep_indices = np.arange(0, T, step)

        if keep_indices[-1] != T - 1:
            keep_indices = np.concatenate([keep_indices, np.array([T - 1])])

        compressed = X_deg[:, keep_indices, :, :, :]
        restored = np.empty_like(X_deg)

        for t in range(T):
            nearest_idx = keep_indices[np.argmin(np.abs(keep_indices - t))]
            restored[:, t, :, :, :] = X_deg[:, nearest_idx, :, :, :]

        return restored

    raise ValueError(f"Unknown degradation mode: {mode}")


# ======================================================================================
# Model
# ======================================================================================

class PoseWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.2):
        super().__init__()

        padding = ((kernel_size - 1) // 2) * dilation

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(channels)

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(channels)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        return F.relu(out + residual)


class CompactTCNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_channels: int,
        num_blocks: int,
        dropout: float,
    ):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Conv1d(input_dim, hidden_channels, kernel_size=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        blocks = []

        for block_idx in range(num_blocks):
            dilation = 2 ** block_idx

            blocks.append(
                TemporalBlock(
                    channels=hidden_channels,
                    kernel_size=5,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        self.tcn = nn.Sequential(*blocks)

        classifier_hidden = max(hidden_channels // 2, 16)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, x):
        # x shape: [batch, time, features]
        x = x.transpose(1, 2)
        x = self.input_projection(x)
        x = self.tcn(x)
        logits = self.classifier(x)
        return logits


def compute_class_weights(y_train: np.ndarray, num_classes: int = 4) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / np.maximum(weights.mean(), 1e-6)
    return torch.tensor(weights, dtype=torch.float32)


def make_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = PoseWindowDataset(X_train, y_train)
    val_ds = PoseWindowDataset(X_val, y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
    )

    return train_loader, val_loader


def evaluate_model(model: nn.Module, loader: DataLoader) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)

            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true_all.append(yb.numpy())
            y_pred_all.append(preds.cpu().numpy())
            y_prob_all.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    y_prob = np.concatenate(y_prob_all)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=sorted(LABEL_ID_TO_NAME.keys()),
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            labels=sorted(LABEL_ID_TO_NAME.keys()),
            average="weighted",
            zero_division=0,
        ),
    }

    return metrics, y_true, y_pred, y_prob


def train_one_model(
    region_name: str,
    region_cfg: Dict[str, List[int]],
    X_features: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    Dict[str, torch.Tensor],
]:
    t0_region = time.time()
    set_all_seeds(seed)

    X_train = X_features[train_idx]
    X_val = X_features[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    input_dim = X_train.shape[-1]
    num_classes = len(LABEL_ID_TO_NAME)

    train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val)

    model = CompactTCNClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_channels=CONFIG["hidden_channels"],
        num_blocks=CONFIG["num_tcn_blocks"],
        dropout=CONFIG["dropout"],
    ).to(DEVICE)

    class_weights = compute_class_weights(y_train, num_classes=num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(CONFIG["epochs"], 1),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    best_macro_f1 = -np.inf
    best_state = None
    history_rows = []

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["gradient_clip_norm"])
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(float(loss.detach().cpu().item()))

        scheduler.step()

        metrics, _, _, _ = evaluate_model(model, val_loader)

        mean_loss = float(np.mean(train_losses)) if len(train_losses) > 0 else np.nan

        history_rows.append({
            "seed": seed,
            "region": region_name,
            "epoch": epoch,
            "train_loss": mean_loss,
            "val_accuracy": metrics["accuracy"],
            "val_balanced_accuracy": metrics["balanced_accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "val_weighted_f1": metrics["weighted_f1"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        })

        print(
            f"[seed {seed} | {region_name} | epoch {epoch:02d}/{CONFIG['epochs']}] "
            f"loss={mean_loss:.4f} | "
            f"acc={metrics['accuracy']:.4f} | "
            f"bal_acc={metrics['balanced_accuracy']:.4f} | "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_state = {
                "epoch": epoch,
                "model_state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
            }

    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])

    final_metrics, y_true, y_pred, y_prob = evaluate_model(model, val_loader)

    precision, recall, f1_values, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=sorted(LABEL_ID_TO_NAME.keys()),
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=sorted(LABEL_ID_TO_NAME.keys()),
    )

    history_df = pd.DataFrame(history_rows)
    history_path = RESULTS_DIR / f"history_seed_{seed}_{region_name}.csv"
    history_df.to_csv(history_path, index=False)

    predictions_df = pd.DataFrame({
        "seed": seed,
        "region": region_name,
        "y_true": y_true,
        "y_pred": y_pred,
    })

    for class_id, class_name in LABEL_ID_TO_NAME.items():
        predictions_df[f"prob_{class_name}"] = y_prob[:, class_id]

    predictions_path = RESULTS_DIR / f"predictions_seed_{seed}_{region_name}.csv"
    predictions_df.to_csv(predictions_path, index=False)

    model_path = MODEL_DIR / f"model_seed_{seed}_{region_name}.pt"
    torch.save(
        {
            "seed": seed,
            "region": region_name,
            "region_cfg": region_cfg,
            "model_state_dict": model.state_dict(),
            "best_epoch": int(best_state["epoch"]) if best_state is not None else np.nan,
            "input_dim": input_dim,
            "config": CONFIG,
            "label_id_to_name": LABEL_ID_TO_NAME,
            "keypoint_id_to_name": KEYPOINT_ID_TO_NAME,
        },
        model_path,
    )

    result_row = {
        "seed": seed,
        "region": region_name,
        "mice": ",".join(map(str, region_cfg["mice"])),
        "keypoints": ",".join(KEYPOINT_ID_TO_NAME[keypoint_id] for keypoint_id in region_cfg["keypoints"]),
        "n_keypoints": len(region_cfg["keypoints"]),
        "n_mice": len(region_cfg["mice"]),
        "input_dim": input_dim,
        "n_train_windows": len(train_idx),
        "n_val_windows": len(val_idx),
        "best_epoch": int(best_state["epoch"]) if best_state is not None else np.nan,
        "accuracy": final_metrics["accuracy"],
        "balanced_accuracy": final_metrics["balanced_accuracy"],
        "macro_f1": final_metrics["macro_f1"],
        "weighted_f1": final_metrics["weighted_f1"],
        "runtime_seconds": time.time() - t0_region,
        "model_path": str(model_path),
        "history_path": str(history_path),
        "predictions_path": str(predictions_path),
    }

    per_class_rows = []

    for i, class_id in enumerate(sorted(LABEL_ID_TO_NAME.keys())):
        per_class_rows.append({
            "seed": seed,
            "region": region_name,
            "class_id": class_id,
            "class_name": LABEL_ID_TO_NAME[class_id],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1_values[i]),
            "support": int(support[i]),
        })

    model_state_cpu = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }

    return result_row, per_class_rows, history_df, predictions_df, cm, model_state_cpu


def evaluate_state_on_features(
    model_state: Dict[str, torch.Tensor],
    input_dim: int,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model = CompactTCNClassifier(
        input_dim=input_dim,
        num_classes=len(LABEL_ID_TO_NAME),
        hidden_channels=CONFIG["hidden_channels"],
        num_blocks=CONFIG["num_tcn_blocks"],
        dropout=CONFIG["dropout"],
    ).to(DEVICE)

    model.load_state_dict(model_state)

    eval_ds = PoseWindowDataset(X_eval, y_eval)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
    )

    return evaluate_model(model, eval_loader)


# ======================================================================================
# Statistics
# ======================================================================================

def summarize_across_seeds(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for region, sub_df in results_df.groupby("region"):
        row = {"region": region, "n_seeds": len(sub_df)}

        for metric in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
            values = sub_df[metric].values.astype(float)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            sem = std / np.sqrt(max(len(values), 1))
            ci95 = 1.96 * sem

            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = ci95
            row[f"{metric}_min"] = float(np.min(values))
            row[f"{metric}_max"] = float(np.max(values))

        rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)
    return summary_df


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_boot: int,
    seed: int,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n = len(y_true)
    values = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)

        try:
            value = metric_fn(y_true[idx], y_pred[idx])

            if np.isfinite(value):
                values.append(value)

        except Exception:
            continue

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    values = np.asarray(values, dtype=np.float64)

    return (
        float(np.mean(values)),
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def compute_pooled_bootstrap_table(prediction_tables: Dict[Tuple[int, str], pd.DataFrame]) -> pd.DataFrame:
    rows = []

    by_region = {}

    for key, pred_df in prediction_tables.items():
        region = pred_df["region"].iloc[0]
        by_region.setdefault(region, [])
        by_region[region].append(pred_df)

    for region, frames in by_region.items():
        pooled = pd.concat(frames, ignore_index=True)

        y_true = pooled["y_true"].values
        y_pred = pooled["y_pred"].values

        macro_mean, macro_low, macro_high = bootstrap_metric_ci(
            y_true=y_true,
            y_pred=y_pred,
            metric_fn=lambda a, b: f1_score(
                a,
                b,
                labels=sorted(LABEL_ID_TO_NAME.keys()),
                average="macro",
                zero_division=0,
            ),
            n_boot=CONFIG["bootstrap_iterations"],
            seed=123,
        )

        bal_mean, bal_low, bal_high = bootstrap_metric_ci(
            y_true=y_true,
            y_pred=y_pred,
            metric_fn=lambda a, b: balanced_accuracy_score(a, b),
            n_boot=CONFIG["bootstrap_iterations"],
            seed=456,
        )

        acc_mean, acc_low, acc_high = bootstrap_metric_ci(
            y_true=y_true,
            y_pred=y_pred,
            metric_fn=lambda a, b: accuracy_score(a, b),
            n_boot=CONFIG["bootstrap_iterations"],
            seed=789,
        )

        rows.append({
            "region": region,
            "pooled_macro_f1_boot_mean": macro_mean,
            "pooled_macro_f1_ci_low": macro_low,
            "pooled_macro_f1_ci_high": macro_high,
            "pooled_balanced_accuracy_boot_mean": bal_mean,
            "pooled_balanced_accuracy_ci_low": bal_low,
            "pooled_balanced_accuracy_ci_high": bal_high,
            "pooled_accuracy_boot_mean": acc_mean,
            "pooled_accuracy_ci_low": acc_low,
            "pooled_accuracy_ci_high": acc_high,
        })

    return pd.DataFrame(rows).sort_values("pooled_macro_f1_boot_mean", ascending=False)


def holm_correction(p_values: List[float]) -> List[float]:
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=float)

    running_max = 0.0

    for rank, idx in enumerate(order):
        multiplier = n - rank
        value = min(1.0, multiplier * p_values[idx])
        running_max = max(running_max, value)
        adjusted[idx] = running_max

    return adjusted.tolist()


def paired_tests_against_full(results_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []

    full_df = results_df[results_df["region"] == "full_body_both_mice"][["seed", metric]]
    full_df = full_df.rename(columns={metric: "full_value"})

    regions = sorted([r for r in results_df["region"].unique() if r != "full_body_both_mice"])

    for region in regions:
        sub_df = results_df[results_df["region"] == region][["seed", metric]]
        sub_df = sub_df.rename(columns={metric: "region_value"})

        merged = pd.merge(full_df, sub_df, on="seed", how="inner")

        if len(merged) < 2:
            continue

        diff = merged["region_value"].values - merged["full_value"].values

        try:
            t_stat, t_p = stats.ttest_rel(merged["region_value"].values, merged["full_value"].values)
        except Exception:
            t_stat, t_p = np.nan, np.nan

        try:
            w_stat, w_p = stats.wilcoxon(merged["region_value"].values, merged["full_value"].values)
        except Exception:
            w_stat, w_p = np.nan, np.nan

        rows.append({
            "metric": metric,
            "comparison": f"{region} minus full_body_both_mice",
            "region": region,
            "n_pairs": len(merged),
            "mean_region": float(np.mean(merged["region_value"].values)),
            "mean_full": float(np.mean(merged["full_value"].values)),
            "mean_difference_region_minus_full": float(np.mean(diff)),
            "std_difference": float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0,
            "paired_t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
            "paired_t_p": float(t_p) if np.isfinite(t_p) else np.nan,
            "wilcoxon_stat": float(w_stat) if np.isfinite(w_stat) else np.nan,
            "wilcoxon_p": float(w_p) if np.isfinite(w_p) else np.nan,
        })

    out = pd.DataFrame(rows)

    if len(out) > 0:
        for p_col in ["paired_t_p", "wilcoxon_p"]:
            valid_mask = out[p_col].notna()
            corrected = [np.nan] * len(out)

            if valid_mask.sum() > 0:
                adjusted_valid = holm_correction(out.loc[valid_mask, p_col].tolist())
                valid_indices = out.index[valid_mask].tolist()

                for idx, value in zip(valid_indices, adjusted_valid):
                    corrected[idx] = value

            out[f"{p_col}_holm"] = corrected

    return out


def make_retention_table(results_df: pd.DataFrame) -> pd.DataFrame:
    full_df = results_df[results_df["region"] == "full_body_both_mice"][
        ["seed", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    ].rename(
        columns={
            "accuracy": "full_accuracy",
            "balanced_accuracy": "full_balanced_accuracy",
            "macro_f1": "full_macro_f1",
            "weighted_f1": "full_weighted_f1",
        }
    )

    merged = pd.merge(results_df, full_df, on="seed", how="left")

    for metric in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
        merged[f"{metric}_retention_percent"] = (
            100.0 * merged[metric] / np.maximum(merged[f"full_{metric}"], 1e-9)
        )

    return merged


# ======================================================================================
# Plotting
# ======================================================================================

def plot_summary_metric(summary_df: pd.DataFrame, metric_mean: str, metric_ci: str, out_path: Path) -> None:
    df = summary_df.sort_values(metric_mean, ascending=True)

    values = df[metric_mean].values
    errors = df[metric_ci].values

    plt.figure(figsize=(10, max(5, 0.55 * len(df))))
    plt.barh(df["region"], values, xerr=errors, capsize=4)
    plt.xlabel(metric_mean.replace("_", " ").title())
    plt.ylabel("Body-region condition")
    plt.xlim(0, min(1.0, max(0.1, float(np.nanmax(values + errors)) * 1.15)))
    plt.tight_layout()
    plt.savefig(out_path, dpi=CONFIG["figure_dpi"], bbox_inches="tight")
    plt.show()

    print(f"[figure] Saved: {out_path}")


def plot_seed_points(results_df: pd.DataFrame, metric: str, out_path: Path) -> None:
    regions = (
        results_df.groupby("region")[metric]
        .mean()
        .sort_values(ascending=True)
        .index
        .tolist()
    )

    region_to_y = {region: i for i, region in enumerate(regions)}

    plt.figure(figsize=(10, max(5, 0.55 * len(regions))))

    for region in regions:
        sub_df = results_df[results_df["region"] == region].sort_values("seed")
        y_base = region_to_y[region]
        jitter = np.linspace(-0.13, 0.13, len(sub_df)) if len(sub_df) > 1 else np.array([0.0])
        plt.scatter(sub_df[metric].values, y_base + jitter, s=40)

    means = results_df.groupby("region")[metric].mean().reindex(regions)
    plt.scatter(means.values, np.arange(len(regions)), marker="D", s=80)

    plt.yticks(np.arange(len(regions)), regions)
    plt.xlabel(metric.replace("_", " ").title())
    plt.ylabel("Body-region condition")
    plt.xlim(0, min(1.0, max(0.1, float(results_df[metric].max()) * 1.15)))
    plt.tight_layout()
    plt.savefig(out_path, dpi=CONFIG["figure_dpi"], bbox_inches="tight")
    plt.show()

    print(f"[figure] Saved: {out_path}")


def plot_retention_summary(retention_df: pd.DataFrame, metric: str, out_path: Path) -> None:
    col = f"{metric}_retention_percent"

    summary = (
        retention_df
        .groupby("region")[col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary["ci95"] = 1.96 * summary["std"] / np.sqrt(summary["count"].clip(lower=1))
    summary = summary.sort_values("mean", ascending=True)

    plt.figure(figsize=(10, max(5, 0.55 * len(summary))))
    plt.barh(summary["region"], summary["mean"], xerr=summary["ci95"], capsize=4)
    plt.axvline(100.0, linestyle="--", linewidth=1)
    plt.xlabel(f"{metric.replace('_', ' ').title()} retained relative to full body (%)")
    plt.ylabel("Body-region condition")
    plt.tight_layout()
    plt.savefig(out_path, dpi=CONFIG["figure_dpi"], bbox_inches="tight")
    plt.show()

    print(f"[figure] Saved: {out_path}")


def plot_per_class_recall_heatmap(per_class_df: pd.DataFrame, out_path: Path) -> None:
    mean_recall_df = (
        per_class_df
        .groupby(["region", "class_name"])["recall"]
        .mean()
        .reset_index()
    )

    pivot = mean_recall_df.pivot(index="region", columns="class_name", values="recall")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    plt.figure(figsize=(8, max(5, 0.55 * len(pivot))))
    plt.imshow(pivot.values, aspect="auto", vmin=0, vmax=1)
    plt.colorbar(label="Mean recall across seeds")

    plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(np.arange(len(pivot.index)), pivot.index)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            plt.text(j, i, f"{value:.2f}", ha="center", va="center")

    plt.xlabel("Behaviour class")
    plt.ylabel("Body-region condition")
    plt.tight_layout()
    plt.savefig(out_path, dpi=CONFIG["figure_dpi"], bbox_inches="tight")
    plt.show()

    print(f"[figure] Saved: {out_path}")


def plot_confusion_matrix_mean(confusion_tables: Dict[Tuple[int, str], np.ndarray], region: str, out_path: Path) -> None:
    cms = [cm for (seed, r), cm in confusion_tables.items() if r == region]

    if len(cms) == 0:
        print(f"[figure] No confusion matrices found for region: {region}")
        return

    cm_mean = np.mean(np.stack(cms, axis=0), axis=0)
    labels = [LABEL_ID_TO_NAME[class_id] for class_id in sorted(LABEL_ID_TO_NAME.keys())]

    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(cm_mean)
    plt.colorbar(label="Mean count across seeds")
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    for i in range(cm_mean.shape[0]):
        for j in range(cm_mean.shape[1]):
            plt.text(j, i, f"{cm_mean[i, j]:.1f}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=CONFIG["figure_dpi"], bbox_inches="tight")
    plt.show()

    print(f"[figure] Saved: {out_path}")


def plot_degradation_curves(degradation_df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    if degradation_df.empty:
        print("[figure] No degradation results found.")
        return

    for mode in sorted(degradation_df["degradation_mode"].unique()):
        sub_mode = degradation_df[degradation_df["degradation_mode"] == mode].copy()

        summary = (
            sub_mode
            .groupby(["region", "degradation_severity"])[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        summary["ci95"] = 1.96 * summary["std"] / np.sqrt(summary["count"].clip(lower=1))

        plt.figure(figsize=(9, 6))

        for region in sorted(summary["region"].unique()):
            sub_region = summary[summary["region"] == region].sort_values("degradation_severity")
            plt.errorbar(
                sub_region["degradation_severity"].values,
                sub_region["mean"].values,
                yerr=sub_region["ci95"].values,
                marker="o",
                linewidth=1.5,
                capsize=3,
                label=region,
            )

        plt.xlabel(f"{mode} severity")
        plt.ylabel(metric.replace("_", " ").title())
        plt.ylim(0, 1.0)
        plt.legend(fontsize=8, ncol=1)
        plt.tight_layout()

        out_path = out_dir / f"degradation_{mode}_{metric}.png"
        plt.savefig(out_path, dpi=CONFIG["figure_dpi"], bbox_inches="tight")
        plt.show()

        print(f"[figure] Saved: {out_path}")


def plot_degradation_retention(degradation_df: pd.DataFrame, out_path: Path) -> None:
    if degradation_df.empty:
        print("[figure] No degradation results found.")
        return

    full = degradation_df[degradation_df["region"] == "full_body_both_mice"][
        ["seed", "degradation_mode", "degradation_severity", "macro_f1"]
    ].rename(columns={"macro_f1": "full_macro_f1"})

    merged = degradation_df.merge(
        full,
        on=["seed", "degradation_mode", "degradation_severity"],
        how="left",
    )

    merged["macro_f1_retention_percent"] = (
        100.0 * merged["macro_f1"] / np.maximum(merged["full_macro_f1"], 1e-9)
    )

    summary = (
        merged
        .groupby(["region", "degradation_mode"])["macro_f1_retention_percent"]
        .mean()
        .reset_index()
    )

    pivot = summary.pivot(index="region", columns="degradation_mode", values="macro_f1_retention_percent")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    plt.figure(figsize=(8, max(5, 0.55 * len(pivot))))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label="Mean macro F1 retention vs full body (%)")

    plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(np.arange(len(pivot.index)), pivot.index)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            plt.text(j, i, f"{value:.1f}", ha="center", va="center")

    plt.xlabel("Degradation mode")
    plt.ylabel("Body-region condition")
    plt.tight_layout()
    plt.savefig(out_path, dpi=CONFIG["figure_dpi"], bbox_inches="tight")
    plt.show()

    retention_path = RESULTS_DIR / "degradation_macro_f1_retention_vs_full.csv"
    merged.to_csv(retention_path, index=False)

    print(f"[figure] Saved: {out_path}")
    print(f"[save] Degradation retention table: {retention_path}")


# ======================================================================================
# LaTeX table export
# ======================================================================================

def save_latex_table(df: pd.DataFrame, path: Path, float_format: str = "%.3f") -> None:
    latex = df.to_latex(index=False, escape=False, float_format=float_format)
    with open(path, "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"[table] Saved LaTeX table: {path}")


def make_main_latex_tables(
    summary_df: pd.DataFrame,
    retention_df: pd.DataFrame,
    paired_tests_df: pd.DataFrame,
    degradation_summary_df: pd.DataFrame,
) -> None:
    clean_table = summary_df[
        [
            "region",
            "macro_f1_mean",
            "macro_f1_ci95",
            "balanced_accuracy_mean",
            "balanced_accuracy_ci95",
            "accuracy_mean",
            "accuracy_ci95",
        ]
    ].copy()

    clean_table = clean_table.rename(
        columns={
            "region": "Region",
            "macro_f1_mean": "Macro F1",
            "macro_f1_ci95": "Macro F1 95\\% CI half-width",
            "balanced_accuracy_mean": "Balanced accuracy",
            "balanced_accuracy_ci95": "Balanced accuracy 95\\% CI half-width",
            "accuracy_mean": "Accuracy",
            "accuracy_ci95": "Accuracy 95\\% CI half-width",
        }
    )

    save_latex_table(clean_table, TABLE_DIR / "table_clean_ablation_summary.tex")

    retention_summary = (
        retention_df
        .groupby("region")[["macro_f1_retention_percent", "balanced_accuracy_retention_percent"]]
        .agg(["mean", "std"])
    )

    retention_summary.columns = [
        "_".join(col).strip()
        for col in retention_summary.columns.values
    ]
    retention_summary = retention_summary.reset_index()
    retention_summary = retention_summary.rename(
        columns={
            "region": "Region",
            "macro_f1_retention_percent_mean": "Macro F1 retention (\\%)",
            "macro_f1_retention_percent_std": "Macro F1 retention SD",
            "balanced_accuracy_retention_percent_mean": "Balanced accuracy retention (\\%)",
            "balanced_accuracy_retention_percent_std": "Balanced accuracy retention SD",
        }
    )

    save_latex_table(retention_summary, TABLE_DIR / "table_retention_summary.tex")

    if not paired_tests_df.empty:
        paired_table = paired_tests_df[
            [
                "metric",
                "region",
                "mean_difference_region_minus_full",
                "paired_t_p_holm",
                "wilcoxon_p_holm",
            ]
        ].copy()

        paired_table = paired_table.rename(
            columns={
                "metric": "Metric",
                "region": "Region",
                "mean_difference_region_minus_full": "Mean difference vs full body",
                "paired_t_p_holm": "Paired t-test Holm $p$",
                "wilcoxon_p_holm": "Wilcoxon Holm $p$",
            }
        )

        save_latex_table(paired_table, TABLE_DIR / "table_paired_tests_vs_full.tex")

    if not degradation_summary_df.empty:
        degradation_table = degradation_summary_df[
            [
                "region",
                "degradation_mode",
                "degradation_severity",
                "macro_f1_mean",
                "macro_f1_ci95",
                "balanced_accuracy_mean",
                "balanced_accuracy_ci95",
            ]
        ].copy()

        degradation_table = degradation_table.rename(
            columns={
                "region": "Region",
                "degradation_mode": "Degradation",
                "degradation_severity": "Severity",
                "macro_f1_mean": "Macro F1",
                "macro_f1_ci95": "Macro F1 95\\% CI half-width",
                "balanced_accuracy_mean": "Balanced accuracy",
                "balanced_accuracy_ci95": "Balanced accuracy 95\\% CI half-width",
            }
        )

        save_latex_table(degradation_table, TABLE_DIR / "table_degradation_summary.tex")


# ======================================================================================
# Degradation experiment
# ======================================================================================

def evaluate_degradation_for_trained_model(
    seed: int,
    region_name: str,
    region_cfg: Dict[str, List[int]],
    model_state: Dict[str, torch.Tensor],
    input_dim: int,
    X_windows: np.ndarray,
    y: np.ndarray,
    val_idx: np.ndarray,
) -> List[Dict[str, Any]]:
    rows = []

    if region_name not in CONFIG["degradation_regions"]:
        return rows

    X_val_raw = X_windows[val_idx]
    y_val = y[val_idx]

    for deg_cfg in CONFIG["degradation_grid"]:
        mode = deg_cfg["mode"]
        severity = float(deg_cfg["severity"])

        deg_seed = seed + int(abs(hash((region_name, mode, severity))) % 100000)

        X_deg_raw = apply_degradation(
            X=X_val_raw,
            mode=mode,
            severity=severity,
            seed=deg_seed,
        )

        X_deg_features = make_temporal_features(
            X_windows=X_deg_raw,
            mice=region_cfg["mice"],
            keypoints=region_cfg["keypoints"],
        )

        metrics, y_true, y_pred, y_prob = evaluate_state_on_features(
            model_state=model_state,
            input_dim=input_dim,
            X_eval=X_deg_features,
            y_eval=y_val,
        )

        precision, recall, f1_values, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=sorted(LABEL_ID_TO_NAME.keys()),
            zero_division=0,
        )

        rows.append({
            "seed": seed,
            "region": region_name,
            "degradation_mode": mode,
            "degradation_severity": severity,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            **{
                f"recall_{LABEL_ID_TO_NAME[class_id]}": float(recall[i])
                for i, class_id in enumerate(sorted(LABEL_ID_TO_NAME.keys()))
            },
            **{
                f"f1_{LABEL_ID_TO_NAME[class_id]}": float(f1_values[i])
                for i, class_id in enumerate(sorted(LABEL_ID_TO_NAME.keys()))
            },
            **{
                f"support_{LABEL_ID_TO_NAME[class_id]}": int(support[i])
                for i, class_id in enumerate(sorted(LABEL_ID_TO_NAME.keys()))
            },
        })

        print(
            f"[degradation] seed={seed} | region={region_name} | "
            f"{mode}={severity} | macro_f1={metrics['macro_f1']:.4f} | "
            f"bal_acc={metrics['balanced_accuracy']:.4f}"
        )

    return rows


def summarize_degradation(degradation_df: pd.DataFrame) -> pd.DataFrame:
    if degradation_df.empty:
        return pd.DataFrame()

    rows = []

    group_cols = ["region", "degradation_mode", "degradation_severity"]

    for keys, sub_df in degradation_df.groupby(group_cols):
        region, mode, severity = keys

        row = {
            "region": region,
            "degradation_mode": mode,
            "degradation_severity": severity,
            "n_seeds": sub_df["seed"].nunique(),
        }

        for metric in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
            values = sub_df[metric].values.astype(float)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ci95 = 1.96 * std / np.sqrt(max(len(values), 1))

            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = ci95

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["degradation_mode", "degradation_severity", "macro_f1_mean"], ascending=[True, True, False])


def paired_degradation_tests_against_full(degradation_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if degradation_df.empty:
        return pd.DataFrame()

    rows = []

    full = degradation_df[degradation_df["region"] == "full_body_both_mice"][
        ["seed", "degradation_mode", "degradation_severity", metric]
    ].rename(columns={metric: "full_value"})

    for region in sorted([r for r in degradation_df["region"].unique() if r != "full_body_both_mice"]):
        region_df = degradation_df[degradation_df["region"] == region][
            ["seed", "degradation_mode", "degradation_severity", metric]
        ].rename(columns={metric: "region_value"})

        merged = region_df.merge(
            full,
            on=["seed", "degradation_mode", "degradation_severity"],
            how="inner",
        )

        for keys, sub in merged.groupby(["degradation_mode", "degradation_severity"]):
            mode, severity = keys

            if len(sub) < 2:
                continue

            diff = sub["region_value"].values - sub["full_value"].values

            try:
                t_stat, t_p = stats.ttest_rel(sub["region_value"].values, sub["full_value"].values)
            except Exception:
                t_stat, t_p = np.nan, np.nan

            try:
                w_stat, w_p = stats.wilcoxon(sub["region_value"].values, sub["full_value"].values)
            except Exception:
                w_stat, w_p = np.nan, np.nan

            rows.append({
                "metric": metric,
                "region": region,
                "degradation_mode": mode,
                "degradation_severity": severity,
                "n_pairs": len(sub),
                "mean_region": float(np.mean(sub["region_value"].values)),
                "mean_full": float(np.mean(sub["full_value"].values)),
                "mean_difference_region_minus_full": float(np.mean(diff)),
                "std_difference": float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0,
                "paired_t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
                "paired_t_p": float(t_p) if np.isfinite(t_p) else np.nan,
                "wilcoxon_stat": float(w_stat) if np.isfinite(w_stat) else np.nan,
                "wilcoxon_p": float(w_p) if np.isfinite(w_p) else np.nan,
            })

    out = pd.DataFrame(rows)

    if len(out) > 0:
        for p_col in ["paired_t_p", "wilcoxon_p"]:
            valid_mask = out[p_col].notna()
            corrected = [np.nan] * len(out)

            if valid_mask.sum() > 0:
                adjusted_valid = holm_correction(out.loc[valid_mask, p_col].tolist())
                valid_indices = out.index[valid_mask].tolist()

                for idx, value in zip(valid_indices, adjusted_valid):
                    corrected[idx] = value

            out[f"{p_col}_holm"] = corrected

    return out


# ======================================================================================
# Main experiment
# ======================================================================================

def main():
    global_start_time = time.time()

    print_banner("CalMS21 Minimum Observable Body Geometry — Paper-grade Experiment")
    require_l4_or_cuda()

    print(f"[project] Project directory : {PROJECT_DIR}")
    print(f"[project] Results directory : {RESULTS_DIR}")
    print(f"[project] Figures directory : {FIG_DIR}")
    print(f"[project] Tables directory  : {TABLE_DIR}")
    print(f"[project] Models directory  : {MODEL_DIR}")
    print(f"[config] Seeds             : {CONFIG['seeds']}")
    print(f"[config] Window size       : {CONFIG['window_size']}")
    print(f"[config] Window stride     : {CONFIG['window_stride']}")
    print(f"[config] Epochs            : {CONFIG['epochs']}")
    print(f"[config] Batch size        : {CONFIG['batch_size']}")
    print(f"[config] Hidden channels   : {CONFIG['hidden_channels']}")
    print(f"[config] TCN blocks        : {CONFIG['num_tcn_blocks']}")

    save_json(CONFIG, RESULTS_DIR / "config.json")

    copy_existing_file_if_available(CONFIG["zip_name"], ZIP_PATH)
    copy_existing_file_if_available(CONFIG["train_json_name"], TRAIN_JSON_PATH)

    download_file_with_fallback(CONFIG["download_urls"], ZIP_PATH)

    extract_train_json(
        zip_path=ZIP_PATH,
        train_json_path=TRAIN_JSON_PATH,
        target_json_name=CONFIG["train_json_name"],
    )

    X_windows, y, groups = sample_or_load_windows()

    dataset_manifest = {
        "X_shape": list(X_windows.shape),
        "y_shape": list(y.shape),
        "groups_shape": list(groups.shape),
        "class_counts": {
            LABEL_ID_TO_NAME[class_id]: int(np.sum(y == class_id))
            for class_id in LABEL_ID_TO_NAME.keys()
        },
        "unique_groups": int(len(np.unique(groups))),
    }
    save_json(dataset_manifest, RESULTS_DIR / "dataset_manifest.json")

    print_banner("Precomputing clean features for all body-region conditions")

    region_features: Dict[str, np.ndarray] = {}

    for region_name, region_cfg in BODY_REGION_CONFIGS.items():
        t0 = time.time()
        print(f"[features] Building clean features for {region_name}...")

        region_features[region_name] = make_temporal_features(
            X_windows=X_windows,
            mice=region_cfg["mice"],
            keypoints=region_cfg["keypoints"],
        )

        print(
            f"[features] {region_name}: {region_features[region_name].shape} | "
            f"time={format_seconds(time.time() - t0)}"
        )

    result_rows = []
    per_class_rows_all = []
    all_history_frames = []
    prediction_tables: Dict[Tuple[int, str], pd.DataFrame] = {}
    confusion_tables: Dict[Tuple[int, str], np.ndarray] = {}
    degradation_rows_all = []
    split_frames = []

    for seed in CONFIG["seeds"]:
        if minutes_elapsed(global_start_time) > CONFIG["global_budget_minutes"]:
            print(f"[budget] Global budget reached at {minutes_elapsed(global_start_time):.1f} min. Stopping.")
            break

        print_banner(f"Seed {seed}")
        set_all_seeds(seed)

        train_idx, val_idx, split_df = make_split_for_seed(X_windows, y, groups, seed)
        split_frames.append(split_df)

        for region_name, region_cfg in BODY_REGION_CONFIGS.items():
            if minutes_elapsed(global_start_time) > CONFIG["global_budget_minutes"]:
                print(f"[budget] Global budget reached at {minutes_elapsed(global_start_time):.1f} min. Stopping.")
                break

            print_banner(f"Clean training | seed={seed} | region={region_name}")

            result_row, per_class_rows, history_df, pred_df, cm, model_state = train_one_model(
                region_name=region_name,
                region_cfg=region_cfg,
                X_features=region_features[region_name],
                y=y,
                train_idx=train_idx,
                val_idx=val_idx,
                seed=seed,
            )

            result_rows.append(result_row)
            per_class_rows_all.extend(per_class_rows)
            all_history_frames.append(history_df)
            prediction_tables[(seed, region_name)] = pred_df
            confusion_tables[(seed, region_name)] = cm

            interim_results_df = pd.DataFrame(result_rows)
            interim_results_df.to_csv(RESULTS_DIR / "clean_results_intermediate.csv", index=False)

            if CONFIG["run_degradation_experiment"] and region_name in CONFIG["degradation_regions"]:
                degradation_rows = evaluate_degradation_for_trained_model(
                    seed=seed,
                    region_name=region_name,
                    region_cfg=region_cfg,
                    model_state=model_state,
                    input_dim=int(result_row["input_dim"]),
                    X_windows=X_windows,
                    y=y,
                    val_idx=val_idx,
                )

                degradation_rows_all.extend(degradation_rows)

                if len(degradation_rows_all) > 0:
                    pd.DataFrame(degradation_rows_all).to_csv(
                        RESULTS_DIR / "degradation_results_intermediate.csv",
                        index=False,
                    )

    if len(result_rows) == 0:
        raise RuntimeError("No clean models were trained.")

    print_banner("Saving clean experiment outputs")

    results_df = pd.DataFrame(result_rows)
    per_class_df = pd.DataFrame(per_class_rows_all)
    all_history_df = pd.concat(all_history_frames, ignore_index=True) if len(all_history_frames) > 0 else pd.DataFrame()
    all_splits_df = pd.concat(split_frames, ignore_index=True) if len(split_frames) > 0 else pd.DataFrame()

    clean_summary_df = summarize_across_seeds(results_df)
    pooled_bootstrap_df = compute_pooled_bootstrap_table(prediction_tables)
    retention_df = make_retention_table(results_df)

    paired_macro_df = paired_tests_against_full(results_df, metric="macro_f1")
    paired_bal_df = paired_tests_against_full(results_df, metric="balanced_accuracy")
    paired_acc_df = paired_tests_against_full(results_df, metric="accuracy")
    paired_tests_df = pd.concat([paired_macro_df, paired_bal_df, paired_acc_df], ignore_index=True)

    results_df.to_csv(RESULTS_DIR / "clean_all_seed_region_results.csv", index=False)
    per_class_df.to_csv(RESULTS_DIR / "clean_per_class_results.csv", index=False)
    all_history_df.to_csv(RESULTS_DIR / "training_histories.csv", index=False)
    all_splits_df.to_csv(RESULTS_DIR / "all_splits.csv", index=False)
    clean_summary_df.to_csv(RESULTS_DIR / "clean_summary_across_seeds.csv", index=False)
    pooled_bootstrap_df.to_csv(RESULTS_DIR / "clean_pooled_bootstrap_ci.csv", index=False)
    retention_df.to_csv(RESULTS_DIR / "clean_retention_vs_full_by_seed.csv", index=False)
    paired_tests_df.to_csv(RESULTS_DIR / "clean_paired_tests_vs_full.csv", index=False)

    print("[clean] Summary across seeds:")
    print(clean_summary_df.to_string(index=False))

    print("[clean] Paired tests against full body:")
    print(paired_tests_df.to_string(index=False))

    degradation_df = pd.DataFrame(degradation_rows_all)

    if len(degradation_df) > 0:
        print_banner("Saving degradation robustness outputs")

        degradation_summary_df = summarize_degradation(degradation_df)

        degradation_tests_macro_df = paired_degradation_tests_against_full(degradation_df, metric="macro_f1")
        degradation_tests_bal_df = paired_degradation_tests_against_full(degradation_df, metric="balanced_accuracy")
        degradation_tests_df = pd.concat([degradation_tests_macro_df, degradation_tests_bal_df], ignore_index=True)

        degradation_df.to_csv(RESULTS_DIR / "degradation_results.csv", index=False)
        degradation_summary_df.to_csv(RESULTS_DIR / "degradation_summary_across_seeds.csv", index=False)
        degradation_tests_df.to_csv(RESULTS_DIR / "degradation_paired_tests_vs_full.csv", index=False)

        print("[degradation] Summary:")
        print(degradation_summary_df.head(40).to_string(index=False))

    else:
        degradation_summary_df = pd.DataFrame()
        degradation_tests_df = pd.DataFrame()
        print("[degradation] No degradation rows were produced.")

    print_banner("Generating figures")

    plot_summary_metric(
        summary_df=clean_summary_df,
        metric_mean="macro_f1_mean",
        metric_ci="macro_f1_ci95",
        out_path=FIG_DIR / "fig_clean_macro_f1_mean_ci95.png",
    )

    plot_summary_metric(
        summary_df=clean_summary_df,
        metric_mean="balanced_accuracy_mean",
        metric_ci="balanced_accuracy_ci95",
        out_path=FIG_DIR / "fig_clean_balanced_accuracy_mean_ci95.png",
    )

    plot_seed_points(
        results_df=results_df,
        metric="macro_f1",
        out_path=FIG_DIR / "fig_clean_seed_points_macro_f1.png",
    )

    plot_seed_points(
        results_df=results_df,
        metric="balanced_accuracy",
        out_path=FIG_DIR / "fig_clean_seed_points_balanced_accuracy.png",
    )

    plot_retention_summary(
        retention_df=retention_df,
        metric="macro_f1",
        out_path=FIG_DIR / "fig_clean_macro_f1_retention_vs_full.png",
    )

    plot_retention_summary(
        retention_df=retention_df,
        metric="balanced_accuracy",
        out_path=FIG_DIR / "fig_clean_balanced_accuracy_retention_vs_full.png",
    )

    plot_per_class_recall_heatmap(
        per_class_df=per_class_df,
        out_path=FIG_DIR / "fig_clean_mean_per_class_recall_heatmap.png",
    )

    plot_confusion_matrix_mean(
        confusion_tables=confusion_tables,
        region="full_body_both_mice",
        out_path=FIG_DIR / "fig_clean_mean_confusion_full_body.png",
    )

    if "hips_tail_both_mice" in BODY_REGION_CONFIGS:
        plot_confusion_matrix_mean(
            confusion_tables=confusion_tables,
            region="hips_tail_both_mice",
            out_path=FIG_DIR / "fig_clean_mean_confusion_hips_tail.png",
        )

    if len(degradation_df) > 0:
        plot_degradation_curves(
            degradation_df=degradation_df,
            metric="macro_f1",
            out_dir=FIG_DIR,
        )

        plot_degradation_curves(
            degradation_df=degradation_df,
            metric="balanced_accuracy",
            out_dir=FIG_DIR,
        )

        plot_degradation_retention(
            degradation_df=degradation_df,
            out_path=FIG_DIR / "fig_degradation_macro_f1_retention_heatmap.png",
        )

    print_banner("Saving LaTeX-ready tables")

    make_main_latex_tables(
        summary_df=clean_summary_df,
        retention_df=retention_df,
        paired_tests_df=paired_tests_df,
        degradation_summary_df=degradation_summary_df,
    )

    manifest = {
        "project_dir": str(PROJECT_DIR),
        "results_dir": str(RESULTS_DIR),
        "figures_dir": str(FIG_DIR),
        "tables_dir": str(TABLE_DIR),
        "models_dir": str(MODEL_DIR),
        "total_runtime_seconds": time.time() - global_start_time,
        "total_runtime_formatted": format_seconds(time.time() - global_start_time),
        "n_clean_models": int(len(results_df)),
        "n_degradation_rows": int(len(degradation_df)) if len(degradation_df) > 0 else 0,
        "device": str(DEVICE),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "main_result_files": [
            str(RESULTS_DIR / "clean_all_seed_region_results.csv"),
            str(RESULTS_DIR / "clean_summary_across_seeds.csv"),
            str(RESULTS_DIR / "clean_retention_vs_full_by_seed.csv"),
            str(RESULTS_DIR / "clean_paired_tests_vs_full.csv"),
            str(RESULTS_DIR / "clean_per_class_results.csv"),
            str(RESULTS_DIR / "clean_pooled_bootstrap_ci.csv"),
            str(RESULTS_DIR / "degradation_results.csv"),
            str(RESULTS_DIR / "degradation_summary_across_seeds.csv"),
            str(RESULTS_DIR / "degradation_paired_tests_vs_full.csv"),
        ],
        "main_figure_files": [
            str(FIG_DIR / "fig_clean_macro_f1_mean_ci95.png"),
            str(FIG_DIR / "fig_clean_balanced_accuracy_mean_ci95.png"),
            str(FIG_DIR / "fig_clean_seed_points_macro_f1.png"),
            str(FIG_DIR / "fig_clean_seed_points_balanced_accuracy.png"),
            str(FIG_DIR / "fig_clean_macro_f1_retention_vs_full.png"),
            str(FIG_DIR / "fig_clean_balanced_accuracy_retention_vs_full.png"),
            str(FIG_DIR / "fig_clean_mean_per_class_recall_heatmap.png"),
            str(FIG_DIR / "fig_clean_mean_confusion_full_body.png"),
            str(FIG_DIR / "fig_clean_mean_confusion_hips_tail.png"),
            str(FIG_DIR / "fig_degradation_macro_f1_retention_heatmap.png"),
        ],
    }

    save_json(manifest, RESULTS_DIR / "run_manifest.json")

    print_banner("Completed paper-grade experiment")
    print(f"Total runtime       : {manifest['total_runtime_formatted']}")
    print(f"Results directory   : {RESULTS_DIR}")
    print(f"Figures directory   : {FIG_DIR}")
    print(f"Tables directory    : {TABLE_DIR}")
    print(f"Models directory    : {MODEL_DIR}")
    print(f"Manifest            : {RESULTS_DIR / 'run_manifest.json'}")

    print("\nPrimary interpretation checks:")
    print("1. Use clean_summary_across_seeds.csv to rank body-region conditions.")
    print("2. Use clean_retention_vs_full_by_seed.csv to report percent retained relative to full body.")
    print("3. Use clean_paired_tests_vs_full.csv for Holm-corrected paired comparisons.")
    print("4. Use clean_per_class_results.csv and the recall heatmap to identify class-specific failures.")
    print("5. Use degradation_summary_across_seeds.csv to test whether compact regions degrade gracefully.")
    print("6. Use LaTeX tables from the tables folder directly in Overleaf.")

    return {
        "results_df": results_df,
        "clean_summary_df": clean_summary_df,
        "per_class_df": per_class_df,
        "retention_df": retention_df,
        "paired_tests_df": paired_tests_df,
        "degradation_df": degradation_df,
        "degradation_summary_df": degradation_summary_df,
        "manifest": manifest,
    }


outputs = main()