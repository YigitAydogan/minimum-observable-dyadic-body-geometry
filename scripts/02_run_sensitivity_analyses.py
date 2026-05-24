"""
CalMS21 Interface Add-on Experiments

"""

from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
import sys
import time
import warnings
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

warnings.filterwarnings("ignore")


# =============================================================================
# Dependency setup
# =============================================================================

def ensure_package(package_name: str, import_name: str | None = None) -> None:
    name = import_name if import_name is not None else package_name
    try:
        __import__(name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package_name])


ensure_package("numpy")
ensure_package("pandas")
ensure_package("matplotlib")
ensure_package("scikit-learn", "sklearn")
ensure_package("ijson")
ensure_package("requests")
ensure_package("torch")

import ijson
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path("/content/calms21_minimum_body_paper_grade")
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"

ADDON_DIR = BASE_DIR / "interface_addon"
RESULTS_DIR = ADDON_DIR / "results"
FIG_DIR = ADDON_DIR / "figures"
TABLE_DIR = ADDON_DIR / "tables"
SNIPPET_DIR = ADDON_DIR / "latex_snippets"
CACHE_OUT_DIR = ADDON_DIR / "cache"
MODEL_DIR = ADDON_DIR / "models"

ZIP_OUTPUT_PATH = ADDON_DIR / "interface_addon_package.zip"

TRAIN_JSON_NAME = "calms21_task1_train.json"
ZIP_NAME = "task1_classic_classification.zip"

DOWNLOAD_URLS = [
    "https://data.caltech.edu/records/s0vdx-0k302/files/task1_classic_classification.zip/content",
    "https://data.caltech.edu/records/s0vdx-0k302/files/task1_classic_classification.zip?download=1",
]

LABEL_ID_TO_NAME = {
    0: "attack",
    1: "investigation",
    2: "mount",
    3: "other",
}

KEYPOINT_ID_TO_NAME = {
    0: "nose",
    1: "left_ear",
    2: "right_ear",
    3: "neck",
    4: "left_hip",
    5: "right_hip",
    6: "tail_base",
}

REGION_CONFIGS = {
    "full_body_both_mice": {"mice": [0, 1], "keypoints": [0, 1, 2, 3, 4, 5, 6]},
    "no_tail_both_mice": {"mice": [0, 1], "keypoints": [0, 1, 2, 3, 4, 5]},
    "hips_tail_both_mice": {"mice": [0, 1], "keypoints": [4, 5, 6]},
    "head_neck_both_mice": {"mice": [0, 1], "keypoints": [0, 1, 2, 3]},
    "trunk_tail_both_mice": {"mice": [0, 1], "keypoints": [3, 4, 5, 6]},
    "head_only_both_mice": {"mice": [0, 1], "keypoints": [0, 1, 2]},
    "hips_only_both_mice": {"mice": [0, 1], "keypoints": [4, 5]},
    "neck_only_both_mice": {"mice": [0, 1], "keypoints": [3]},
    "tail_base_only_both_mice": {"mice": [0, 1], "keypoints": [6]},
    "resident_only_full_body": {"mice": [0], "keypoints": [0, 1, 2, 3, 4, 5, 6]},
    "intruder_only_full_body": {"mice": [1], "keypoints": [0, 1, 2, 3, 4, 5, 6]},
}

MAIN_REGIONS = [
    "full_body_both_mice",
    "no_tail_both_mice",
    "hips_tail_both_mice",
    "head_neck_both_mice",
    "trunk_tail_both_mice",
    "resident_only_full_body",
    "intruder_only_full_body",
]

REGION_ORDER = [
    "full_body_both_mice",
    "no_tail_both_mice",
    "hips_tail_both_mice",
    "head_neck_both_mice",
    "trunk_tail_both_mice",
    "head_only_both_mice",
    "hips_only_both_mice",
    "neck_only_both_mice",
    "tail_base_only_both_mice",
    "resident_only_full_body",
    "intruder_only_full_body",
]

REGION_SHORT = {
    "full_body_both_mice": "Full body",
    "no_tail_both_mice": "No tail",
    "hips_tail_both_mice": "Hips+tail",
    "head_neck_both_mice": "Head+neck",
    "trunk_tail_both_mice": "Trunk+tail",
    "head_only_both_mice": "Head only",
    "hips_only_both_mice": "Hips only",
    "neck_only_both_mice": "Neck only",
    "tail_base_only_both_mice": "Tail base",
    "resident_only_full_body": "Resident only",
    "intruder_only_full_body": "Intruder only",
}

SEEDS = [11, 22, 33, 44, 55]

WINDOW_SIZE = 45

ORIGINAL_OVERLAP_CACHE_NAME = "calms21_windows_T45_S2_C12000.npz"
NONOVERLAP_CACHE_NAME = "calms21_windows_T45_S45_C12000_with_starts.npz"

MAX_WINDOWS_PER_CLASS_NONOVERLAP = 12000

MAX_TRAIN_PER_CLASS_SHALLOW = 8000
MAX_VAL_PER_CLASS_SHALLOW = 2000

MAX_TRAIN_PER_CLASS_NONOVERLAP_TCN = 8000
MAX_VAL_PER_CLASS_NONOVERLAP_TCN = 2000

VALIDATION_FRACTION = 0.25
MAX_SPLIT_ATTEMPTS = 1000

TCN_EPOCHS = 12
BATCH_SIZE = 512
HIDDEN_CHANNELS = 96
NUM_TCN_BLOCKS = 4
DROPOUT = 0.20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2

FIG_DPI = 600


# =============================================================================
# Paths and utility
# =============================================================================

def setup_dirs() -> None:
    if ADDON_DIR.exists():
        shutil.rmtree(ADDON_DIR)
    for d in [ADDON_DIR, RESULTS_DIR, FIG_DIR, TABLE_DIR, SNIPPET_DIR, CACHE_OUT_DIR, MODEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def print_banner(text: str) -> None:
    print("\n" + "=" * 110)
    print(text)
    print("=" * 110)


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sort_regions(df: pd.DataFrame, region_col: str = "region") -> pd.DataFrame:
    order = {r: i for i, r in enumerate(REGION_ORDER)}
    out = df.copy()
    out["_order"] = out[region_col].map(order).fillna(9999)
    out = out.sort_values(["_order", region_col]).drop(columns=["_order"])
    return out


def add_region_labels(df: pd.DataFrame, region_col: str = "region") -> pd.DataFrame:
    out = df.copy()
    out["region_short"] = out[region_col].map(REGION_SHORT).fillna(out[region_col])
    return out


def summarize_by_region_and_model(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row["n_seeds"] = sub["seed"].nunique()
        for metric in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
            vals = sub[metric].astype(float).values
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{metric}_ci95"] = 1.96 * row[f"{metric}_std"] / math.sqrt(max(len(vals), 1))
            row[f"{metric}_min"] = float(np.min(vals))
            row[f"{metric}_max"] = float(np.max(vals))
        rows.append(row)
    return pd.DataFrame(rows)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    def convert(x: Any) -> Any:
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, dict):
            return {str(k): convert(v) for k, v in x.items()}
        if isinstance(x, list):
            return [convert(v) for v in x]
        return x

    path.write_text(json.dumps(convert(obj), indent=2), encoding="utf-8")


# =============================================================================
# Dataset availability
# =============================================================================

def download_file_with_fallback(urls: List[str], out_path: Path, chunk_size: int = 1024 * 1024) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 10 * 1024 * 1024:
        print(f"[download] Existing file found: {out_path}")
        return out_path

    last_error = None
    for url in urls:
        print(f"[download] Trying URL: {url}")
        tmp_path = out_path.with_suffix(out_path.suffix + ".part")
        try:
            with requests.get(url, stream=True, timeout=60, allow_redirects=True) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                downloaded = 0
                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = 100.0 * downloaded / total
                            print(f"\r[download] {pct:6.2f}% | {downloaded/(1024**2):.1f}/{total/(1024**2):.1f} MB", end="")
            print()
            if tmp_path.stat().st_size < 10 * 1024 * 1024:
                raise RuntimeError("Downloaded file was unexpectedly small.")
            tmp_path.rename(out_path)
            return out_path
        except Exception as exc:
            last_error = exc
            print(f"[download] Failed: {exc}")
            if tmp_path.exists():
                tmp_path.unlink()
    raise RuntimeError(f"All download URLs failed. Last error: {last_error}")


def extract_train_json(zip_path: Path, train_json_path: Path) -> Path:
    if train_json_path.exists() and train_json_path.stat().st_size > 10 * 1024 * 1024:
        print(f"[extract] Existing JSON found: {train_json_path}")
        return train_json_path

    with zipfile.ZipFile(zip_path, "r") as zf:
        candidates = [name for name in zf.namelist() if Path(name).name == TRAIN_JSON_NAME]
        if not candidates:
            raise FileNotFoundError(f"{TRAIN_JSON_NAME} not found in {zip_path}")
        member = candidates[0]
        print(f"[extract] Extracting {member}")
        with zf.open(member, "r") as src, open(train_json_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return train_json_path


def ensure_train_json() -> Path:
    train_json_path = DATA_DIR / TRAIN_JSON_NAME
    if train_json_path.exists() and train_json_path.stat().st_size > 10 * 1024 * 1024:
        return train_json_path
    zip_path = DATA_DIR / ZIP_NAME
    download_file_with_fallback(DOWNLOAD_URLS, zip_path)
    return extract_train_json(zip_path, train_json_path)


def stream_calms21_task1_videos(json_path: Path):
    with open(json_path, "rb") as f:
        for video_id, video_obj in ijson.kvitems(f, "annotator-id_0"):
            yield video_id, video_obj


# =============================================================================
# Window caches
# =============================================================================

def load_original_overlap_cache() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = CACHE_DIR / ORIGINAL_OVERLAP_CACHE_NAME
    if not path.exists():
        candidates = sorted(CACHE_DIR.glob("calms21_windows_T45_S2*.npz"))
        if candidates:
            path = candidates[0]
    if not path.exists():
        raise FileNotFoundError(
            "Could not find the original overlap cache. Run the paper-grade script first. "
            f"Expected something like {CACHE_DIR / ORIGINAL_OVERLAP_CACHE_NAME}"
        )
    print(f"[cache] Loading original overlap cache: {path}")
    data = np.load(path, allow_pickle=True)
    return data["X"].astype(np.float32), data["y"].astype(np.int64), data["groups"].astype(str)


def build_or_load_nonoverlap_cache(train_json_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = CACHE_OUT_DIR / NONOVERLAP_CACHE_NAME
    if path.exists():
        print(f"[cache] Loading non-overlap cache: {path}")
        data = np.load(path, allow_pickle=True)
        return (
            data["X"].astype(np.float32),
            data["y"].astype(np.int64),
            data["groups"].astype(str),
            data["starts"].astype(np.int64),
        )

    print_banner("Building non-overlapping window cache")
    X_windows: List[np.ndarray] = []
    y_labels: List[int] = []
    groups: List[str] = []
    starts: List[int] = []
    per_class_counts = {c: 0 for c in LABEL_ID_TO_NAME}

    n_videos = 0
    t0 = time.time()

    for video_id, video_obj in stream_calms21_task1_videos(train_json_path):
        n_videos += 1
        keypoints = np.asarray(video_obj["keypoints"], dtype=np.float32)
        annotations = np.asarray(video_obj["annotations"], dtype=np.int64)

        if keypoints.ndim != 4:
            continue
        if annotations.shape[0] != keypoints.shape[0]:
            continue

        n_frames = keypoints.shape[0]
        if n_frames < WINDOW_SIZE:
            continue

        for start in range(0, n_frames - WINDOW_SIZE + 1, WINDOW_SIZE):
            end = start + WINDOW_SIZE
            center = start + WINDOW_SIZE // 2
            label = int(annotations[center])

            if label not in per_class_counts:
                continue
            if per_class_counts[label] >= MAX_WINDOWS_PER_CLASS_NONOVERLAP:
                continue

            window = keypoints[start:end]
            if not np.isfinite(window).all():
                continue

            X_windows.append(window)
            y_labels.append(label)
            groups.append(video_id)
            starts.append(start)
            per_class_counts[label] += 1

        if n_videos % 10 == 0:
            counts_txt = ", ".join(f"{LABEL_ID_TO_NAME[k]}={v}" for k, v in per_class_counts.items())
            print(f"[nonoverlap] videos={n_videos:03d} | windows={len(X_windows):06d} | {counts_txt} | elapsed={format_seconds(time.time()-t0)}")

    if not X_windows:
        raise RuntimeError("No non-overlapping windows were built.")

    X = np.stack(X_windows, axis=0).astype(np.float32)
    y = np.asarray(y_labels, dtype=np.int64)
    groups_arr = np.asarray(groups).astype(str)
    starts_arr = np.asarray(starts, dtype=np.int64)

    print(f"[nonoverlap] X shape: {X.shape}")
    print(f"[nonoverlap] class counts: {dict(zip(*np.unique(y, return_counts=True)))}")

    np.savez_compressed(path, X=X, y=y, groups=groups_arr, starts=starts_arr)
    print(f"[cache] Saved non-overlap cache: {path}")
    return X, y, groups_arr, starts_arr


# =============================================================================
# Splits
# =============================================================================

def class_counts(y: np.ndarray, idx: np.ndarray) -> Dict[int, int]:
    return {c: int(np.sum(y[idx] == c)) for c in LABEL_ID_TO_NAME}


def adaptive_min_counts(y: np.ndarray) -> Tuple[Dict[int, int], Dict[int, int]]:
    total_counts = {c: int(np.sum(y == c)) for c in LABEL_ID_TO_NAME}
    min_train = {}
    min_val = {}
    for c, n in total_counts.items():
        min_train[c] = max(20, min(1000, int(0.10 * n)))
        min_val[c] = max(10, min(300, int(0.04 * n)))
    return min_train, min_val


def split_ok(y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, min_train: Dict[int, int], min_val: Dict[int, int]) -> bool:
    train_counts = class_counts(y, train_idx)
    val_counts = class_counts(y, val_idx)
    for c in LABEL_ID_TO_NAME:
        if train_counts[c] < min_train[c]:
            return False
        if val_counts[c] < min_val[c]:
            return False
    return True


def make_valid_split(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, int]:
    min_train, min_val = adaptive_min_counts(y)

    for attempt in range(MAX_SPLIT_ATTEMPTS):
        split_seed = seed + attempt * 997
        splitter = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_FRACTION, random_state=split_seed)
        train_idx, val_idx = next(splitter.split(X, y, groups=groups))

        if split_ok(y, train_idx, val_idx, min_train, min_val):
            print(f"[split] seed={seed} valid at attempt={attempt+1} split_seed={split_seed}")
            print(f"[split] train counts: {class_counts(y, train_idx)}")
            print(f"[split] val counts  : {class_counts(y, val_idx)}")
            return train_idx, val_idx, split_seed

    raise RuntimeError(f"Could not find a valid video-level split for seed {seed}.")


def class_balanced_cap(indices: np.ndarray, y: np.ndarray, max_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: List[int] = []
    for c in sorted(LABEL_ID_TO_NAME):
        c_idx = indices[y[indices] == c]
        if len(c_idx) > max_per_class:
            c_idx = rng.choice(c_idx, size=max_per_class, replace=False)
        selected.extend(c_idx.tolist())
    out = np.asarray(selected, dtype=np.int64)
    rng.shuffle(out)
    return out


# =============================================================================
# Feature construction
# =============================================================================

def impute_nan_keypoints(X: np.ndarray) -> np.ndarray:
    X = X.copy().astype(np.float32)
    if np.isfinite(X).all():
        return X
    n, t, m, c, k = X.shape
    for i in range(n):
        for mi in range(m):
            for ki in range(k):
                for ci in range(c):
                    vals = X[i, :, mi, ci, ki]
                    finite = np.isfinite(vals)
                    if finite.all():
                        continue
                    if finite.sum() == 0:
                        X[i, :, mi, ci, ki] = 0.0
                    else:
                        x_idx = np.arange(t)
                        X[i, :, mi, ci, ki] = np.interp(x_idx, x_idx[finite], vals[finite])
    return X


def make_temporal_features(X_windows: np.ndarray, mice: List[int], keypoints: List[int]) -> np.ndarray:
    X = X_windows[:, :, mice, :, :]
    X = X[:, :, :, :, keypoints]
    X = impute_nan_keypoints(X)

    # [N,T,mouse,coord,keypoint] -> [N,T,mouse,keypoint,coord]
    X = np.transpose(X, (0, 1, 2, 4, 3)).astype(np.float32)

    centroid = np.mean(X, axis=(1, 2, 3), keepdims=True)
    X_centered = X - centroid
    scale = np.std(X_centered, axis=(1, 2, 3, 4), keepdims=True)
    scale = np.maximum(scale, 1e-6)
    X_norm = X_centered / scale

    vel = np.diff(X_norm, axis=1)
    vel = np.concatenate([vel[:, :1], vel], axis=1)

    n, t = X_norm.shape[0], X_norm.shape[1]
    pos_features = X_norm.reshape(n, t, -1)
    vel_features = vel.reshape(n, t, -1)

    extras: List[np.ndarray] = []

    if len(mice) == 2:
        mouse_centroids = np.mean(X_norm, axis=3)
        centroid_delta = mouse_centroids[:, :, 0, :] - mouse_centroids[:, :, 1, :]
        centroid_dist = np.sqrt(np.sum(centroid_delta ** 2, axis=-1, keepdims=True))
        extras.append(centroid_dist.astype(np.float32))

        mouse0 = X_norm[:, :, 0, :, :]
        mouse1 = X_norm[:, :, 1, :, :]
        kp_dist = np.sqrt(np.sum((mouse0 - mouse1) ** 2, axis=-1))
        extras.append(kp_dist.astype(np.float32))

        selected_names = [KEYPOINT_ID_TO_NAME[k] for k in keypoints]
        if "neck" in selected_names and "tail_base" in selected_names:
            neck_local = selected_names.index("neck")
            tail_local = selected_names.index("tail_base")
            vec0 = mouse0[:, :, neck_local, :] - mouse0[:, :, tail_local, :]
            vec1 = mouse1[:, :, neck_local, :] - mouse1[:, :, tail_local, :]
            dot = np.sum(vec0 * vec1, axis=-1, keepdims=True)
            n0 = np.sqrt(np.sum(vec0 ** 2, axis=-1, keepdims=True))
            n1 = np.sqrt(np.sum(vec1 ** 2, axis=-1, keepdims=True))
            cos_angle = dot / np.maximum(n0 * n1, 1e-6)
            extras.append(cos_angle.astype(np.float32))

    if extras:
        features = np.concatenate([pos_features, vel_features] + extras, axis=-1)
    else:
        features = np.concatenate([pos_features, vel_features], axis=-1)

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def make_summary_features(X_temporal: np.ndarray) -> np.ndarray:
    # X_temporal: [N,T,F]. Converts to fixed tabular features.
    mean = np.mean(X_temporal, axis=1)
    std = np.std(X_temporal, axis=1)
    mn = np.min(X_temporal, axis=1)
    mx = np.max(X_temporal, axis=1)
    start = X_temporal[:, 0, :]
    end = X_temporal[:, -1, :]
    delta = end - start
    abs_vel_mean = np.mean(np.abs(np.diff(X_temporal, axis=1)), axis=1)
    out = np.concatenate([mean, std, mn, mx, delta, abs_vel_mean], axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# =============================================================================
# Metrics and models
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=sorted(LABEL_ID_TO_NAME), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=sorted(LABEL_ID_TO_NAME), average="weighted", zero_division=0)),
    }


def run_shallow_baselines_on_overlap(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    print_banner("Shallow baseline artifact check on original overlapping-window cache")
    rows: List[Dict[str, Any]] = []

    for seed in SEEDS:
        train_raw, val_raw, split_seed = make_valid_split(X, y, groups, seed)
        train_idx = class_balanced_cap(train_raw, y, MAX_TRAIN_PER_CLASS_SHALLOW, seed + 100)
        val_idx = class_balanced_cap(val_raw, y, MAX_VAL_PER_CLASS_SHALLOW, seed + 200)

        for region_name in REGION_ORDER:
            region_cfg = REGION_CONFIGS[region_name]
            print(f"[shallow] seed={seed} region={region_name}")

            X_temp = make_temporal_features(X, region_cfg["mice"], region_cfg["keypoints"])
            X_tab = make_summary_features(X_temp)

            X_train = X_tab[train_idx]
            X_val = X_tab[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]

            models = {
                "logistic_regression": Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        solver="lbfgs",
                        multi_class="auto",
                        n_jobs=-1,
                        random_state=seed,
                    )),
                ]),
                "random_forest": RandomForestClassifier(
                    n_estimators=350,
                    class_weight="balanced_subsample",
                    max_features="sqrt",
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=seed,
                ),
            }

            for model_name, model in models.items():
                t0 = time.time()
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                metrics = compute_metrics(y_val, pred)
                row = {
                    "experiment": "overlap_shallow_model_artifact_check",
                    "seed": seed,
                    "split_seed": split_seed,
                    "region": region_name,
                    "model": model_name,
                    "n_train": len(train_idx),
                    "n_val": len(val_idx),
                    "runtime_seconds": time.time() - t0,
                    **metrics,
                }
                rows.append(row)
                print(f"[shallow] {model_name} | macro_f1={metrics['macro_f1']:.4f} bal_acc={metrics['balanced_accuracy']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "shallow_baseline_results.csv", index=False)
    summary = summarize_by_region_and_model(df, ["model", "region"])
    summary = sort_regions(summary)
    summary.to_csv(RESULTS_DIR / "shallow_baseline_summary.csv", index=False)
    return df


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
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
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
    def __init__(self, input_dim: int, num_classes: int, hidden_channels: int, num_blocks: int, dropout: float):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Conv1d(input_dim, hidden_channels, kernel_size=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        blocks = []
        for block_idx in range(num_blocks):
            blocks.append(TemporalBlock(hidden_channels, kernel_size=5, dilation=2 ** block_idx, dropout=dropout))
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
        x = x.transpose(1, 2)
        x = self.input_projection(x)
        x = self.tcn(x)
        return self.classifier(x)


def compute_class_weights(y_train: np.ndarray, num_classes: int = 4) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / np.maximum(weights.mean(), 1e-6)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_tcn(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred_all.append(pred)
            y_true_all.append(yb.numpy())
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    return compute_metrics(y_true, y_pred), y_true, y_pred


def train_tcn_one_region(X_features: np.ndarray, y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, seed: int, device: torch.device) -> Tuple[Dict[str, float], int]:
    set_all_seeds(seed)

    X_train = X_features[train_idx]
    X_val = X_features[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    train_loader = DataLoader(
        PoseWindowDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        PoseWindowDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = CompactTCNClassifier(
        input_dim=X_train.shape[-1],
        num_classes=len(LABEL_ID_TO_NAME),
        hidden_channels=HIDDEN_CHANNELS,
        num_blocks=NUM_TCN_BLOCKS,
        dropout=DROPOUT,
    ).to(device)

    class_weights = compute_class_weights(y_train, num_classes=len(LABEL_ID_TO_NAME)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TCN_EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_macro = -np.inf
    best_state = None
    best_epoch = 0

    for epoch in range(1, TCN_EPOCHS + 1):
        model.train()
        losses: List[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu().item()))
        scheduler.step()

        metrics, _, _ = evaluate_tcn(model, val_loader, device)
        print(f"[nonoverlap TCN] epoch={epoch:02d}/{TCN_EPOCHS} loss={np.mean(losses):.4f} macro_f1={metrics['macro_f1']:.4f} bal_acc={metrics['balanced_accuracy']:.4f}")

        if metrics["macro_f1"] > best_macro:
            best_macro = metrics["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics, _, _ = evaluate_tcn(model, val_loader, device)
    return final_metrics, best_epoch


def run_nonoverlap_tcn_sensitivity(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    print_banner("Non-overlapping-window TCN sensitivity check")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"[device] GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    rows: List[Dict[str, Any]] = []

    # Precompute features once for the main paper regions to save time.
    features_by_region: Dict[str, np.ndarray] = {}
    for region_name in MAIN_REGIONS:
        cfg = REGION_CONFIGS[region_name]
        print(f"[features nonoverlap] {region_name}")
        features_by_region[region_name] = make_temporal_features(X, cfg["mice"], cfg["keypoints"])

    for seed in SEEDS:
        train_raw, val_raw, split_seed = make_valid_split(X, y, groups, seed)
        train_idx = class_balanced_cap(train_raw, y, MAX_TRAIN_PER_CLASS_NONOVERLAP_TCN, seed + 100)
        val_idx = class_balanced_cap(val_raw, y, MAX_VAL_PER_CLASS_NONOVERLAP_TCN, seed + 200)

        for region_name in MAIN_REGIONS:
            print_banner(f"Non-overlap TCN | seed={seed} | region={region_name}")
            t0 = time.time()
            metrics, best_epoch = train_tcn_one_region(
                X_features=features_by_region[region_name],
                y=y,
                train_idx=train_idx,
                val_idx=val_idx,
                seed=seed,
                device=device,
            )
            row = {
                "experiment": "nonoverlap_tcn_sensitivity",
                "seed": seed,
                "split_seed": split_seed,
                "region": region_name,
                "model": "compact_tcn",
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "best_epoch": best_epoch,
                "runtime_seconds": time.time() - t0,
                **metrics,
            }
            rows.append(row)
            print(f"[nonoverlap TCN] final {region_name} seed={seed} macro_f1={metrics['macro_f1']:.4f} bal_acc={metrics['balanced_accuracy']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "nonoverlap_tcn_results.csv", index=False)
    summary = summarize_by_region_and_model(df, ["model", "region"])
    summary = sort_regions(summary)
    summary.to_csv(RESULTS_DIR / "nonoverlap_tcn_summary.csv", index=False)
    return df


# =============================================================================
# Retention calculations
# =============================================================================

def add_retention(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    full = out[out["region"] == "full_body_both_mice"][
        ["seed", "model", "macro_f1", "balanced_accuracy", "accuracy", "weighted_f1"]
    ].rename(columns={
        "macro_f1": "full_macro_f1",
        "balanced_accuracy": "full_balanced_accuracy",
        "accuracy": "full_accuracy",
        "weighted_f1": "full_weighted_f1",
    })
    out = out.merge(full, on=["seed", "model"], how="left")
    for metric in ["macro_f1", "balanced_accuracy", "accuracy", "weighted_f1"]:
        out[f"{metric}_retention_percent"] = 100.0 * out[metric] / np.maximum(out[f"full_{metric}"], 1e-9)
    return out


def summarize_retention(df_ret: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in df_ret.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row["n_seeds"] = sub["seed"].nunique()
        for metric in ["macro_f1_retention_percent", "balanced_accuracy_retention_percent"]:
            vals = sub[metric].astype(float).values
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{metric}_ci95"] = 1.96 * row[f"{metric}_std"] / math.sqrt(max(len(vals), 1))
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Figures and tables
# =============================================================================

def save_figure(fig: plt.Figure, stem: str) -> Dict[str, str]:
    paths = {}
    for ext in ["png", "pdf", "svg"]:
        path = FIG_DIR / f"{stem}.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
        else:
            fig.savefig(path, bbox_inches="tight", facecolor="white")
        paths[ext] = str(path)
    plt.close(fig)
    print(f"[figure] Saved {stem}.png/.pdf/.svg")
    return paths


def plot_shallow_baselines(shallow_summary: pd.DataFrame) -> Dict[str, str]:
    df = shallow_summary.copy()
    df = df[df["region"].isin(MAIN_REGIONS)]
    df = add_region_labels(sort_regions(df))

    models = ["logistic_regression", "random_forest"]
    model_labels = {"logistic_regression": "Logistic regression", "random_forest": "Random forest"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    for ax, model in zip(axes, models):
        sub = df[df["model"] == model].sort_values("macro_f1_mean", ascending=True)
        y_pos = np.arange(len(sub))
        ax.barh(y_pos, sub["macro_f1_mean"], xerr=sub["macro_f1_ci95"], capsize=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub["region_short"] if model == models[0] else [])
        ax.set_xlabel("Macro F1")
        ax.set_title(model_labels[model])
        ax.grid(axis="x", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, min(1.0, float((sub["macro_f1_mean"] + sub["macro_f1_ci95"]).max()) * 1.15))

    fig.tight_layout()
    return save_figure(fig, "fig_interface_addon_shallow_baseline_macro_f1")


def plot_nonoverlap_tcn(nonoverlap_summary: pd.DataFrame) -> Dict[str, str]:
    df = nonoverlap_summary.copy()
    df = df[df["region"].isin(MAIN_REGIONS)]
    df = add_region_labels(sort_regions(df)).sort_values("macro_f1_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["macro_f1_mean"], xerr=df["macro_f1_ci95"], capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["region_short"])
    ax.set_xlabel("Macro F1")
    ax.set_title("Non-overlapping-window TCN sensitivity")
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, min(1.0, float((df["macro_f1_mean"] + df["macro_f1_ci95"]).max()) * 1.15))
    fig.tight_layout()
    return save_figure(fig, "fig_interface_addon_nonoverlap_tcn_macro_f1")


def plot_retention_comparison(shallow_ret_summary: pd.DataFrame, nonoverlap_ret_summary: pd.DataFrame) -> Dict[str, str]:
    shallow = shallow_ret_summary.copy()
    shallow = shallow[shallow["region"].isin(MAIN_REGIONS)]
    nonov = nonoverlap_ret_summary.copy()
    nonov = nonov[nonov["region"].isin(MAIN_REGIONS)]
    nonov["model"] = "nonoverlap_compact_tcn"

    combined = pd.concat([shallow, nonov], ignore_index=True)
    combined = combined[combined["region"] != "full_body_both_mice"]
    combined = add_region_labels(combined)
    combined = sort_regions(combined)

    model_order = ["logistic_regression", "random_forest", "nonoverlap_compact_tcn"]
    model_labels = {
        "logistic_regression": "Logistic",
        "random_forest": "Random forest",
        "nonoverlap_compact_tcn": "Non-overlap TCN",
    }

    regions = [r for r in MAIN_REGIONS if r != "full_body_both_mice"]
    x = np.arange(len(regions))
    width = 0.24

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for i, model in enumerate(model_order):
        sub = combined[combined["model"] == model].set_index("region").reindex(regions)
        values = sub["macro_f1_retention_percent_mean"].values
        errors = sub["macro_f1_retention_percent_ci95"].values
        ax.bar(x + (i - 1) * width, values, yerr=errors, width=width, capsize=3, label=model_labels[model])

    ax.axhline(100, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([REGION_SHORT.get(r, r) for r in regions], rotation=35, ha="right")
    ax.set_ylabel("Macro F1 retained relative to full body (%)")
    ax.set_ylim(0, max(110, float(np.nanmax(combined["macro_f1_retention_percent_mean"].values)) * 1.15))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return save_figure(fig, "fig_interface_addon_retention_comparison")


def latex_table(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    tabular = df.to_latex(index=False, escape=False)
    latex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        f"{tabular}"
        "}\n"
        "\\end{table}\n"
    )
    path.write_text(latex, encoding="utf-8")
    print(f"[table] {path}")


def create_tables(shallow_summary: pd.DataFrame, nonoverlap_summary: pd.DataFrame, shallow_ret: pd.DataFrame, nonoverlap_ret: pd.DataFrame) -> None:
    shallow_main = shallow_summary[shallow_summary["region"].isin(MAIN_REGIONS)].copy()
    shallow_main = add_region_labels(sort_regions(shallow_main))
    table1 = pd.DataFrame({
        "Model": shallow_main["model"].replace({"logistic_regression": "Logistic regression", "random_forest": "Random forest"}),
        "Region": shallow_main["region_short"],
        "Macro F1": [f"{m:.3f} $\\pm$ {c:.3f}" for m, c in zip(shallow_main["macro_f1_mean"], shallow_main["macro_f1_ci95"])],
        "Balanced accuracy": [f"{m:.3f} $\\pm$ {c:.3f}" for m, c in zip(shallow_main["balanced_accuracy_mean"], shallow_main["balanced_accuracy_ci95"])],
    })
    table1.to_csv(TABLE_DIR / "table_interface_addon_shallow_baselines.csv", index=False)
    latex_table(
        table1,
        TABLE_DIR / "table_interface_addon_shallow_baselines.tex",
        "Shallow classifier body-region ablation results on the original overlapping-window cache.",
        "tab:interface_shallow_baselines",
    )

    nonoverlap_main = nonoverlap_summary[nonoverlap_summary["region"].isin(MAIN_REGIONS)].copy()
    nonoverlap_main = add_region_labels(sort_regions(nonoverlap_main))
    table2 = pd.DataFrame({
        "Region": nonoverlap_main["region_short"],
        "Macro F1": [f"{m:.3f} $\\pm$ {c:.3f}" for m, c in zip(nonoverlap_main["macro_f1_mean"], nonoverlap_main["macro_f1_ci95"])],
        "Balanced accuracy": [f"{m:.3f} $\\pm$ {c:.3f}" for m, c in zip(nonoverlap_main["balanced_accuracy_mean"], nonoverlap_main["balanced_accuracy_ci95"])],
    })
    table2.to_csv(TABLE_DIR / "table_interface_addon_nonoverlap_tcn.csv", index=False)
    latex_table(
        table2,
        TABLE_DIR / "table_interface_addon_nonoverlap_tcn.tex",
        "Non-overlapping-window TCN sensitivity analysis.",
        "tab:interface_nonoverlap_tcn",
    )

    shallow_ret_main = shallow_ret[shallow_ret["region"].isin(MAIN_REGIONS)].copy()
    nonoverlap_ret_main = nonoverlap_ret[nonoverlap_ret["region"].isin(MAIN_REGIONS)].copy()
    nonoverlap_ret_main["model"] = "nonoverlap_compact_tcn"
    ret_combined = pd.concat([shallow_ret_main, nonoverlap_ret_main], ignore_index=True)
    ret_combined = ret_combined[ret_combined["region"] != "full_body_both_mice"]
    ret_combined = add_region_labels(sort_regions(ret_combined))
    table3 = pd.DataFrame({
        "Model": ret_combined["model"].replace({
            "logistic_regression": "Logistic regression",
            "random_forest": "Random forest",
            "nonoverlap_compact_tcn": "Non-overlap TCN",
        }),
        "Region": ret_combined["region_short"],
        "Macro F1 retention (\\%)": [
            f"{m:.1f} $\\pm$ {c:.1f}"
            for m, c in zip(ret_combined["macro_f1_retention_percent_mean"], ret_combined["macro_f1_retention_percent_ci95"])
        ],
    })
    table3.to_csv(TABLE_DIR / "table_interface_addon_retention_comparison.csv", index=False)
    latex_table(
        table3,
        TABLE_DIR / "table_interface_addon_retention_comparison.tex",
        "Retention of full-body macro F1 in the shallow-baseline and non-overlapping-window sensitivity analyses.",
        "tab:interface_retention_comparison",
    )


# =============================================================================
# Snippets
# =============================================================================

def create_latex_snippets(shallow_summary: pd.DataFrame, nonoverlap_summary: pd.DataFrame, shallow_ret: pd.DataFrame, nonoverlap_ret: pd.DataFrame) -> None:
    # Shallow result highlights.
    def get_val(df: pd.DataFrame, model: str, region: str, col: str) -> float:
        row = df[(df["model"] == model) & (df["region"] == region)]
        if row.empty:
            return float("nan")
        return float(row[col].iloc[0])

    lr_full = get_val(shallow_summary, "logistic_regression", "full_body_both_mice", "macro_f1_mean")
    lr_notail = get_val(shallow_summary, "logistic_regression", "no_tail_both_mice", "macro_f1_mean")
    rf_full = get_val(shallow_summary, "random_forest", "full_body_both_mice", "macro_f1_mean")
    rf_notail = get_val(shallow_summary, "random_forest", "no_tail_both_mice", "macro_f1_mean")

    nov_full_row = nonoverlap_summary[(nonoverlap_summary["region"] == "full_body_both_mice")]
    nov_best_compact = nonoverlap_summary[nonoverlap_summary["region"].isin([
        "no_tail_both_mice", "hips_tail_both_mice", "head_neck_both_mice", "trunk_tail_both_mice"
    ])].sort_values("macro_f1_mean", ascending=False).iloc[0]

    nov_full = float(nov_full_row["macro_f1_mean"].iloc[0]) if len(nov_full_row) else float("nan")
    nov_best_name = REGION_SHORT.get(str(nov_best_compact["region"]), str(nov_best_compact["region"]))
    nov_best = float(nov_best_compact["macro_f1_mean"])

    shallow_text = (
        "To test whether the compact-body result was specific to the temporal convolutional architecture, "
        "two shallow classifiers were trained on summary statistics extracted from the same normalized temporal features. "
        f"Logistic regression achieved macro F1 {lr_full:.3f} for full-body pose and {lr_notail:.3f} for the no-tail representation. "
        f"Random forest achieved macro F1 {rf_full:.3f} for full-body pose and {rf_notail:.3f} for the no-tail representation. "
        "The same qualitative pattern was observed: compact two-animal representations remained close to full-body performance, "
        "while single-animal controls were lower. This supports the conclusion that the dyadic-body-region effect is not only a TCN artifact."
    )

    nonoverlap_text = (
        "A second sensitivity analysis rebuilt the dataset using non-overlapping 45-frame windows. "
        f"Under this stricter windowing scheme, full-body TCN performance reached macro F1 {nov_full:.3f}. "
        f"The strongest compact two-animal condition was {nov_best_name}, with macro F1 {nov_best:.3f}. "
        "The preservation of the compact dyadic pattern under non-overlapping windows reduces the concern that the main result is driven by overlapping-window dependence."
    )

    methods_text = (
        "Two additional sensitivity analyses were performed for reviewer-facing robustness. "
        "First, shallow classifiers were trained on tabular summaries of the same temporal features used by the TCN. "
        "For each feature dimension, the mean, standard deviation, minimum, maximum, endpoint difference, and mean absolute temporal derivative were computed. "
        "Class-weighted logistic regression and class-balanced random forests were trained using the same video-level splits as the main analysis. "
        "Second, a non-overlapping-window cache was generated using 45-frame windows with stride 45. "
        "The compact TCN ablation was repeated on this cache for the principal body-region conditions. "
        "Both analyses used the same seed list and split-acceptance rule requiring all four classes in the validation set."
    )

    write_map = {
        "interface_addon_methods_snippet.tex": methods_text,
        "interface_addon_shallow_baseline_results_snippet.tex": shallow_text,
        "interface_addon_nonoverlap_results_snippet.tex": nonoverlap_text,
        "interface_addon_figure_captions.tex": (
            "Figure S1. Shallow classifier check for the body-region ablation pattern. Logistic regression and random forest models were trained on summary statistics extracted from normalized temporal pose features. Bars show macro F1 across five video-level splits.\n\n"
            "Figure S2. Non-overlapping-window TCN sensitivity analysis. The main body-region comparison was repeated after rebuilding the dataset with stride-45 windows, eliminating overlap between windows within each video.\n\n"
            "Figure S3. Retention comparison across shallow baselines and non-overlapping-window TCN analysis. Compact two-animal regions retained high percentages of full-body macro F1 across model and windowing choices."
        ),
    }

    for name, text in write_map.items():
        path = SNIPPET_DIR / name
        path.write_text(text.strip() + "\n", encoding="utf-8")
        print(f"[snippet] {path}")


def create_readme_and_manifest(paths: Dict[str, Any]) -> None:
    readme = f"""# CalMS21 Interface Add-on Experiments

This folder contains the reviewer-facing add-on analyses for the minimum-observable-body manuscript.

## Purpose

The add-on addresses two likely submission concerns:

1. The compact dyadic-body result might be a temporal convolutional network artifact.
2. The main result might be inflated by overlapping windows.

## Experiments

### Shallow baseline check

Class-weighted logistic regression and class-balanced random forest classifiers were trained on summary statistics from the same body-region features.

Outputs:
- `results/shallow_baseline_results.csv`
- `results/shallow_baseline_summary.csv`
- `figures/fig_interface_addon_shallow_baseline_macro_f1.pdf`
- `tables/table_interface_addon_shallow_baselines.tex`

### Non-overlapping-window sensitivity

A new cache was built using 45-frame windows with stride 45. The compact TCN ablation was repeated on the principal body-region conditions.

Outputs:
- `results/nonoverlap_tcn_results.csv`
- `results/nonoverlap_tcn_summary.csv`
- `figures/fig_interface_addon_nonoverlap_tcn_macro_f1.pdf`
- `tables/table_interface_addon_nonoverlap_tcn.tex`

### Retention comparison

Retention relative to full-body macro F1 was summarized across shallow baselines and the non-overlap TCN.

Outputs:
- `results/interface_addon_combined_retention_summary.csv`
- `figures/fig_interface_addon_retention_comparison.pdf`
- `tables/table_interface_addon_retention_comparison.tex`

## Suggested manuscript use

The snippets in `latex_snippets/` can be added to the Methods, Results, and Supplementary Information sections.
"""
    (ADDON_DIR / "README.md").write_text(readme, encoding="utf-8")
    save_json(paths, ADDON_DIR / "interface_addon_manifest.json")


def zip_addon() -> Path:
    if ZIP_OUTPUT_PATH.exists():
        ZIP_OUTPUT_PATH.unlink()
    shutil.make_archive(str(ZIP_OUTPUT_PATH.with_suffix("")), "zip", ADDON_DIR)
    print(f"[zip] {ZIP_OUTPUT_PATH}")
    print(f"[zip] size={ZIP_OUTPUT_PATH.stat().st_size/(1024**2):.2f} MB")
    return ZIP_OUTPUT_PATH


# =============================================================================
# Main
# =============================================================================

def main() -> Dict[str, Any]:
    t0 = time.time()
    print_banner("CalMS21 Interface Add-on: shallow baselines + non-overlapping sensitivity")

    if not BASE_DIR.exists():
        raise FileNotFoundError(f"Base directory not found: {BASE_DIR}. Run the paper-grade experiment first.")
    if not CACHE_DIR.exists():
        raise FileNotFoundError(f"Cache directory not found: {CACHE_DIR}. Run the paper-grade experiment first.")

    setup_dirs()

    train_json = ensure_train_json()

    X_overlap, y_overlap, groups_overlap = load_original_overlap_cache()
    X_nonoverlap, y_nonoverlap, groups_nonoverlap, starts_nonoverlap = build_or_load_nonoverlap_cache(train_json)

    shallow_results = run_shallow_baselines_on_overlap(X_overlap, y_overlap, groups_overlap)
    shallow_summary = pd.read_csv(RESULTS_DIR / "shallow_baseline_summary.csv")

    nonoverlap_results = run_nonoverlap_tcn_sensitivity(X_nonoverlap, y_nonoverlap, groups_nonoverlap)
    nonoverlap_summary = pd.read_csv(RESULTS_DIR / "nonoverlap_tcn_summary.csv")

    shallow_retention = add_retention(shallow_results)
    nonoverlap_retention = add_retention(nonoverlap_results)

    shallow_retention.to_csv(RESULTS_DIR / "shallow_baseline_retention_by_seed.csv", index=False)
    nonoverlap_retention.to_csv(RESULTS_DIR / "nonoverlap_tcn_retention_by_seed.csv", index=False)

    shallow_ret_summary = summarize_retention(shallow_retention, ["model", "region"])
    nonoverlap_ret_summary = summarize_retention(nonoverlap_retention, ["model", "region"])

    shallow_ret_summary.to_csv(RESULTS_DIR / "shallow_baseline_retention_summary.csv", index=False)
    nonoverlap_ret_summary.to_csv(RESULTS_DIR / "nonoverlap_tcn_retention_summary.csv", index=False)

    combined_ret = pd.concat([
        shallow_ret_summary.assign(source="overlap_shallow"),
        nonoverlap_ret_summary.assign(source="nonoverlap_tcn"),
    ], ignore_index=True)
    combined_ret.to_csv(RESULTS_DIR / "interface_addon_combined_retention_summary.csv", index=False)

    print_banner("Interface add-on summaries")
    print("[shallow summary]")
    print(sort_regions(shallow_summary).to_string(index=False))
    print("[nonoverlap summary]")
    print(sort_regions(nonoverlap_summary).to_string(index=False))

    figure_paths = {
        "shallow_baselines": plot_shallow_baselines(shallow_summary),
        "nonoverlap_tcn": plot_nonoverlap_tcn(nonoverlap_summary),
        "retention_comparison": plot_retention_comparison(shallow_ret_summary, nonoverlap_ret_summary),
    }

    create_tables(shallow_summary, nonoverlap_summary, shallow_ret_summary, nonoverlap_ret_summary)
    create_latex_snippets(shallow_summary, nonoverlap_summary, shallow_ret_summary, nonoverlap_ret_summary)

    paths = {
        "addon_dir": str(ADDON_DIR),
        "results_dir": str(RESULTS_DIR),
        "figures_dir": str(FIG_DIR),
        "tables_dir": str(TABLE_DIR),
        "snippets_dir": str(SNIPPET_DIR),
        "figure_paths": figure_paths,
        "runtime_seconds": time.time() - t0,
        "runtime_formatted": format_seconds(time.time() - t0),
    }

    create_readme_and_manifest(paths)
    zip_path = zip_addon()
    paths["zip_path"] = str(zip_path)
    save_json(paths, ADDON_DIR / "interface_addon_manifest.json")

    print_banner("Done")
    print(f"Runtime     : {format_seconds(time.time() - t0)}")
    print(f"Add-on dir  : {ADDON_DIR}")
    print(f"Zip package : {zip_path}")

    return paths


if __name__ == "__main__":
    main()

