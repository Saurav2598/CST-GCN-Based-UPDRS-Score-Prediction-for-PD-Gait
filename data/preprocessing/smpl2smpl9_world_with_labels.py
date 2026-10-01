"""Convert CARE-PD SMPL walks to a labeled nine-joint gait skeleton.

Run from anywhere (the CARE-PD repository must contain this file at
data/preprocessing/smpl2smpl9_world_with_labels.py):

    python data/preprocessing/smpl2smpl9_world_with_labels.py -db BMCLab
    python data/preprocessing/smpl2smpl9_world_with_labels.py --all
    python data/preprocessing/smpl2smpl9_world_with_labels.py --all \
        --root-normalization first_frame

Each non-reserved NPZ key is a walk of shape (T, 9, 3), float32. The joint
order matches the CST-GCN graph, with native SMPL indices in parentheses:
    0 pelvis (0), 1 left hip (1), 2 left knee (4), 3 left ankle (7),
    4 left foot (10), 5 right hip (2), 6 right knee (5),
    7 right ankle (8), 8 right foot (11).

When scores exist, __UPDRS_GAIT__ is a scalar JSON string mapping walk keys
to {UPDRS_GAIT, subject_id, walk_id}. Unlabeled walks remain in the archive.

Normalization modes:
  per_frame   Subtract the pelvis XYZ in every frame (default; CST-GCN style).
  first_frame Put the lowest selected SMPL joint at Y=0, then subtract only
              the first pelvis XZ position. This retains travel and height.

SMPL foot joints are internal joints, not sole contact markers. In the
first_frame mode, Y=0 is therefore only an approximate floor reference.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from smplx.body_models import SMPL
from tqdm import tqdm

# Resolve CARE-PD imports when invoked by absolute path from another directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from const.const import _DEVICE, DATASET_ORIGINAL_FPS, SUPPORTED_DATASETS
from const import path
from data.preprocessing.trajectory_correction import (
    transform_seq_so_it_has_no_slope_AMASS,
)


SMPL9_INDICES = [0, 1, 4, 7, 10, 2, 5, 8, 11]
SLOPE_CORRECTION_DATASETS = {"T-LTC", "T-SDU", "T-SDU-PD"}
TARGET_FPS = 30
MIN_OUTPUT_FRAMES = 30
LABELS_KEY = "__UPDRS_GAIT__"


def label_for_walk(smpl_data, walk_name, subject_id, walk_id):
    """Return the original walk's UPDRS-GAIT score when it is 0, 1, 2 or 3."""
    score = smpl_data.get("UPDRS_GAIT")
    if score is None:
        return None
    try:
        if isinstance(score, (bool, np.bool_)) or not np.isfinite(float(score)):
            raise ValueError
        score_num = float(score)
        if score_num not in (0.0, 1.0, 2.0, 3.0):
            raise ValueError
    except (TypeError, ValueError):
        print(f"\nWARNING: no valid UPDRS_GAIT for {walk_name}: {score!r}")
        return None
    return {
        "UPDRS_GAIT": int(score_num),
        "subject_id": str(subject_id),
        "walk_id": str(walk_id),
    }


def reconstruct_smpl24_world(smpl_model, sequence, down_sample_rate, phase):
    """Return native SMPL joints (T, 24, 3) with matched per-frame trans.

    The model is created with create_transl=False. Thus its output joints
    contain body pose and shape but not the walk's global translation.
    """
    poses = np.asarray(sequence["pose"], dtype=np.float32).reshape(-1, 24, 3)
    n_frames = poses.shape[0]
    translations = np.asarray(sequence["trans"], dtype=np.float32)
    if translations.shape != (n_frames, 3):
        raise ValueError(f"Expected trans shape ({n_frames}, 3), got {translations.shape}")

    betas = np.asarray(sequence["beta"], dtype=np.float32)
    if betas.ndim == 1:
        betas = np.broadcast_to(betas, (n_frames, betas.shape[0]))
    elif betas.ndim == 2 and betas.shape[0] == 1:
        betas = np.broadcast_to(betas, (n_frames, betas.shape[1]))
    elif betas.ndim != 2 or betas.shape[0] != n_frames:
        raise ValueError(f"Cannot match beta shape {betas.shape} to {n_frames} frames")

    # Identical slicing preserves pose/shape/translation frame alignment.
    poses = poses[phase::down_sample_rate]
    betas = betas[phase::down_sample_rate]
    translations = translations[phase::down_sample_rate]

    with torch.no_grad():
        pose_tensor = torch.as_tensor(poses, device=_DEVICE)
        beta_tensor = torch.as_tensor(betas.copy(), device=_DEVICE)
        trans_tensor = torch.as_tensor(translations, device=_DEVICE)
        output = smpl_model(
            betas=beta_tensor,
            global_orient=pose_tensor[:, 0, :],
            body_pose=pose_tensor[:, 1:, :].reshape(len(poses), -1),
            return_verts=False,
        )
        if output.joints.shape[1] < 24:
            raise ValueError(f"SMPL returned only {output.joints.shape[1]} joints")
        joints_world = output.joints[:, :24, :] + trans_tensor[:, None, :]
        return joints_world.cpu().numpy().astype(np.float32)


def align_initial_facing_yaw(smpl24_world, n_initial_frames=15):
    """Rotate the entire walk around +Y until initial body facing is +Z.

    Use full-SMPL right/left hips (2/1) and shoulders (17/16). Average the
    horizontal anatomical directions across early frames, and use the two
    foot-to-ankle vectors to disambiguate which side is forward. Apply a
    single proper rotation to every frame; never mirror left and right.
    """
    early = smpl24_world[: min(n_initial_frames, len(smpl24_world))]
    across = (early[:, 2] - early[:, 1]) + (early[:, 17] - early[:, 16])
    across[:, 1] = 0.0
    across_norm = np.linalg.norm(across, axis=1)
    valid = across_norm > 1e-6
    if not np.any(valid):
        raise ValueError("Cannot estimate facing: degenerate hip/shoulder axis")

    unit_across = across[valid] / across_norm[valid, None]
    across_mean = unit_across.mean(axis=0)
    if np.linalg.norm(across_mean) < 0.2:
        raise ValueError("Cannot estimate facing: conflicting early-frame directions")
    forward = np.cross(np.array([0.0, 1.0, 0.0]), across_mean)
    forward[1] = 0.0
    forward /= np.linalg.norm(forward)

    toe_direction = ((early[:, 10] - early[:, 7]) +
                     (early[:, 11] - early[:, 8])).mean(axis=0)
    toe_direction[1] = 0.0
    if np.linalg.norm(toe_direction) > 1e-6 and np.dot(forward, toe_direction) < 0:
        forward = -forward

    # R_y(-atan2(forward_x, forward_z)) maps the horizontal forward to +Z.
    yaw = np.arctan2(forward[0], forward[2])
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, 0.0, -s],
                         [0.0, 1.0, 0.0],
                         [s, 0.0, c]], dtype=np.float32)
    return smpl24_world @ rotation.T


def normalize_nine_joints(smpl9_world, mode):
    """Normalize a (T, 9, 3) walk according to the requested mode."""
    joints = np.asarray(smpl9_world, dtype=np.float32).copy()
    if mode == "per_frame":
        joints -= joints[:, 0:1, :].copy()
    elif mode == "first_frame":
        joints[:, :, 1] -= joints[:, :, 1].min()
        first_root_xz = joints[0, 0, [0, 2]].copy()
        joints[:, :, [0, 2]] -= first_root_xz
    else:
        raise ValueError(f"Unknown root-normalization mode: {mode}")
    return joints


def process_dataset(dataset, root_normalization="per_frame", facing_frames=15):
    project_root = Path(path.PROJECT_ROOT)
    data_file = project_root / "assets" / "datasets" / "raw" / f"{dataset}.pkl"
    model_file = (project_root / "data" / "preprocessing" / "common" /
                  "body_models" / "smpl" / "SMPL_NEUTRAL.pkl")
    if not data_file.is_file():
        raise FileNotFoundError(data_file)
    if not model_file.is_file():
        raise FileNotFoundError(model_file)

    output_dir = project_root / "assets" / "datasets" / "processed" / "smpl9" 
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{dataset}_smpl9_3d_30f_or_longer_{root_normalization}.npz"

    smpl_model = SMPL(
        model_path=str(model_file), num_betas=10, create_transl=False,
    ).to(_DEVICE)
    smpl_model.eval()
    all_smpls = joblib.load(data_file)
    original_fps = DATASET_ORIGINAL_FPS[dataset]
    down_sample_rate = max(1, int(original_fps / TARGET_FPS))
    print(f"\nDataset: {dataset}; input: {data_file}; output: {output_file}")
    print(f"FPS: {original_fps}; phase count: {down_sample_rate}; "
          f"normalization: {root_normalization}")

    result_world = {}
    result_labels = {}
    skipped_trimmed = skipped_short = failed = total_walks = 0

    for subject_id in tqdm(all_smpls, desc=f"{dataset} subjects"):
        for walk_id, sequence in all_smpls[subject_id].items():
            total_walks += 1
            if "Trimmed" in str(walk_id):
                skipped_trimmed += 1
                continue
            if np.asarray(sequence["pose"]).shape[0] < MIN_OUTPUT_FRAMES:
                skipped_short += 1
                continue

            for phase in range(down_sample_rate):
                walk_name = f"{subject_id}__{walk_id}"
                if down_sample_rate > 1:
                    walk_name += f"_down{phase}"
                try:
                    smpl24_world = reconstruct_smpl24_world(
                        smpl_model, sequence, down_sample_rate, phase,
                    )
                    if len(smpl24_world) < MIN_OUTPUT_FRAMES:
                        skipped_short += 1
                        continue

                    if dataset in SLOPE_CORRECTION_DATASETS:
                        # CARE-PD's AMASS function uses full SMPL indices and
                        # mutates its input root Y; pass an isolated copy.
                        smpl24_world = transform_seq_so_it_has_no_slope_AMASS(
                            smpl24_world.copy(),
                            n_frames_est_mov_dir=15,
                            window_size=90,
                            polynomial=4,
                        )

                    smpl24_world = align_initial_facing_yaw(
                        smpl24_world, n_initial_frames=facing_frames,
                    )
                    smpl9_world = normalize_nine_joints(
                        smpl24_world[:, SMPL9_INDICES, :], root_normalization,
                    )
                    if not np.isfinite(smpl9_world).all():
                        raise ValueError("Non-finite joint coordinates")
                    if walk_name in result_world or walk_name == LABELS_KEY:
                        raise ValueError(f"Duplicate or reserved walk key: {walk_name}")

                    result_world[walk_name] = smpl9_world
                    label = label_for_walk(sequence, walk_name, subject_id, walk_id)
                    if label is not None:
                        result_labels[walk_name] = label
                except Exception as exc:
                    failed += 1
                    print(f"\nWARNING: failed {walk_name}: {exc}")

    contents = dict(result_world)
    if result_labels:
        contents[LABELS_KEY] = np.array(json.dumps(result_labels, ensure_ascii=False))
    np.savez_compressed(output_file, **contents)

    print(f"Raw walks: {total_walks}; saved sequences: {len(result_world)}; "
          f"labeled sequences: {len(result_labels)}")
    print(f"Skipped Trimmed: {skipped_trimmed}; skipped short: {skipped_short}; "
          f"failed: {failed}")
    if result_world:
        first_key = next(iter(result_world))
        print(f"Example: {first_key} -> {result_world[first_key].shape}")
    print(f"Saved: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="CARE-PD .pkl -> labeled SMPL-native nine-joint .npz"
    )
    select = parser.add_mutually_exclusive_group(required=True)
    select.add_argument("-db", "--dataset", choices=SUPPORTED_DATASETS)
    select.add_argument("--all", action="store_true", help="Process all available cohort PKLs")
    parser.add_argument(
        "--root-normalization", choices=("per_frame", "first_frame"),
        default="per_frame",
    )
    parser.add_argument("--facing-frames", type=int, default=15)
    args = parser.parse_args()
    if args.facing_frames < 1:
        parser.error("--facing-frames must be positive")
    if args.all:
        datasets = [name for name in SUPPORTED_DATASETS if (
            Path(path.PROJECT_ROOT) / "assets" / "datasets" / "raw" / f"{name}.pkl"
        ).is_file()]
        if not datasets:
            parser.error("No supported cohort PKLs were found")
    else:
        datasets = [args.dataset]
    for dataset in datasets:
        process_dataset(dataset, args.root_normalization, args.facing_frames)


if __name__ == "__main__":
    main()
