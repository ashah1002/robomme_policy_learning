"""Convert per-frame .npy feature files to per-episode .bin files for fast loading.

Each resolution (8x8, 4x4, 2x2) is stored in a separate directory so only the
needed resolution is read during training.

Output layout (features_bin/):
  image_emb_8x8/episode_{idx}.bin  — bf16 raw bytes, shape (T, V, 64, 2048) per episode
  image_emb_4x4/episode_{idx}.bin  — bf16 raw bytes, shape (T, V, 16, 2048)
  image_emb_2x2/episode_{idx}.bin  — bf16 raw bytes, shape (T, V, 4,  2048)
  pos_emb_8x8/episode_{idx}.bin    — f32  raw bytes, shape (T, V, 64, 768)
  pos_emb_4x4/episode_{idx}.bin    — f32  raw bytes, shape (T, V, 16, 768)
  pos_emb_2x2/episode_{idx}.bin    — f32  raw bytes, shape (T, V, 4,  768)
  state/episode_{idx}.bin          — f32  raw bytes, shape (T, state_dim)
  kept_indices/episode_{idx}.json  — copied from original features/ dir (used by token dropping)
  metadata.json                    — {episode_name: {frames, num_views, state_dim}}

NOTE: pos_emb MUST stay in f32.  bf16 causes distinct positional embeddings to
collapse onto the same value (the spacing between consecutive pos values at
frame indices near `max_steps` falls below bf16 resolution), which breaks the
temporal signal the model relies on.

Usage:
    uv run scripts/convert_features_to_bin.py \\
        --features_dir data/robomme_preprocessed_data/features \\
        --output_dir data/robomme_preprocessed_data/features_bin
"""

import argparse
import json
import os
import shutil

import numpy as np
import torch


IMG_DIM = 2048
POS_DIM = 768
RESOLUTIONS = {
    "8x8": 64,
    "4x4": 16,
    "2x2": 4,
}
SUBDIRS = (
    [f"image_emb_{r}" for r in RESOLUTIONS]
    + [f"pos_emb_{r}" for r in RESOLUTIONS]
    + ["state", "kept_indices"]
)


def convert_episode(epis_dir, output_dir):
    epis_name = os.path.basename(epis_dir)

    # Collect .npy files sorted by step index
    npy_files = {}
    for f in os.listdir(epis_dir):
        if f.startswith("token_emb_") and f.endswith(".npy"):
            step = int(f.replace("token_emb_", "").replace(".npy", ""))
            npy_files[step] = os.path.join(epis_dir, f)

    if not npy_files:
        return None

    T = max(npy_files.keys()) + 1
    if len(npy_files) != T:
        print(f"  WARNING: {epis_name} has gaps — expected {T} files, found {len(npy_files)}, skipping")
        return None

    # Peek first frame for shapes
    sample = np.load(npy_files[0], allow_pickle=True).item()
    V = np.asarray(sample["image_emb_8x8"]).shape[0] # num_views = 1 in all our experiments, we use front view only for memory
    state_dim = np.asarray(sample["state_emb"]).shape[0]

    # Pre-allocate per-resolution arrays
    img_arrs = {r: np.zeros((T, V, tok, IMG_DIM), dtype=np.float32) for r, tok in RESOLUTIONS.items()}
    pos_arrs = {r: np.zeros((T, V, tok, POS_DIM), dtype=np.float32) for r, tok in RESOLUTIONS.items()}
    state_arr = np.zeros((T, state_dim), dtype=np.float32)

    for step in range(T):
        feat = np.load(npy_files[step], allow_pickle=True).item()
        for r in RESOLUTIONS:
            img_arrs[r][step] = np.asarray(feat[f"image_emb_{r}"], dtype=np.float32)
            pos_arrs[r][step] = np.asarray(feat[f"pos_emb_{r}"], dtype=np.float32)
        state_arr[step] = np.asarray(feat["state_emb"], dtype=np.float32)[:state_dim]

    # Write image embeddings as bf16 (source data is already bf16 — round-trip exact)
    for r in RESOLUTIONS:
        img_bf16 = torch.from_numpy(img_arrs[r]).to(torch.bfloat16)
        img_bf16.contiguous().view(torch.int16).numpy().tofile(
            os.path.join(output_dir, f"image_emb_{r}", f"{epis_name}.bin"))

    # Write pos embeddings as f32 — must NOT be downcast to bf16 (see module docstring)
    for r in RESOLUTIONS:
        pos_arrs[r].tofile(
            os.path.join(output_dir, f"pos_emb_{r}", f"{epis_name}.bin"))

    # Write state as f32
    state_arr.tofile(os.path.join(output_dir, "state", f"{epis_name}.bin"))

    # Copy kept_indices.json (used by perceptual token-dropping)
    src_kept = os.path.join(epis_dir, "kept_indices.json")
    if os.path.exists(src_kept):
        shutil.copyfile(
            src_kept,
            os.path.join(output_dir, "kept_indices", f"{epis_name}.json"),
        )

    return epis_name, T, V, state_dim


def main():
    parser = argparse.ArgumentParser(description="Convert .npy features to .bin format")
    parser.add_argument("--features_dir", type=str, required=True,
                        help="Input features/ directory with episode_* subdirs")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for .bin files (e.g. features_bin/)")
    args = parser.parse_args()

    for subdir in SUBDIRS:
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)

    # Find episode directories
    epis_dirs = sorted(
        [os.path.join(args.features_dir, d)
         for d in os.listdir(args.features_dir)
         if d.startswith("episode_") and os.path.isdir(os.path.join(args.features_dir, d))],
        key=lambda x: int(os.path.basename(x).split("_")[1]),
    )
    print(f"Found {len(epis_dirs)} episodes in {args.features_dir}")

    metadata = {}
    for i, epis_dir in enumerate(epis_dirs):
        result = convert_episode(epis_dir, args.output_dir)
        if result is None:
            continue
        name, T, V, state_dim = result
        metadata[name] = {"frames": T, "num_views": V, "state_dim": state_dim}
        if (i + 1) % 50 == 0 or i == len(epis_dirs) - 1:
            print(f"[{i + 1}/{len(epis_dirs)}] {name}: {T} frames")

    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Converted {len(metadata)} episodes to {args.output_dir}")


if __name__ == "__main__":
    main()
