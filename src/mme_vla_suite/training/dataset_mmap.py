"""RoboMME dataset with memory-mapped .bin loading for history features.

Drop-in replacement for RoboMMEDataset — overrides only _gather_history_feat.
Expects features_bin/ created by scripts/convert_features_to_bin.py.

Uses np.memmap for zero-copy virtual-memory access.  The OS pages data in/out
automatically — 500 GB of .bin files works fine on a 64-bit system even with
limited physical RAM.  Best performance on local SSD; works on NFS but may
be slower due to page-fault latency.
"""

import json
import math
import os

import ml_dtypes
import numpy as np

from mme_vla_suite.training.dataset import RoboMMEDataset

IMG_DIM = 2048
POS_DIM = 768
RESOLUTION_TOKENS = {"8x8": 64, "4x4": 16, "2x2": 4}

# Storage dtypes in features_bin/:
#   image_emb_*  : bf16  (matches the source .npy dtype; round-trip exact)
#   pos_emb_*    : f32   (bf16 collapses distinct positional values, breaks the model)
#   state        : f32


def _resolve_resolution(history_config):
    """Determine which spatial resolution this config needs."""
    if history_config is None:
        return None, None
    rep = history_config.representation_type
    if rep == "recurrent":
        return "8x8", 64
    if rep == "perceptual":
        if history_config.perceptual_memory.type == "token_dropping":
            return "8x8", 64
        tpi = history_config.token_per_image
        s = int(math.sqrt(tpi))
        key = f"{s}x{s}"
        return key, RESOLUTION_TOKENS[key]
    return None, None


class RoboMMEDatasetMmap(RoboMMEDataset):
    """Loads history features from per-episode .bin files via mmap."""

    def __init__(self, *args, bin_features_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bin_dir = bin_features_dir or os.path.join(
            self.dataset.dataset_path, "features_bin")

        meta_path = os.path.join(self.bin_dir, "metadata.json")
        assert os.path.exists(meta_path), f"No metadata.json in {self.bin_dir}"
        with open(meta_path) as f:
            self.bin_meta = json.load(f)

        self._res_key, self._res_tokens = _resolve_resolution(self.history_config)

        # Lazy mmap cache (per-worker after DataLoader fork)
        self._mmaps = {}

    def _get_mmaps(self, epis_idx):
        key = f"episode_{epis_idx}"
        if key not in self._mmaps:
            meta = self.bin_meta[key]
            T = meta["frames"]
            V = meta["num_views"]
            state_dim = meta["state_dim"]
            tokens = self._res_tokens
            res = self._res_key

            # Image: memmap directly as bfloat16 via ml_dtypes
            img_mm = np.memmap(
                os.path.join(self.bin_dir, f"image_emb_{res}", f"{key}.bin"),
                dtype=ml_dtypes.bfloat16, mode="r",
                shape=(T, V, tokens, IMG_DIM))

            pos_mm = np.memmap(
                os.path.join(self.bin_dir, f"pos_emb_{res}", f"{key}.bin"),
                dtype=np.float32, mode="r",
                shape=(T, V, tokens, POS_DIM))

            state_mm = np.memmap(
                os.path.join(self.bin_dir, "state", f"{key}.bin"),
                dtype=np.float32, mode="r",
                shape=(T, state_dim))

            self._mmaps[key] = (img_mm, pos_mm, state_mm)
        return self._mmaps[key]

    def prepare_token_drop(self, epis_idx, step_idx):
        token_budget = self.history_config.budget
        kept_path = os.path.join(self.bin_dir, "kept_indices", f"episode_{epis_idx}.json")
        with open(kept_path) as f:
            kept_indices = json.load(f)
        return self.mem_buffer.prepare_token_dropping(
            step_idx, token_budget, self._gather_history_feat,
            kept_indices=kept_indices, epis_idx=epis_idx)

    def _gather_history_feat(self, indices_to_load, epis_idx):
        img_mm, pos_mm, state_mm = self._get_mmaps(epis_idx)
        res = self._res_key
        img_key = f"image_emb_{res}"
        pos_key = f"pos_emb_{res}"

        history_feats = {}
        for frame_idx in indices_to_load:
            # Return zero-copy views into the mmap.  Downstream code in
            # MemoryBuffer.prepare_* always goes through np.stack/np.concatenate,
            # which allocate new arrays, so views are safe here.  They also pickle
            # correctly across DataLoader workers (tested).
            history_feats[frame_idx] = {
                img_key: img_mm[frame_idx],
                pos_key: pos_mm[frame_idx],
                "state_emb": state_mm[frame_idx],
            }

        return history_feats
