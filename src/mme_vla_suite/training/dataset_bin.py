"""RoboMME dataset with seek-based .bin loading for history features.

Drop-in replacement for RoboMMEDataset — overrides only _gather_history_feat.
Expects features_bin/ created by scripts/convert_features_to_bin.py.

Each resolution (8x8, 4x4, 2x2) has its own .bin file per episode.
Only the resolution needed by the current config is read from disk.
"""

import json
import math
import os

import ml_dtypes
import numpy as np

from mme_vla_suite.training.dataset import RoboMMEDataset

IMG_DIM = 2048
POS_DIM = 768
BF16_ELEM = 2
F32_ELEM = 4

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


class RoboMMEDatasetBin(RoboMMEDataset):
    """Loads history features from per-episode .bin files via seek."""

    def __init__(self, *args, bin_features_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bin_dir = bin_features_dir or os.path.join(
            self.dataset.dataset_path, "features_bin")

        meta_path = os.path.join(self.bin_dir, "metadata.json")
        assert os.path.exists(meta_path), f"No metadata.json in {self.bin_dir}"
        with open(meta_path) as f:
            self.bin_meta = json.load(f)

        # Determine the single resolution we need
        self._res_key, self._res_tokens = _resolve_resolution(self.history_config)

        # Pre-compute frame byte sizes (set once we know V from first episode)
        self._frame_sizes_ready = False
        self._img_frame_bytes = 0
        self._pos_frame_bytes = 0
        self._state_frame_bytes = 0

        # Lazy file handle cache (per-worker after DataLoader fork)
        self._img_fds = {}
        self._pos_fds = {}
        self._state_fds = {}

    def _ensure_frame_sizes(self, V, state_dim):
        if self._frame_sizes_ready:
            return
        tokens = self._res_tokens
        self._img_frame_bytes = V * tokens * IMG_DIM * BF16_ELEM
        self._pos_frame_bytes = V * tokens * POS_DIM * F32_ELEM
        self._state_frame_bytes = state_dim * F32_ELEM
        self._V = V
        self._state_dim = state_dim
        self._frame_sizes_ready = True

    def _open_episode(self, epis_key):
        if epis_key not in self._img_fds:
            res = self._res_key
            self._img_fds[epis_key] = os.open(
                os.path.join(self.bin_dir, f"image_emb_{res}", f"{epis_key}.bin"), os.O_RDONLY)
            self._pos_fds[epis_key] = os.open(
                os.path.join(self.bin_dir, f"pos_emb_{res}", f"{epis_key}.bin"), os.O_RDONLY)
            self._state_fds[epis_key] = os.open(
                os.path.join(self.bin_dir, "state", f"{epis_key}.bin"), os.O_RDONLY)
        return self._img_fds[epis_key], self._pos_fds[epis_key], self._state_fds[epis_key]

    def prepare_token_drop(self, epis_idx, step_idx):
        token_budget = self.history_config.budget
        kept_path = os.path.join(self.bin_dir, "kept_indices", f"episode_{epis_idx}.json")
        with open(kept_path) as f:
            kept_indices = json.load(f)
        return self.mem_buffer.prepare_token_dropping(
            step_idx, token_budget, self._gather_history_feat,
            kept_indices=kept_indices, epis_idx=epis_idx)

    def _gather_history_feat(self, indices_to_load, epis_idx):
        epis_key = f"episode_{epis_idx}"
        meta = self.bin_meta[epis_key]
        V = meta["num_views"]
        state_dim = meta["state_dim"]
        self._ensure_frame_sizes(V, state_dim)

        img_fd, pos_fd, state_fd = self._open_episode(epis_key)
        tokens = self._res_tokens
        res = self._res_key
        img_key = f"image_emb_{res}"
        pos_key = f"pos_emb_{res}"

        history_feats = {}
        for frame_idx in indices_to_load:
            # os.pread bypasses Python's BufferedReader — one syscall, no extra buffering.
            # np.frombuffer(bytes, ...) returns a read-only view; downstream code only reads
            # or calls np.stack/concatenate (both allocate new arrays), so no copy is needed.
            raw = os.pread(img_fd, self._img_frame_bytes, frame_idx * self._img_frame_bytes)
            img = np.frombuffer(raw, dtype=ml_dtypes.bfloat16).reshape(V, tokens, IMG_DIM)

            raw = os.pread(pos_fd, self._pos_frame_bytes, frame_idx * self._pos_frame_bytes)
            pos = np.frombuffer(raw, dtype=np.float32).reshape(V, tokens, POS_DIM)

            raw = os.pread(state_fd, self._state_frame_bytes, frame_idx * self._state_frame_bytes)
            state = np.frombuffer(raw, dtype=np.float32).reshape(state_dim)

            history_feats[frame_idx] = {
                img_key: img,
                pos_key: pos,
                "state_emb": state,
            }

        return history_feats
