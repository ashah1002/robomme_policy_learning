"""RoboMME dataset with seek-based .bin loading for history features.

Drop-in replacement for RoboMMEDataset — overrides only _gather_history_feat.
Expects features_bin/ created by scripts/convert_features_to_bin.py.

Each resolution (8x8, 4x4, 2x2) has its own .bin file per episode.
Only the resolution needed by the current config is read from disk.
"""

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor

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
    """Loads history features from per-episode .bin files via seek.

    Parameters
    ----------
    num_read_threads : int
        If > 1, parallelize per-frame reads across this many threads.  Each task
        does all 3 preads (img + pos + state) for one frame, so NFS latency is
        overlapped.  Recommended ~4-16 for recurrent configs (64 frames); keep at
        1 for frame-sampling (3-8 frames) since thread dispatch overhead dominates.
    """

    def __init__(self, *args, bin_features_dir=None, num_read_threads: int = 1, **kwargs):
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

        # Read-parallelism config. Executor is lazy so it's created *after* the
        # DataLoader fork (avoids sharing threads across worker processes).
        self._num_read_threads = int(num_read_threads)
        self._executor = None

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

    def _get_executor(self):
        """Lazily create a ThreadPoolExecutor per worker process (post-fork)."""
        if self._num_read_threads <= 1:
            return None
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self._num_read_threads)
        return self._executor

    def _read_one_frame(self, frame_idx, V, tokens, state_dim,
                        img_fd, pos_fd, state_fd, img_key, pos_key):
        """Read img + pos + state for a single frame. Thread-safe (uses pread)."""
        img_raw = os.pread(img_fd, self._img_frame_bytes,
                           frame_idx * self._img_frame_bytes)
        img = np.frombuffer(img_raw, dtype=ml_dtypes.bfloat16).reshape(V, tokens, IMG_DIM)

        pos_raw = os.pread(pos_fd, self._pos_frame_bytes,
                           frame_idx * self._pos_frame_bytes)
        pos = np.frombuffer(pos_raw, dtype=np.float32).reshape(V, tokens, POS_DIM)

        state_raw = os.pread(state_fd, self._state_frame_bytes,
                             frame_idx * self._state_frame_bytes)
        state = np.frombuffer(state_raw, dtype=np.float32).reshape(state_dim)

        return frame_idx, {img_key: img, pos_key: pos, "state_emb": state}

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

        executor = self._get_executor() if len(indices_to_load) > 1 else None
        history_feats = {}

        if executor is not None:
            # Parallel path: each task does 3 preads for one frame. os.pread is
            # thread-safe (atomic at offset), and the GIL is released during the
            # syscall, so NFS latency is overlapped across threads.
            futures = [
                executor.submit(
                    self._read_one_frame, fi, V, tokens, state_dim,
                    img_fd, pos_fd, state_fd, img_key, pos_key,
                )
                for fi in indices_to_load
            ]
            for future in futures:
                fi, feats = future.result()
                history_feats[fi] = feats
        else:
            # Sequential path (low overhead for small frame counts).
            for frame_idx in indices_to_load:
                _, history_feats[frame_idx] = self._read_one_frame(
                    frame_idx, V, tokens, state_dim,
                    img_fd, pos_fd, state_fd, img_key, pos_key,
                )

        return history_feats
