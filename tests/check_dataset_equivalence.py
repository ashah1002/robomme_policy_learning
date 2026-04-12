"""Correctness check: verify that npy, bin, and mmap datasets return identical samples.

Runs all three loaders side-by-side across every history_config variant (frame
sampling, token dropping, recurrent) and compares the outputs key-by-key.

Exact equality is required for every field.  Both ``image_emb`` (bf16 in storage,
round-trip exact against the source .npy which is already bf16) and ``pos_emb``
(f32 in storage) must match the original loader bitwise.

Usage:
    uv run tests/check_dataset_equivalence.py \\
        --dataset_path data/robomme_preprocessed_data_sample \\
        --num_samples 50
"""

import dataclasses
import random
import sys

import numpy as np
import tyro
from omegaconf import OmegaConf

import mme_vla_suite.training.config as _config
from mme_vla_suite.training.dataset import RoboMMEDataset
from mme_vla_suite.training.dataset_bin import RoboMMEDatasetBin
from mme_vla_suite.training.dataset_mmap import RoboMMEDatasetMmap

# (label, yaml_file_or_None) — None disables history entirely
HISTORY_CONFIGS = [
    # ("no-history",          None),
    ("framesamp-modul 4x4", "perceptual-framesamp-modul.yaml"),
    ("tokendrop-modul 8x8", "perceptual-tokendrop-modul.yaml"),
    ("recurrent-ttt  8x8",  "recurrent-ttt-modul.yaml"),
]

SKIP_KEYS = {"prompt", "simple_subgoal", "grounded_subgoal"}


def _load_history_config(yaml_name):
    if yaml_name is None:
        return None
    path = f"src/mme_vla_suite/models/config/robomme/{yaml_name}"
    with open(path) as f:
        return OmegaConf.load(f)


def _compare_samples(s_ref, s_other, label_other, idx, config_label):
    """Compare a single (s_ref, s_other) pair. Returns list of error strings."""
    errors = []
    ref_keys = {k for k, v in s_ref.items() if v is not None and k not in SKIP_KEYS}
    other_keys = {k for k, v in s_other.items() if v is not None and k not in SKIP_KEYS}
    
    import pdb; pdb.set_trace()

    if ref_keys != other_keys:
        extra = other_keys - ref_keys
        missing = ref_keys - other_keys
        if extra:
            errors.append(f"  [{config_label}] idx={idx}: {label_other} has extra keys {extra}")
        if missing:
            errors.append(f"  [{config_label}] idx={idx}: {label_other} missing keys {missing}")

    for key in ref_keys & other_keys:
        v_ref = np.asarray(s_ref[key], dtype=np.float32)
        v_other = np.asarray(s_other[key], dtype=np.float32)

        if v_ref.shape != v_other.shape:
            errors.append(
                f"  [{config_label}] idx={idx} {key}: shape mismatch "
                f"{v_ref.shape} vs {v_other.shape} ({label_other})"
            )
            continue

        if not np.array_equal(v_ref, v_other):
            max_err = float(np.max(np.abs(v_ref - v_other)))
            errors.append(
                f"  [{config_label}] idx={idx} {key}: "
                f"value mismatch, max_err={max_err:.6g} ({label_other})"
            )
    return errors


def main(
    dataset_path: str = "data/robomme_preprocessed_data",
    num_samples: int = 50,
    sample_stride: int = 1000,
    seed: int = 42,
):
    """Verify equivalence across npy / bin / mmap datasets for every history config.

    Args:
        dataset_path: Path to the preprocessed dataset (must contain both features/ and features_bin/).
        num_samples: How many sample indices to check per history config.
        sample_stride: Stride between sample indices (first sample at idx=0).
        seed: Random seed for reproducible sampling decisions inside the dataset.
    """
    config = _config.get_config("mme_vla_suite")
    config = dataclasses.replace(config, data=dataclasses.replace(config.data, repo_id="robomme"))
    data_config = config.data.create(config.assets_dirs, config.model)

    base_kwargs = dict(
        dataset_path=dataset_path,
        data_config=data_config,
        action_horizon=config.model.action_horizon,
        compute_norm_stats=True,
    )

    overall_errors = []

    for config_label, yaml_name in HISTORY_CONFIGS:
        print(f"\n=== {config_label} ===")
        history_config = _load_history_config(yaml_name)
        kwargs = dict(base_kwargs, history_config=history_config)

        try:
            ds_npy = RoboMMEDataset(**kwargs)
            ds_bin = RoboMMEDatasetBin(**kwargs)
            ds_mmap = RoboMMEDatasetMmap(**kwargs)
        except Exception as e:
            print(f"  ERROR: failed to instantiate datasets: {e}")
            overall_errors.append(f"{config_label}: instantiation failed")
            continue

        ds_len = min(len(ds_npy), len(ds_bin), len(ds_mmap))
        indices = [i * sample_stride for i in range(num_samples) if i * sample_stride < ds_len]
        config_errors = []

        for idx in indices:
            # Reset RNG before each dataset call so any internal randomness
            # (e.g. subgoal_online sampling, grounding augmentation) is identical.
            random.seed(seed + idx); s_npy = ds_npy[idx]
            random.seed(seed + idx); s_bin = ds_bin[idx]
            random.seed(seed + idx); s_mmap = ds_mmap[idx]

            config_errors += _compare_samples(s_npy, s_bin, "bin", idx, config_label)
            config_errors += _compare_samples(s_npy, s_mmap, "mmap", idx, config_label)

        if config_errors:
            print(f"  FAIL — {len(config_errors)} mismatches in {len(indices)} samples")
            for err in config_errors[:10]:
                print(err)
            if len(config_errors) > 10:
                print(f"  ... and {len(config_errors) - 10} more")
            overall_errors.extend(config_errors)
        else:
            # Report which keys were compared so it's clear what we actually checked
            sample_keys = sorted(
                k for k, v in ds_npy[indices[0]].items()
                if v is not None and k not in SKIP_KEYS
            )
            print(f"  OK — {len(indices)} samples, keys checked: {sample_keys}")

    print("\n" + "=" * 60)
    if overall_errors:
        print(f"FAILED: {len(overall_errors)} total mismatches across all configs")
        sys.exit(1)
    else:
        print("PASSED: all configs produce equivalent outputs across npy/bin/mmap")


if __name__ == "__main__":
    tyro.cli(main)
