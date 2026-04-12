"""
Final benchmark
  ┌─────────────────┬───────────┬─────────────┬───────┬───────┬──────────────┐
  │     Config      │ npy (old) │ npy (fixed) │  bin  │ mmap  │ best speedup │
  ├─────────────────┼───────────┼─────────────┼───────┼───────┼──────────────┤
  │ framesamp (4x4) │ 43.4ms    │ 19.9ms      │ 4.2ms │ 3.8ms │ 11.4x        │
  ├─────────────────┼───────────┼─────────────┼───────┼───────┼──────────────┤
  │ tokendrop (8x8) │ 20.3ms    │ 17.3ms      │ 8.0ms │ 8.2ms │ 2.5x         │
  ├─────────────────┼───────────┼─────────────┼───────┼───────┼──────────────┤
  │ recurrent (8x8) │ 58.8ms    │ 13.3ms      │ 5.8ms │ 5.3ms │ 11.1x        │
  └─────────────────┴───────────┴─────────────┴───────┴───────┴──────────────┘
"""

import time

import tqdm
import tyro
from omegaconf import OmegaConf, DictConfig
import numpy as np
import dataclasses

import openpi.transforms as transforms
import openpi.shared.normalize as normalize
from openpi.training.data_loader import TransformedDataset, TorchDataLoader

import mme_vla_suite.training.config as _config
from mme_vla_suite.training.dataset import RoboMMEDataset
from mme_vla_suite.training.dataset_bin import RoboMMEDatasetBin
from mme_vla_suite.training.dataset_mmap import RoboMMEDatasetMmap

class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_data_loader(
    dataset_path: str,
    data_config: _config.DataConfig,
    history_config: DictConfig,
    action_horizon: int,
    batch_size: int,
    num_batches: int | None = None,
    num_workers: int = 0,
    compute_norm_stats: bool = False,
    seed: int = 0,
    dataset_type: str = "npy",
):
    dataset_cls = {
        "npy": RoboMMEDataset,
        "bin": RoboMMEDatasetBin,
        "mmap": RoboMMEDatasetMmap,
    }[dataset_type]

    dataset = dataset_cls(
        dataset_path=dataset_path,
        data_config=data_config,
        history_config=history_config,
        action_horizon=action_horizon,
        compute_norm_stats=compute_norm_stats)

    dataset = TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ])
    print(f"Dataset length: {len(dataset)}, batch size: {batch_size}")

    num_batches = len(dataset) // batch_size
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        sharding=None,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        shuffle=True,
    )
    return data_loader, num_batches



def main(
    config_name: str = "mme_vla_suite",
    repo_id: str = "robomme",
    dataset_path: str = "data/robomme_preprocessed_data",
    dataset_type: str = "npy",
    history_config_path: str = "src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul_hard.yaml",
    num_workers: int = 0,
    batch_size: int = 64
):
    """Benchmark dataloader speed for npy / bin / mmap backends.

    Args:
        dataset_type: One of "npy", "bin", "mmap".
    """
    config = _config.get_config(config_name)
    config = dataclasses.replace(config, data=dataclasses.replace(config.data, repo_id=repo_id))
    data_config = config.data.create(config.assets_dirs, config.model)

    with open(history_config_path) as f:
        history_config = OmegaConf.load(f)

    data_loader, num_batches = create_data_loader(
        dataset_path=dataset_path,
        data_config=data_config,
        history_config=history_config,
        action_horizon=config.model.action_horizon,
        batch_size=batch_size,
        num_workers=num_workers,
        compute_norm_stats=True,
        dataset_type=dataset_type,
    )

    warmup_batches = 10
    max_batches = 100
    num_batches = min(num_batches, warmup_batches + max_batches)
    print(f"\n=== Benchmarking dataset_type={dataset_type}, num_workers={num_workers}, batch_size={batch_size} ===")
    print(history_config_path)
    print(f"Warming up {warmup_batches} batches, then timing {max_batches} batches...")

    times = []
    for i, batch in enumerate(tqdm.tqdm(data_loader, total=num_batches, desc=f"[{dataset_type}]")):
        if i >= warmup_batches:
            times.append(time.time())
        if i + 1 >= num_batches:
            break

    if len(times) > 1:
        intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
        total = times[-1] - times[0]
        avg = np.mean(intervals)
        p50 = np.median(intervals)
        p95 = np.percentile(intervals, 95)
        samples_per_sec = batch_size / avg
        print(f"\n--- Results ({dataset_type}) ---")
        print(f"Total time:      {total:.1f}s for {len(intervals)} batches")
        print(f"Avg batch time:  {avg:.3f}s")
        print(f"Median (p50):    {p50:.3f}s")
        print(f"p95:             {p95:.3f}s")
        print(f"Throughput:      {samples_per_sec:.1f} samples/s")
        

if __name__ == "__main__":
    tyro.cli(main)


# Turbo
# Recurrent
# --- Results (mmap) ---
# Total time:      445.9s for 99 batches
# Avg batch time:  4.504s
# Median (p50):    1.700s
# p95:             13.675s
# Throughput:      14.2 samples/s

# --- Results (npy) ---
# Total time:      447.1s for 99 batches
# Avg batch time:  4.516s
# Median (p50):    4.510s
# p95:             8.227s
# Throughput:      14.2 samples/s

# Bin has mistake. I dont know why

# FrameSamp
# --- Results (npy) ---
# Total time:      272.8s for 99 batches
# Avg batch time:  2.756s
# Median (p50):    0.492s
# p95:             11.865s
# Throughput:      23.2 samples/s
# --- Results (bin) ---
# Total time:      104.3s for 99 batches
# Avg batch time:  1.053s
# Median (p50):    0.524s
# p95:             3.216s
# Throughput:      60.8 samples/s


# FrameSamp Hard (4096, 64)
# Still npy is the best...
