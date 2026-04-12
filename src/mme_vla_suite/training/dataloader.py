"""
We implemented our own data loader,
which can be 5-10x faster than LeRobot dataloader and can avoid memory explosion issue

--
Update: We improve our dataloader by using np.memmap to load history features from per-episode .bin files.
see dataset_mmap.py for details.
"""

import jax
import logging
from omegaconf import DictConfig
from openpi.models import model as _model
from openpi.training.data_loader import DataLoader, TorchDataLoader,transform_dataset
import openpi.training.config as _config

from mme_vla_suite.training.dataset import RoboMMEDataset
from mme_vla_suite.training.dataset_bin import RoboMMEDatasetBin
from mme_vla_suite.training.dataset_mmap import RoboMMEDatasetMmap
from mme_vla_suite.models.integration.history_observation import HistAugObservation
from mme_vla_suite.models.config.utils import get_history_config


DATASET_CLASSES = {
    "npy": RoboMMEDataset,
    "bin": RoboMMEDatasetBin,
    "mmap": RoboMMEDatasetMmap,
}



class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader
        self._total_samples = len(data_loader._data_loader.dataset)

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield HistAugObservation.from_dict(batch), batch["actions"]
            

def create_data_loader(
    dataset_path: str,
    data_config: _config.DataConfig,
    history_config: str | DictConfig | None,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    dataset_type: str = "npy",
    num_read_threads: int = 1,
) -> DataLoader[tuple[HistAugObservation, _model.Actions]]:

    history_config = get_history_config(history_config)

    if dataset_type not in DATASET_CLASSES:
        raise ValueError(
            f"Unknown dataset_type '{dataset_type}', must be one of {list(DATASET_CLASSES)}"
        )
    dataset_cls = DATASET_CLASSES[dataset_type]
    logging.info(f"Using dataset class: {dataset_cls.__name__} (dataset_type='{dataset_type}')")

    dataset_kwargs = dict(
        dataset_path=dataset_path,
        data_config=data_config,
        history_config=history_config,
        action_horizon=action_horizon,
    )
    # Only the bin loader supports multi-threaded reads (mmap uses the kernel
    # page cache, npy uses its own ThreadPoolExecutor already).
    if dataset_type == "bin":
        dataset_kwargs["num_read_threads"] = num_read_threads

    dataset = dataset_cls(**dataset_kwargs)
    
    dataset = transform_dataset(
        dataset, data_config, skip_norm_stats=skip_norm_stats)

    local_batch_size = batch_size // jax.process_count()
    logging.info(f"local_batch_size: {local_batch_size}")
    
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework="jax",
    )

    return DataLoaderImpl(data_config, data_loader)