from collections.abc import Sequence
import json
import logging
import pathlib
import time
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

from mme_vla_suite.models.integration.history_observation import HistAugObservation
from mme_vla_suite.models.integration.history_pi0 import HistoryPi0
from mme_vla_suite.shared.mem_buffer import MemoryBuffer, MemoryBufferRecurrent


logger = logging.getLogger(__name__)

class MME_VLA_Policy:
    def __init__(
        self,
        model: HistoryPi0,
        *,
        seed: int = 42,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        norm_stats: dict[str, _transforms.NormStats] | None = None,
        use_quantiles: bool = False,
        stats_dir: str | pathlib.Path | None = None,
    ):
        self._model = model
        self._seed = seed
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._sample_actions_with_stats = nnx_utils.module_jit(
            model.sample_actions_with_stats,
            static_argnames=("num_steps",),
        )
        self._vision_encode = nnx_utils.module_jit(model.vision_encode)
        
        
        self.config = model.history_config
        self.mem_buffer = None
        
        self.state_norm_stats = norm_stats['state']
        self.use_quantiles = use_quantiles
        self._stats_dir = pathlib.Path(stats_dir) if stats_dir is not None else None
        self._stats_enabled = self._stats_dir is not None and self.config is not None
        self._stats_global_step = 0
        self._stats_episode_idx = -1
        self._stats_step_in_episode = 0
        self._stats_aggregate: dict[str, Any] = {}
        if self._stats_enabled:
            self._stats_dir.mkdir(parents=True, exist_ok=True)
            (self._stats_dir / "raw").mkdir(exist_ok=True)
            (self._stats_dir / "summary").mkdir(exist_ok=True)
        
        self.reset()
        
    
    def _prepare_mem_buffer(self):
        if self.config is None or self.config.representation_type == "symbolic":
            self.mem_buffer = None
        elif self.config.representation_type == "recurrent":
            self.mem_buffer = MemoryBufferRecurrent(
                num_views=self.config.num_views,
                img_emb_dim=self.config.memory_feature.img.input_dim,
                pos_emb_dim=self.config.memory_feature.pos.input_dim,
                state_emb_dim=self.config.memory_feature.state.input_dim,
                input_obs_horizon=self.config.streaming_obs_horizon,
                max_recur_steps=self.config.recurrent_memory.max_recur_steps,
                max_video_steps=self.config.recurrent_memory.max_pretraj_steps,
                prepare_buffer=True, vision_enc_fn=self._vision_encode,
            )
        else:
            self.mem_buffer = MemoryBuffer(
                num_views=self.config.num_views,
                img_emb_dim=self.config.memory_feature.img.input_dim,
                pos_emb_dim=self.config.memory_feature.pos.input_dim,
                state_emb_dim=self.config.memory_feature.state.input_dim,
                compute_token_drop_score = self.config.perceptual_memory.type == "token_dropping",
                token_drop_stride=self.config.streaming_obs_horizon // 2,
                prepare_buffer=True, vision_enc_fn=self._vision_encode,
            )

    @override
    def infer(self, obs: dict) -> dict:
        if self.config is not None and self.config.representation_type != "symbolic":
            assert len(self.mem_buffer._history_feats) > 0, \
                "history feats is empty, add buffer first"
                                        
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._prepare_history(inputs)
        inputs = self._input_transform(inputs)
        observation = HistAugObservation.from_dict(
            jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        )
        self._rng, sample_rng = jax.random.split(self._rng)
    
        start_time = time.monotonic()
        debug_summary = None
        if self._stats_enabled:
            actions, debug_stats = self._sample_actions_with_stats(
                sample_rng, observation, **self._sample_kwargs
            )
            debug_summary = self._record_debug_stats(debug_stats)
        else:
            actions = self._sample_actions(sample_rng, observation, **self._sample_kwargs)

        outputs = {
            "state": observation.state,
            "actions": actions,
        }
        model_time = time.monotonic() - start_time
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)      
        outputs = self._output_transform(outputs)
        outputs["infer_time_ms"] = model_time * 1000
        if debug_summary is not None:
            outputs["memory_debug_stats"] = debug_summary
        
        return outputs
    
    @override
    def reset(self) -> None:
        del self.mem_buffer
        self._prepare_mem_buffer()
        self.step_idx = -1  
        self.exec_start_idx = 0
        self._rng = jax.random.key(self._seed)
        if self._stats_enabled:
            self._stats_episode_idx += 1
            self._stats_step_in_episode = 0
            
    
    def add_buffer(self, obs: dict) -> None:
        if self.mem_buffer is None:
            return
        images = obs["images"]
        states = obs["state"]
        if obs.get("exec_start_idx", 0) > 0: # has video
            self.exec_start_idx = obs["exec_start_idx"]
        
        step_idx_list = list(range(self.step_idx+1, self.step_idx + len(images) + 1))
        self.mem_buffer.add_buffer(images, states, step_idx_list)
        self.step_idx += len(images)

    def _normalize_state(self, state):
        if self.use_quantiles:
            return (state - self.state_norm_stats.q01) / (self.state_norm_stats.q99 - self.state_norm_stats.q01 + 1e-6) * 2.0 - 1.0
        else:
            return (state - self.state_norm_stats.mean) / (self.state_norm_stats.std + 1e-6)

    def _prepare_history(self, inputs: dict) -> dict:
        if self.config is None or self.config.representation_type == "symbolic":
            return inputs
        
        if self.config.representation_type == "recurrent":
            history_feats_gather_fn = self.mem_buffer.default_history_feats_gather_fn
            recur_image_emb, recur_pos_emb, recur_state_emb, recur_mask = \
                self.mem_buffer.prepare_token_recurrent(
                    self.step_idx, self.exec_start_idx, history_feats_gather_fn)
            inputs["recur_image_emb"] = recur_image_emb
            inputs["recur_pos_emb"] = recur_pos_emb
            inputs["recur_state_emb"] = self._normalize_state(recur_state_emb)
            inputs["recur_mask"] = recur_mask
        elif self.config.representation_type == "perceptual":
            history_feats_gather_fn = self.mem_buffer.default_history_feats_gather_fn
            token_budget = self.config.budget
            
            if self.config.perceptual_memory.type == "token_dropping":
                static_image_emb, static_pos_emb, static_state_emb, static_mask = \
                    self.mem_buffer.prepare_token_dropping(
                        self.step_idx, token_budget, history_feats_gather_fn)
            else:
                token_per_image = self.config.token_per_image
                static_image_emb, static_pos_emb, static_state_emb, static_mask = \
                    self.mem_buffer.prepare_frame_sampling(
                        self.step_idx, token_budget, token_per_image, history_feats_gather_fn)
            
            inputs["static_image_emb"] = static_image_emb
            inputs["static_pos_emb"] = static_pos_emb
            inputs["static_state_emb"] = self._normalize_state(static_state_emb)
            inputs["static_mask"] = static_mask
        else:
            raise ValueError(f"Not supported representation type: {self.config.representation_type}")
        
    
        return inputs

    def _stats_paths(self) -> tuple[pathlib.Path, pathlib.Path]:
        stem = (
            f"episode_{self._stats_episode_idx:04d}_"
            f"step_{self._stats_step_in_episode:05d}_"
            f"global_{self._stats_global_step:06d}"
        )
        return self._stats_dir / "raw" / f"{stem}.npz", self._stats_dir / "summary" / f"{stem}.json"

    def _array_distribution(self, values: np.ndarray) -> dict[str, float]:
        flat = values.astype(np.float32).reshape(-1)
        return {
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "p01": float(np.percentile(flat, 1)),
            "p05": float(np.percentile(flat, 5)),
            "p50": float(np.percentile(flat, 50)),
            "p95": float(np.percentile(flat, 95)),
            "p99": float(np.percentile(flat, 99)),
        }

    def _update_attention_aggregate(
        self, region_means: np.ndarray, labels: tuple[str, ...]
    ) -> dict[str, Any]:
        if not self._stats_aggregate:
            self._stats_aggregate = {
                "type": "attention",
                "labels": labels,
                "num_infers": 0,
                "sum": np.zeros(len(labels), dtype=np.float64),
                "count": 0,
                "per_layer_sum": np.zeros((region_means.shape[1], len(labels)), dtype=np.float64),
                "per_layer_count": 0,
            }

        self._stats_aggregate["num_infers"] += 1
        self._stats_aggregate["sum"] += region_means.sum(axis=(0, 1))
        self._stats_aggregate["count"] += region_means.shape[0] * region_means.shape[1]
        self._stats_aggregate["per_layer_sum"] += region_means.sum(axis=0)
        self._stats_aggregate["per_layer_count"] += region_means.shape[0]

        return {
            "type": "attention",
            "labels": list(labels),
            "num_infers": int(self._stats_aggregate["num_infers"]),
            "overall_mean": {
                label: float(total / self._stats_aggregate["count"])
                for label, total in zip(labels, self._stats_aggregate["sum"], strict=True)
            },
            "per_layer_mean": {
                label: (
                    self._stats_aggregate["per_layer_sum"][:, idx]
                    / self._stats_aggregate["per_layer_count"]
                ).tolist()
                for idx, label in enumerate(labels)
            },
        }

    def _update_modulation_aggregate(
        self, scale: np.ndarray, shift: np.ndarray
    ) -> dict[str, Any]:
        if not self._stats_aggregate:
            self._stats_aggregate = {
                "type": "modulation",
                "num_infers": 0,
                "scale_sum": 0.0,
                "scale_sq_sum": 0.0,
                "scale_count": 0,
                "scale_min": np.inf,
                "scale_max": -np.inf,
                "shift_sum": 0.0,
                "shift_sq_sum": 0.0,
                "shift_count": 0,
                "shift_min": np.inf,
                "shift_max": -np.inf,
            }

        scale_f = scale.astype(np.float64).reshape(-1)
        shift_f = shift.astype(np.float64).reshape(-1)
        self._stats_aggregate["num_infers"] += 1
        self._stats_aggregate["scale_sum"] += float(scale_f.sum())
        self._stats_aggregate["scale_sq_sum"] += float(np.square(scale_f).sum())
        self._stats_aggregate["scale_count"] += scale_f.size
        self._stats_aggregate["scale_min"] = min(self._stats_aggregate["scale_min"], float(scale_f.min()))
        self._stats_aggregate["scale_max"] = max(self._stats_aggregate["scale_max"], float(scale_f.max()))
        self._stats_aggregate["shift_sum"] += float(shift_f.sum())
        self._stats_aggregate["shift_sq_sum"] += float(np.square(shift_f).sum())
        self._stats_aggregate["shift_count"] += shift_f.size
        self._stats_aggregate["shift_min"] = min(self._stats_aggregate["shift_min"], float(shift_f.min()))
        self._stats_aggregate["shift_max"] = max(self._stats_aggregate["shift_max"], float(shift_f.max()))

        scale_mean = self._stats_aggregate["scale_sum"] / self._stats_aggregate["scale_count"]
        shift_mean = self._stats_aggregate["shift_sum"] / self._stats_aggregate["shift_count"]
        scale_var = (
            self._stats_aggregate["scale_sq_sum"] / self._stats_aggregate["scale_count"]
            - scale_mean**2
        )
        shift_var = (
            self._stats_aggregate["shift_sq_sum"] / self._stats_aggregate["shift_count"]
            - shift_mean**2
        )
        return {
            "type": "modulation",
            "num_infers": int(self._stats_aggregate["num_infers"]),
            "scale": {
                "mean": float(scale_mean),
                "std": float(np.sqrt(max(scale_var, 0.0))),
                "min": float(self._stats_aggregate["scale_min"]),
                "max": float(self._stats_aggregate["scale_max"]),
            },
            "shift": {
                "mean": float(shift_mean),
                "std": float(np.sqrt(max(shift_var, 0.0))),
                "min": float(self._stats_aggregate["shift_min"]),
                "max": float(self._stats_aggregate["shift_max"]),
            },
        }

    def _write_json(self, path: pathlib.Path, payload: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _record_debug_stats(self, debug_stats: dict[str, Any]) -> dict[str, Any]:
        assert self._stats_dir is not None
        raw_path, summary_path = self._stats_paths()
        debug_stats = jax.tree.map(np.asarray, debug_stats)

        if "attention_region_means" in debug_stats:
            labels = ("memory_tokens", "image_tokens", "language_tokens")
            region_means = debug_stats["attention_region_means"].astype(np.float32)
            summary = {
                "type": "attention",
                "episode_idx": self._stats_episode_idx,
                "step_in_episode": self._stats_step_in_episode,
                "global_step": self._stats_global_step,
                "labels": list(labels),
                "key_region_lengths": debug_stats["key_region_lengths"].astype(int).tolist(),
                "overall_mean": {
                    label: float(value)
                    for label, value in zip(labels, region_means.mean(axis=(0, 1)), strict=True)
                },
                "per_layer_mean": {
                    label: region_means.mean(axis=0)[:, idx].tolist()
                    for idx, label in enumerate(labels)
                },
            }
            aggregate_summary = self._update_attention_aggregate(region_means, labels)
            np.savez_compressed(
                raw_path,
                attention_region_means=region_means,
                key_region_lengths=debug_stats["key_region_lengths"].astype(np.int32),
            )
            logger.info(
                "Attention stats ep=%d step=%d: %s",
                self._stats_episode_idx,
                self._stats_step_in_episode,
                ", ".join(
                    f"{label}={summary['overall_mean'][label]:.4f}" for label in labels
                ),
            )
        else:
            scale = debug_stats["modulation_scale"].astype(np.float16)
            shift = debug_stats["modulation_shift"].astype(np.float16)
            summary = {
                "type": "modulation",
                "episode_idx": self._stats_episode_idx,
                "step_in_episode": self._stats_step_in_episode,
                "global_step": self._stats_global_step,
                "scale": self._array_distribution(scale.astype(np.float32)),
                "shift": self._array_distribution(shift.astype(np.float32)),
            }
            aggregate_summary = self._update_modulation_aggregate(
                scale.astype(np.float32), shift.astype(np.float32)
            )
            np.savez_compressed(raw_path, modulation_scale=scale, modulation_shift=shift)
            logger.info(
                "Modulation stats ep=%d step=%d: scale(mean=%.4f,std=%.4f) shift(mean=%.4f,std=%.4f)",
                self._stats_episode_idx,
                self._stats_step_in_episode,
                summary["scale"]["mean"],
                summary["scale"]["std"],
                summary["shift"]["mean"],
                summary["shift"]["std"],
            )

        self._write_json(summary_path, summary)
        self._write_json(self._stats_dir / "aggregate_summary.json", aggregate_summary)
        self._stats_global_step += 1
        self._stats_step_in_episode += 1
        return summary

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata