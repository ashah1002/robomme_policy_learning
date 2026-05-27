import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at
from mme_vla_suite.models.config.utils import history_flag
from mme_vla_suite.models.representation.mem_encoder import FeatureEncoder
from mme_vla_suite.models.representation.temporal_memory_encoder import (
    TemporalMemoryEncoder,
)
from mme_vla_suite.models.representation.utils import kernel_init


class MemoryRouter(nnx.Module):
    def __init__(
        self,
        hidden_dim: int,
        rngs: nnx.Rngs,
        dtype: at.DTypeLike = jnp.float32,
    ):
        self.proj = nnx.Linear(
            hidden_dim, 1, rngs=rngs, dtype=dtype, kernel_init=kernel_init
        )

    def __call__(
        self,
        tokens: at.Float[at.Array, "b s d"],
        mask: at.Bool[at.Array, "b s"],
    ) -> at.Float[at.Array, "b 1"]:
        m = mask.astype(tokens.dtype)[..., None]
        pooled = (tokens * m).sum(axis=1) / jnp.clip(m.sum(axis=1), a_min=1e-6)
        return jax.nn.sigmoid(self.proj(pooled))


class PerceptualMemory(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs, dtype: at.DTypeLike = jnp.float32):
        self.config = config
        self.dtype = dtype

        self.mem_type = config.perceptual_memory.type

        self.feature_encoder = FeatureEncoder(
            rngs=rngs,
            dtype=dtype,
            image_input_dim=self.config.memory_feature.img.input_dim,
            pos_input_dim=self.config.memory_feature.pos.input_dim,
            state_input_dim=self.config.memory_feature.state.input_dim,
            pos_output_dim=self.config.memory_feature.pos.hidden_dim,
            state_output_dim=self.config.memory_feature.state.hidden_dim,
            ouput_dim_for_recur=None,
            output_dim_for_percep=self.config.memory_token_dim,
            use_pos_emb=self.config.use_pos_emb,
            use_state_emb=self.config.use_state_emb,
        )

        self.temporal_encoder = None
        if history_flag(config, "temporal_encoder"):
            self.temporal_encoder = TemporalMemoryEncoder(
                config, rngs=rngs, dtype=dtype
            )

        self.mem_router = None
        if history_flag(config, "memory_gate"):
            self.mem_router = MemoryRouter(
                self.config.memory_token_dim, rngs=rngs, dtype=dtype
            )

    def __call__(
        self,
        static_image_emb: at.Float[at.Array, "b l d1"],
        static_pos_emb: at.Float[at.Array, "b l d2"],
        static_state_emb: at.Float[at.Array, "b l d3"],
        static_mask: at.Bool[at.Array, "b l"] | None = None,
    ):
        assert static_image_emb.shape[1] == self.config.budget

        hidden_states = self.feature_encoder.encode_perceptual_memory(
            static_image_emb, static_pos_emb, static_state_emb
        )

        stats = {}
        if self.temporal_encoder is not None:
            hidden_states = self.temporal_encoder(hidden_states, static_mask)
        if self.mem_router is not None:
            if static_mask is None:
                static_mask = jnp.ones(
                    hidden_states.shape[:2], dtype=jnp.bool_
                )
            g = self.mem_router(hidden_states, static_mask)
            stats["mem_gate"] = g

        return hidden_states, None, stats if stats else None
