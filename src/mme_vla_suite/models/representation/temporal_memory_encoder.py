import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at
from mme_vla_suite.models.representation.utils import kernel_init, kernel_init_out_proj


class TemporalMemoryEncoderLayer(nnx.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        rngs: nnx.Rngs,
        dtype: at.DTypeLike = jnp.float32,
    ):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        hidden_dim = int(dim * mlp_ratio)

        self.norm1 = nnx.LayerNorm(dim, rngs=rngs, dtype=dtype)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs, dtype=dtype)
        self.q_proj = nnx.Linear(
            dim, dim, use_bias=False, rngs=rngs, dtype=dtype, kernel_init=kernel_init
        )
        self.k_proj = nnx.Linear(
            dim, dim, use_bias=False, rngs=rngs, dtype=dtype, kernel_init=kernel_init
        )
        self.v_proj = nnx.Linear(
            dim, dim, use_bias=False, rngs=rngs, dtype=dtype, kernel_init=kernel_init
        )
        self.o_proj = nnx.Linear(
            dim,
            dim,
            use_bias=False,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init_out_proj,
        )
        self.fc1 = nnx.Linear(
            dim, hidden_dim, rngs=rngs, dtype=dtype, kernel_init=kernel_init
        )
        self.fc2 = nnx.Linear(
            hidden_dim,
            dim,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init_out_proj,
        )

    def __call__(
        self,
        x: at.Float[at.Array, "b s d"],
        mask: at.Bool[at.Array, "b s"],
    ) -> at.Float[at.Array, "b s d"]:
        h = self.norm1(x)
        q = einops.rearrange(
            self.q_proj(h), "b s (nh hd) -> b nh s hd", nh=self.num_heads
        )
        k = einops.rearrange(
            self.k_proj(h), "b s (nh hd) -> b nh s hd", nh=self.num_heads
        )
        v = einops.rearrange(
            self.v_proj(h), "b s (nh hd) -> b nh s hd", nh=self.num_heads
        )

        attn_mask = mask[:, None, None, :]
        encoded = jax.nn.dot_product_attention(
            query=q,
            key=k,
            value=v,
            mask=attn_mask,
        )
        encoded = einops.rearrange(encoded, "b nh s hd -> b s (nh hd)")
        x = x + self.o_proj(encoded)

        h = self.norm2(x)
        x = x + self.fc2(nnx.gelu(self.fc1(h)))
        return x


class TemporalMemoryEncoder(nnx.Module):
    """Two-layer masked self-attention over memory tokens."""

    def __init__(
        self,
        config,
        rngs: nnx.Rngs,
        dtype: at.DTypeLike = jnp.float32,
    ):
        dim = config.memory_token_dim
        num_layers = int(getattr(config.temporal_encoder, "num_layers", 2))
        num_heads = int(getattr(config.temporal_encoder, "num_heads", 4))
        mlp_ratio = float(getattr(config.temporal_encoder, "mlp_ratio", 4))

        self.num_layers = num_layers
        for i in range(num_layers):
            setattr(
                self,
                f"layer_{i}",
                TemporalMemoryEncoderLayer(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    rngs=rngs,
                    dtype=dtype,
                ),
            )

    def __call__(
        self,
        x: at.Float[at.Array, "b s d"],
        mask: at.Bool[at.Array, "b s"] | None = None,
    ) -> at.Float[at.Array, "b s d"]:
        if mask is None:
            mask = jnp.ones(x.shape[:2], dtype=jnp.bool_)
        for i in range(self.num_layers):
            x = getattr(self, f"layer_{i}")(x, mask)
        return x
