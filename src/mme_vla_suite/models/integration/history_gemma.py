import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

from openpi.models.gemma import PALIGEMMA_VOCAB_SIZE
from openpi.models.gemma import Config
from openpi.models.gemma import Embedder
from openpi.models.gemma import KVCache
from openpi.models.gemma import RMSNorm
from openpi.models.gemma import _apply_rope
from openpi.models.gemma import _gated_residual
from openpi.models.gemma import Attention
import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding


from mme_vla_suite.models.representation.utils import kernel_init_out_proj
from mme_vla_suite.models.integration.utils import _name
from mme_vla_suite.models.integration.utils import Attention_with_MemoryExpert
from mme_vla_suite.models.integration.utils import get_config, Variant


def _empty_stats() -> at.Float[at.Array, " n"]:
    return jnp.zeros((0,), dtype=jnp.float32)


def _summarize_attention_probs(
    probs: at.Float[at.Array, "b k g t s"], key_region_lengths: Sequence[int] | None
) -> at.Float[at.Array, " n"]:
    if key_region_lengths is None:
        return _empty_stats()

    region_means = []
    start = 0
    for length in key_region_lengths:
        end = start + length
        region_means.append(jnp.mean(jnp.sum(probs[..., start:end], axis=-1), dtype=jnp.float32))
        start = end
    return jnp.stack(region_means, axis=0)


@at.typecheck
class MemoryRMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x, cond=None, *, capture_stats: bool = False):
        dtype = x.dtype  # original dtype, could be half-precision
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))
        if cond is None:
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
            normed_inputs = normed_inputs * (1 + scale)
            return normed_inputs.astype(dtype), None
        
        # modulation = nn.Dense(x.shape[-1] * 2, kernel_init=nn.initializers.zeros, dtype=dtype)(cond)
        modulation = nn.Dense(x.shape[-1] * 2, kernel_init=kernel_init_out_proj, dtype=dtype)(cond) # add small randomness instead of pure zeros
        scale, shift = jnp.split(modulation, 2, axis=-1)
        normed_inputs = normed_inputs * (1 + scale) + shift
        stats = None
        if capture_stats:
            stats = {
                "scale": scale.astype(jnp.float32),
                "shift": shift.astype(jnp.float32),
            }
        return normed_inputs.astype(dtype), stats

class MemoryAttention(nn.Module):
    """
    Cross Attention for Memory Modulation
    Use action sequence to attend memory sequence.
    """
    @nn.compact
    def __call__(self, x, mem_seq, mem_mask):
        # x: [B, T, D], mem_seq: [B, S, D], mem_mask: [B, S]
        B, mem_len, mem_width = mem_seq.shape
        B, x_len, x_width = x.shape
        # Let's hardcode the values for now
        num_heads, num_kv_heads, head_dim, width = (
            4,
            1,
            256,
            1024,
        )  # same dim as the action expert in pi05
        assert mem_width == x_width == width
        q_einsum = lora.Einsum(
            shape=(num_heads, width, head_dim),
            name="q_einsum_mem",
            init_fn=nn.initializers.lecun_normal(
                in_axis=-2, out_axis=-1, batch_axis=(0,)
            ),
        )
        kv_einsum = lora.Einsum(
            shape=(2, num_kv_heads, width, head_dim),
            name="kv_einsum_mem",
            init_fn=nn.initializers.lecun_normal(
                in_axis=-2, out_axis=-1, batch_axis=(0, 1)
            ),
        )
        rms_norm = MemoryRMSNorm(name="mem_rms_norm")
        x, _ = rms_norm(x)
        q = q_einsum("BTD,NDH->BTNH", x)
        
        mem_seq, _ = rms_norm(mem_seq)
        k, v = kv_einsum("BSD,2KDH->2BSKH", mem_seq)
        
        q_positions = einops.repeat(
            jnp.arange(mem_len, x_len + mem_len), "t -> b t", b=B
        )
        k_positions = einops.repeat(jnp.arange(mem_len), "t -> b t", b=B)
        
        q = _apply_rope(q, positions=q_positions)
        q *= head_dim**-0.5
        k = _apply_rope(k, positions=k_positions)
        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=num_kv_heads)

        logits = jnp.einsum(
            "BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32
        )
        attn_mask = mem_mask[:, None, None, None, :]  # (B, 1, 1, 1, S)
        masked_logits = jnp.where(attn_mask, logits, -2.3819763e38)
        probs = jax.nn.softmax(masked_logits, axis=-1).astype(x.dtype)
        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        out_einsum = lora.Einsum(
            shape=(num_heads, head_dim, width),
            name="out_einsum_mem",
            init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
        )
        return out_einsum("BTNH,NHD->BTD", encoded)


@at.typecheck
class HistoryBlock(nn.Module):
    """Transformer block."""

    configs: tuple[Config, ...]

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    integration_type: str | None = None

    @nn.compact
    def __call__(
        self,
        xs,
        kv_cache,
        positions,
        attn_mask,
        adarms_cond,
        mem_seq,
        mem_mask,
        attention_stats_spec=None,
        capture_modulation_stats: bool = False,
        deterministic=True,
    ):  # noqa: FBT002

        if self.integration_type == "modulation":
            mem_attn = MemoryAttention(name="mem_attn")

        xs = sharding.activation_sharding_constraint(xs)
        drop = (
            nn.Dropout(self.dropout, self.dropout_bdims)
            if self.dropout
            else lambda x, _: x
        )
        
        if self.integration_type == "expert":
            attn = Attention_with_MemoryExpert(configs=self.configs, name="attn")
        else:
            attn = Attention(configs=self.configs, name="attn")

        pre_attn = []
        gates = []
        for i, x in enumerate(xs):
            if x is not None:
                name = _name("pre_attention_norm", i) if self.integration_type != "expert" else _name("pre_attention_norm", i-1)
                x, gate = RMSNorm(name=name)(
                    x, adarms_cond[i]
                )  # noqa: PLW2901
            pre_attn.append(x)
            gates.append(gate if x is not None else None)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        post_attn, kv_cache, attn_stats = attn(
            pre_attn,
            positions,
            attn_mask,
            kv_cache,
            attention_stats_spec=attention_stats_spec,
        )
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)
        xs = [
            _gated_residual(x, y, gate)
            for x, y, gate in zip(xs, post_attn, gates, strict=True)
        ]
        xs = sharding.activation_sharding_constraint(xs)
        

        out = []
        gates = []
        modulation_stats = None
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                # Add Memory Modulation before FFN
                if i == len(xs) - 1 and self.integration_type == "modulation":
                    mem_mod_vec = mem_attn(x, mem_seq[-1], mem_mask[-1])
                    x, modulation_stats = MemoryRMSNorm(name="mem_rms_norm_ffn")(
                        x, mem_mod_vec, capture_stats=capture_modulation_stats
                    )
                
                name=_name("pre_ffw_norm", i) if self.integration_type != "expert" else _name("pre_ffw_norm", i-1)
                x, gate = RMSNorm(name=name)(
                    x, adarms_cond[i]
                )  # noqa: PLW2901
                                
                name = _name("mlp", i) if self.integration_type != "expert" else _name("mlp", i-1)
                x = lora.FeedForward(  # noqa: PLW2901
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    name=name,
                    lora_config=config.lora_configs.get("ffn"),
                )(x)

            out.append(x)
            gates.append(gate if x is not None else None)

        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        xs = [
            _gated_residual(x, y, gate)
            for x, y, gate in zip(xs, out, gates, strict=True)
        ]
        xs = sharding.activation_sharding_constraint(xs)
        
        layer_stats = {
            "attention_region_means": attn_stats,
            "modulation_scale": (
                modulation_stats["scale"] if modulation_stats is not None else _empty_stats()
            ),
            "modulation_shift": (
                modulation_stats["shift"] if modulation_stats is not None else _empty_stats()
            ),
        }
        return xs, (kv_cache, layer_stats)


KVCache: TypeAlias = tuple[
    at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]
]


@at.typecheck
class Module(nn.Module):
    """Transformer model, supporting a mixture of different weights for different tokens."""

    configs: Sequence[Config]  # list of configs, one for each expert
    embed_dtype: str

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()  # Every float is dropped independently.
    adarms: bool = False
    
    integration_type: str | None = None

    def setup(self):
        # all experts must have the same depth
        assert all(config.depth == self.configs[0].depth for config in self.configs)
        embed_dim = self.configs[0].width if self.integration_type != "expert" else self.configs[1].width
        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=embed_dim,  # embedder for first expert only
            name="embedder",
        )
        block_cls = nn.remat(
            HistoryBlock,
            prevent_cse=False,
            static_argnums=(7, 8, 9),
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(
                0,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
            ),  # 0=kv_cache, 1=positions, 2=mask, 3=adarms_cond, 4=mem_seq, 5=mem_mask, 6=attention_stats_spec, 7=capture_modulation_stats, 8=deterministic
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
            integration_type=self.integration_type,
        )
        self.final_norms = [
            RMSNorm(name=_name("final_norm", i) if self.integration_type != "expert" else _name("final_norm", i-1)) for i in range(len(self.configs))
        ]

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    @at.typecheck
    def __call__(
        self,
        # list of token arrays, one for each expert, or None if that expert should not be run
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        adarms_cond: Sequence[at.Float[at.Array, "b _d"] | None] | None = None,
        *,
        kv_cache: KVCache | None = None,
        mem_seq: Sequence[at.Float[at.Array, "b lmem _d"] | None] | None = None,
        mem_mask: Sequence[at.Bool[at.Array, "b lmem"] | None] | None = None,
        attention_stats_spec: dict[str, Any] | None = None,
        capture_modulation_stats: bool = False,
        deterministic: bool = True,
        return_stats: bool = False,
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache] | tuple[
        Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache, dict[str, at.Array]
    ]:
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        embedded, (kv_cache, layer_stats) = self.layers(
            embedded,
            kv_cache,
            positions,
            mask,
            adarms_cond,
            mem_seq,
            mem_mask,
            attention_stats_spec,
            capture_modulation_stats,
            deterministic,
        )

        assert all(
            e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None
        )

        embedded = [
            f(e, a)[0] if e is not None else e
            for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ]
        if return_stats:
            return embedded, kv_cache, layer_stats
        return embedded, kv_cache

    def init(self, use_adarms: Sequence[bool], mem_mods: Sequence[bool]):
        """Convenience method for initializing all parameters, necessary due to the quirks of linen."""
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[
                jnp.zeros((1, c.width)) if u else None
                for u, c in zip(use_adarms, self.configs, strict=True)
            ],
            mem_seq=[
                jnp.zeros((1, 4, c.width)) if m else None
                for c, m in zip(self.configs, mem_mods, strict=True)
            ],
            mem_mask=[jnp.ones((1, 4), dtype=bool) if m else None for m in mem_mods],
        )
