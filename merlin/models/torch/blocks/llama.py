from dataclasses import dataclass
from typing import Optional, TypeVar

import torch
import torch.nn as nn

from merlin.models.torch.blocks.attention import (
    AttentionMask,
    CausalSelfAttention,
    RotaryEmbeddings,
    create_attention_mask,
)
from merlin.models.torch.blocks.mlp import PositionwiseFeedForward
from merlin.models.torch.transforms.regularization import RMSNorm
from merlin.models.torch.utils.llama_utils import (
    convert_checkpoint,
    find_multiple,
    llama_model_lookup,
)

Self = TypeVar("Self", bound="LlamaBlock")


@dataclass
class LlamaConfig:
    max_seq_length: int = 2048
    vocab_size: int = 32_000
    padded_vocab_size: Optional[int] = None
    num_layers: int = 32
    num_heads: int = 32
    embedding_dim: int = 4096

    def __post_init__(self):
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, 64)

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(**LLAMA_CONFIGS[name])


LLAMA_CONFIGS = {
    "7B": dict(num_layers=32, num_heads=32, embedding_dim=4096),
    "13B": dict(num_layers=40, num_heads=40, embedding_dim=5120),
    "30B": dict(num_layers=60, num_heads=52, embedding_dim=6656),
    "65B": dict(num_layers=80, num_heads=64, embedding_dim=8192),
}


class LlamaBlock(nn.Module):
    def __init__(self, config: LlamaConfig, max_seq_length: Optional[int] = None) -> None:
        super().__init__()

        assert config.padded_vocab_size is not None

        self.config = config
        self.max_seq_length = max_seq_length or config.max_seq_length

        self.transformer = LlamaTransformer(config)
        self.output_embeddings = nn.Linear(
            config.embedding_dim, config.padded_vocab_size, bias=False
        )

    def forward(
        self,
        inputs: torch.Tensor,
        max_seq_length: Optional[int] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.transformer(inputs, positions=positions)
        logits = self.output_embeddings(outputs)
        return logits

    @classmethod
    def from_name(cls, model_size: str) -> Self:
        return cls(LlamaConfig.from_name(model_size))

    @classmethod
    def from_checkpoint(
        cls, checkpoint_dir, model_size: Optional[str] = None, device=None, dtype=None
    ):
        state_dict = convert_checkpoint(checkpoint_dir, model_size)
        model_size = model_size or llama_model_lookup(state_dict)
        model = cls.from_name(model_size)
        model.load_state_dict(state_dict)
        return model

    def reset_cache(self) -> None:
        for head in self.transformer.heads:
            head.kv_cache = None


class LlamaTransformer(nn.Module):
    def __init__(self, config: LlamaConfig, max_seq_length: Optional[int] = None) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None

        self.config = config
        self.max_seq_length = max_seq_length or config.max_seq_length

        self.rotary_embeds = RotaryEmbeddings(
            self.config.embedding_dim // self.config.num_heads,
            self.config.max_seq_length,
        )
        self.mask_cache = AttentionMask(create_attention_mask(max_seq_length=self.max_seq_length))

        self.token_embeddings = nn.Embedding(config.padded_vocab_size, config.embedding_dim)
        self.heads = nn.ModuleList(
            LlamaAttentionHead(
                num_heads=config.num_heads,
                embedding_dim=config.embedding_dim,
                max_seq_length=self.max_seq_length,
                rotary_embeds=self.rotary_embeds,
            )
            for _ in range(config.num_layers)
        )
        self.layernorm = RMSNorm(config.embedding_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_length = inputs.size()

        if positions is not None:
            mask = self.mask_cache.select_position(positions)
        else:
            mask = self.mask_cache.select(seq_length)

        x = self.token_embeddings(inputs)

        if positions is None:
            for block in self.heads:
                x = block(
                    x,
                    mask=mask,
                )
        else:
            for head in self.heads:
                x = head(
                    x,
                    positions=positions,
                    mask=mask,
                )

        x = self.layernorm(x)

        return x

    @classmethod
    def from_name(cls, model_size: str) -> Self:
        return cls(LlamaConfig.from_name(model_size))


class LlamaAttentionHead(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        max_seq_length: int,
        rotary_embeds: Optional[RotaryEmbeddings] = None,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.max_seq_length = max_seq_length
        self.rotary_embeds = rotary_embeds

        self.input_layernorm = RMSNorm(self.embedding_dim)
        self.attention = CausalSelfAttention(
            num_heads=self.num_heads,
            embedding_dim=self.embedding_dim,
            max_seq_length=self.max_seq_length,
            rotary_embeds=self.rotary_embeds,
        )
        self.post_attention_layernorm = RMSNorm(self.embedding_dim)

        hidden_dim = 4 * self.embedding_dim
        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = find_multiple(n_hidden, 256)
        self.mlp = PositionwiseFeedForward(
            self.embedding_dim, n_hidden, bias=False, activation=nn.SiLU
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        mask: Optional[AttentionMask] = None,
    ) -> torch.Tensor:
        x = x + self.attention(
            self.input_layernorm(x),
            positions=positions,
            mask=mask,
        )
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
