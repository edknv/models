from dataclasses import dataclass
from typing import List, Optional, Tuple, TypeVar, Union

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
from merlin.models.torch.utils.llama_utils import convert_checkpoint, find_multiple

Self = TypeVar("Self", bound="LlamaBlock")


@dataclass
class LlamaConfig:
    block_size: int = 2048
    vocab_size: int = 32_000
    padded_vocab_size: Optional[int] = None
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096

    def __post_init__(self):
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, 64)

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(**LLAMA_CONFIGS[name])


LLAMA_CONFIGS = {
    "7B": dict(n_layer=32, n_head=32, n_embd=4096),
    "13B": dict(n_layer=40, n_head=40, n_embd=5120),
    "30B": dict(n_layer=60, n_head=52, n_embd=6656),
    "65B": dict(n_layer=80, n_head=64, n_embd=8192),
}


class LlamaBlock(nn.Module):
    def __init__(self, config: LlamaConfig, max_seq_length: Optional[int] = None) -> None:
        super().__init__()

        assert config.padded_vocab_size is not None

        self.config = config
        self.max_seq_length = max_seq_length or config.block_size

        self.transformer = LlamaTransformer(config)
        self.output_embeddings = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)

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
    def from_checkpoint(cls, checkpoint_dir, model_size="7B", device=None, dtype=None):
        model = cls.from_name(model_size)
        state_dict = convert_checkpoint(checkpoint_dir, model_size)
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
        self.max_seq_length = max_seq_length or config.block_size

        self.rotary_embeds = RotaryEmbeddings(
            self.config.n_embd // self.config.n_head,
            self.config.block_size,
        )
        self.mask_cache = AttentionMask(create_attention_mask(max_seq_length=self.max_seq_length))

        self.token_embeddings = nn.Embedding(config.padded_vocab_size, config.n_embd)
        self.heads = nn.ModuleList(
            LlamaAttentionHead(
                num_heads=config.n_head,
                embedding_dim=config.n_embd,
                max_seq_length=self.max_seq_length,
                rotary_embeds=self.rotary_embeds,
            )
            for _ in range(config.n_layer)
        )
        self.layernorm = RMSNorm(config.n_embd)

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
