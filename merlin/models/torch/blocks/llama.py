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
        return cls(**llama_configs[name])


llama_configs = {
    "7B": dict(n_layer=32, n_head=32, n_embd=4096),
    "13B": dict(n_layer=40, n_head=40, n_embd=5120),
    "30B": dict(n_layer=60, n_head=52, n_embd=6656),
    "65B": dict(n_layer=80, n_head=64, n_embd=8192),
}


class LlamaBlock(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.transformer = LlamaTransformer(config)
        self.output_embeddings = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)

    def forward(
        self,
        inputs: torch.Tensor,
        max_seq_length: Optional[int] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.transformer(inputs, max_seq_length=max_seq_length, positions=positions)
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
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.token_embeddings = nn.Embedding(config.padded_vocab_size, config.n_embd)
        self.heads = nn.ModuleList(LlamaAttentionHead(config) for _ in range(config.n_layer))
        self.layernorm = RMSNorm(config.n_embd)

        self.rotary_embeds: Optional[RotaryEmbeddings] = None
        self.mask_cache: Optional[AttentionMask] = None

    def forward(
        self,
        inputs: torch.Tensor,
        max_seq_length: Optional[int] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_length = inputs.size()

        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size

        if self.rotary_embeds is None:
            self.rotary_embeds = RotaryEmbeddings(
                self.config.n_embd // self.config.n_head,
                self.config.block_size,
            )

        if self.mask_cache is None:
            self.mask_cache = AttentionMask(
                create_attention_mask(max_seq_length=max_seq_length, device=inputs.device)
            )

        if positions is not None:
            mask = self.mask_cache.select_position(positions)
        else:
            mask = self.mask_cache.select(seq_length)

        rope = self.rotary_embeds

        x = self.token_embeddings(inputs)

        if positions is None:
            for block in self.heads:
                x = block(x, rope, mask, max_seq_length)
        else:
            for i, block in enumerate(self.heads):
                x = block(
                    x,
                    rope,
                    mask,
                    max_seq_length,
                    positions,
                )

        x = self.layernorm(x)

        return x

    @classmethod
    def from_name(cls, model_size: str) -> Self:
        return cls(LlamaConfig.from_name(model_size))


class LlamaAttentionHead(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()

        self.input_layernorm = RMSNorm(config.n_embd)
        self.attention = CausalSelfAttention(
            num_heads=config.n_head,
            embedding_dim=config.n_embd,
            max_seq_length=config.block_size,
        )
        self.post_attention_layernorm = RMSNorm(config.n_embd)

        hidden_dim = 4 * config.n_embd
        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = find_multiple(n_hidden, 256)
        self.mlp = PositionwiseFeedForward(config.n_embd, n_hidden, bias=False, activation=nn.SiLU)

    def forward(
        self,
        x: torch.Tensor,
        rope,
        mask: Optional[AttentionMask],
        max_seq_length: int,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.attention(self.input_layernorm(x), rope, mask, max_seq_length, positions)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
