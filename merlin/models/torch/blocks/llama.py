"""Full definition of a Llama Language Model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
"""
# mypy: ignore-errors
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, TypeVar, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from merlin.models.torch.blocks.attention import RotaryEmbeddings
from merlin.models.torch.blocks.mlp import PositionwiseFeedForward
from merlin.models.torch.transforms.regularization import RMSNorm
from merlin.models.torch.utils.llama_utils import convert_checkpoint, find_multiple

Self = TypeVar("Self", bound="LlamaBlock")

MaskCache = torch.Tensor
KVCache = Tuple[torch.Tensor, torch.Tensor]


@dataclass
class LlamaConfig:
    block_size: int = 2048
    vocab_size: int = 32000
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


@dataclass
class AttentionMask:
    def __init__(self, bool_mask: Optional[torch.Tensor] = None) -> None:
        self.bool_mask = bool_mask

    def select(self, seq_length: int) -> torch.Tensor:
        return self.bool_mask[:, :, :seq_length, :seq_length]

    def select_position(self, position: torch.Tensor) -> torch.Tensor:
        return self.bool_mask.index_select(2, position)


def create_attention_mask(
    max_seq_length: int, device: Optional[torch.device] = None
) -> torch.Tensor:
    ones = torch.ones(
        (max_seq_length, max_seq_length),
        device=device,
        dtype=torch.bool,
    )
    return torch.tril(ones).unsqueeze(0).unsqueeze(0)


class LlamaBlock(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=RMSNorm(config.n_embd),
            )
        )

        self.rotary_embeds: Optional[RotaryEmbeddings] = None
        self.mask_cache: Optional[MaskCache] = None
        self.kv_caches: List[KVCache] = []

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer)
            )
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer)
            )

    def forward(
        self,
        idx: torch.Tensor,
        max_seq_length: Optional[int] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[KVCache]]]:
        B, T = idx.size()

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
                create_attention_mask(max_seq_length=max_seq_length, device=idx.device)
            )

        if input_pos is not None:
            # mask = self.mask_cache.index_select(2, input_pos)
            # mask = mask[:, :, :, :max_seq_length]
            mask = self.mask_cache.select_position(input_pos)
        else:
            mask = self.mask_cache[:, :, :T, :T]

        rope = self.rotary_embeds

        # forward the model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        if input_pos is None:  # proxy for use_cache=False
            for block in self.transformer.h:
                x, _ = block(x, rope, mask, max_seq_length)
        else:
            if not self.kv_caches:
                head_size = self.config.n_embd // self.config.n_head
                cache_shape = (B, self.config.n_head, max_seq_length, head_size)
                self.kv_caches = [
                    (
                        torch.zeros(cache_shape, device=x.device, dtype=x.dtype),
                        torch.zeros(cache_shape, device=x.device, dtype=x.dtype),
                    )
                    for _ in range(self.config.n_layer)
                ]
            for i, block in enumerate(self.transformer.h):
                x, self.kv_caches[i] = block(
                    x, rope, mask, max_seq_length, input_pos, self.kv_caches[i]
                )

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)  # (b, t, vocab_size)

        return logits

    @classmethod
    def from_name(cls, model_size: str) -> Self:
        return cls(LlamaConfig.from_name(model_size))

    @classmethod
    def from_checkpoint(cls, checkpoint_dir, model_size="7B", device=None, dtype=None):
        # model = cls.from_name(model_size)
        # state_dict = torch.load(checkpoint_dir)
        # model.load_state_dict(state_dict)

        model = cls.from_name(model_size)
        state_dict = convert_checkpoint(checkpoint_dir, model_size)
        model.load_state_dict(state_dict)
        return model

    def build_mask_cache(self, idx: torch.Tensor) -> MaskCache:
        ones = torch.ones(
            (self.config.block_size, self.config.block_size), device=idx.device, dtype=torch.bool
        )
        return torch.tril(ones).unsqueeze(0).unsqueeze(0)

    def reset_cache(self) -> None:
        self.kv_caches.clear()
        # if self.mask_cache.device.type == "xla":
        #    # https://github.com/Lightning-AI/lit-parrot/pull/83#issuecomment-1558150179
        #    self.rope_embeds = None
        #    self.mask_cache = None


class Block(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.rms_2 = RMSNorm(config.n_embd)

        hidden_dim = 4 * config.n_embd
        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = find_multiple(n_hidden, 256)
        self.mlp = PositionwiseFeedForward(config.n_embd, n_hidden, bias=False, activation=nn.SiLU)

    def forward(
        self,
        x: torch.Tensor,
        rope,
        mask: MaskCache,
        max_seq_length: int,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        h, new_kv_cache = self.attn(self.rms_1(x), rope, mask, max_seq_length, input_pos, kv_cache)
        x = x + h
        x = x + self.mlp(self.rms_2(x))
        return x, new_kv_cache


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size

    def forward(
        self,
        x: torch.Tensor,
        rope,
        mask: MaskCache,
        max_seq_length: int,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch
        # and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size)
        q = q.view(B, T, self.n_head, head_size)
        v = v.view(B, T, self.n_head, head_size)

        # q = apply_rope(q, rope)
        # k = apply_rope(k, rope)
        q = rope(q, input_pos)
        k = rope(k, input_pos)

        k = k.transpose(1, 2)  # (B, nh, T, hs)
        q = q.transpose(1, 2)  # (B, nh, T, hs)
        v = v.transpose(1, 2)  # (B, nh, T, hs)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            # check if reached token limit
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                # shift 1 position to the left
                cache_k = torch.roll(cache_k, -1, dims=2)
                cache_v = torch.roll(cache_v, -1, dims=2)
            k = cache_k.index_copy(2, input_pos, k)
            v = cache_v.index_copy(2, input_pos, v)
            kv_cache = k, v

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        #  att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #  att = att.masked_fill(mask[:,:,:T,:T] == 0, float('-inf'))
        #  att = F.softmax(att, dim=-1)
        #  y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)

        return y, kv_cache
