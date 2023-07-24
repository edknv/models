import torch

import merlin.models.torch as mm
from merlin.models.torch.utils import module_utils


class TestLlamaBlock:
    def setup_method(self):
        self.llama_config = mm.LlamaConfig(
            max_seq_length=64,
            vocab_size=100,
            num_layers=1,
            num_heads=1,
            embedding_dim=128,
        )
        self.llama = mm.LlamaBlock(self.llama_config)
        self.input_dict = {
            "token": torch.tensor([[1, 3, 36, 2, 10]]),
            "position": torch.tensor([0, 1, 2, 3, 4]),
        }

    def test_forward(self):
        out = self.llama(self.input_dict)
        assert isinstance(out, torch.Tensor)
        assert out.shape[:-1] == self.input_dict["token"].shape
        assert out.shape[-1] == self.llama_config.embedding_dim

    def test_forward_torchscript(self):
        out = module_utils.module_test(self.llama, self.input_dict)
        assert isinstance(out, torch.Tensor)
        assert out.shape[:-1] == self.input_dict["token"].shape
        assert out.shape[-1] == self.llama_config.embedding_dim
