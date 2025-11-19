# https://www.youtube.com/watch?v=l8pRSuU81PU

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F




@dataclass
class GPTConfiguration:
    block_size: int = 256
    vocab_size: int = 65
    total_layers: int = 6
    total_heads: int = 6
    total_embeddings: int = 384

class GPT(nn.Module):

    def __int__(self, configuration: GPTConfiguration):
        super().__init__()
        self.configuration = configuration

        # nn.ModuleDict is a Module that allows to index the submodules using keys, like a dictionary
        # nn.Embedding is a wrapper Module around a single array of numbers. It allows to access the tensor's elements by index
        # nn.ModuleList like a ModuleDict, but we can index it using integers
        self.transformer = nn.ModuleDict(
            dict(
                # Weights of the Token Embeddings
                wte = nn.Embedding(configuration.vocab_size, configuration.total_embeddings),

                # Weights of the Position Embeddings
                wpe = nn.Embedding(configuration.block_size, configuration.total_embeddings),

                # Hidden layers
                h = nn.ModuleList( [Block(configuration) for _ in range(configuration.total_layers)] ),

                # Linear layer
                ln_f = nn.LayerNorm(configuration.total_embeddings)
            )
        )

        # Final classifier: Language Model Head
        self.lm_head = nn.Linear( configuration.total_embeddings, configuration.vocab_size, bias=False )

