import torch
from torch import nn

import numpy as np
import math


def _get_postitional_embeddings(max_len, embedding_size):
    pe = torch.zeros(max_len, embedding_size)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.embedding_size = config.embedding_size

        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.embedding_size,
            padding_idx=config.pad_id,
        )
        self.type_embeddings = nn.Embedding(
            config.type_vocab_size,
            config.embedding_size,
            padding_idx=config.pad_id,
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.embedding_size,
            padding_idx=config.pad_id,
        )

        if config.fixed_pe:
            pe = _get_postitional_embeddings(config.max_position_embeddings - 1, config.embedding_size)
            pe = torch.cat([torch.zeros((1, config.embedding_size)), pe])  # padding_idx = 0
            self.position_embeddings.weight.data.copy_(pe)
            for param in self.position_embeddings.parameters():
                param.requires_grad = False

        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps, elementwise_affine=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, token_ids, type_ids, position_ids):
        """
        :param token_ids: max_len x batch_size (S, N)
        :param type_ids: max_len x batch_size (S, N)
        :param position_ids: max_len x batch_size (S, N)
        :return: embedded inputs
        """

        token_embeddings = self.word_embeddings(token_ids)
        type_embeddings = self.type_embeddings(type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = token_embeddings + type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings   # (S, N, H)
