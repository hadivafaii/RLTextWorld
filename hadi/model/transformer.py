import os
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from .embeddings import Embeddings
from .preprocessing import get_nlp, preproc


# TODO: add TransformerEncoderLayer and then if share_weights=True then go ALBERT style
#  in TransformerEncoder, if not then have separate laters and define _clone_layers function
class TransformerEncoder(nn.Module):

    def __init__(self, config):
        super(TransformerEncoder, self).__init__()

        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)

        self.self_attn = nn.MultiheadAttention(
            config.hidden_size, config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob)

        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)

        self.norm1 = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout1 = nn.Dropout(config.attention_probs_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

        self.activation = _get_activation_fn(config.hidden_act)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoder, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # be sure to transpose embedded inside forward of Transformer to get: embedded size = (S, N, E)
        src = self.embedding_hidden_mapping_in(src)     # (S, N, H)

        outputs, attn_outputs = (src,), ()

        for _ in range(config.num_hidden_layers):
            src2, attn_weights = self.self_attn(
                src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
            src = src + self.dropout1(src2)
            src = self.norm1(src)

            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

            outputs += (src,)
            attn_outputs += (attn_weights,)

        return src, outputs, attn_outputs


class Language(object):
    def __init__(self, data_config):
        lang_load_ = os.path.join(
            data_config.processed_dir,
            'lang_data_max_len={:d}.npy'.format(data_config.max_len))
        lang_data_all = np.load(lang_load_, allow_pickle=True).item()
        lang_data = lang_data_all['eps={:.2f}'.format(data_config.eps)]

        self.w2i = lang_data['w2i']
        self.i2w = lang_data['i2w']
        self.vocab = list(self.w2i.keys())
        self.vocab_size = len(self.vocab)

        self.entity2indx = lang_data['entity2indx']
        self.indx2entity = lang_data['indx2entity']

        self.verb2indx = lang_data['verb2indx']
        self.indx2verb = lang_data['indx2verb']

        self.tokenizer = get_nlp().tokenizer

    def add_word(self, word):
        if word in self.w2i:
            raise RuntimeError("{} already exists in vocab".format(word))
        else:
            self.w2i.update({word: self.vocab_size})
            self.i2w.update({self.vocab_size: word})
            self.vocab = list(self.w2i.keys())
            self.vocab_size = len(self.w2i)

    def preproc(self, string):
        return preproc(string, self.tokenizer)


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class Transformer(nn.Module):
    def __init__(self, config, data_config):
        super(Transformer, self).__init__()

        self.config = config
        self.data_config = data_config
        self.nlp = Language(data_config)

        self.embeddings = Embeddings(config)
        self.encoder = TransformerEncoder(config)

        # self.trainer = Trainer()
        # self.discriminator_head = DiscriminatorHead(config)
        # self.generator_head = GeneratorHead(config)

    def forward(self, x):
        embedded = self.embeddings(x)
        src = self.encoder(embedded)
        return src
