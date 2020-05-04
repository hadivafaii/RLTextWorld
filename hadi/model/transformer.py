import os
import numpy as np
from copy import deepcopy as dc

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

        self.config = config
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

        for _ in range(self.config.num_hidden_layers):
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


class GeneratorHead(nn.Module):
    def __init__(self, config, nlp, pretrain_modes):
        super(GeneratorHead, self).__init__()

        # TODO: make it work for more than 1 pretrain modes
        pretrain_mode = pretrain_modes[0]
        if 'ENTITY' in pretrain_mode:
            self.conversion_dict = nlp.indx2entity
        elif 'VERB' in pretrain_mode:
            self.conversion_dict = nlp.indx2verb
        else:
            self.conversion_dict = None

        self.pretrain_mode = pretrain_mode
        self.config = config

        self.dense = nn.Linear(config.hidden_size, config.embedding_size, bias=True)
        self.activation = _get_activation_fn(config.hidden_act)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, transformer_hidden, objects_embedded, labels):
        predictions = self.activation(self.dense(transformer_hidden)) @ objects_embedded.T
        num_classes = len(objects_embedded)
        predictions = predictions.view(-1, num_classes)

        probabilities = F.softmax(predictions, dim=1)
        sampled_indxs = probabilities.multinomial(num_samples=1).view(labels.shape)

        return predictions, sampled_indxs

    def embed_objects(self, word_embeddings, object_indxs=None, reduction='mean', use_cuda=True):
        if object_indxs is None:
            object_indxs = list(self.conversion_dict.keys())

        object_vectors_list = []
        for indx in object_indxs:
            if use_cuda:
                obj = torch.tensor(self.conversion_dict[indx], dtype=torch.long).cuda()
            else:
                obj = torch.tensor(self.conversion_dict[indx], dtype=torch.long)  # , device=)#, device=device)
            obj_v = word_embeddings(obj)

            if reduction == 'mean':
                obj_v = torch.mean(obj_v, dim=0, keepdim=True)
            elif reduction == 'sum':
                obj_v = torch.sum(obj_v, dim=0, keepdim=True)

            object_vectors_list.append(obj_v)

        return torch.cat(object_vectors_list, dim=0)

    def get_x_corrupt(self, x_masked, labels, sampled_indxs):
        if self.conversion_dict is None:  # this corresponds to MLM
            x_corrupt = dc(x_masked)
            x_corrupt[labels != -100] = sampled_indxs.flatten()[labels != -100]

        else:  # corresponds to entity and verb recignition
            x_corrupt = np.zeros(x_masked.shape)
            for i in range(len(x_masked)):
                unk_pos = np.where(labels[i] != -100)[0]
                x_corrupt[i] = _replace_objects(
                    x_masked[i],
                    indices=unk_pos,
                    replace_with=sampled_indxs[i][unk_pos],
                    conversion_dict=self.conversion_dict,
                    unk_id=self.config.unk_id,
                )[:x_masked.shape[-1]]

        return x_corrupt.astype(int)


class Transformer(nn.Module):
    def __init__(self, config, data_config):
        super(Transformer, self).__init__()

        self.config = config
        self.data_config = data_config
        self.nlp = Language(data_config)

        self.embeddings = Embeddings(config)
        self.encoder = TransformerEncoder(config)

        self.generator_head = GeneratorHead(config, self.nlp, pretrain_modes=data_config.pretrain_modes)

        # self.trainer = Trainer()
        # self.discriminator_head = DiscriminatorHead(config)
        # self.generator_head = GeneratorHead(config)

    def forward(self, inputs):
        embedded = self.embeddings(*inputs).transpose(1, 0)  # (S, N, E)
        last_hidden, hiddens, attn_weights = self.encoder(embedded)
        return last_hidden, hiddens, attn_weights


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

        self.data_config = data_config
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


# Helper functions
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def _replace_objects(x, indices, replace_with, conversion_dict, unk_id=3):
    x_replaced = dc(x)
    replacements = [conversion_dict[item] for item in replace_with]

    if not any([len(item) - 1 for item in replacements]):  # if len any of objects is longer than 1
        x_replaced[indices] = [item[0] for item in replacements]

    else:
        #   print([[nlp.i2w[t] for t in ent] for ent in replacements])
        extension = 0
        for i, item in zip(indices, replacements):
            x_replaced = np.insert(x_replaced, extension + i + 1, [unk_id] * (len(item) - 1))
            extension += len(item) - 1

        replacements_flattened = [item for sublist in replacements for item in sublist]
        x_replaced[x_replaced == unk_id] = replacements_flattened

    return x_replaced
