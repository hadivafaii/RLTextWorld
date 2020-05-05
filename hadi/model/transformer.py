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

        self.config = config
        self.pretrain_modes = pretrain_modes

        conversion_dicts = []
        for mode_ in pretrain_modes:
            if 'ENTITY' in mode_:
                conversion_dicts.append(nlp.indx2entity)
            elif 'VERB' in mode_:
                conversion_dicts.append(nlp.indx2verb)
            elif 'MLM' in mode_:
                conversion_dicts.append(nlp.i2w)
            else:   # the rest is PERMUTE data
                continue

        self.conversion_dicts = conversion_dicts

        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.linear2 = nn.Linear(config.hidden_size, config.embedding_size, bias=True)
        self.norm1 = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.activation = _get_activation_fn(config.hidden_act)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, transformer_hidden, objects_embedded, labels):
        if type(objects_embedded) is not list:
            objects_embedded = [objects_embedded]

        if type(labels) is not list:
            labels = [labels]

        x = self.activation(self.linear1(transformer_hidden))
        x = self.linear2(self.norm1(x))

        predictions, sampled_indxs = (), ()
        for objs, lbls in zip(objects_embedded, labels):
            preds = x @ objs.T
            num_classes = len(objs)
            preds = preds.view(-1, num_classes)

            probs = F.softmax(preds, dim=1)
            sampled_ = probs.multinomial(num_samples=1).view(lbls.shape)

            predictions += (preds,)
            sampled_indxs += (sampled_,)

        return predictions, sampled_indxs

    def embed_objects(self, word_embeddings, indxs=None, reduction='mean', use_cuda=True):
        object_vectors_tuple = ()
        for pretrain_mode, conversion_dict in zip(self.pretrain_modses, self.conversion_dicts):
            if indxs is None:
                indxs = list(conversion_dict.keys())

            if pretrain_mode == 'MLM':
                if use_cuda:
                    object_vectors = word_embeddings(torch.tensor(indxs, dtype=torch.long).cuda())
                else:
                    object_vectors = word_embeddings(torch.tensor(indxs, dtype=torch.long))

            else:
                object_vectors_list_ = []
                for i in indxs:
                    if use_cuda:
                        obj_is = torch.tensor(conversion_dict[i], dtype=torch.long).cuda()
                    else:
                        obj_is = torch.tensor(conversion_dict[i], dtype=torch.long)  # , device=)#, device=device)
                    obj_v = word_embeddings(obj_is)

                    if reduction == 'mean':
                        obj_v = torch.mean(obj_v, dim=0, keepdim=True)
                    elif reduction == 'sum':
                        obj_v = torch.sum(obj_v, dim=0, keepdim=True)

                    object_vectors_list_.append(obj_v)
                object_vectors = torch.cat(object_vectors_list_, dim=0)

            object_vectors_tuple += (object_vectors,)

        return object_vectors_tuple

    def get_x_corrupt(self, x_masked, labels, sampled_indxs):
        x_corrupt_tuple = ()
        for pretrain_mode, conversion_dict, lbl, sampled_ in zip(
                self.pretrain_modes, self.conversion_dicts, labels, sampled_indxs):

            if pretrain_mode == 'MLM':  # this corresponds to MLM
                x_corrupt = dc(x_masked)
                x_corrupt[lbl != -100] = sampled_.flatten()[lbl != -100]

            else:  # corresponds to entity and verb recignition
                x_corrupt = np.zeros(x_masked.shape)
                for i in range(len(x_masked)):
                    unk_pos = np.where(lbl[i] != -100)[0]
                    x_corrupt[i] = _replace_objects(
                        x_masked[i],
                        indices=unk_pos,
                        replace_with=sampled_[i][unk_pos],
                        conversion_dict=conversion_dict,
                        unk_id=self.config.unk_id,
                    )[:x_masked.shape[-1]]

            x_corrupt_tuple += (x_corrupt.astype(int),)

        return x_corrupt_tuple


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

    def forward(self, inputs):
        embedded = self.embeddings(*inputs).transpose(1, 0)  # (S, N, E)
        last_hidden, hiddens, attn_weights = self.encoder(embedded)
        return last_hidden, hiddens, attn_weights


class Language(object):
    def __init__(self, data_config):
        lang_load_ = os.path.join(
            data_config.processed_dirs[0],
            'lang_data_max_len={:d}.npy'.format(data_config.max_len))
        lang_data_all = np.load(lang_load_, allow_pickle=True).item()

        max_vocab_size = 0
        winnder_eps = 1.00
        for eps in data_config.epsilons:
            lang_data_ = lang_data_all['eps={:.2f}'.format(eps)]
            if len(lang_data_['w2i']) > max_vocab_size:
                max_vocab_size = len(lang_data_['w2i'])
                winnder_eps = eps

        lang_data = lang_data_all['eps={:.2f}'.format(winnder_eps)]

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
