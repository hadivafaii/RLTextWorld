import os
import numpy as np
from copy import deepcopy as dc
from itertools import chain, compress
from datetime import datetime
import yaml

import torch
from torch import nn
import torch.nn.functional as F

from .embeddings import Embeddings
from .preprocessing import get_nlp, preproc
from .configuration import TransformerConfig, DataConfig
import sys; sys.path.append('..')
from utils.gen_pretrain_data import get_ranges


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

      #  self.pooler = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
      #  self.poooler_activation = nn.Tanh()

        self.config = config
        self.activation = _get_activation_fn(config.hidden_act)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoder, self).__setstate__(state)

    def forward(self, src, src_mask, src_key_padding_mask=None):
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

    def create_attention_mask(self, mask_bool):
        """
        :param mask_bool: batch_size x max_len
        :return: mask_square_additive: bath_size x max_len x max_len
        """
        if type(mask_bool) == torch.Tensor:
            mask = mask_bool.float()
        else:
            mask = torch.tensor(mask_bool, dtype=torch.float)

        mask_square = torch.einsum('bij,bjk->bik', torch.ones(mask.unsqueeze(-1).size(), device=mask.device), mask.unsqueeze(-2))

        mask_square_additive = mask_square.masked_fill(
            mask_square == 0, float('-inf')).masked_fill(mask_square == 1, float(0.0))

        return torch.repeat_interleave(mask_square_additive, repeats=self.config.num_attention_heads, dim=0)


class Generator(nn.Module):
    def __init__(self, config, nlp, pretrain_modes):
        super(Generator, self).__init__()

        self.config = config
        self.pretrain_modes = pretrain_modes

        conversion_dicts = {}
        for mode_ in pretrain_modes:
            if 'ENTITY' in mode_:
                conversion_dicts.update({mode_: nlp.indx2entity})
            elif 'VERB' in mode_:
                conversion_dicts.update({mode_: nlp.indx2verb})
            elif 'MLM' in mode_:
                conversion_dicts.update({mode_: nlp.i2w})
            else:   # the rest is PERMUTE data
                continue

        self.conversion_dicts = conversion_dicts

        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.linear2 = nn.Linear(config.hidden_size, config.embedding_size, bias=True)
        self.norm1 = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.embedding_size, config.layer_norm_eps)
        self.activation = _get_activation_fn(config.hidden_act)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, transformer_hiddens, objects_embedded, labels):
        # x = self.activation(self.linear1(transformer_hiddens))
        # x = self.linear2(self.norm1(x))
        x = self.norm2(self.activation(self.linear2(transformer_hiddens)))

        num_classes = len(objects_embedded)
        predictions = x @ objects_embedded.T
        predictions = predictions.view(-1, num_classes)

        if self.config.generator_temperature != 1.0:
            predictions = predictions / self.config.generator_temperature
        probs = F.softmax(predictions, dim=1)
        sampled_indxs = probs.multinomial(num_samples=1).view(labels.shape).detach()

        return predictions, sampled_indxs

    def embed_objects(self, word_embeddings, pretrain_mode, indxs=None, reduction='mean'):
        conversion_dict = self.conversion_dicts[pretrain_mode]

        _device = word_embeddings.weight.device

        if indxs is None:
            indxs = list(conversion_dict.keys())

        if pretrain_mode == 'MLM':
            _illegal_indices = [self.config.pad_id, self.config.obs_id, self.config.act_id, self.config.unk_id]
            indxs = list(filter(lambda i: i not in _illegal_indices, indxs))
            object_vectors = word_embeddings(torch.tensor(indxs, dtype=torch.long, device=_device))

        elif pretrain_mode in ['ACT_ENTITY', 'ACT_VERB', 'OBS_ENTITY', 'OBS_VERB']:
            object_vectors_list_ = []
            for i in indxs:
                obj_is = torch.tensor(conversion_dict[i], dtype=torch.long, device=_device)
                obj_v = word_embeddings(obj_is)

                if reduction == 'mean':
                    obj_v = torch.mean(obj_v, dim=0, keepdim=True)
                elif reduction == 'sum':
                    obj_v = torch.sum(obj_v, dim=0, keepdim=True)

                object_vectors_list_.append(obj_v)
            object_vectors = torch.cat(object_vectors_list_, dim=0)

        else:
            raise ValueError("Invalid pretrain mode, {}, encountered in generator".format(pretrain_mode))

        return object_vectors

    def get_x_corrupt(self, x_masked, labels, sampled_indxs, pretrain_mode):
        if pretrain_mode == 'MLM':  # this corresponds to MLM
            x_corrupt = dc(x_masked)
            x_corrupt[labels != -100] = sampled_indxs[labels != -100]

        elif pretrain_mode in ['ACT_ENTITY', 'ACT_VERB', 'OBS_ENTITY', 'OBS_VERB']:
            x_corrupt = np.zeros(x_masked.shape, dtype=int)
            for i in range(len(x_masked)):
                unk_pos = np.where(labels[i] != -100)[0]
                x_corrupt[i] = _replace_objects(
                    x_masked[i],
                    indices=unk_pos,
                    replace_with=sampled_indxs[i][unk_pos],
                    conversion_dict=self.conversion_dicts[pretrain_mode],
                    unk_id=self.config.unk_id,
                )[:x_masked.shape[-1]]
        else:
            raise ValueError("Invalid pretrain mode, {}, encountered in generator".format(pretrain_mode))

        return x_corrupt


class Discriminator(nn.Module):
    def __init__(self, config, nlp, pretrain_modes):
        super(Discriminator, self).__init__()

        self.config = config
        self.pretrain_modes = pretrain_modes

        conversion_dicts = {}
        for mode_ in pretrain_modes:
            if 'ENTITY' in mode_:
                conversion_dicts.update({mode_: nlp.entity2indx})
            elif 'VERB' in mode_:
                conversion_dicts.update({mode_: nlp.verb2indx})
            elif 'MLM' in mode_:
                conversion_dicts.update({mode_: nlp.w2i})
            elif 'ORDER' in mode_:
                raise NotImplementedError
            elif 'PREDICT' in mode_:
                raise NotImplementedError
            else:
                raise ValueError("invalid pretrain mode")

        self.conversion_dicts = conversion_dicts

        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.activation = _get_activation_fn(config.hidden_act)

        # TODO: have a single weight for different modes or give each unique w?
        if set(pretrain_modes).intersection({'ACT_ENTITY', 'ACT_VERB', 'OBS_ENTITY', 'OBS_VERB', 'MLM'}):
            self.lin_proj = nn.Linear(config.hidden_size, 1, bias=True)
        elif set(pretrain_modes).intersection({'ACT_ORDER', 'OBS_ORDER', 'ACT_PREDICT', 'OBS_PREDICT'}):
            # self.rnn_proj = nn.GRUCell(config.hidden_size, 1) ## for when 2nd stage attn is needed
            self.rnn_proj = nn.LSTM(config.hidden_size, 1)

        #  self.linear2 = nn.Linear(config.hidden_size, config.embedding_size, bias=True)
        #  self.norm1 = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)
        #  self.activation = _get_activation_fn(config.hidden_act)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, transformer_hidden, flat_indices, pretrain_mode):
        # these are pre-sigmoid, the loss has sigmoid in it so no need to apply it here
        if pretrain_mode in ['ACT_ENTITY', 'ACT_VERB', 'OBS_ENTITY', 'OBS_VERB', 'MLM']:
            if pretrain_mode != 'MLM' and len(np.unique([len(item) for item in flat_indices])) > 1:
                h = torch.cat(list(map(lambda z: torch.mean(transformer_hidden.view(-1, self.config.hidden_size)[z], dim=0, keepdim=True), flat_indices)))
            else:
                try:    # first make sure flat_indices is a list, not a list of len 1 arrays
                    flat_indices = [inds.item() for inds in flat_indices]
                except AttributeError:
                    pass
                h = transformer_hidden.view(-1, self.config.hidden_size)[flat_indices]
            predictions = self.lin_proj(self.activation(self.dense(h)))

        elif pretrain_mode in ['ACT_ORDER', 'OBS_ORDER']:
            predictions = self.rnn_proj(hidden)

        elif pretrain_mode in ['ACT_PREDICT', 'OBS_PREDICT']:
            raise NotImplemented

        else:
            raise ValueError('invalid pretrain mdoe')
        #  predictions += do contrastive s

        return predictions.flatten()

    def get_discriminator_labels(self, corrupted_token_ids, masked_token_ids, generator_replaced_labels, pretrain_mode):
        conversion_dict = self.conversion_dicts[pretrain_mode]

        if pretrain_mode == 'MLM':
            x_corrupt_flat = corrupted_token_ids.flatten()
            x_masked_flat = masked_token_ids.flatten()

            labels = np.ones(len(x_corrupt_flat))
            unk_indices = np.where(x_masked_flat == self.config.unk_id)[0]
            labels[unk_indices] = 0

            _illegal_indices = [self.config.pad_id]     # discriminator loss will run on these
            flat_indices = [tup[0] for tup in enumerate(x_corrupt_flat) if tup[1] not in _illegal_indices]
            final_discriminator_labels = labels[flat_indices]

        else:
            ranges_chained, corrupted_ranges_labels = _extract_object_info(
                corrupted_token_ids,
                conversion_dict,
                pretrain_mode,
                self.config)

            corrupted_ranges_labels = sorted(corrupted_ranges_labels, key=lambda tup: tup[0].start)
            all_lbls_found_corrupt = [tup[1] for tup in corrupted_ranges_labels]

            _, masked_ranges_labels = _extract_object_info(
                masked_token_ids,
                conversion_dict,
                pretrain_mode,
                self.config)

            masked_ranges_labels = sorted(masked_ranges_labels, key=lambda tup: tup[0].start)
            all_lbls_found_masked = [tup[1] for tup in masked_ranges_labels]

            discriminator_labels = np.ones(len(all_lbls_found_corrupt), dtype=int) * -1

            pos_i = 0
            neg_i = 0
            for i, lbl in enumerate(all_lbls_found_corrupt):
                if neg_i < len(generator_replaced_labels) and lbl == generator_replaced_labels[neg_i]:
                    discriminator_labels[i] = 0
                    neg_i += 1
                elif pos_i < len(all_lbls_found_masked) and lbl == all_lbls_found_masked[pos_i]:
                    discriminator_labels[i] = 1
                    pos_i += 1

            valid_obj_indices = np.where(discriminator_labels >= 0)[0]
            final_discriminator_labels = discriminator_labels[valid_obj_indices]
            final_ranges_labels = [
                tup[0] for tup in zip(corrupted_ranges_labels, final_discriminator_labels) if tup[1] >= 0]
            flat_indices = [np.array(ranges_chained)[tup[0]] for tup in final_ranges_labels]

        return torch.tensor(final_discriminator_labels, dtype=torch.float), flat_indices


class Transformer(nn.Module):
    def __init__(self, config, data_config):
        super(Transformer, self).__init__()

        self.config = config
        self.data_config = data_config
        self.nlp = Language(data_config)

        self.embeddings = Embeddings(config)
        self.encoder = TransformerEncoder(config)
        # self.decoder = TransformerDecoder(config)

        self.generator = Generator(config, self.nlp, pretrain_modes=data_config.pretrain_modes)
        self.discriminator = Discriminator(config, self.nlp, pretrain_modes=data_config.pretrain_modes)

        self.init_weights()

    def forward(self, inputs, attention_masks=None):
        embedded = self.embeddings(*inputs).transpose(1, 0)  # (S, N, E)
        if attention_masks is None:
            attention_masks = self.encoder.create_attention_mask(inputs[2] > 0)
        last_hidden, hiddens, attn_weights = self.encoder(embedded, src_mask=attention_masks)
        return last_hidden, hiddens, attn_weights

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def save(self):
        config_dict = vars(self.config)
        data_config_dict = vars(self.data_config)

        to_hash_dict_ = dc(config_dict)
        to_hash_dict_.update(data_config_dict)
        hashed_info = hash(frozenset(sorted(to_hash_dict_)))

        save_dir = os.path.join(
            self.data_config.model_save_dir, "[{}]_{:s}".format(hashed_info, datetime.now().strftime("[%Y_%m_%d_%H:%M]")))
        os.makedirs(save_dir, exist_ok=True)

        torch.save(self.state_dict(), os.path.join(save_dir, 'model.bin'))

        with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config_dict, f)

        with open(os.path.join(save_dir, 'data_config.yaml'), 'w') as f:
            yaml.dump(data_config_dict, f)

    @staticmethod
    def load(load_dir=None, verbose=True):
        if load_dir is None:
            _dir = os.path.join(os.environ['HOME'], 'Documents/FTWP/SAVED_MODELS')
            available_models = os.listdir(_dir)
            if verbose:
                print('Available models to load:\n', available_models)
            load_dir = os.path.join(_dir, available_models[-1])

        if verbose:
            print('\nLoading from:\n', load_dir)

        with open(os.path.join(load_dir, 'config.yaml'), 'r') as stream:
            try:
                config_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        with open(os.path.join(load_dir, 'data_config.yaml'), 'r') as stream:
            try:
                data_config_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        loaded_config = TransformerConfig(**config_dict)
        loaded_data_config = DataConfig(
            pretrain_modes=data_config_dict['pretrain_modes'],
            game_type=data_config_dict['game_types'][0].split('/')[0],
            game_spec=data_config_dict['game_spec'],
            k=data_config_dict['k'],
            mask_prob=data_config_dict['mask_prob'],
            mlm_mask_prob=data_config_dict['mlm_mask_prob'],
            max_len=data_config_dict['max_len'],
            eps=data_config_dict['epsilons'][0],
            train_valid_test=data_config_dict['train_valid_test'])

        loaded_tmr = Transformer(loaded_config, loaded_data_config)
        loaded_tmr.load_state_dict(torch.load(os.path.join(load_dir, 'model.bin')))

        return loaded_tmr


class Language(object):
    def __init__(self, data_config):
        lang_load_ = os.path.join(
            data_config.processed_dirs[0],
            'lang_data_max_len={:d}.npy'.format(data_config.max_len))
        lang_data_all = np.load(lang_load_, allow_pickle=True).item()

        # TODO: different epssilons have different dictionaries so this is not gonna work as is. One hacky way out
        #  is to choose the winner epsilon and fix the dictionary for that epsilon, but the when loading the data
        #  in trainer, remember to translate trajectories (token_ids) from other epsilons to this fixed dictionary
        #  so that you would have a universally consitent dictionary

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
    replacements = [conversion_dict[obj.item()] for obj in replace_with]

    if not any([len(item) - 1 for item in replacements]):  # if len any of objects is longer than 1
        x_replaced[indices] = torch.from_numpy(np.array([item[0] for item in replacements]))

    else:
        #   print([[nlp.i2w[t] for t in ent] for ent in replacements])
        extension = 0
        for i, item in zip(indices, replacements):
            x_replaced = np.insert(x_replaced, extension + i + 1, [unk_id] * (len(item) - 1))
            extension += len(item) - 1

        replacements_flattened = [item for sublist in replacements for item in sublist]
        x_replaced[x_replaced == unk_id] = torch.from_numpy(np.array(replacements_flattened))

    return x_replaced


def _extract_object_info(x, conversion_dict, pretrain_mode, config):
    obs_ranges_flat, act_ranges_flat = get_ranges(x, config, flatten=True)

    if 'ACT' in pretrain_mode:
        ranges_chained = list(chain(*act_ranges_flat))
    elif 'OBS' in pretrain_mode:
        ranges_chained = list(chain(*obs_ranges_flat))
    else:
        raise NotImplementedError

    x_of_interest = x.flatten()[ranges_chained]

    founds_dict = {}
    for obj_tuple, obj_label in conversion_dict.items():
        subseq = list(obj_tuple)
        seq = list(x_of_interest)
        founds_dict.update({obj_label: list(_get_index(subseq, seq))})

    nonzero_foudns_dict = {}
    _ = list(map(
        lambda z: nonzero_foudns_dict.update({z[0]: z[1]}),
        filter(lambda tup: len(tup[1]), founds_dict.items())
    ))

    detected_ranges_dilated = np.ones(len(x.flatten())) * -100
    for lbl_, rngs_list in nonzero_foudns_dict.items():
        detected_ranges_dilated[np.array(ranges_chained)[list(chain(*rngs_list))]] = lbl_

    ultimate_lables, ultiamte_ranges = [], []
    for lbl_, rngs_list in nonzero_foudns_dict.items():
        ultimate_lables.extend([lbl_] * len(rngs_list))
        ultiamte_ranges.extend(rngs_list)

    return ranges_chained, list(zip(ultiamte_ranges, ultimate_lables))


def _get_index(subseq, seq):
    i, n, m = -1, len(seq), len(subseq)
    try:
        while True:
            i = seq.index(subseq[0], i + 1, n - m + 1)
            if subseq == seq[i:i + m]:
                yield range(i, i + m)
    except ValueError:
        return -1
