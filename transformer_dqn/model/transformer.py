import os
from os.path import join as pjoin
import numpy as np
from copy import deepcopy as dc
from itertools import chain, compress
from datetime import datetime
from prettytable import PrettyTable
import yaml

import torch
from torch import nn
import torch.nn.functional as F

from .embeddings import Embeddings
from .preprocessing import get_tokenizer, preproc
from .configuration import TransformerConfig, DataConfig
# import sys; sys.path.append('..')
# from utils.gen_pretrain_data import get_ranges


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
        )

        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)

        self.norm1 = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout1 = nn.Dropout(config.attention_probs_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

        self.activation = _get_activation_fn(config.hidden_act)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, need_attention_weights=False):
        src2, attention_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=need_attention_weights,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src, attention_weights


class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.decoder_hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.decoder_hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
        )

        self.linear1 = nn.Linear(config.decoder_hidden_size, config.decoder_intermediate_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear2 = nn.Linear(config.decoder_intermediate_size, config.decoder_hidden_size)

        self.norm1 = nn.LayerNorm(config.decoder_hidden_size, config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.decoder_hidden_size, config.layer_norm_eps)
        self.norm3 = nn.LayerNorm(config.decoder_hidden_size, config.layer_norm_eps)
        self.dropout1 = nn.Dropout(config.attention_probs_dropout_prob)
        self.dropout2 = nn.Dropout(config.attention_probs_dropout_prob)
        self.dropout3 = nn.Dropout(config.hidden_dropout_prob)

        self.activation = _get_activation_fn(config.hidden_act)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, need_attention_weights=False):

        # self attn
        tgt2, self_attention_weights = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=need_attention_weights,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross sttn
        tgt2, cross_attention_weights = self.cross_attn(
            tgt, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=need_attention_weights,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, self_attention_weights, cross_attention_weights


class TransformerEncoder(nn.Module):
    def __init__(self, config, encoder_layer):
        super(TransformerEncoder, self).__init__()

        if config.tie_weights:
            self.encoder_layer = encoder_layer
        else:
            self.encoder_layers = _get_clones(encoder_layer, config.num_hidden_layers)

        self.tie_weights = config.tie_weights
        self.num_hidden_layers = config.num_hidden_layers

        self.norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, need_attention_weights=False):
        attention_outputs = ()

        if self.tie_weights:
            for _ in range(self.num_hidden_layers):
                src, attention_weights = self.encoder_layer(
                    src, src_mask=src_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    need_attention_weights=need_attention_weights)
                attention_outputs += (attention_weights,)
        else:
            for layer in self.encoder_layers:
                src, attention_weights = layer(
                    src, src_mask=src_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    need_attention_weights=need_attention_weights)
                attention_outputs += (attention_weights,)

        src = self.norm(src)

        return src, attention_outputs


class TransformerDecoder(nn.Module):
    def __init__(self, config, decoder_layer):

        super(TransformerDecoder, self).__init__()
        if config.tie_weights:
            self.decoder_layer = decoder_layer
        else:
            self.decoder_layers = _get_clones(decoder_layer, config.decoder_num_hidden_layers)

        self.tie_weights = config.tie_weights
        self.num_hidden_layers = config.decoder_num_hidden_layers

        self.norm = nn.LayerNorm(config.decoder_hidden_size, config.layer_norm_eps)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, need_attention_weights=False):

        self_attention_outputs, cross_attention_outputs = (), ()

        if self.tie_weights:
            for _ in range(self.num_hidden_layers):
                tgt, self_attention_weights, cross_attention_weights = self.decoder_layer(
                    tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    need_attention_weights=need_attention_weights)

                self_attention_outputs += (self_attention_weights,)
                cross_attention_outputs += (cross_attention_weights,)

        else:
            for layer in self.decoder_layers:
                tgt, self_attention_weights, cross_attention_weights = layer(
                    tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    need_attention_weights=need_attention_weights)

                self_attention_outputs += (self_attention_weights,)
                cross_attention_outputs += (cross_attention_weights,)

        tgt = self.norm(tgt)

        return tgt, self_attention_outputs, cross_attention_outputs


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size, bias=True)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # hidden_states: (S x N x H)
        first_token_tensor = hidden_states[0]   # (N x H)
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class Transformer(nn.Module):
    def __init__(self, config, data_config):
        super(Transformer, self).__init__()

        self.config = config
        self.data_config = data_config
        self.nlp = Language(data_config)

        self.embeddings = Embeddings(config)

        encoder_layer = TransformerEncoderLayer(config)
        self.encoder = TransformerEncoder(config, encoder_layer)
        self.encoder_pooler = Pooler(config.hidden_size)

        if config.embedding_size != config.hidden_size:
            self.encoder_embedding_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)

        decoder_layer = TransformerDecoderLayer(config)
        self.decoder = TransformerDecoder(config, decoder_layer)
        self.decoder_pooler = Pooler(config.decoder_hidden_size)
        # TODO: remove decoder pooler (not necessary).
        if config.embedding_size != config.decoder_hidden_size:
            self.decoder_embedding_mapping_in = nn.Linear(config.embedding_size, config.decoder_hidden_size)

        if config.hidden_size != config.decoder_hidden_size:
            self.decoder_src_mapping_in = nn.Linear(config.hidden_size, config.decoder_hidden_size)

        # TODO: the only scenario where this is inefficient is when enc_dim == dec_dim != emb_dim
        #  in that case there are two separate mapping in layers from emb that have identical shapes

        self.pretrain_category_dict = {}

        self.pretrain_category_dict.update(dict.fromkeys(['MLM', 'MOM'], 'mlm'))
        self.pretrain_category_dict.update(dict.fromkeys(['ACT_PRED', 'OBS_PRED'], 'pred'))
        self.pretrain_category_dict.update(dict.fromkeys(['ACT_ELIM'], 'ae'))
        self.pretrain_category_dict.update(dict.fromkeys(['ACT_GEN'], 'ag'))

        current_categories = list(np.unique([self.pretrain_category_dict[x] for x in data_config.pretrain_modes]))

        generator_dicts, discriminator_dicts = {}, {}
        if config.unique_top_layers:
            for pretrain_mode in data_config.pretrain_modes:
                if pretrain_mode in ['MLM', 'MOM']:
                    generator_dicts.update({pretrain_mode: Generator(config)})
                    discriminator_dicts.update({pretrain_mode: Discriminator(config)})
                elif pretrain_mode in ['ACT_PRED', 'OBS_PRED', 'ACT_ELIM']:
                    discriminator_dicts.update(
                        {pretrain_mode: Discriminator(config, pretrain_category=self.pretrain_category_dict[pretrain_mode])})
                elif pretrain_mode == 'ACT_GEN':
                    continue
                else:
                    raise ValueError("Invalid pretrain mode, '{}', encountered in discriminator".format(pretrain_mode))
        else:
            for category in current_categories:
                if category == 'mlm':
                    generator_dicts.update({category: Generator(config)})
                    discriminator_dicts.update({category: Discriminator(config, pretrain_category=category)})
                elif category in ['pred', 'ae']:
                    discriminator_dicts.update({category: Discriminator(config, pretrain_category=category)})
                elif category == 'ag':
                    continue
                else:
                    raise ValueError("Invalid pretrain category, '{}', encountered in discriminator".format(category))

        self.generators = nn.ModuleDict(generator_dicts)
        self.discriminators = nn.ModuleDict(discriminator_dicts)

        self.init_weights()
        self.print_num_params()

    def forward(self, src_inputs, tgt_inputs=None,
                src_mask=None, tgt_mask=None, memory_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None,
                need_pooler_outputs=False, need_attention_weights=False):
        """Take in and process masked source/target sequences.

        Args:
            src_inputs: tuple of tensors (token_ids, type_ids, position_ids) each has size: (S, N)
            tgt_inputs: tuple of tensors (token_ids, type_ids, position_ids) each has size: (T, N)
                can be None when pretraining only the encoder
            src_mask: additive mask for the src sequence.
            tgt_mask: additive mask for the tgt sequence.
            memory_mask: the additive mask for the encoder output. will be set to src_mask if not provided.
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).
            need_pooler_outputs: it True will add output from pooler
            need_attention_weights: if True the output will contain attention_weights (None if False)

        Returns:
            (encoer_outputs, decoder_outputs), where:
                encoer_outputs = (
                    encoder_last_hidden,
                    encoder_self_attentions,
                    pooled_encoder_output,
                )
                decoder_outputs = (
                    decoder_last_hidden,
                    decoder_self_attentions,
                    decoder_cross_attentions,
                    pooled_decoder_output,
                )
        """

        # embed src_intputs
        src_embedded = self.embeddings(*src_inputs)     # (S, N, E)
        if self.config.embedding_size != self.config.hidden_size:
            src_embedded = self.encoder_embedding_mapping_in(src_embedded)      # (S, N, H_enc)

        if src_key_padding_mask is None:
            src_key_padding_mask = src_inputs[0].T == self.config.pad_id

        encoder_outputs = self.encoder(src_embedded, src_mask=src_mask,
                                       src_key_padding_mask=src_key_padding_mask,
                                       need_attention_weights=need_attention_weights)
        if need_pooler_outputs:
            encoder_outputs += (self.pooler(encoder_outputs[0]),)

        if tgt_inputs is not None:
            # embed tgt_intputs
            tgt_embedded = self.embeddings(*tgt_inputs)     # (T, N, E)
            if self.config.embedding_size != self.config.decoder_hidden_size:
                tgt_embedded = self.decoder_embedding_mapping_in(tgt_embedded)  # (T, N, H_dec)

            # memory: (L, N, H_enc)
            memory = encoder_outputs[0]
            if self.config.hidden_size != self.config.decoder_hidden_size:
                memory = self.decoder_src_mapping_in(memory)    # memory: (L, N, H_dec)

            if src_embedded.size(1) != tgt_embedded.size(1):
                raise RuntimeError("the batch number of src and tgt must be equal")
            if not (memory.size(2) == tgt_embedded.size(2) == self.config.hidden_size):
                raise RuntimeError("the feature number of memory and tgt must be equal to hidden size of the model")

            if tgt_key_padding_mask is None:
                tgt_key_padding_mask = tgt_inputs[0].T == self.config.pad_id

            if memory_key_padding_mask is None:
                memory_key_padding_mask = src_inputs[0].T == self.config.pad_id

            decoder_outputs = self.decoder(tgt, memory=memory,
                                           tgt_mask=tgt_mask, memory_mask=memory_mask,
                                           tgt_key_padding_mask=tgt_key_padding_mask,
                                           memory_key_padding_mask=memory_key_padding_mask,
                                           need_attention_weights=need_attention_weights)

        else:
            decoder_outputs = None

        return encoder_outputs, decoder_outputs

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

    def get_word_embeddings(self, device):
        indxs = torch.tensor(list(self.nlp.i2w.keys()), dtype=torch.long, device=device)
        return self.embeddings.word_embeddings(indxs)

    def print_num_params(self):
        t = PrettyTable(['Module Name', 'Num Params'])

        for name, m in self.named_modules():
            total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
            if '.' not in name:
                if isinstance(m, type(self)):
                    t.add_row(["Total", "{} k".format(np.round(total_params / 1000, decimals=1))])
                    t.add_row(['---', '---'])
                else:
                    t.add_row([name, "{} k".format(np.round(total_params / 1000, decimals=1))])
        print(t, '\n\n')

    def create_attention_mask(self, token_ids, mask_unk=False):
        """
        :param token_ids: max_len x batch_size
        :param mask_unk: if True the <UNK> positions will be masked
        :return: mask_square_additive: bath_size x max_len x max_len
        """
        # TODO: this is wrong, completely unnecessary
        if mask_unk:
            mask = torch.logical_and(token_ids != self.config.pad_id,
                                     token_ids != self.config.unk_id).float()
        else:
            mask = (token_ids != self.config.pad_id).float()

        max_len, batch_size = mask.size()
        mask_square = mask.expand(max_len, max_len, batch_size).permute(2, 0, 1)    # (N, S, S)

        mask_square_additive = mask_square.masked_fill(
            mask_square == 0, float('-inf')).masked_fill(mask_square == 1, float(0.0))

        mask_repeated = torch.repeat_interleave(
            mask_square_additive, repeats=self.config.num_attention_heads, dim=0)

        return mask_repeated

    def save(self, prefix=None, comment=None):
        config_dict = vars(self.config)
        data_config_dict = vars(self.data_config)

        to_hash_dict_ = dc(config_dict)
        to_hash_dict_.update(data_config_dict)
        hashed_info = str(hash(frozenset(sorted(to_hash_dict_))))

        if prefix is None:
            prefix = 'chkpt:0'

        save_dir = pjoin(
            self.data_config.model_save_dir,
            "[{}_{:s}]".format(comment, hashed_info),
            "{}_{:s}".format(prefix, datetime.now().strftime("[%Y_%m_%d_%H:%M]")))

        os.makedirs(save_dir, exist_ok=True)

        torch.save(self.state_dict(), pjoin(save_dir, 'model.bin'))

        with open(pjoin(save_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config_dict, f)

        with open(pjoin(save_dir, 'data_config.yaml'), 'w') as f:
            yaml.dump(data_config_dict, f)

    @staticmethod
    def load(model_id=-1, chkpt_id=-1, config=None, data_config=None, load_dir=None, verbose=True):
        if load_dir is None:
            _dir = pjoin(os.environ['HOME'], 'Documents/FTWP/SAVED_MODELS')
            available_models = os.listdir(_dir)
            if verbose:
                print('Available models to load:\n', available_models)
            model_dir = pjoin(_dir, available_models[model_id])
            available_chkpts = os.listdir(model_dir)
            if verbose:
                print('\nAvailable chkpts to load:\n', available_chkpts)
            load_dir = pjoin(model_dir, available_chkpts[chkpt_id])

        if verbose:
            print('\nLoading from:\n{}\n'.format(load_dir))

        if config is None:
            with open(pjoin(load_dir, 'config.yaml'), 'r') as stream:
                try:
                    config_dict = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)
            config = TransformerConfig(**config_dict)

        if data_config is None:
            with open(pjoin(load_dir, 'data_config.yaml'), 'r') as stream:
                try:
                    data_config_dict = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)

            data_config = DataConfig(
                pretrain_modes=data_config_dict['pretrain_modes'],
                game_type=data_config_dict['game_types'][0].split('/')[0],
                game_spec=data_config_dict['game_spec'],
                k=data_config_dict['k'],
                mlm_mask_prob=data_config_dict['mlm_mask_prob'],
                mom_mask_prob=data_config_dict['mom_mask_prob'],
                max_len=data_config_dict['max_len'],
                eps=data_config_dict['epsilons'][0],
                train_valid_test=data_config_dict['train_valid_test'])

        loaded_tmr = Transformer(config, data_config)
        loaded_tmr.load_state_dict(torch.load(pjoin(load_dir, 'model.bin')))

        return loaded_tmr


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()

        self.linear = nn.Linear(config.hidden_size, config.embedding_size, bias=True)
        self.norm = nn.LayerNorm(config.embedding_size, config.layer_norm_eps)
        self.activation = _get_activation_fn(config.hidden_act)

        self.temperature = config.generator_temperature
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, hiddens, objects_embedded, labels):
        x = self.norm(self.activation(self.linear(hiddens)))

        num_classes = len(objects_embedded)
        predictions = x @ objects_embedded.T
        predictions = predictions.view(-1, num_classes)

        if self.temperature != 1.0:
            predictions = predictions / config.generator_temperature
        probs = F.softmax(predictions, dim=1).detach()
        sampled_indxs = probs.multinomial(num_samples=1).view(labels.shape)

        return predictions, sampled_indxs

    @staticmethod
    def get_x_corrupt(x_masked, labels, sampled_indxs):
        x_corrupt = dc(x_masked)
        x_corrupt[labels != -100] = sampled_indxs[labels != -100]

        return x_corrupt


class Discriminator(nn.Module):
    def __init__(self, config, pretrain_category=None):
        super(Discriminator, self).__init__()

        self.config = config

        if pretrain_category == 'ae':
            self.hidden_dim = config.decoder_hidden_size
        else:
            self.hidden_dim = config.hidden_size

        self.dense = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.activation = _get_activation_fn(config.hidden_act)

        self.lin_proj = nn.Linear(self.hidden_dim, 1, bias=True)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, hiddens, flat_indices=None):
        if flat_indices is not None:
            h_flat = hiddens.view(-1, self.hidden_dim)[flat_indices]
        else:
            h_flat = hiddens.view(-1, self.hidden_dim)

        predictions = self.lin_proj(self.activation(self.dense(h_flat))).flatten()
        return predictions

    def get_discriminator_labels(self, corrupted_token_ids, masked_token_ids,
                                 generator_replaced_labels, gold_labels):
        x_masked_flat = masked_token_ids.flatten()
        x_corrupt_flat = corrupted_token_ids.flatten()

        labels = torch.ones(len(x_corrupt_flat))
        labels[x_masked_flat == self.config.unk_id] = 0
        labels[generator_replaced_labels.flatten() == gold_labels.flatten()] = 1

        flat_indices = (x_corrupt_flat != self.config.pad_id).nonzero().flatten()
        final_discriminator_labels = labels[flat_indices].float()

        return final_discriminator_labels, flat_indices


class Language(object):
    def __init__(self, data_config):
        lang_load_file = pjoin(data_config.lang_dir, 'lang_data_max_len={:d}.npy'.format(data_config.max_len))
        lang_data_all = np.load(lang_load_file, allow_pickle=True).item()

        # TODO: different epssilons have different dictionaries so
        #  multiple epsilon scenarios are not going to work as is.

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

        self.act2indx = lang_data['act2indx']
        self.indx2act = lang_data['indx2act']

        self.data_config = data_config
        self.tokenizer = get_tokenizer()

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


def _get_clones(module, num_copies):
    return nn.ModuleList([dc(module) for _ in range(num_copies)])


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
