import os
from os.path import join as pjoin
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

# Here lies the old TransformerEncoder
"""
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

    def forward(self, src, src_mask, src_key_padding_mask=None, output_attentions=True):
        # type: (Tensor, Optional[Tensor], Optional[Tensor], Optional[Bool]) -> Tensor
        Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            output_attentions: sss

        Shape:
            see the docs in Transformer class.
        

        # be sure to transpose embedded inside forward of Transformer to get: embedded size = (S, N, E)
        src 
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

    def forward(self, src, src_mask, src_key_padding_mask=None, output_attentions=True):
        # type: (Tensor, Optional[Tensor], Optional[Tensor], Optional[Bool]) -> Tensor
        Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            output_attentions: sss

        Shape:
            see the docs in Transformer class.
        

        # be sure to transpose embedded inside forward of Transformer to get: embedded size = (S, N, E)
        src = self.embedding_hidden_mapping_in(src)     # (S, N, H)

        outputs, attn_outputs = (src,), ()

        for _ in range(self.config.num_hidden_layers):
            src2, attn_weights = self.self_attn(
                src, src, src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=output_attentions,
            )
            src = src + self.dropout1(src2)
            src = self.norm1(src)

            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

            outputs += (src,)
            attn_outputs += (attn_weights,)

        return src, outputs, attn_outputs

= self.embedding_hidden_mapping_in(src)     # (S, N, H)

        outputs, attn_outputs = (src,), ()

        for _ in range(self.config.num_hidden_layers):
            src2, attn_weights = self.self_attn(
                src, src, src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=output_attentions,
            )
            src = src + self.dropout1(src2)
            src = self.norm1(src)

            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

            outputs += (src,)
            attn_outputs += (attn_weights,)

        return src, outputs, attn_outputs
"""


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

        decoder_layer = TransformerDecoderLayer(config)
        self.decoder = TransformerDecoder(config, decoder_layer)
        self.decoder_pooler = Pooler(config.decoder_hidden_size)

        self.generator = Generator(config, self.nlp, pretrain_modes=data_config.pretrain_modes)
        self.discriminator = Discriminator(config, self.nlp, pretrain_modes=data_config.pretrain_modes)

        if config.embedding_size != config.hidden_size:
            self.encoder_embedding_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)

        if config.embedding_size != config.decoder_hidden_size:
            self.decoder_embedding_mapping_in = nn.Linear(config.embedding_size, config.decoder_hidden_size)

        if config.hidden_size != config.decoder_hidden_size:
            self.decoder_src_mapping_in = nn.Linear(config.hidden_size, config.decoder_hidden_size)

        self.init_weights()

    def forward(self, src_inputs, tgt_inputs=None,
                src_mask=None, tgt_mask=None, memory_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None,
                need_pooler_outputs=False, need_attention_weights=False):
        """Take in and process masked source/target sequences.

        Args:
            src_inputs: tuple of tensors (token_ids, type_ids, position_ids) each has size: (S, N)
            tgt_inputs: tuple of tensors (token_ids, type_ids, position_ids) each has size: (T, N)
                can be None when pretraining encoder
            src_mask: additive mask for the src sequence. prevents attending to <PAD> and <UNK> tokens
            tgt_mask: additive mask for the tgt sequence. prevents attending to <PAD> and <UNK> tokens
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

        # create attention masks if not provided
        if src_mask is None:
            src_mask = self.create_attention_mask(src_inputs[2] > 0)
        # TODO: verify this below is actualy what memory_mask is meant to do:
        if memory_mask is None:
            memory_mask = src_mask

        encoder_outputs = self.encoder(src_embedded, src_mask=src_mask,
                                       src_key_padding_mask=src_key_padding_mask,
                                       need_attention_weights=need_attention_weights)
        if need_pooler_outputs:
            encoder_outputs += (self.encoder_pooler(encoder_outputs[0]),)

        if tgt_inputs is not None:
            # embed tgt_intputs
            tgt_embedded = self.embeddings(*tgt_inputs)     # (T, N, E)
            if self.config.embedding_size != self.config.decoder_hidden_size:
                tgt_embedded = self.decoder_embedding_mapping_in(tgt_embedded)  # (S, N, H_dec)

            memory = encoder_outputs[0]
            if self.config.hidden_size != self.config.decoder_hidden_size:
                memory = self.decoder_src_mapping_in(memory)

            if src_embedded.size(1) != tgt_embedded.size(1):
                raise RuntimeError("the batch number of src and tgt must be equal")
            if not (memory.size(2) == tgt_embedded.size(2) == self.config.hidden_size):
                raise RuntimeError("the feature number of memory and tgt must be equal to hidden size of the model")

            if tgt_mask is None:
                tgt_mask = self.create_attention_mask(tgt_inputs[2] > 0)

            decoder_outputs = self.decoder(tgt, memory=memory,
                                           tgt_mask=tgt_mask, memory_mask=memory_mask,
                                           tgt_key_padding_mask=tgt_key_padding_mask,
                                           memory_key_padding_mask=memory_key_padding_mask,
                                           need_attention_weights=need_attention_weights)
            if need_pooler_outputs:
                decoder_outputs += (self.decoder_pooler(decoder_outputs[0]),)

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

    def create_attention_mask(self, token_ids, mask_unk=False):
        """
        :param token_ids: max_len x batch_size
        :param mask_unk: if True the <UNK> positions will be masked
        :return: mask_square_additive: bath_size x max_len x max_len
        """
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

    def save(self, prefix=None):
        config_dict = vars(self.config)
        data_config_dict = vars(self.data_config)

        to_hash_dict_ = dc(config_dict)
        to_hash_dict_.update(data_config_dict)
        hashed_info = str(hash(frozenset(sorted(to_hash_dict_))))

        if prefix is None:
            prefix = 'chkpt:0'

        save_dir = pjoin(
            self.data_config.model_save_dir,
            "[{:s}]".format(hashed_info),
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
                mask_prob=data_config_dict['mask_prob'],
                mlm_mask_prob=data_config_dict['mlm_mask_prob'],
                mom_mask_prob=data_config_dict['mom_mask_prob'],
                max_len=data_config_dict['max_len'],
                eps=data_config_dict['epsilons'][0],
                train_valid_test=data_config_dict['train_valid_test'])

        loaded_tmr = Transformer(config, data_config)
        loaded_tmr.load_state_dict(torch.load(pjoin(load_dir, 'model.bin')))

        return loaded_tmr


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
            elif 'MLM' in mode_ or 'MOM' in mode_:
                conversion_dicts.update({mode_: nlp.i2w})
            else:   # the rest is PERMUTE data
                continue

        self.conversion_dicts = conversion_dicts

        self.linear = nn.Linear(config.hidden_size, config.embedding_size, bias=True)
        self.norm = nn.LayerNorm(config.embedding_size, config.layer_norm_eps)
        self.activation = _get_activation_fn(config.hidden_act)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, encoder_outputs, objects_embedded, labels):
        x = self.norm(self.activation(self.linear(encoder_outputs)))

        num_classes = len(objects_embedded)
        predictions = x @ objects_embedded.T
        predictions = predictions.view(-1, num_classes)

        if self.config.generator_temperature != 1.0:
            predictions = predictions / self.config.generator_temperature
        probs = F.softmax(predictions, dim=1).detach()
        sampled_indxs = probs.multinomial(num_samples=1).view(labels.shape)

        return predictions, sampled_indxs

    def embed_objects(self, word_embeddings, pretrain_mode, indxs=None, reduction='mean'):
        conversion_dict = self.conversion_dicts[pretrain_mode]

        _device = word_embeddings.weight.device

        if indxs is None:
            indxs = list(conversion_dict.keys())

        if pretrain_mode in ['MLM', 'MOM']:
            # TODO: well there is no illegal indices, the model should learn not to predict these
            # _illegal_indices = [self.config.pad_id, self.config.obs_id, self.config.act_id, self.config.unk_id]
            # indxs = list(filter(lambda i: i not in _illegal_indices, indxs))
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
        if pretrain_mode in ['MLM', 'MOM']:  # this corresponds to MLM
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
            elif 'MLM' in mode_ or 'MOM' in mode_:
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
        if set(pretrain_modes).intersection({'ACT_ENTITY', 'ACT_VERB', 'OBS_ENTITY', 'OBS_VERB', 'MLM', 'MOM'}):
            # TODO: experiment with bias=False (mathematically there should be no bias here)
            #  and btw it doesn really matter, maybe this bias helps with that flat line you get initially?
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
        if pretrain_mode in ['ACT_ENTITY', 'ACT_VERB', 'OBS_ENTITY', 'OBS_VERB', 'MLM', 'MOM']:
            if pretrain_mode not in ['MLM', 'MOM'] and len(np.unique([len(item) for item in flat_indices])) > 1:
                h = torch.cat(list(map(
                    lambda z: torch.mean(transformer_hidden.view(-1, self.config.hidden_size)[z], dim=0, keepdim=True), flat_indices
                )))
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

    def get_discriminator_labels(self, corrupted_token_ids, masked_token_ids,
                                 generator_replaced_labels, gold_labels, pretrain_mode):
        conversion_dict = self.conversion_dicts[pretrain_mode]

        if pretrain_mode in ['MLM', 'MOM']:
            x_masked_flat = masked_token_ids.flatten()
            x_corrupt_flat = corrupted_token_ids.flatten()

            labels = np.ones(len(x_corrupt_flat))
            unk_indices = np.where(x_masked_flat == self.config.unk_id)[0]
            assert len(unk_indices) == len(generator_replaced_labels) == len(gold_labels), "Otherwise something wrong"
            remaining_fake_indices = np.delete(unk_indices, np.where(generator_replaced_labels == gold_labels)[0])
            labels[remaining_fake_indices] = 0

            _ignore_indices = [self.config.pad_id]     # discriminator loss will not run on these
            flat_indices = [tup[0] for tup in enumerate(x_corrupt_flat) if tup[1] not in _ignore_indices]
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
                masked_token_ids.T,
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


class Language(object):
    def __init__(self, data_config):
        lang_load_file = pjoin(data_config.lang_dir, 'lang_data_max_len={:d}.npy'.format(data_config.max_len))
        lang_data_all = np.load(lang_load_file, allow_pickle=True).item()

        # TODO: different epssilons have different dictionaries so multiple epsilon scenarios are not going to work as is.

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
