import numpy as np
from tqdm import tqdm
from tqdm.notebook import tnrange
from time import sleep
from pprint import pprint
import torch
from torch import nn
from torch.optim import Adam
from .forward import *
from .optimizer import Lamb, ScheduledOptim
import sys; sys.path.append('..')
from utils.gen_pretrain_data import load_data
from utils.utils import to_np


class OfflineTrainer:
    def __init__(self,
                 transformer,
                 train_config,
                 use_cuda=True,
                 data_on_cuda=False,
                 seed=665,
                 ):

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.rng = np.random.RandomState(seed)

        cuda_condition = torch.cuda.is_available() and use_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")
        self.data_on_cuda = data_on_cuda

        self.model = transformer.to(self.device)
        self.train_config = train_config
        self.config = transformer.config
        self.data_config = transformer.data_config

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if use_cuda and torch.cuda.device_count() > 1:
            print("Using {:d} GPUS".format(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Load data
        loaded_data_all = self._batchify(self._load_data())
        train_data, valid_data, test_data = {}, {}, {}
        pretrain_modes = []
        for type_key, data_tuple in loaded_data_all.items():
            try:
                mode_, _type = type_key.split('/')
                pretrain_mode = mode_.split('-')[0]
                pretrain_modes.append(pretrain_mode)
                if _type == 'train':
                    train_data.update({pretrain_mode: data_tuple})
                elif _type == 'valid':
                    valid_data.update({pretrain_mode: data_tuple})
                elif _type == 'test':
                    test_data.update({pretrain_mode: data_tuple})
                else:
                    raise ValueError("Invalid game type encountered")
            except IndexError:
                continue

        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

        self.pretrain_modes = list(np.unique(pretrain_modes))

        self.optim = None
        self.optim_schedule = None
        self.setup_optim()

        print("\nTotal Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, rounds, epochs_per_round=None):
        self.model.train()

        if epochs_per_round is None:
            epochs_per_round = {pretrain_mode: 1 for pretrain_mode in self.pretrain_modes}

        assert type(rounds) in [int, range], "Please provide either range or int"

        rounds = range(rounds) if isinstance(rounds, int) else rounds
        for r in rounds:
            for mode in self.pretrain_modes:
                for epoch in range(epochs_per_round[mode]):
                    self.iteration(self.train_data, pretrain_mode=mode,
                                   r=r, epoch=epoch, train=True)

            if (r + 1) % self.train_config.chkpt_freq == 0:
                print('Saving chkpt:{:d}'.format(r+1))
                self.model.save('chkpt:{:d}'.format(r+1))

    def valid(self):
        self.model.eval()
        with torch.no_grad():
            for mode in self.pretrain_modes:
                self.iteration(self.valid_data, mode)

    def test(self):
        self.model.eval()
        with torch.no_grad():
            for mode in self.pretrain_modes:
                self.iteration(self.test_data, mode)

    def iteration(self, data_dict, pretrain_mode=None, r=0, epoch=0, train=False):
        # get data tuple: (token_ids, type_ids, position_ids, labels)
        data_tuple = data_dict[pretrain_mode]

        # shuffle data
        num_batches = len(data_tuple[0])
        shuffled_indices = self.rng.permutation(num_batches)
        data_tuple = tuple(map(lambda z: z[shuffled_indices], data_tuple))

        # put data on cuda if it's not there already
        if not self.data_on_cuda and self.device.type == 'cuda':
            data_tuple = tuple(map(lambda z: z.to(self.device), data_dict[pretrain_mode]))

        inputs = data_tuple[:3]
        labels = data_tuple[3]

        cuml_loss = 0.0
        cuml_gen_corrects = 0.0
        cuml_disc_corrects = 0.0

        pbar = tqdm(range(num_batches))
        for i in pbar:
            batch_inputs = tuple(map(lambda z: z[i], inputs))
            batch_labels = labels[i]

            batch_hiddens, _ = self.model(src_inputs=batch_inputs)[0]

            if pretrain_mode in ['MLM', 'MOM']:
                losses, correct_prediction_stats, _ = corrupted_fwd(
                    model=self.model,
                    masked_hiddens=batch_hiddens,
                    masked_inputs=batch_inputs,
                    masked_labels=batch_labels,
                    pretrain_mode=pretrain_mode,
                    loss_imbalance_lambda=self.train_config.loss_imbalance_lambda)

            elif pretrain_mode in ['ACT_ORDER', 'OBS_ORDER']:
                losses, extra_outputs = self.permuted_fwd(self.model)
                raise NotImplementedError
            elif pretrain_mode in ['ACT_PRED', 'OBS_PRED', 'PAIR_PRED']:
                losses, extra_outputs = self.pred_fwd(self.model)
                raise NotImplementedError
            elif pretrain_mode in ['ACT_ELIM']:
                losses, extra_outputs = self.act_elim_fwd(self.model)
                raise NotImplementedError
            elif pretrain_mode in ['ACT_GEN']:
                losses, extra_outputs = self.act_gen_fwd(self.model)
                raise NotImplementedError
            else:
                raise ValueError("Invalid pretrain mode: {}".format(pretrain_mode))

            loss = sum(loss for loss in losses.values())

            # backward and optimization only in train
            if train:
                if self.train_config.optim_choice == 'adam_with_warmup':
                    self.optim_schedule.zero_grad()
                    loss.backward()
                    self.optim_schedule.step_and_update_lr()
                else:
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

            msg0 = '{:s}, round {:d}. epoch {:d}.'.format(pretrain_mode, r, epoch)
            msg1 = ""
            for k, v in losses.items():
                msg1 += "{}: {:.3f}, ".format(k, v.item())
            msg1 += "tot_loss: {:.3f}".format(loss.item())

            desc1 = msg0 + '\t|\t' + msg1
            pbar.set_description(desc1)

            cuml_loss += loss.item()
            cuml_gen_corrects += 100 * (
                        correct_prediction_stats['num_gen_corrects'] / correct_prediction_stats['tot_masked'])
            cuml_disc_corrects += 100 * (
                        correct_prediction_stats['num_disc_corrects'] / correct_prediction_stats['tot_tokens'])

            #   train_log = {
            #      "pretrain_mode": pretrain_mode,
            #      "epoch": epoch,
            #      "iter": i,
            #      "cuml_loss": cuml_loss,
            #      "cuml_acc": cuml_percent_corrects,
            #      "loss": loss.item(),
            # }

            #  for k, v in losses.items():
            #     train_log.update({k: v.item()})

            if i + 1 == num_batches:
                desc2 = 'round # {:d}, {:s}, avg_loss: {:.4f}, avg_gen_corrects: {:.3f} {:s}, avg_disc_corrects: {:.3f} {:s}'
                desc2 = desc2.format(
                    r, pretrain_mode, cuml_loss / num_batches,
                    cuml_gen_corrects / num_batches, '%',
                    cuml_disc_corrects / num_batches, '%'
                )
                pbar.set_description(desc2)

    def setup_optim(self, large_lr_parameters_keywords=None, freeze_parameters_keywords=None):
        if large_lr_parameters_keywords is None:
            large_lr_parameters_keywords = ['generators', 'discriminators']
        else:
            assert isinstance(large_lr_parameters_keywords, list), "Must provide a list of keywords"
        if freeze_parameters_keywords is None:
            freeze_parameters_keywords = []
        else:
            assert isinstance(freeze_parameters_keywords, list), "Must provide a list of keywords"

        # should be changed into torch.nn.ParameterList()
        param_freeze_list = []
        param_small_lr_list = []
        param_large_lr_list = []

        for k, v in self.model.named_parameters():
            small_lr = True
            for large_keyword in large_lr_parameters_keywords:
                if large_keyword in k:
                    param_large_lr_list.append(v)
                    small_lr = False
                    break
            for freeze_keyword in freeze_parameters_keywords:
                if freeze_keyword in k:
                    param_freeze_list.append(v)
                    small_lr = False
                    break
            if small_lr:
                param_small_lr_list.append(v)

        param_freeze_list = nn.ParameterList(param_freeze_list)
        for param in param_freeze_list:
            param.requires_grad = False

        param_small_lr_list = nn.ParameterList(param_small_lr_list)
        param_large_lr_list = nn.ParameterList(param_large_lr_list)

        params_dict = [
            {'params': param_small_lr_list, 'lr': self.train_config.lr},
            {'params': param_large_lr_list, 'lr': 25 * self.train_config.lr},
        ]

        print('Total number of freeze params: {}'.format(
            sum(p.numel() for p in self.model.parameters() if not p.requires_grad)))
        print('Total number of params with small lr: {}'.format(
            sum(p.numel() for p in param_small_lr_list if p.requires_grad)))
        print('Total number of params with large lr: {}'.format(
            sum(p.numel() for p in param_large_lr_list if p.requires_grad)))

        if self.train_config.optim_choice == 'lamb':
            self.optim = Lamb(
                params_dict,
                lr=self.train_config.lr,
                betas=self.train_config.betas,
                weight_decay=self.train_config.weight_decay,
                adam=False,
            )

        elif self.train_config.optim_choice == 'adam':
            self.optim = Adam(
                params_dict,
                lr=self.train_config.lr,
                betas=self.train_config.betas,
                weight_decay=self.train_config.weight_decay,
            )

        elif self.train_config.optim_choice == 'adam_with_warmup':
            self.optim = Adam(
                params_dict,
                lr=self.train_config.lr,
                betas=self.train_config.betas,
                weight_decay=self.train_config.weight_decay,
            )
            self.optim_schedule = ScheduledOptim(
                self.optim, self.model.config.hidden_size, n_warmup_steps=self.train_config.warmup_steps)
        else:
            raise ValueError("Invalid optimizer choice: {}".format(train_config.optim_chioce))

    def _load_data(self, only_load_best_eps=True):
        data_dict_dict, loaded_from = load_data(self.data_config, load_extra_stuff=False, verbose=False)

        data_dict_cat_eps = {}
        for type_key, data_dict in data_dict_dict.items():
            token_ids_list = []
            type_ids_list = []
            position_ids_list = []
            labels_list = []
            for eps in self.data_config.epsilons:
                if only_load_best_eps and eps < 1.00:
                    continue
                key = 'max_len={:d},eps={:.2f}'.format(self.data_config.max_len, eps)
                token_ids, type_ids, position_ids, labels = data_dict[key]

                token_ids_list.append(token_ids)
                type_ids_list.append(type_ids)
                position_ids_list.append(position_ids)
                labels_list.append(labels)

            token_ids = np.concatenate(token_ids_list)
            type_ids = np.concatenate(type_ids_list)
            position_ids = np.concatenate(position_ids_list)
            labels = np.concatenate(labels_list)

            # TODO: later fix gen_pretrain kind of functions
            #  so that you don't have to put .T on all of these here
            data_typle = ((token_ids.T, type_ids.T, position_ids.T), labels.T)
            data_dict_cat_eps.update({type_key: data_typle})

        redundant_str = '/home/hadi/Documents/FTWP/DATA'
        print('Data loaded from:')
        try:
            pprint(list(map(lambda s: s.replace(redundant_str, '~'), loaded_from)))
            print('\n')
        except AttributeError:
            pass
        return data_dict_cat_eps

    def _batchify(self, data_dict):
        batched_data_dict = {}

        for type_key, data_tuple in data_dict.items():
            inputs, labels = data_tuple
            assert inputs[0].shape == inputs[1].shape == inputs[2].shape == labels.shape, "something wrong"

            num_samples = inputs[0].shape[1]
            num_batches = int(np.ceil(num_samples / self.train_config.batch_size))

            empty_arr = np.empty((num_batches, inputs[0].shape[0], self.train_config.batch_size), dtype=int)
            batched_token_ids = empty_arr.copy()
            batched_type_ids = empty_arr.copy()
            batched_position_ids = empty_arr.copy()
            batched_labels = empty_arr.copy()

            shuffled_indices = self.rng.permutation(num_samples)
            inputs = tuple(map(lambda z: z[:, shuffled_indices], inputs))
            labels = labels[:, shuffled_indices]

            for b in range(num_batches):
                batch_indices = slice(b * self.train_config.batch_size, (b + 1) * self.train_config.batch_size)
                if b == num_batches - 1 and num_samples % self.train_config.batch_size != 0:
                    batch_indices = slice(num_samples - self.train_config.batch_size, num_samples)
                batched_token_ids[b] = inputs[0][:, batch_indices]
                batched_type_ids[b] = inputs[1][:, batch_indices]
                batched_position_ids[b] = inputs[2][:, batch_indices]
                batched_labels[b] = labels[:, batch_indices]

            if self.data_on_cuda:
                batched_data_tuple = (
                    torch.tensor(batched_token_ids, dtype=torch.long, device=self.device),
                    torch.tensor(batched_type_ids, dtype=torch.long, device=self.device),
                    torch.tensor(batched_position_ids, dtype=torch.long, device=self.device),
                    torch.tensor(batched_labels, dtype=torch.long, device=self.device),
                )
            else:
                batched_data_tuple = (
                    torch.tensor(batched_token_ids, dtype=torch.long),
                    torch.tensor(batched_type_ids, dtype=torch.long),
                    torch.tensor(batched_position_ids, dtype=torch.long),
                    torch.tensor(batched_labels, dtype=torch.long),
                )

            batched_data_dict.update({type_key: batched_data_tuple})

        return batched_data_dict
