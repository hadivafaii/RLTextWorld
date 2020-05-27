import numpy as np
from tqdm import tqdm
# from tqdm.notebook import tnrange
from os.path import join as pjoin
from time import sleep
from datetime import datetime
from pprint import pprint

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from .forward import *
from .dataset import *
from .optimizer import Lamb, log_lamb_rs, ScheduledOptim

import sys; sys.path.append('..')
from utils.gen_pretrain_data import load_data
from utils.utils import to_np


class OfflineTrainer:
    def __init__(self,
                 transformer,
                 train_config,
                 seed=665,
                 ):

        torch.manual_seed(seed)
        np.random.seed(seed)

        cuda_condition = torch.cuda.is_available() and train_config.use_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")

        self.model = transformer.to(self.device)
        self.train_config = train_config
        self.config = transformer.config
        self.data_config = transformer.data_config

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if train_config.use_cuda and torch.cuda.device_count() > 1:
            print("Using {:d} GPUS".format(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        self.pretrain_modes = list()

        self.train_datasets = dict()
        self.valid_datasets = dict()
        self.test_datasets = dict()

        self.train_dataloaders = dict()
        self.valid_dataloaders = dict()
        self.test_dataloaders = dict()

        # Load data and update datasets/loaders
        self._setup_datasets(self._load_data())
        self._setup_dataloaders()

        self.writer = SummaryWriter()

        self.optim = None
        self.optim_schedule = None
        self.setup_optim()

        print("\nTotal Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, nb_epochs, comment=None):
        assert type(nb_epochs) in [int, range], "Please provide either range or int"

        if comment is None:
            comment = ""
            for pretrain_modes in self.pretrain_modes:
                comment += "{}+".format(pretrain_modes)
            comment += 'lr:{:.1e}'.format(self.train_config.lr)
            comment += 'lr:{:.1e}'.format(self.train_config.batch_size)

        self.writer = SummaryWriter(
            pjoin(self.train_config.runs_dir, "{}_{}".format(comment, datetime.now().strftime("[%Y_%m_%d_%H:%M]"))))

        self.model.train()

        epochs_range = range(nb_epochs) if isinstance(nb_epochs, int) else nb_epochs
        for epoch in epochs_range:
            self.iteration(self.train_dataloaders, epoch=epoch, train=True)

            if (epoch + 1) % self.train_config.chkpt_freq == 0:
                print('Saving chkpt:{:d}'.format(epoch+1))
                self.model.save('chkpt:{:d}'.format(epoch+1), comment=comment)

    def valid(self):
        self.model.eval()
        with torch.no_grad():
            for mode in self.pretrain_modes:
                self.iteration(self.valid_dataloaders, mode)

    def test(self):
        self.model.eval()
        with torch.no_grad():
            for mode in self.pretrain_modes:
                self.iteration(self.test_dataloaders, mode)

    def iteration(self, dataloader_dict, epoch=0, train=False):
        cuml_loss = 0.0
        cuml_gen_corrects = 0.0
        cuml_disc_corrects = 0.0

        data_iterators = {k: iter(v) for k, v in dataloader_dict.items()}
        max_num_batches = max([len(dataloader) for dataloader in dataloader_dict.values()])

        pbar = tqdm(range(max_num_batches))
        for i in pbar:
            pretrain_losses = {}
            for pretrain_mode in self.pretrain_modes:
                try:
                    data_tuple = next(data_iterators[pretrain_mode])
                except StopIteration:
                    data_iterators[pretrain_mode] = iter(dataloader_dict[pretrain_mode])
                    data_tuple = next(data_iterators[pretrain_mode])

                batch_data_tuple = transpose_send_to_cuda(data_tuple, self.device)

                batch_inputs = batch_data_tuple[:3]
                batch_labels = batch_data_tuple[3]

                if pretrain_mode in ['MLM', 'MOM']:
                    losses, correct_prediction_stats, _ = corrupted_fwd(
                        model=self.model,
                        masked_inputs=batch_inputs,
                        masked_labels=batch_labels,
                        pretrain_mode=pretrain_mode,
                        loss_imbalance_lambda=self.train_config.loss_imbalance_lambda)

                elif pretrain_mode in ['ACT_ORDER', 'OBS_ORDER']:
                    # losses, extra_outputs = self.permuted_fwd(self.model)
                    raise NotImplementedError
                elif pretrain_mode in ['ACT_PRED', 'OBS_PRED', 'PAIR_PRED']:
                    # losses, extra_outputs = self.pred_fwd(self.model)
                    raise NotImplementedError
                elif pretrain_mode in ['ACT_ELIM']:
                    # losses, extra_outputs = self.act_elim_fwd(self.model)
                    raise NotImplementedError
                elif pretrain_mode in ['ACT_GEN']:
                    # losses, extra_outputs = self.act_gen_fwd(self.model)
                    raise NotImplementedError
                else:
                    raise ValueError("Invalid pretrain mode: {}".format(pretrain_mode))

                loss = sum(x for x in losses.values()) / len(self.pretrain_modes)
                pretrain_losses.update({pretrain_mode: loss})

                cuml_loss += loss.item()

                try:
                    percent_gen_corrects = 100.0 * (
                                correct_prediction_stats['num_gen_cocommentrrects'] / correct_prediction_stats['tot_masked'])
                except KeyError:
                    percent_gen_corrects = 0.0

                try:
                    percent_disc_corrects = 100.0 * (
                            correct_prediction_stats['num_disc_corrects'] / correct_prediction_stats['tot_tokens'])
                except KeyError:
                    percent_disc_corrects = 0.0

                cuml_gen_corrects += percent_gen_corrects
                cuml_disc_corrects += percent_disc_corrects

                global_step = epoch * max_num_batches + i
                if (global_step + 1) % self.train_config.log_freq == 0:
                    # add losses to writer
                    for k, v in losses.items():
                        self.writer.add_scalar("{}/{}".format(pretrain_mode, k), v.item(), global_step)

                    # add accuracy performance to writer
                    self.writer.add_scalar('{}/gen_corrects'.format(pretrain_mode), percent_gen_corrects, global_step)
                    self.writer.add_scalar('{}/disc_corrects'.format(pretrain_mode), percent_disc_corrects, global_step)

                    # add optim state to writer
                    if self.train_config.optim_choice == 'adam_with_warmup':
                        self.writer.add_scalar('lr', self.optim_schedule.current_lr, global_step)
                    else:
                        log_lamb_rs(self.optim, self.writer, global_step)

            final_loss = sum(x for x in pretrain_losses.values())

            # backward and optimization only in train
            if train:
                if self.train_config.optim_choice == 'adam_with_warmup':
                    self.optim_schedule.zero_grad()
                    final_loss.backward()
                    self.optim_schedule.step_and_update_lr()
                else:
                    self.optim.zero_grad()
                    final_loss.backward()
                    self.optim.step()

            msg0 = 'epoch {:d}'.format(epoch)
            msg1 = ""
            for k, v in pretrain_losses.items():
                msg1 += "{}: {:.3f}, ".format(k, v.item())
            msg1 += "tot: {:.3f}".format(final_loss.item())

            desc1 = msg0 + '\t|\t' + msg1
            pbar.set_description(desc1)

            global_step = epoch * max_num_batches + i
            if i + 1 == max_num_batches:
                desc2 = 'epoch # {:d}, avg_loss: {:.4f}, avg_gen_corrects: {:.3f} {:s}, avg_disc_corrects: {:.3f} {:s}'
                desc2 = desc2.format(
                    epoch, cuml_loss / max_num_batches / len(self.pretrain_modes),
                    cuml_gen_corrects / max_num_batches / len(self.pretrain_modes), '%',
                    cuml_disc_corrects / max_num_batches / len(self.pretrain_modes), '%',
                )
                pbar.set_description(desc2)

                self.writer.add_embedding(
                    self.model.get_word_embeddings(self.device),
                    metadata=list(self.model.nlp.i2w.values()),
                    global_step=global_step,
                    tag='word_emb')

    def setup_optim(self):
        freeze_parameters_keywords = self.train_config.freeze_parameters_keywords
        large_lr_parameters_keywords = self.train_config.large_lr_parameters_keywords

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
            {'params': param_large_lr_list, 'lr': self.train_config.lr_ratio * self.train_config.lr},
        ]

        print('Total number of freeze params: {:.1f} k'.format(
            sum(p.numel() for p in self.model.parameters() if not p.requires_grad) / 1000))
        print('Total number of params with small lr: {:.1f} k'.format(
            sum(p.numel() for p in param_small_lr_list if p.requires_grad) / 1000))
        print('Total number of params with large lr ({:.1f} x): {:.1f} k'.format(
            self.train_config.lr_ratio, sum(p.numel() for p in param_large_lr_list if p.requires_grad) / 1000))

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
                filter(lambda p: p.requires_grad, self.model.parameters()),
                betas=self.train_config.betas,
                weight_decay=self.train_config.weight_decay,
            )
            self.optim_schedule = ScheduledOptim(
                self.optim,
                self.model.config.hidden_size,
                n_warmup_steps=self.train_config.warmup_steps,
            )

            print('\nUsing {} with {} warmup steps.  Large vs small lr doesnt apply. . . '.format(
                self.train_config.optim_choice, self.train_config.warmup_steps))

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

            data_tuple = (token_ids, type_ids, position_ids, labels)
            data_dict_cat_eps.update({type_key: data_tuple})

        redundant_str = '/home/hadi/Documents/FTWP/DATA'
        print('Data loaded from:')
        try:
            pprint(list(map(lambda s: s.replace(redundant_str, '~'), loaded_from)))
            print('\n')
        except AttributeError:
            pass

        return data_dict_cat_eps

    def _setup_datasets(self, data_dict):
        for type_key, data_tuple in data_dict.items():
            try:
                mode_, _type = type_key.split('/')
                pretrain_mode = mode_.split('-')[0]
                self.pretrain_modes.append(pretrain_mode)
                if _type == 'train':
                    self.train_datasets.update({pretrain_mode: OfflineDataset(data_tuple, pretrain_mode)})
                elif _type == 'valid':
                    self.valid_datasets.update({pretrain_mode: OfflineDataset(data_tuple, pretrain_mode)})
                elif _type == 'test':
                    self.test_datasets.update({pretrain_mode: OfflineDataset(data_tuple, pretrain_mode)})
                else:
                    raise ValueError("Invalid game type encountered")
            except IndexError:
                continue

        self.pretrain_modes = list(np.unique(self.pretrain_modes))

    def _setup_dataloaders(self):
        num_workers = int(np.floor(30 / len(self.pretrain_modes)))
        for pretrain_mode in self.pretrain_modes:
            self.train_dataloaders.update(
                {pretrain_mode: DataLoader(
                    self.train_datasets[pretrain_mode],
                    batch_size=self.train_config.batch_size,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=num_workers,
                )}
            )
            self.valid_dataloaders.update(
                {pretrain_mode: DataLoader(
                    self.valid_datasets[pretrain_mode],
                    batch_size=self.train_config.batch_size,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=num_workers,
                )}
            )
            self.test_dataloaders.update(
                {pretrain_mode: DataLoader(
                    self.test_datasets[pretrain_mode],
                    batch_size=self.train_config.batch_size,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=num_workers,
                )}
            )
