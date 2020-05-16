import numpy as np
from tqdm import tqdm
from tqdm.notebook import tnrange
from time import sleep
from pprint import pprint
import torch
from torch import nn
from torch.optim import Adam
from .optimizer import Lamb, ScheduledOptim
import sys;

sys.path.append('..')
from utils.gen_pretrain_data import compute_type_position_ids, load_data
from utils.utils import to_np


class OfflineTrainer:
    def __init__(self,
                 transformer,
                 train_config,
                 pretrain_focus=None,
                 use_cuda=True,
                 load_masks=False,
                 seed=665,
                 ):

        self.rng = np.random.RandomState(seed)

        cuda_condition = torch.cuda.is_available() and use_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")

        self.model = transformer.to(self.device)
        self.train_config = train_config
        self.config = transformer.config
        self.data_config = transformer.data_config

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if use_cuda and torch.cuda.device_count() > 1:
            print("Using {:d} GPUS".format(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        if pretrain_focus is not None and not isinstance(pretrain_focus, list):
            pretrain_focus = [pretrain_focus]
        self.pretrain_focus = pretrain_focus

        # Load data
        loaded_data_all = self._batchify(self._load_data(load_masks=load_masks))
        train_data, valid_data, test_data = {}, {}, {}
        for type_key, data_tuple in loaded_data_all.items():
            try:
                _type = type_key.split('/')[1]
                if _type == 'train':
                    train_data.update({type_key: data_tuple})
                elif _type == 'valid':
                    valid_data.update({type_key: data_tuple})
                elif _type == 'test':
                    test_data.update({type_key: data_tuple})
                else:
                    raise ValueError("Invalid game type encountered")
            except IndexError:
                continue

        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

        # Setting the optimizer with hyper-param
        if train_config.optim_choice == 'lamb':
            self.optim = Lamb(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=train_config.lr,
                betas=train_config.betas,
                weight_decay=train_config.weight_decay,
                adam=False,
            )

        elif train_config.optim_choice == 'adam':
            self.optim = Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=train_config.lr,
                betas=train_config.betas,
                weight_decay=train_config.weight_decay,
            )

        elif train_config.optim_choice == 'adam_with_warmup':
            self.optim = Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=train_config.lr,
                betas=train_config.betas,
                weight_decay=train_config.weight_decay,
            )
            self.optim_schedule = ScheduledOptim(
                self.optim, transformer.config.hidden_size, n_warmup_steps=self.train_config.warmup_steps)

        else:
            raise ValueError("Invalid optimizer choice: {}".format(train_config.optim_chioce))

        self.log_freq = train_config.log_freq

        print("\nTotal Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, nb_epochs):
        self.model.train()
        for epoch in range(nb_epochs):
            self.iteration(self.train_data, train=True, epoch=epoch)

    def valid(self):
        self.model.eval()
        with torch.no_grad():
            self.iteration(self.valid_data)

    def test(self):
        self.model.eval()
        with torch.no_grad():
            self.iteration(self.test_data)

    def iteration(self, data_dict, train=False, epoch=0):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch
        :param epoch: current epoch index
        :param data_dict: data_dict
        :param train: boolean value of is train or test
        """

        for type_key, data_tuple in data_dict.items():
            pretrain_mode = type_key.split('-')[0]

            if self.pretrain_focus is not None and pretrain_mode not in self.pretrain_focus:
                continue

            inputs, masks, labels = data_tuple
            num_batches = len(inputs[0])

            cuml_loss = 0.0
            cuml_gen_corrects = 0.0
            cuml_disc_corrects = 0.0

            pbar = tqdm(range(num_batches))
            for i in pbar:
                batch_inputs = tuple(map(lambda z: z[i], inputs))
                if masks is not None:
                    batch_masks = masks[i]
                else:
                    batch_masks = self.model.encoder.create_attention_mask(batch_inputs[2] > 0)
                batch_labels = labels[i]

                hiddens, _, _ = self.model(batch_inputs, batch_masks)

                if pretrain_mode in ['ACT_ENTITY', 'ACT_VERB', 'OBS_ENTITY', 'OBS_VERB', 'MLM']:
                    if train:
                        losses, correct_prediction_stats, _ = self.corrupted_fwd(
                            hiddens=hiddens,
                            masked_inputs=batch_inputs,
                            masked_labels=batch_labels,
                            pretrain_mode=pretrain_mode,
                            return_extras=False)
                    else:
                        losses, correct_prediction_stats, extra_outputs = self.corrupted_fwd(
                            hiddens=hiddens,
                            masked_inputs=batch_inputs,
                            masked_labels=batch_labels,
                            pretrain_mode=pretrain_mode,
                            return_extras=True)

                elif pretrain_mode in ['ACT_ORDER', 'OBS_ORDER']:
                    losses, extra_outputs = self.permuted_fwd()
                    raise NotImplementedError
                elif pretrain_mode in ['ACT_PREDICT', 'OBS_PREDICT', 'PAIR_PRED']:
                    losses, extra_outputs = self.pred_fwd()
                    raise NotImplementedError
                elif pretrain_mode in ['ACT_ELIM']:
                    losses, extra_outputs = self.act_elim_fwd()
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

                msg0 = '{:s}, epoch # {:d}'.format(pretrain_mode, epoch)
                msg1 = ""
                for k, v in losses.items():
                    msg1 += "{}: {:.3f}, ".format(k, v.item())
                msg1 += "tot_loss: {:.3f}".format(loss.item())

                # msg2 = "corrects: {:d}, total = {:d}, percent correct = {:.2f} {:s}"
                # msg2 = msg2.format(num_corrects, num_total, 100 * (num_corrects / num_total), '%')

                desc1 = msg0 + '\t|\t' + msg1  # + ' |\t' + msg2
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
                    desc2 = '{:s}, epoch # {:d}, avg_loss: {:.4f}, avg_gen_corrects: {:.3f} {:s}, avg_disc_corrects: {:.3f} {:s}'
                    desc2 = desc2.format(
                        pretrain_mode, epoch, cuml_loss / num_batches,
                        cuml_gen_corrects / num_batches, '%',
                        cuml_disc_corrects / num_batches, '%'
                    )
                    pbar.set_description(desc2)

    def corrupted_fwd(self, hiddens, masked_inputs, masked_labels, pretrain_mode, return_extras=False):
        objects_embedded = self.model.generator.embed_objects(
            self.model.embeddings.word_embeddings, pretrain_mode, reduction='mean')

        masked_labels.transpose_(0, 1)  # max_len x batch_size

        gen_preds, sampled_indxs = self.model.generator(hiddens, objects_embedded, masked_labels)
        generator_loss = self.model.generator.loss_fn(gen_preds, masked_labels.flatten())

        x_corrupt = self.model.generator.get_x_corrupt(
            to_np(masked_inputs[0]), to_np(masked_labels).T, to_np(sampled_indxs).T, pretrain_mode)

        corrupt_type_ids, corrupt_position_ids = compute_type_position_ids(
            x_corrupt, self.model.config, starting_position_ids=to_np(masked_inputs[2][:, 0]))

        corrupt_inputs = (x_corrupt, corrupt_type_ids, corrupt_position_ids)
        corrupt_inputs = tuple(
            map(
                lambda z: torch.tensor(z, dtype=torch.long, device=self.device) if type(
                    z) is not torch.Tensor else z.to(self.device),
                corrupt_inputs
            )
        )
        corrupt_mask = self.model.encoder.create_attention_mask(corrupt_inputs[2] > 0)

        corrupt_hiddens, _, _ = self.model(corrupt_inputs, corrupt_mask)

        disc_labels, flat_indices = self.model.discriminator.get_discriminator_labels(
            to_np(x_corrupt).T,
            to_np(masked_inputs[0]).T,
            to_np(sampled_indxs[masked_labels != -100]),
            to_np(masked_labels[masked_labels != -100]),
            pretrain_mode)

        disc_preds = self.model.discriminator(corrupt_hiddens, flat_indices, pretrain_mode)
        discriminator_loss = self.model.discriminator.loss_fn(disc_preds, disc_labels.to(self.device))

        losses = {
            'gen_loss': generator_loss,
            'disc_loss': discriminator_loss * self.train_config.loss_imbalance_lambda,
        }
        correct_prediction_stats = {
            'num_gen_corrects': masked_labels.eq(sampled_indxs).sum().item(),
            'tot_masked': (~masked_labels.eq(-100)).sum().item(),
            'num_disc_corrects': (torch.sigmoid(disc_preds).cpu().ge(0.5).float().eq(disc_labels)).sum().item(),
            'tot_tokens': len(disc_labels),
        }
        if return_extras:
            extra_outputs = {
                'generator_predictions': gen_preds,
                'generator_sampled_labels': sampled_indxs[masked_labels != -100],
                'x_corrupt': x_corrupt, 'flat_indices': flat_indices,
                'discriminator_predictions': disc_preds,
                'discriminator_gold_labels': disc_labels,
            }
            outputs = (losses, correct_prediction_stats, extra_outputs)
        else:
            outputs = (losses, correct_prediction_stats, None)

        return outputs

    def permuted_fwd(self):
        raise NotImplementedError

    def pred_fwd(self):
        raise NotImplementedError

    def act_elim_fwd(self):
        raise NotImplementedError

    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path
        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

    def _load_data(self, only_load_best_eps=True, load_masks=False):
        data_dict_dict, loaded_from = load_data(self.data_config, load_extra_stuff=False, verbose=False)

        data_dict_cat_eps = {}
        for type_key, data_dict in data_dict_dict.items():
            token_ids_list = []
            type_ids_list = []
            position_ids_list = []
            mask_list = []
            labels_list = []
            for eps in self.data_config.epsilons:
                if only_load_best_eps and eps < 1.00:
                    continue
                key = 'max_len={:d},eps={:.2f}'.format(self.data_config.max_len, eps)
                token_ids, type_ids, position_ids, mask, labels = data_dict[key]

                token_ids_list.append(token_ids)
                type_ids_list.append(type_ids)
                position_ids_list.append(position_ids)
                mask_list.append(mask)
                labels_list.append(labels)

            token_ids = np.concatenate(token_ids_list)
            type_ids = np.concatenate(type_ids_list)
            position_ids = np.concatenate(position_ids_list)
            mask = np.concatenate(mask_list)
            labels = np.concatenate(labels_list)

            if load_masks:
                data_typle = ((token_ids, type_ids, position_ids), mask, labels)
            else:
                data_typle = ((token_ids, type_ids, position_ids), None, labels)
            data_dict_cat_eps.update({type_key: data_typle})

        redundant_str = '/home/hadi/Documents/FTWP/DATA'
        print('Data loaded from:')
        try:
            pprint(list(map(lambda s: s.replace(redundant_str, '~'), loaded_from)))
        except AttributeError:
            pass
        return data_dict_cat_eps

    def _batchify(self, data_dict):
        batched_data_dict = {}

        for type_key, data_tuple in data_dict.items():
            inputs, masks, labels = data_tuple
            assert inputs[0].shape == inputs[1].shape == inputs[2].shape == labels.shape, "something wrong"

            num_samples = len(inputs[0])
            num_batches = int(np.ceil(num_samples / self.train_config.batch_size))

            empty_arr = np.empty((num_batches, self.train_config.batch_size, inputs[0].shape[-1]), dtype=int)
            batched_token_ids = empty_arr.copy()
            batched_type_ids = empty_arr.copy()
            batched_position_ids = empty_arr.copy()
            batched_masks = []
            batched_labels = empty_arr.copy()

            for b in range(num_batches):
                batch_indices = self.rng.choice(num_samples, size=self.train_config.batch_size)
                batched_token_ids[b] = inputs[0][batch_indices]
                batched_type_ids[b] = inputs[1][batch_indices]
                batched_position_ids[b] = inputs[2][batch_indices]
                if masks is not None:
                    batched_masks.append(self.model.encoder.create_attention_mask(masks[batch_indices]).unsqueeze(0))
                batched_labels[b] = labels[batch_indices]

            if batched_masks:
                batched_data_tuple = (
                    (
                        torch.tensor(batched_token_ids, dtype=torch.long, device=self.device),
                        torch.tensor(batched_type_ids, dtype=torch.long, device=self.device),
                        torch.tensor(batched_position_ids, dtype=torch.long, device=self.device),
                    ),
                    torch.cat(batched_masks).to(self.device),
                    torch.tensor(batched_labels, dtype=torch.long, device=self.device),
                )
            else:
                batched_data_tuple = (
                    (
                        torch.tensor(batched_token_ids, dtype=torch.long, device=self.device),
                        torch.tensor(batched_type_ids, dtype=torch.long, device=self.device),
                        torch.tensor(batched_position_ids, dtype=torch.long, device=self.device),
                    ),
                    None,
                    torch.tensor(batched_labels, dtype=torch.long, device=self.device),
                )

            batched_data_dict.update({type_key: batched_data_tuple})

        return batched_data_dict
