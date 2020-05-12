import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import Adam
import sys; sys.path.append('..')
from utils.gen_pretrain_data import compute_type_position_ids, load_data
from utils.utils import to_np


class ScheduledOptim:
    def __init__(self, optimizer, hidden_size, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(hidden_size, -0.5)

    def step_and_update_lr(self):
        """Step with the inner optimizer"""
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer"""
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        """Learning rate scheduling per step"""

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class OfflineTrainer:
    def __init__(self,
                 transformer,
                 train_config,
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

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=train_config.lr,
            betas=train_config.betas,
            weight_decay=train_config.weight_decay,
        )
        self.optim_schedule = ScheduledOptim(
            self.optim, transformer.config.hidden_size, n_warmup_steps=self.train_config.warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        #    self.criterion = nn.NLLLoss(ignore_index=0)

        self.log_freq = train_config.log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.model.train()
        self.iteration(epoch, self.train_data, mode='train')

    def valid(self, epoch):
        self.model.eval()
        self.iteration(epoch, self.valid_data, mode='valid')

    def test(self, epoch):
        self.model.eval()
        self.iteration(epoch, self.test_data, mode='test')

    def iteration(self, epoch, data_dict, mode):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch
        :param epoch: current epoch index
        :param data_dict: data_dict
        :param mode: boolean value of is train or test
        """

        for type_key, data_tuple in data_dict.items():
            # if 'ENTITY' not in type_key:
            #    continue
            pretrain_mode = type_key.split('-')[0]
            inputs, masks, labels = data_tuple

            num_batches = len(inputs[0])

            for i in tqdm(range(num_batches)):
                # if i != 0:
                #    continue

                batch_inputs = tuple(map(lambda z: z[i], inputs))
                if masks is not None:
                    batch_masks = masks[i]
                else:
                    batch_masks = self.model.encoder.create_attention_mask(batch_inputs[2] > 0)
                batch_labels = labels[i]

                hiddens, _, _ = self.model(batch_inputs, batch_masks)

                if pretrain_mode in ['ACT_ENTITY', 'ACT_VERB', 'OBS_ENTITY', 'OBS_VERB', 'MLM']:
                    # * TODO: mask <UNK> tokens
                    losses, extra_outputs = self.corrupted_fwd(
                        hiddens=hiddens,
                        masked_inputs=batch_inputs,
                        masked_labels=batch_labels,
                        pretrain_mode=pretrain_mode)

                    return losses, extra_outputs, (batch_inputs, batch_masks, batch_labels)

                elif pretrain_mode in ['ACT_ORDER', 'OBS_ORDER']:
                    losses, extra_outputs = self.permuted_fwd()
                elif pretrain_mode in ['ACT_PREDICT', 'OBS_PREDICT']:
                    losses, extra_outputs = self.pred_fwd()
                elif pretrain_mode in ['ACT_ELIM']:
                    losses, extra_outputs = self.act_elim_fwd()

                avg_loss = 0.0
                total_correct = 0
                total_element = 0

                # TODO: write this function
                # batch_corrupted_labels = detect_objects(batch_corrupted_inputs)

                # TODO: write discriminator
                # discriminator_outputs = self.model.discriminator_head(batch_corrupted_inputs, batch_corrupted_labels)
                # discriminator_loss = self.model.discriminator_head.loss_fn(discriminator_outputs, batch_corrupted_labels)

                # feed this as input to the discriminator head and get
                # loss = generator_losses + self.train_config.loss_imbalance_lambda * discriminator_loss
                # loss1, loss2, loss3 = 1, 2, 3
                # losses = [loss1, loss2, loss3]
                # total_loss = sum(loss for loss in losses)

                # backward and optimization only in train
                if mode == 'train':
                    self.optim_schedule.zero_grad()
                    loss.backward()
                    self.optim_schedule.step_and_update_lr()

                # next sentence prediction accuracy
                correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
                avg_loss += loss.item()
                total_correct += correct
                total_element += data["is_next"].nelement()

                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "avg_acc": total_correct / total_element * 100,
                    "loss": loss.item()
                }

                if i % self.log_freq == 0:
                    data_iter.write(str(post_fix))

                print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "total_acc=",
                      total_correct * 100.0 / total_element)

    def corrupted_fwd(self, hiddens, masked_inputs, masked_labels, pretrain_mode):
        objects_embedded = self.model.generator.embed_objects(
            self.model.embeddings.word_embeddings, pretrain_mode, reduction='mean')

        gen_preds, sampled_indxs = self.model.generator(hiddens, objects_embedded, masked_labels)
        generator_loss = self.model.generator.loss_fn(gen_preds, masked_labels.flatten())

        x_corrupt = self.model.generator.get_x_corrupt(
            masked_inputs[0], masked_labels, sampled_indxs, pretrain_mode)

        corrupt_type_ids, corrupt_position_ids = compute_type_position_ids(
            x_corrupt, self.model.config, starting_position_ids=masked_inputs[2][:, 0])

        corrupt_inputs = (x_corrupt, corrupt_type_ids, corrupt_position_ids)
        corrupt_mask = self.model.encoder.create_attention_mask(corrupt_position_ids > 0)

        corrupt_hiddens, _, _ = self.model(corrupt_inputs, corrupt_mask)

        ranges_chained, disc_labels, ranges_labels = self.model.discriminator.get_discriminator_labels(
            to_np(x_corrupt), to_np(masked_inputs[0]), to_np(sampled_indxs[masked_labels != -100]), pretrain_mode)

        flat_indices = [np.array(ranges_chained)[tup[0]] for tup in ranges_labels]
        disc_preds = self.model.discriminator(corrupt_hiddens, flat_indices, pretrain_mode)

        discriminator_loss = self.model.discriminator.loss_fn(disc_preds, disc_labels.to(self.device))

        losses = {
            'generator_loss': generator_loss,
            'discriminator_loss': discriminator_loss
        }
        extra_outputs = {
            'generator_predictions': gen_preds,
            'generator_sampled_labels': sampled_indxs[masked_labels != -100],
            'x_corrupt': x_corrupt,
            'discriminator_predictions': disc_preds,
            'discriminator_gold_labels': disc_labels
        }

        return losses, extra_outputs

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
        data_dict_dict, _ = load_data(self.data_config, load_extra_stuff=False, verbose=False)

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
                        torch.tensor(batched_token_ids, dtype=torch.long),
                        torch.tensor(batched_type_ids, dtype=torch.long),
                        torch.tensor(batched_position_ids, dtype=torch.long),
                    ),
                    torch.cat(batched_masks),
                    torch.tensor(batched_labels, dtype=torch.long),
                )
            else:
                batched_data_tuple = (
                    (
                        torch.tensor(batched_token_ids, dtype=torch.long),
                        torch.tensor(batched_type_ids, dtype=torch.long),
                        torch.tensor(batched_position_ids, dtype=torch.long),
                    ),
                    None,
                    torch.tensor(batched_labels, dtype=torch.long),
                )

            batched_data_dict.update({type_key: batched_data_tuple})

        return batched_data_dict
