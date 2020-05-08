import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import sys; sys.path.append('..')
from utils.gen_pretrain_data import compute_type_position_ids, load_data


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


class Trainer:
    def __init__(self,
                 transformer,
                 train_config,
                 use_cuda=True,
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
        loaded_data_all = self._batchify(self._load_data())
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
        self.iteration(epoch, self.train_data, mode='train')

    def valid(self, epoch):
        self.iteration(epoch, self.valid_data, mode='valid')

    def test(self, epoch):
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

        #  for pretrain_mode, data_dict in zip(self.data_config, data_dict_tuple):

        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        num_batches = len(data_dict['token_ids'])

        for i in tqdm(range(num_batches)):
            # TODO: this is not complete
            batch_inputs = (
                data_dict['mode...'][0][0][i],
                data_dict[0][1][i],
                data_dict[0][2][i])
            batch_labels = data_dict['labels'][i]

            # push data through transformer
            last_hiddens = [my_trainer.model(item)[0] for item in batch_inputs]

            # get the embedded objects
            objects_embedded = my_trainer.model.generator_head.embed_objects(
                my_trainer.model.embeddings.word_embeddings, reduction='mean',
                use_cuda=my_trainer.device.type == 'cuda')

            # calculate the generator predictions to get loss + sample to get predicted indices
            gen_preds, sampled_indxs = my_trainer.model.generator_head(last_hiddens, objects_embedded, batch_labels)

            # generator loss
            generator_losses = [my_trainer.model.generator_head.loss_fn(tup[0], tup[1].flatten())
                                for tup in zip(gen_preds, batch_labels)]

            # generate x_corrupt to be fed into the discriminator
            x_corrupts = my_trainer.model.generator_head.get_x_corrupt(
                list(map(lambda z: to_np(z[0]), batch_inputs)),
                list(map(lambda z: to_np(z), batch_labels)),
                list(map(lambda z: to_np(z), sampled_indxs)),
            )

            corrupt_type_ids, corrupt_position_ids = zip(*[compute_type_position_ids(
                tup[0], config, starting_position_ids=to_np(tup[1][2][:, 0])) for tup in zip(x_corrupts, batch_inputs)])

            batch_corrupted_inputs = [
                tuple(map(lambda z: torch.tensor(z, dtype=torch.long, device=my_trainer.device), tup))
                for tup in zip(x_corrupts, corrupt_type_ids, corrupt_position_ids)]

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

    def _load_data(self):
        data_dict_dict, _ = load_data(self.data_config, load_extra_stuff=False, verbose=False)

        data_dict_cat_eps = {}
        for type_key, data_dict in data_dict_dict.items():
            token_ids_list = []
            type_ids_list = []
            position_ids_list = []
            labels_list = []
            for eps in self.data_config.epsilons:
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

            data_dict_cat_eps.update({type_key: ((token_ids, type_ids, position_ids), labels)})
        return data_dict_cat_eps

    def _batchify(self, data_dict):
        batched_data_dict = {}

        for type_key, data_tuple in data_dict.items():
            inputs, labels = data_tuple
            assert inputs[0].shape == inputs[1].shape == inputs[2].shape == labels.shape, "something wrong"

            num_samples = len(inputs[0])
            num_batches = int(np.ceil(num_samples / self.train_config.batch_size))

            empty_arr = np.empty((num_batches, self.train_config.batch_size, inputs[0].shape[-1]), dtype=int)
            batched_token_ids = empty_arr.copy()
            batched_type_ids = empty_arr.copy()
            batched_position_ids = empty_arr.copy()
            batched_labels = empty_arr.copy()

            for b in range(num_batches):
                batch_indices = self.rng.choice(num_samples, size=self.train_config.batch_size)
                batched_token_ids[b] = inputs[0][batch_indices]
                batched_type_ids[b] = inputs[1][batch_indices]
                batched_position_ids[b] = inputs[2][batch_indices]
                batched_labels[b] = labels[batch_indices]

            batched_data_tuple = (
                (
                    torch.tensor(batched_token_ids, dtype=torch.long, device=self.device),
                    torch.tensor(batched_type_ids, dtype=torch.long, device=self.device),
                    torch.tensor(batched_position_ids, dtype=torch.long, device=self.device),
                ),
                torch.tensor(batched_labels, dtype=torch.long, device=self.device)
            )
            batched_data_dict.update({type_key: batched_data_tuple})

        return batched_data_dict
