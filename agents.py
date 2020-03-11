import re
from typing import Mapping, List, Any, Optional
from collections import defaultdict

import numpy as np

import textworld
import textworld.gym
from textworld import EnvInfos

from transformers import BertTokenizer, BertModel

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class RandomAgent(textworld.gym.Agent):
    """ Agent that randomly selects a command from the admissible ones. """

    def __init__(self, seed=1234):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

    @property
    def infos_to_request(self) -> textworld.EnvInfos:
        return textworld.EnvInfos(admissible_commands=True)

    def act(self, obs: str, score: int, done: bool, infos: Mapping[str, Any]) -> str:
        return self.rng.choice(infos["admissible_commands"])


class ExpertAgent(textworld.gym.Agent):
    """ Agent """

    def __init__(self):
        print('Step aside, StrawExpert coming')
    @property
    def infos_to_request(self) -> textworld.EnvInfos:
        return textworld.EnvInfos(policy_commands=True)

    def act(self, obs: str, score: int, done: bool, infos: Mapping[str, Any]) -> str:
        return infos["policy_commands"][0]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CommandScorer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CommandScorer, self).__init__()
        torch.manual_seed(42)  # For reproducibility
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.encoder_gru = nn.GRU(hidden_size, hidden_size)
        self.cmd_encoder_gru = nn.GRU(hidden_size, hidden_size)
        self.state_gru = nn.GRU(hidden_size, hidden_size)
        self.hidden_size = hidden_size
        self.state_hidden = torch.zeros(1, 1, hidden_size, device=device)
        self.critic = nn.Linear(hidden_size, 1)
        self.att_cmd = nn.Linear(hidden_size * 2, 1)

    def forward(self, obs, commands, **kwargs):
        input_length = obs.size(0)
        batch_size = obs.size(1)
        nb_cmds = commands.size(1)

        #batch_size = len(obs)

        embedded = self.embedding(obs)

        obs_h_0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        cmds_h_0 = torch.zeros(1, nb_cmds, self.hidden_size, device=device)

        encoder_output, encoder_hidden = self.encoder_gru(embedded, obs_h_0)
        state_output, state_hidden = self.state_gru(encoder_hidden, self.state_hidden)
        self.state_hidden = state_hidden
        value = self.critic(state_output)

        # Attention network over the commands.
        cmds_embedding = self.embedding.forward(commands)
        _, cmds_encoding_last_states = self.cmd_encoder_gru.forward(cmds_embedding, cmds_h_0)  # 1 x cmds x hidden

        # Same observed state for all commands.
        # cmd_selector_input = torch.stack([state_hidden] * nb_cmds, 2)  # 1 x batch x cmds x hidden
        cmd_selector_input = torch.stack([self.state_hidden] * nb_cmds, 2)

        # Same command choices for the whole batch.
        cmds_encoding_last_states = torch.stack([cmds_encoding_last_states] * batch_size, 1)  # 1 x batch x cmds x hidden

        # Concatenate the observed state and command encodings.
        cmd_selector_input = torch.cat([cmd_selector_input, cmds_encoding_last_states], dim=-1)

        # Compute one score per command.
        scores = F.relu(self.att_cmd(cmd_selector_input)).squeeze(-1)  # 1 x Batch x cmds

        probs = F.softmax(scores, dim=2)  # 1 x Batch x cmds
        index = probs[0].multinomial(num_samples=1).unsqueeze(0)  # 1 x batch x indx
        return scores, index, value

    def reset_hidden(self, batch_size):
        self.state_hidden = torch.zeros(1, batch_size, self.hidden_size, device=device)


class SoftActorCritic:
    """ Simple Neural Agent for playing TextWorld games. """
    MAX_VOCAB_SIZE = 1000
    UPDATE_FREQUENCY = 10
    LOG_FREQUENCY = 1000
    GAMMA = 0.9

    def __init__(self, hidden_size=128, reg=None) -> None:
        self._initialized = False
        self._epsiode_has_started = False
        self.transitions = []
        self.stats = {"max": defaultdict(list), "mean": defaultdict(list)}
        self.last_score = 0
        self.no_train_step = 0

        self.id2word = ["<PAD>", "<UNK>"]
        self.word2id = {w: i for i, w in enumerate(self.id2word)}

        self.model = CommandScorer(input_size=self.MAX_VOCAB_SIZE, hidden_size=hidden_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), 0.00003)

        if reg is None:
            self.reg = {'lambda_pol': 1.0, 'lambda_val': 0.5, 'lambda_ent': 0.1}
        else:
            self.reg = reg

        self.huber_loss = nn.SmoothL1Loss(reduction='sum')

        self.mode = "test"

    def train(self):
        self.mode = "train"
        self.model.reset_hidden(1)

    def test(self):
        self.mode = "test"
        self.model.reset_hidden(1)

    @property
    def infos_to_request(self) -> EnvInfos:
        return EnvInfos(description=True, inventory=True, admissible_commands=True,
                        won=True, lost=True)

    def _get_word_id(self, word):
        if word not in self.word2id:
            if len(self.word2id) >= self.MAX_VOCAB_SIZE:
                return self.word2id["<UNK>"]

            self.id2word.append(word)
            self.word2id[word] = len(self.word2id)

        return self.word2id[word]

    def _tokenize(self, text):
        # Simple tokenizer: strip out all non-alphabetic characters.
        text = re.sub("[^a-zA-Z0-9\- ]", " ", text)
        # text = re.sub("[^a-zA-Z0-9-'.]", " ", text)
        word_ids = list(map(self._get_word_id, text.split()))
        return word_ids

    def _process(self, texts):
        texts = list(map(self._tokenize, texts))
        max_len = max(len(l) for l in texts)
        padded = np.ones((len(texts), max_len)) * self.word2id["<PAD>"]

        for i, text in enumerate(texts):
            padded[i, :len(text)] = text

        padded_tensor = torch.from_numpy(padded).type(torch.long).to(device)
        padded_tensor = padded_tensor.permute(1, 0)  # Batch x Seq => Seq x Batch
        return padded_tensor

    def _discount_rewards(self, last_values):
        returns, advantages = [], []
        R = last_values.data
        for t in reversed(range(len(self.transitions))):
            rewards, _, _, values = self.transitions[t]
            R = rewards + self.GAMMA * R
            adv = R - values
            returns.append(R)
            advantages.append(adv)

        return returns[::-1], advantages[::-1]

    def act(self, obs: str, score: int, done: bool, infos: Mapping[str, Any]) -> Optional[str]:

        # Build agent's observation: feedback + look + inventory.
        input_ = "{}\n{}\n{}".format(obs, infos["description"], infos["inventory"])

        # Tokenize and pad the input and the commands to chose from.
        input_tensor = self._process([input_])
        commands_tensor = self._process(infos["admissible_commands"])

        # Get our next action and value prediction.
        outputs, indexes, values = self.model(input_tensor, commands_tensor)
        action = infos["admissible_commands"][indexes[0]]

        if self.mode == "test":
            if done:
                self.model.reset_hidden(1)
            return action

        self.no_train_step += 1

        if self.transitions:
            reward = score - self.last_score  # Reward is the gain/loss in score.
            self.last_score = score
            if infos["won"]:
                reward += 100
            if infos["lost"]:
                reward -= 100

            self.transitions[-1][0] = reward  # Update reward information.

        self.stats["max"]["score"].append(score)
        if self.no_train_step % self.UPDATE_FREQUENCY == 0:
            # Update model
            returns, advantages = self._discount_rewards(values)

            loss = 0
            for transition, ret, advantage in zip(self.transitions, returns, advantages):
                reward, indexes_, outputs_, values_ = transition

                advantage = advantage.detach()  # Block gradients flow here.
                probs = F.softmax(outputs_, dim=2)
                log_probs = torch.log(probs)
                log_action_probs = log_probs.gather(2, indexes_)
                policy_loss = (-log_action_probs * advantage).sum()
                # value_loss = (.5 * (values_ - ret) ** 2.).sum()
                value_loss = self.huber_loss(values_, ret)
                entropy = (-probs * log_probs).sum()
                loss += (self.reg['lambda_pol'] * policy_loss +
                         self.reg['lambda_val'] * value_loss -
                         self.reg['lambda_ent'] * entropy)

                self.stats["mean"]["reward"].append(reward)
                self.stats["mean"]["policy"].append(policy_loss.item())
                self.stats["mean"]["value"].append(value_loss.item())
                self.stats["mean"]["entropy"].append(entropy.item())
                self.stats["mean"]["confidence"].append(torch.exp(log_action_probs).item())

            if self.no_train_step % self.LOG_FREQUENCY == 0:
                msg = "{}. ".format(self.no_train_step)
                msg += "  ".join("{}: {:.3f}".format(k, np.mean(v)) for k, v in self.stats["mean"].items())
                msg += "  " + "  ".join("{}: {}".format(k, np.max(v)) for k, v in self.stats["max"].items())
                msg += "  vocab: {}".format(len(self.id2word))
                print(msg)
                self.stats = {"max": defaultdict(list), "mean": defaultdict(list)}

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 40)
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.transitions = []
            self.model.reset_hidden(1)
        else:
            # Keep information about transitions for Truncated Backpropagation Through Time.
            self.transitions.append([None, indexes, outputs, values])  # Reward will be set on the next call

        if done:
            self.last_score = 0  # Will be starting a new episode. Reset the last score.

        return action


class CommandScorerBERT(nn.Module):
    def __init__(self, hidden_size, reduced_dim=None, bidirectional=False):
        super(CommandScorerBERT, self).__init__()
        torch.manual_seed(42)  # For reproducibility

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        with torch.no_grad():
            self.bert = BertModel.from_pretrained('bert-base-uncased')#.to(device)
            self.bert.eval()
        input_size = self.bert.config.hidden_size

        if reduced_dim is not None:
            self.fc_dim_reduction = nn.Linear(input_size, reduced_dim)
            input_size = reduced_dim
        else:
            self.fc_dim_reduction = None

        self.obs_encoder_gru = nn.GRU(input_size, hidden_size, bidirectional=bidirectional)
        self.cmd_encoder_gru = nn.GRU(input_size, hidden_size, bidirectional=bidirectional)
        self.state_gru = nn.GRU(hidden_size, hidden_size)
        self.hidden_size = hidden_size
        self.state_hidden = torch.zeros(1, 1, hidden_size, device=device)
        self.critic = nn.Linear(hidden_size, 1)
        self.att_cmd = nn.Linear(hidden_size * 2, 1)

    def _process(self, doc):
        # get list of tokenized sentences in doc
        if type(doc) is not list:
            doc_resub = re.sub("[^a-zA-Z0-9\-'.]", " ", doc).strip()
            # doc = [re.sub("[^a-zA-Z0-9\-']", " ", doc).strip()]
            doc = list(filter(None, doc_resub.split(".")))
        tokenized_doc = [self.tokenizer.tokenize("[CLS]" + sentence + "[SEP]") for sentence in doc]

        # pad
        max_sentence_len = len(max(tokenized_doc, key=len))
        padded_tokenized_doc = []
        for sentence in tokenized_doc:
            while len(sentence) < max_sentence_len:
                sentence.append('[PAD]')
            padded_tokenized_doc.append(sentence)

        enc_doc = [self.tokenizer.encode(sentence, add_special_tokens=False, return_tensors='pt')
                   for sentence in padded_tokenized_doc] # (= get token id)
        enc_doc = torch.cat(enc_doc)
        return enc_doc.to(device)

    def forward(self, obs: str, commands: List[str], **kwargs):
        batch_size = 1
        nb_cmds = len(commands)

        obs_tensor = self._process(obs) # nb_sentences x max_sentence_length
        cmds_tensor = self._process(commands) # nb_cmds x max_cmd_length

        with torch.no_grad():
            obs_embedded = self.bert(obs_tensor)[0] # nb_sentences x max_sentence_len x 768
            obs_embedded = obs_embedded.view(-1, obs_embedded.size(-1)).unsqueeze(1) # seq_len x 1 x 768
            cmds_embedded = self.bert(cmds_tensor)[0].transpose(0, 1) # seq_len x nb_cmds x 768

        if self.fc_dim_reduction is not None:
            obs_embedded = self.fc_dim_reduction(obs_embedded)
            cmds_embedded = self.fc_dim_reduction(cmds_embedded)

        obs_h_0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        cmds_h_0 = torch.zeros(1, nb_cmds, self.hidden_size, device=device)

        _, obs_encoder_hidden = self.obs_encoder_gru(obs_embedded, obs_h_0) # num_dir x 1 x hidden_size (since batch = 1)
        _, cmds_encoder_hidden = self.cmd_encoder_gru.forward(cmds_embedded, cmds_h_0) # num_dir * nb_cmds x 1 x hidden_size

        state_output, state_hidden = self.state_gru(obs_encoder_hidden, self.state_hidden)
        self.state_hidden = state_hidden
        value = self.critic(state_output)

        # Same observed state for all commands.
        # cmd_selector_input = torch.stack([state_hidden] * nb_cmds, 2)  # 1 x batch x cmds x hidden
        cmd_selector_input = torch.stack([self.state_hidden] * nb_cmds, 2)

        # Same command choices for the whole batch.
        cmds_encoder_hidden = torch.stack([cmds_encoder_hidden] * batch_size, 1)  # 1 x batch x cmds x hidden

        # Concatenate the observed state and command encodings.
        cmd_selector_input = torch.cat([cmd_selector_input, cmds_encoder_hidden], dim=-1)

        # Compute one score per command.
        scores = F.relu(self.att_cmd(cmd_selector_input)).squeeze(-1)  # 1 x batch x cmds

        probs = F.softmax(scores, dim=2)  # 1 x batch x cmds
        index = probs[0].multinomial(num_samples=1).unsqueeze(0)  # 1 x batch x indx (= 1 x 1 x 1 when batch = 1)

        return scores, index, value

    def reset_hidden(self, batch_size):
        self.state_hidden = torch.zeros(1, batch_size, self.hidden_size, device=device)


class SoftActorCriticBERT:
    """ Simple Neural Agent for playing TextWorld games. """
    UPDATE_FREQUENCY = 10
    LOG_FREQUENCY = 1000
    GAMMA = 0.9

    def __init__(self, hidden_size=128, reduced_dim=None, reg=None, bidirectional=False) -> None:
        self._initialized = False
        self._epsiode_has_started = False
        self.transitions = []
        self.stats = {"max": defaultdict(list), "mean": defaultdict(list)}
        self.last_score = 0
        self.no_train_step = 0

        self.model = CommandScorerBERT(
            hidden_size=hidden_size, reduced_dim=reduced_dim,
            bidirectional=bidirectional).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), 0.00003)

        if reg is None:
            self.reg = {'lambda_pol': 1.0, 'lambda_val': 1.0, 'lambda_ent': 0}
        else:
            self.reg = reg

        self.huber_loss = nn.SmoothL1Loss(reduction='sum')

        self.mode = "test"

    def train(self):
        self.mode = "train"
        self.model.reset_hidden(1)

    def test(self):
        self.mode = "test"
        self.model.reset_hidden(1)

    @property
    def infos_to_request(self) -> EnvInfos:
        return EnvInfos(description=True, inventory=True, admissible_commands=True,
                        won=True, lost=True)

    def _discount_rewards(self, last_values):
        returns, advantages = [], []
        R = last_values.data
        for t in reversed(range(len(self.transitions))):
            rewards, _, _, values = self.transitions[t]
            R = rewards + self.GAMMA * R
            adv = R - values
            returns.append(R)
            advantages.append(adv)

        return returns[::-1], advantages[::-1]

    def act(self, obs: str, score: int, done: bool, infos: Mapping[str, Any]) -> Optional[str]:

        # Build agent's observation: feedback + look + inventory.
        input_ = "{}\n{}\n{}".format(obs, infos["description"], infos["inventory"])

        # Get our next action and value prediction.
        outputs, indexes, values = self.model(input_, infos["admissible_commands"])
        action = infos["admissible_commands"][indexes[0]]

        if self.mode == "test":
            if done:
                self.model.reset_hidden(1)
            return action

        self.no_train_step += 1

        if self.transitions:
            reward = score - self.last_score  # Reward is the gain/loss in score.
            self.last_score = score
            if infos["won"]:
                reward += 100
            if infos["lost"]:
                reward -= 100

            self.transitions[-1][0] = reward  # Update reward information.

        self.stats["max"]["score"].append(score)
        if self.no_train_step % self.UPDATE_FREQUENCY == 0:
            # Update model
            returns, advantages = self._discount_rewards(values)

            loss = 0
            for transition, ret, advantage in zip(self.transitions, returns, advantages):
                reward, indexes_, outputs_, values_ = transition

                advantage = advantage.detach()  # Block gradients flow here.
                probs = F.softmax(outputs_, dim=2)
                log_probs = torch.log(probs)
                log_action_probs = log_probs.gather(2, indexes_)
                policy_loss = (-log_action_probs * advantage).sum()
                # value_loss = (.5 * (values_ - ret) ** 2.).sum()
                value_loss = self.huber_loss(values_, ret)
                entropy = (-probs * log_probs).sum()
                loss += (self.reg['lambda_pol'] * policy_loss +
                         self.reg['lambda_val'] * value_loss -
                         self.reg['lambda_ent'] * entropy)

                self.stats["mean"]["reward"].append(reward)
                self.stats["mean"]["policy"].append(policy_loss.item())
                self.stats["mean"]["value"].append(value_loss.item())
                self.stats["mean"]["entropy"].append(entropy.item())
                self.stats["mean"]["confidence"].append(torch.exp(log_action_probs).item())

            if self.no_train_step % self.LOG_FREQUENCY == 0:
                msg = "{}. ".format(self.no_train_step)
                msg += "  ".join("{}: {:.3f}".format(k, np.mean(v)) for k, v in self.stats["mean"].items())
                msg += "  " + "  ".join("{}: {}".format(k, np.max(v)) for k, v in self.stats["max"].items())
                print(msg)
                self.stats = {"max": defaultdict(list), "mean": defaultdict(list)}

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 40)
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.transitions = []
            self.model.reset_hidden(1)
        else:
            # Keep information about transitions for Truncated Backpropagation Through Time.
            self.transitions.append([None, indexes, outputs, values])  # Reward will be set on the next call

        if done:
            self.last_score = 0  # Will be starting a new episode. Reset the last score.

        return action
