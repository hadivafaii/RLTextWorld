import os
from collections import namedtuple

class TransformerConfig(object):
    def __init__(
        self,
            vocab_size=1000,
            type_vocab_size=3,
            embedding_size=32,
            hidden_size=128,
            num_hidden_layers=3,
            num_attention_heads=2,
            intermediate_size=512,
            max_position_embeddings=4096,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_id=0,
            obs_id=1,
            act_id=2,
            unk_id=3,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_id = pad_id
        self.obs_id = obs_id
        self.act_id = act_id
        self.unk_id = unk_id



class PretrainConfig(object):
    def __init__(
            self,
            pretrain_mode='ACT_ENTITY',
            game_type='tw_simple/train',
            rewards='dense',
            goal='detailed',
            world_size=5,
            nb_objects=10,
            quest_len=5,
            k=3,
            mask_prob=0.30,
            batch_size=128,
            max_len=512,
            base_dir='/home/hadivafa/Documents/FTWP',
            base_processed_dir='processed_trajectories',
            base_pretrain_dir='pretraining_data',
    ):
        super().__init__()

        ALLOWED_MODES = [
            'ACT_ORDER', 'ACT_ENTITY', 'ACT_VERB',
            'OBS_ORDER', 'OBS_ENTITY', 'OBS_VERB',
            'MLM', ]
        # TODO: add 'ALL' here and figure out a way to jointly train on all datasets
        # TODO: add MLM also

        if pretrain_mode in ['ACT_ORDER', 'OBS_ORDER']:
            pretrain_dir = 'k={:d}'.format(k)
        elif pretrain_mode in ['ACT_ENTITY', 'ACT_VERB', 'OBS_ENTITY', 'OBS_VERB', 'MLM']:
            pretrain_dir = 'mask_prob={:.2f}'.format(mask_prob)
        else:
            raise ValueError('incorrect pretrain type.  allowed opetions: \n{}'.format(ALLOWED_MODES))

        self.pretrain_mode = pretrain_mode

        if game_type.split('/')[0] == 'tw_simple':
            GameConfig = namedtuple('GameConfig', ['goal', 'rewards'])
            game_config = GameConfig(goal, rewards)
            games_dir = 'goal={:s}-rewards={:s}'.format(goal, rewards)

        elif game_type.split('/')[0] == 'custom':
            GameConfig = namedtuple('GameConfig', ['goal', 'wsz', 'nbobj', 'qlen'])
            game_config = GameConfig(goal, world_size, nb_objects, quest_len)
            games_dir = '{:s}/wsz={:d}-nbobj={:d}-qlen={:d}'.format(goal, world_size, nb_objects, quest_len)

        elif game_type.split('/')[0] == 'tw_cooking':
            game_config = None
            games_dir = ''
        else:
            raise ValueError("Enter correct game type")

        self.game_type = game_type
        self.game_config = game_config

        self.k = k
        self.mask_prob = mask_prob
        self.batch_size = batch_size
        self.max_len = max_len

        data_base_dir = os.path.join(base_dir, 'trajectories', game_type)
        self.processed_dir = os.path.join(data_base_dir, base_processed_dir)
        self.pretrain_dir = os.path.join(data_base_dir, base_pretrain_dir, pretrain_dir)
        self.games_dir = os.path.join(base_dir, 'games', game_type, games_dir)
