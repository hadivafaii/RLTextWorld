import os
import yaml
from collections import namedtuple


class TransformerConfig:
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
        super(TransformerConfig).__init__()

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


class DataConfig:
    def __init__(
            self,
            pretrain_modes='ACT_ENTITY',
            game_type='tw_simple/train',
            game_spec='dd',
            k=3,
            mask_prob=0.30,
            batch_size=128,
            max_len=512,
            base_dir='Documents/FTWP/DATA',
            base_processed_dir='processed_trajectories',
            base_pretrain_dir='pretraining_data',
    ):
        super(DataConfig).__init__()

        _allowed_modes = [
            'ACT_ORDER', 'ACT_ENTITY', 'ACT_VERB',
            'OBS_ORDER', 'OBS_ENTITY', 'OBS_VERB',
            'MLM',
        ]
        _allowed_tw_simple_specs = [
            'ns', 'nb', 'nd',
            'bs', 'bb', 'bd',
            'ds', 'db', 'dd',
        ]
        _allowed_custom_specs = [
            'b-joke', 'b-tiny', 'b-small', 'b-medium', 'b-large', 'b-xlarge', 'b-xxlarge', 'b-ultra',
            'd-joke', 'd-tiny', 'd-small', 'd-medium', 'd-large', 'd-xlarge', 'd-xxlarge', 'd-ultra',
        ]

        # TODO: add 'ALL' here and figure out a way to jointly train on all datasets
        # TODO: add MLM also

        base_dir = os.path.join(os.environ['HOME'], base_dir)
        name_ = game_type.split('/')[0]
        yaml_dir = os.path.join(base_dir, '{:s}/{:s}_game_specs.yaml'.format(name_, name_))
        # load yaml file
        with open(yaml_dir, 'r') as stream:
            try:
                game_specs_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        specs_xtracted = list(map(lambda x: x.split('='), game_specs_dict[game_spec].split('-')))

        if name_ == 'tw_simple':
            if game_spec not in _allowed_tw_simple_specs:
                raise ValueError('incorrect game spec for {:s}.  allowed opetions: \n{}'.format(
                    name_, _allowed_tw_simple_specs))
            GameSpecs = namedtuple('GameSpecs', ['goal', 'rewards', 'alias'])
            goal = specs_xtracted[0][1]
            rewards = specs_xtracted[1][1]
            game_specs = GameSpecs(goal, rewards, game_spec)
            spec_dir = 'goal={:s}-rewards={:s}'.format(goal, rewards)

        elif game_type.split('/')[0] == 'custom':
            if game_spec not in _allowed_custom_specs:
                raise ValueError('incorrect game spec for {:s}.  allowed opetions: \n{}'.format(
                    name_, _allowed_custom_specs))
            GameSpecs = namedtuple('GameSpecs', ['goal', 'wsz', 'nbobj', 'qlen', 'alias'])
            goal = specs_xtracted[0][1]
            wsz = int(specs_xtracted[1][1])
            nbobj = int(specs_xtracted[2][1])
            qlen = int(specs_xtracted[3][1])
            game_specs = GameSpecs(goal, wsz, nbobj, qlen, game_spec)
            spec_dir = '{:s}/{:s}'.format(goal, game_spec.split('-')[1])

        elif game_type.split('/')[0] == 'tw_cooking':
            game_specs = None
            spec_dir = ''
        else:
            raise ValueError("Enter correct game type")

        self.base_dir = os.path.join(base_dir, game_type, spec_dir)

        if type(pretrain_modes) is not list:
            pretrain_modes = [pretrain_modes]
        self.pretrain_modes = pretrain_modes

        pretrain_dirs = []
        for mode in self.pretrain_modes:
            if mode in ['ACT_ORDER', 'OBS_ORDER']:
                pretrain_dir = 'k={:d}'.format(k)
            elif mode in ['ACT_ENTITY', 'ACT_VERB', 'OBS_ENTITY', 'OBS_VERB', 'MLM']:
                pretrain_dir = 'mask_prob={:.2f}'.format(mask_prob)
            else:
                raise ValueError('incorrect pretrain type.  allowed opetions: \n{}'.format(_allowed_modes))

            pretrain_dirs.append(os.path.join(self.base_dir, base_pretrain_dir, pretrain_dir))
        self.pretrain_dirs = pretrain_dirs

        self.game_type = game_type
        self.game_specs = game_specs

        self.k = k
        self.mask_prob = mask_prob
        self.batch_size = batch_size
        self.max_len = max_len

        self.games_dir = os.path.join(self.base_dir, 'games')
        self.processed_dir = os.path.join(self.base_dir, base_processed_dir)
