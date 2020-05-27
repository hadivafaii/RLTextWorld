import os
import yaml


class TransformerConfig:
    def __init__(
        self,
            vocab_size=1024,
            type_vocab_size=3,
            max_position_embeddings=512 + 1,
            embedding_size=32,
            hidden_size=128,
            intermediate_size=512,
            num_attention_heads=2,
            num_hidden_layers=3,
            decoder_hidden_size=None,
            decoder_intermediate_size=None,
            decoder_num_hidden_layers=None,
            unique_top_layers=True,
            tie_weights=True,
            fixed_pe=False,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            generator_temperature=1.0,
            pad_id=0,
            obs_id=1,
            act_id=2,
            unk_id=3,
            traj_id=4,
    ):
        super(TransformerConfig).__init__()

        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers

        if decoder_hidden_size is None:
            self.decoder_hidden_size = hidden_size
        else:
            self.decoder_hidden_size = decoder_hidden_size

        if decoder_intermediate_size is None:
            self.decoder_intermediate_size = intermediate_size
        else:
            self.decoder_intermediate_size = decoder_intermediate_size

        if decoder_num_hidden_layers is None:
            self.decoder_num_hidden_layers = num_hidden_layers
        else:
            self.decoder_num_hidden_layers = decoder_num_hidden_layers

        self.unique_top_layers = unique_top_layers
        self.tie_weights = tie_weights
        self.fixed_pe = fixed_pe
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.generator_temperature = generator_temperature
        self.pad_id = pad_id
        self.obs_id = obs_id
        self.act_id = act_id
        self.unk_id = unk_id
        self.traj_id = traj_id


class DataConfig:
    def __init__(
            self,
            pretrain_modes='ACT_ENTITY',
            game_type='custom',
            game_spec='b-small',
            k=3,
            mlm_mask_prob=0.15,
            mom_mask_prob=None,
            max_len=512,
            eps=0.8,
            train_valid_test=True,
            base_dir='Documents/FTWP/DATA',
            model_save_dir='Documents/FTWP/SAVED_MODELS',
            base_processed_dir='processed_trajectories',
            base_pretrain_dir='pretraining_data',
    ):
        super(DataConfig).__init__()

        _allowed_modes = [
            'MLM', 'MOM',
            'ACT_PRED', 'OBS_PRED',
            'ACT_ELIM', 'ACT_GEN',
            # 'ACT_ORDER', 'OBS_ORDER',
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

        if train_valid_test:
            _types = ['train', 'valid', 'test']
        else:
            _types = ['train']

        if '/' in game_type:
            game_type = game_type.split('/')[0]

        if mom_mask_prob is None:
            mom_mask_prob = mlm_mask_prob

        self.train_valid_test = train_valid_test

        base_dir = os.path.join(os.environ['HOME'], base_dir)
        self.lang_dir = os.path.join(base_dir, game_type, 'lang_data')

        if 'tw_cooking' not in game_type:
            yaml_dir = os.path.join(base_dir, '{:s}/{:s}_game_specs.yaml'.format(game_type, game_type))
            # load yaml file
            with open(yaml_dir, 'r') as stream:
                try:
                    game_specs_dict = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)
            specs_xtracted = list(map(lambda x: x.split('='), game_specs_dict[game_spec].split('-')))
        else:
            specs_xtracted = None

        if game_type == 'tw_simple':
            if game_spec not in _allowed_tw_simple_specs:
                raise ValueError('incorrect game spec for {:s}.  allowed opetions: \n{}'.format(
                    game_type, _allowed_tw_simple_specs))
            goal = specs_xtracted[0][1]
            rewards = specs_xtracted[1][1]
            spec_dir = 'goal={:s}-rewards={:s}'.format(goal, rewards)

        elif game_type == 'custom':
            if game_spec not in _allowed_custom_specs:
                raise ValueError('incorrect game spec for {:s}.  allowed opetions: \n{}'.format(
                    game_type, _allowed_custom_specs))
            goal = specs_xtracted[0][1]
            spec_dir = '{:s}/{:s}'.format(goal, game_spec.split('-')[1])

        elif game_type == 'tw_cooking':
            spec_dir = ''
        else:
            raise ValueError("Invalid game type value encountered")

        base_dirs = []
        for _type in _types:
            base_dirs.append(os.path.join(base_dir, game_type, _type, spec_dir))
        self.base_dirs = base_dirs

        if type(pretrain_modes) is not list:
            pretrain_modes = [pretrain_modes]
        self.pretrain_modes = pretrain_modes

        pretrain_dirs = []
        for mode in self.pretrain_modes:
            if mode in ['ACT_ORDER', 'OBS_ORDER']:
                pretrain_dir = 'k={:d}'.format(k)
            elif mode == 'MLM':
                pretrain_dir = 'mask_prob={:.2f}'.format(mlm_mask_prob)
            elif mode == 'MOM':
                pretrain_dir = 'mask_prob={:.2f}'.format(mom_mask_prob)
            else:
                raise ValueError('incorrect pretrain type.  allowed opetions: \n{}'.format(_allowed_modes))

            for base_dir_ in self.base_dirs:
                pretrain_dirs.append(os.path.join(base_dir_, base_pretrain_dir, pretrain_dir))

        self.pretrain_dirs = pretrain_dirs

        self.game_types = [os.path.join(game_type, _type) for _type in _types]
        self.game_spec = game_spec

        self.k = k
        self.mlm_mask_prob = mlm_mask_prob
        self.mom_mask_prob = mom_mask_prob
        self.max_len = max_len

        if type(eps) is not list:
            eps = [eps]
        self.epsilons = eps

        games_dirs, processed_dirs = [], []
        for base_dir_ in self.base_dirs:
            games_dirs.append(os.path.join(base_dir_, 'games'))
            processed_dirs.append(os.path.join(base_dir_, base_processed_dir))
        self.model_save_dir = os.path.join(os.environ['HOME'], model_save_dir)
        self.games_dirs = games_dirs
        self.processed_dirs = processed_dirs


class TrainConfig:
    def __init__(
            self,
            optim_choice='lamb',
            lr=1e-3,
            betas=(0.9, 0.999),
            weight_decay: float = 0.01,
            warmup_steps: int = 1000,
            use_cuda: bool = True,
            log_freq: int = 10,
            chkpt_freq: int = 5,
            batch_size: int = 128,
            loss_imbalance_lambda: float = 10.0,
            runs_dir: str = '/home/hadi/Documents/FTWP/runs',
            lr_ratio: float = 3.0,
            large_lr_parameters_keywords: list = None,
            freeze_parameters_keywords: list = None,
    ):
        super(TrainConfig).__init__()

        _allowed_optim_choices = ['lamb', 'adam', 'adam_with_warmup']
        assert optim_choice in _allowed_optim_choices, "Invalid optimzer choice, allowed options:\n{}".format(_allowed_optim_choices)

        self.optim_choice = optim_choice
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.use_cuda = use_cuda
        self.log_freq = log_freq
        self.chkpt_freq = chkpt_freq
        self.batch_size = batch_size
        self.lr_ratio = lr_ratio
        self.loss_imbalance_lambda = loss_imbalance_lambda
        self.runs_dir = runs_dir

        if large_lr_parameters_keywords is None:
            large_lr_parameters_keywords = ['generators', 'discriminators']
        else:
            assert isinstance(large_lr_parameters_keywords, list), "Must provide a list of keywords"
        self.large_lr_parameters_keywords = large_lr_parameters_keywords

        if freeze_parameters_keywords is None:
            freeze_parameters_keywords = []
        else:
            assert isinstance(freeze_parameters_keywords, list), "Must provide a list of keywords"
        self.freeze_parameters_keywords = freeze_parameters_keywords
