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
