from transformers.configuration_utils import PretrainedConfig


class AlignConfig(PretrainedConfig):

    def __init__(
        self,
        enc: str = 'DKYoon/mt5-base-lm-adapt',
        lm: str = 'facebook/opt-125m',
        dim_enc: int = 2048,
        dim_lm: int = 4096,
        freeze_language_model: bool = True,
        freeze_encoder: bool = True,
        alignments: str = 'linear',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.lm = lm
        self.enc = enc
        self.dim_enc = dim_enc
        self.dim_lm = dim_lm
        self.freeze_language_model = freeze_language_model
        self.freeze_encoder = freeze_encoder
        self.alignments = alignments
        self.hidden_size = dim_lm