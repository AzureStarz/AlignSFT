from transformers.configuration_utils import PretrainedConfig
from typing import Iterable, Optional, Union

class VEDAlignConfig(PretrainedConfig):

    def __init__(
        self,
        enc: str = 'DKYoon/mt5-base-lm-adapt',
        lm: str = 'facebook/opt-125m',
        dim_enc: int = 768,
        dim_lm: int = 768,
        freeze_language_model: bool = True,
        freeze_encoder: bool = True,
        freeze_alignment: bool = True,
        alignments: str = 'linear',
        bottleneck_num_attention_heads: int = 32,
        bottleneck_model_dim: int = 768,
        bottleneck_beta_individual: float = 0.01,
        bottleneck_alpha_aggregate: float = 0.2,
        bottleneck_individual_posterior_kernel: str = 'kl_div',
        bottleneck_use_fina_linear: bool = False,
        bottleneck_loss_weight: float = 1.0,
        divergence_kernel_individual_posterior_kernel: str = 'l2wass',
        divergence_kernel_scaler: Optional[Union[Iterable, float]] = [5e-05, 5e-02],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.lm = lm
        self.enc = enc
        self.dim_enc = dim_enc
        self.dim_lm = dim_lm
        self.freeze_language_model = freeze_language_model
        self.freeze_encoder = freeze_encoder
        self.freeze_alignment = freeze_alignment
        self.alignments = alignments
        self.hidden_sizes = [dim_lm, dim_enc]
        self.bottleneck_num_attention_heads = bottleneck_num_attention_heads
        self.bottleneck_model_dim = bottleneck_model_dim
        self.bottleneck_beta_individual = bottleneck_beta_individual
        self.bottleneck_alpha_aggregate = bottleneck_alpha_aggregate
        self.bottleneck_individual_posterior_kernel = bottleneck_individual_posterior_kernel
        self.bottleneck_use_fina_linear = bottleneck_use_fina_linear
        self.bottleneck_loss_weight = bottleneck_loss_weight
        self.divergence_kernel_individual_posterior_kernel = divergence_kernel_individual_posterior_kernel
        self.divergence_kernel_scaler = divergence_kernel_scaler