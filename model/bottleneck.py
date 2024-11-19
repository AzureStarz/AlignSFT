from argparse import ArgumentError
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .multiheadedpooling import MultiHeadedPooling

from abc import ABC

def gaussian_kl(mu, logvar):
    """
    Kullback-Liebler divergence between a Gaussian and the Prior Gaussian N(0,I).
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

# def gaussian_kl(mu, logvar, mask):
#     """
#     Kullback-Leibler divergence between a Gaussian and the prior Gaussian N(0, I),
#     with padding masked out.

#     Args:
#         mu (torch.Tensor): Mean of the posterior distribution, shape [bsz, seq_len, hidden_dim].
#         logvar (torch.Tensor): Log variance of the posterior distribution, shape [bsz, seq_len, hidden_dim].
#         mask (torch.Tensor): Padding mask, shape [bsz, seq_len], where True indicates valid tokens.
    
#     Returns:
#         torch.Tensor: Summed KL divergence per batch, shape [bsz].
#     """
#     # Compute KL divergence for each element
#     exponential = 1 + logvar - mu.pow(2) - logvar.exp()
#     kl_div = -0.5 * exponential  # Shape: [bsz, seq_len, hidden_dim]
    
#     # Sum over hidden dimensions
#     kl_div = torch.sum(kl_div, dim=-1)  # Shape: [bsz, seq_len]
    
#     # Apply the mask (only sum over valid tokens)
#     kl_div_masked = kl_div * mask  # Mask out padding tokens
#     return torch.sum(kl_div_masked, dim=-1)  # Summed KL divergence per batch, shape [bsz]


def reparameterize_gaussian(mu, logvar, var_weight=1.0):
    """
    Sample a Gaussian from mu, logvar from the Encoder
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)

    return mu + eps * std * var_weight

class BNScalar(nn.Module):
    """特殊的scale层"""
    def __init__(self, model_dim, tau=0.5):
        super(BNScalar, self).__init__()
        self.tau = tau
        self.scale = nn.Parameter(torch.zeros(model_dim), requires_grad=True)  # scale parameter will be created during forward

    def forward(self, inputs, mode='positive'):
        if mode == 'positive':
            scale = self.tau + (1 - self.tau) * torch.sigmoid(self.scale)
        else:
            scale = (1 - self.tau) * torch.sigmoid(-self.scale)

        return inputs * torch.sqrt(scale)

    def extra_repr(self):
        return f'tau={self.tau}'

class Bottleneck(ABC, nn.Module):
    """
    A `Bottleneck` class which imposes some constraints on the output
    of an encoder as an intermediate transform between Encoder and Decoder.

    This is useful for Pooling, Variational Logic etc....

    Realistically we inherit from `_EncoderBase` as this logic
    could also be some form of `Seq2SeqEncoder` except:
    (a) the `Bottleneck` will not do any direct encoding and may be lossy
    (b) cleaner than defining some `AnotherEncoder` class
    """
    def get_input_dim(self) -> int:
        """
        Returns the dimension of the vector input for each element in the sequence input
        to a `Seq2SeqEncoder`. This is `not` the shape of the input tensor, but the
        last element of that shape.
        """
        raise NotImplementedError

    def get_output_dim(self) -> int:
        """
        Returns the dimension of each vector in the sequence output by this `Seq2SeqEncoder`.
        This is `not` the shape of the returned tensor, but the last element of that shape.
        """
        raise NotImplementedError

    def is_bidirectional(self) -> bool:
        raise False

class VariationalBottleneck(Bottleneck):
    """
    Variational Sampling + Reparameterisation

    \mu = Encoder Outputs
    \log\var = PooledSampleOutputs

    std = exp(0.5*\logvar)
    z = \mu + std * torch.randn_like(std)

    This model does **not** pool to a single output representation
    for the sequence. For this we will use `VariationalPoolingBottleneck`

    Note: _may_ need an extra `Linear()` output layer (https://github.com/clementchadebec/benchmark_VAE/blob/main/src/pythae/models/nn/default_architectures.py#L61)
    (Use use_final_linear=True)
    """
    def __init__(
    self,
    num_attention_heads: int,
    model_dim: int,
    dropout: Optional[float] = 0.1,
    model_dim_out: Optional[int] = None,
    use_final_linear: Optional[bool] = True,
    use_bilinear: Optional[bool] = False,
    use_layer_norm: Optional[bool] = False
    ) -> None:
        super().__init__()
        self.logvar_pooling = MultiHeadedPooling(
            num_heads=num_attention_heads,
            model_dim=model_dim,
            dropout=dropout,
            model_dim_out=model_dim_out,
            use_final_linear=use_final_linear,
            use_bilinear=use_bilinear,
            use_layer_norm=use_layer_norm
        )

    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor = None, return_encoder_states: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoding = inputs      
        var_weight = 1.0 # for now mimic hosking

        # shape: (batch_size, 1, encoder_output_dim)
        logvar = self.logvar_pooling(key=encoding, value=encoding, mask=mask)
        
        # Reparameterise over a sequence
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        gaussian_encoding = reparameterize_gaussian(mu=encoding, logvar=logvar, var_weight=var_weight)

        # if not self.training:
            # return gaussian_encoding, mask, 0, logvar
            
        # Calculate KL term from gaussian_kl. Add to returned output
        kl_losses = gaussian_kl(encoding, logvar)
        kl_losses.mul_(mask) # Multiply by mask to ignore loss on padding
        kl_loss = torch.mean(kl_losses, dim=(0,1))

        return gaussian_encoding, mask, kl_loss, logvar


class WassersteinBottleneck(Bottleneck):
    """
    Wasserstein-distance based b'neck using Maximum Mean Discrepancy based minimization (Optimal Transport Theory based model)
    From: https://arxiv.org/abs/1711.01558
    Based on https://github.com/clementchadebec/benchmark_VAE/blob/main/src/pythae/models/wae_mmd/wae_mmd_model.py
    and https://github.com/tolstikhin/wae/blob/master/
    """
    def __init__(
    self,
    num_attention_heads: int,
    model_dim: int,
    mmd_kernel: Optional[str] = "imq",
    dropout: Optional[float] = 0.1,
    model_dim_out: Optional[int] = None,
    use_final_linear: Optional[bool] = True,
    use_bilinear: Optional[bool] = False,
    use_layer_norm: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.kernel_bandwidth = 1.0 # 
        self.mmd_kernel = mmd_kernel
        self.logvar_pooling = MultiHeadedPooling(
            num_heads=num_attention_heads,
            model_dim=model_dim,
            dropout=dropout,
            model_dim_out=model_dim_out,
            use_final_linear=use_final_linear,
            use_bilinear=use_bilinear,
            use_layer_norm=use_layer_norm
        )
        self.bn_scalar = BNScalar(model_dim)
        # self.z_mean_bn = nn.BatchNorm1d(model_dim, affine=False, eps=1e-8)
        self.z_mean_ln = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-8)
        # self.z_std_bn = nn.BatchNorm1d(model_dim, affine=False, eps=1e-8)
        self.z_std_ln = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-8)
        self.z_mean_dense = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2, True),
            nn.GELU(),
            nn.Linear(model_dim * 2, model_dim, True),
        )

    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor = None, return_encoder_states: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoding = inputs
        var_weight = 1.0 # for now mimic hosking

        # shape: (batch_size, 1, encoder_output_dim)
        logvar = self.logvar_pooling(key=encoding, value=encoding, mask=mask)
        
        # Reparameterise over a sequence
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        gaussian_encoding = reparameterize_gaussian(mu=encoding, logvar=logvar, var_weight=var_weight)

        # if not self.training:
            # return gaussian_encoding, mask, 0

        # 1. Sample prior from randn N(0, I)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        z_prior = torch.randn_like(gaussian_encoding)

        # 2. Get Kernel between Q(z|x) and P(z)
        mmd_loss = self.sequence_imq_kernel(gaussian_encoding, z_prior)
        
        return gaussian_encoding, mask, mmd_loss, logvar

    def sequence_imq_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Inverse MultiQuadratic Kernel based on https://github.com/schelotto/Wasserstein-AutoEncoders/blob/master/wae_mmd.py

        Standard formulation computes a (bsz, bsz) matrix and computes a kernel over this. We have an additional `seq_len`
        dimension which we treat as a batch dim here and compute the kernel per time-step before averaging at the end.

        e.g. x and y are of shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        """
        batch_size, seq_len, encoder_dim = x.size()

        # (seq_len_x, batch_size, encoder_output_dim)
        x = x.contiguous().permute(1, 0, 2)

        # (seq_len_y, batch_size, encoder_output_dim)
        y = y.contiguous().permute(1, 0, 2)

        # (seq_len_x, batch_size, 1)
        norms_x = x.pow(2).sum(dim=2, keepdim=True)

        # (seq_len_y, batch_size, 1)
        norms_y = y.pow(2).sum(dim=2, keepdim=True)

        # (seq_len_x, batch_size, batch_size)
        prods_x = torch.bmm(x, x.transpose(1, 2))
        # (seq_len_y, batch_size, batch_size)
        prods_y = torch.bmm(y, y.transpose(1, 2))

        # (seq_len_x, batch_size, batch_size)
        dists_x = norms_x + norms_x.transpose(1, 2) - 2 * prods_x
        # (seq_len_y, batch_size, batch_size)
        dists_y = norms_y + norms_y.transpose(1, 2) - 2 * prods_y

        # (seq_len_x, batch_size, batch_size)
        dot_prod = torch.bmm(x, y.transpose(1, 2))
        dist_c   = norms_x + norms_y.transpose(1, 2) - 2 * dot_prod

        res = torch.zeros((seq_len)).to(x.device)
        eps = torch.ones_like(dists_x).to(x.device) * float("1e-9")
        scales = [0.1, 0.2, 0.5, 1., 2., 5., 10.]
        for scale in scales:
            C = 2 * encoder_dim * self.kernel_bandwidth * scale

            res1 = C / (C + dists_x + eps)    # (seq_len, batch_size, batch_size)
            res1 += C / (C + dists_y + eps)   # (seq_len, batch_size, batch_size)
            
            eye = (1 - torch.eye(batch_size, device=res1.device)).unsqueeze(0) # (1, batch_size, batch_size)

            res1 = res1 * eye           # (seq_len, batch_size, batch_size)
            res1 = torch.sum(res1, dim=(-1, -2)).div_(batch_size * (batch_size - 1)) # (seq_len, )

            res2 = C / (C + dist_c + eps)     # (seq_len, batch_size, batch_size)
            res2 = torch.sum(res2, dim=(-1, -2)) * 2 / (batch_size ** 2) # (seq_len, )

            # (seq_len, )
            res += res1 - res2 

        return res.mean() # scalar

    def rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Returns a matrix of shape (batch_size, batch_size) containing the pairwise kernel computation
        according to the Radial Basis Function
        """
        raise ArgumentError("RBF Kernel is currently not supported. Use IMQ Kernel.")


class JointPosteriorIndividualAggregateWassersteinBottleneck(WassersteinBottleneck):
    """
    Joint Individual and Aggregate Posterior Alignment from https://arxiv.org/abs/1812.02833 equation (7)

    L = loglike(y|z) - Beta * Individual Posterior - alpha * Aggregate Posterior

    Individual Posterior - q(z|x) is a parametric alignment over each element in q - KLDiv

    Aggregate Posterior - q(z) is a Monte Carlo sample using MMD over the whole Z space.
    """

    def __init__(
    self,
    num_attention_heads: int,
    model_dim: int,
    beta_individual: float,
    alpha_aggregate: float,  
    individual_posterior_kernel: str = "kl_div",
    mmd_kernel: Optional[str] = "imq",
    dropout: Optional[float] = 0.1,
    model_dim_out: Optional[int] = None,
    use_final_linear: Optional[bool] = True,
    use_bilinear: Optional[bool] = False,
    use_layer_norm: Optional[bool] = False,
    ) -> None:
        super(JointPosteriorIndividualAggregateWassersteinBottleneck, self).__init__(
            num_attention_heads,
            model_dim,
            mmd_kernel,
            dropout, 
            model_dim_out, 
            use_final_linear, 
            use_bilinear, 
            use_layer_norm
        )

        self.beta_individual = beta_individual
        self.alpha_aggregate = alpha_aggregate
        if individual_posterior_kernel == "kl_div":
            self.posterior_individual = self.posterior_kl_divergence
        elif individual_posterior_kernel == "l2wass":
            raise ArgumentError(f"Not implemented yet!")
        else:
            raise ArgumentError(f"Argument for individual_posterior_kernel:{individual_posterior_kernel} not recognised!")

    def posterior_kl_divergence(self, encoding_mu, encoding_logvar):
        return gaussian_kl(encoding_mu, encoding_logvar)

    def forward(
        self, 
        inputs: torch.Tensor, 
        mask: torch.BoolTensor = None, 
        return_encoder_states: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:

        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoding = self.z_mean_dense(inputs)
        var_weight = 1.0 # for now mimic hosking

        # shape: (batch_size, 1, encoder_output_dim)
        logvar = self.logvar_pooling(key=encoding, value=encoding, mask=mask)
        
        encoding = self.z_mean_ln(encoding)
        encoding = self.bn_scalar(encoding, mode='positive')
        
        logvar = self.z_std_ln(logvar)
        logvar = self.bn_scalar(logvar, mode='negative')
        
        # Reparameterise over a sequence
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        gaussian_encoding = reparameterize_gaussian(mu=encoding, logvar=logvar, var_weight=var_weight)

        ## Individual Posterior
        posterior_individual_loss = gaussian_kl(encoding, logvar)
        posterior_individual_loss.mul_(mask) # Multiply by mask to ignore loss on padding
        posterior_individual_loss = torch.mean(posterior_individual_loss, dim=(0,1))

        ## Aggregate Posterior
        # 1. Sample prior from randn N(0, I)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        z_prior = torch.randn_like(gaussian_encoding)

        # 2. Get Kernel between Q(z|x) and P(z)
        posterior_aggregate_loss = self.sequence_imq_kernel(gaussian_encoding, z_prior)
        
        total_loss = self.beta_individual * posterior_individual_loss + self.alpha_aggregate * posterior_aggregate_loss

        return gaussian_encoding, mask, total_loss, logvar
