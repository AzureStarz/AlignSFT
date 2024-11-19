# code adapted from flamingo-mini https://github.com/dhansmair/flamingo-mini

from __future__ import annotations
from abc import ABC
from typing import Any, Dict, List, Iterable
import contextlib
import logging
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from transformers import PreTrainedModel, AutoModel, AutoConfig, PreTrainedTokenizer, MT5EncoderModel, UMT5EncoderModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast
)

from .configuration_ved_align import VEDAlignConfig
from .alignment_modules import Linear, LinearWithAddedEos, PerceiverResampler, FFNWithAddedEos, FeedForward
from .bottleneck import JointPosteriorIndividualAggregateWassersteinBottleneck
from .divergence_kernel import DivergenceKernel, JointPosteriorIndividualAggregateWassersteinKernel, L2WassersteinDivergenceKernel

@contextlib.contextmanager
def suppress_model_loading_warnings(suppress: bool = True):
    if suppress:
        logger = logging.getLogger('transformers.modeling_utils')
        level = logger.level
        logger.setLevel(logging.CRITICAL)
        yield
        logger.setLevel(level)
    else:
        yield


class VEDBaseModel(ABC, PreTrainedModel):

    config: VEDAlignConfig
    enc: PreTrainedModel
    bottleneck: nn.Module
    lm: PreTrainedModel
    lm_head: nn.Linear
    embeddings: nn.Embedding
    divergence_kernel: DivergenceKernel
    divergence_kernel_scaler: Iterable
    enc_eos: nn.Parameter
    alignment: nn.Module

    config_class = VEDAlignConfig

    def __init__(self, config: VEDAlignConfig, random_init=True, suppress_warnings=True):
        super().__init__(config)
        if 'umt5' in config.enc.lower():
            enc_class = UMT5EncoderModel
        elif 'mt5' in config.enc.lower():
            enc_class = MT5EncoderModel
        else:
            enc_class = AutoModel

        with suppress_model_loading_warnings(suppress_warnings):
            if random_init:
                enc_config = AutoConfig.from_pretrained(
                    config.enc)
                self.enc = enc_class(config=enc_config)
            else:
                print('loading encoder from pretrained')
                self.enc = enc_class.from_pretrained(config.enc)

        # self.enc.gradient_checkpointing_enable(
            # gradient_checkpointing_kwargs={'use_reentrant': False})

        if config.alignments == 'linear':  # default
            self.alignment = nn.Linear(config.dim_enc, config.dim_lm, bias=True)
            self.bottelneck_alignment = nn.Linear(config.dim_enc, config.dim_lm, bias=True)
            self.enc_eos = nn.Parameter(torch.randn(config.dim_lm))
            # self.alignment = LinearWithAddedEos(
            #     dim=config.dim_enc, out_dim=config.dim_lm)
        elif config.alignments == 'ffn':  # mlp
            self.alignment = FFNWithAddedEos(
                dim=config.dim_enc, out_dim=config.dim_lm)
        elif config.alignments == 'latent':
            self.alignment = PerceiverResampler(
                dim=config.dim_enc, out_dim=config.dim_lm, num_latents=config.num_latents)
        else:
            raise ValueError(
                f'unknown alignment type {config.alignments}')

        self.bottleneck = JointPosteriorIndividualAggregateWassersteinBottleneck(
            num_attention_heads=config.bottleneck_num_attention_heads,
            model_dim=config.bottleneck_model_dim,
            model_dim_out=config.bottleneck_model_dim,
            beta_individual=config.bottleneck_beta_individual,
            alpha_aggregate=config.bottleneck_alpha_aggregate,
            individual_posterior_kernel=config.bottleneck_individual_posterior_kernel,
            use_final_linear=config.bottleneck_use_fina_linear,
        )
        self.bottleneck_loss_weight = config.bottleneck_loss_weight
        
        # self.divergence_kernel = JointPosteriorIndividualAggregateWassersteinKernel(
        #     individual_posterior_kernel=config.divergence_kernel_individual_posterior_kernel
        # )
        
        # self.divergence_kernel_scaler = [0.5, 0.01]

    def freeze_encoder(self):
        """freeze vision model """
        for param in self.enc.parameters():
            param.requires_grad = False

    def freeze_lm(self):
        for param in self.lm.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False

    def unfreeze_lm(self):
        for param in self.lm.parameters():
            param.requires_grad = True
        for param in self.lm_head.parameters():
            param.requires_grad = True
    
    def freeze_alignment(self):
        for param in self.alignment.parameters():
            param.requires_grad = False
        for param in self.bottelneck_alignment.parameters():
            param.requires_grad = False
        for param in self.bottleneck.parameters():
            param.requires_grad = False
        self.enc_eos.requires_grad = False

    # get soft prompts
    def get_encoder_features(self, enc_ids: torch.Tensor, enc_mask: torch.Tensor) -> torch.Tensor:
        if self.config.freeze_encoder:
            with torch.no_grad():
                enc_features = self.enc(
                    input_ids=enc_ids, attention_mask=enc_mask).last_hidden_state  # (b, s, d)
        else:
            enc_features = self.enc(
                input_ids=enc_ids, attention_mask=enc_mask).last_hidden_state
        # enc_features = self.alignment(enc_features, enc_mask)
        return enc_features

    def _add_enc_eos(self, x):
        eos = repeat(self.enc_eos, 'd -> b d', b=x.shape[0])
        eos = rearrange(eos, 'b d -> b 1 d')
        x = torch.cat((x, eos), dim=1)
        return x

    # def _project_enc_features(self, x):
    #     return self._add_enc_eos(self.alignment(x))

    def _bottleneck(self, enc_features: torch.Tensor, enc_mask: torch.Tensor, return_encoder_states: bool = False) -> Dict[str, torch.Tensor]:

        ##
        # `Bottleneck` logic
        ##      
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        # shape: (batch_size, <bottleneck_output_dim>, encoder_output_dim)

        bneck_outputs, bneck_source_mask, bottleneck_loss, bottleneck_logvar = self.bottleneck(enc_features, enc_mask, return_encoder_states=return_encoder_states)

        output_dict = {"source_mask": bneck_source_mask, "encoder_outputs": bneck_outputs, "bottleneck_loss": bottleneck_loss}

        if return_encoder_states:
            output_dict['bottleneck_logvar'] = bottleneck_logvar
        else:
            del bottleneck_logvar
        torch.cuda.empty_cache()

        return output_dict

    def compute_crosslingual_alignmenmt_divergence(self, source_outputs, target_outputs):
        divergence_kernel_output = self.divergence_kernel.compute_kernel(source_outputs, target_outputs)

        # Some kernels return Tuples. This is used for JointPosteriorIndividualAggregateWassersteinBottleneck
        if isinstance(divergence_kernel_output, tuple):
            if not isinstance(self.divergence_kernel_scaler, Iterable): # Divergence_kernel_scaler = (Beta_indiv, Alpha_agg)
                self.divergence_kernel_scaler = defaultdict(lambda: self.divergence_kernel_scaler)

            divergence_kernel_loss = divergence_kernel_output[0] * self.divergence_kernel_scaler[0] + \
                divergence_kernel_output[1] * self.divergence_kernel_scaler[1]
        else: # Output is a scalar
            divergence_kernel_loss = divergence_kernel_output * self.divergence_kernel_scaler
        
        torch.cuda.empty_cache()
        return divergence_kernel_loss

    def forward(
        self,
        enc_ids: torch.Tensor | None = None,
        enc_mask: torch.Tensor | None = None,
        src_enc_ids: torch.Tensor | None = None,
        src_enc_mask: torch.Tensor | None = None,
        tgt_enc_ids: torch.Tensor | None = None,
        tgt_enc_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = True,
        past_key_values: tuple | None = None,
        return_dict: bool = True,
        labels: torch.Tensor | None = None,
        loss_reduction: str = 'mean',
        **kwargs
    ) -> CausalLMOutputWithPast:
        # sanity check
        assert return_dict, "can only use return_dict=True at the moment!"

        # find the input shape
        batch_size, seq_length = input_ids.shape[:2] if input_ids is not None else enc_ids.shape[:2]
        device = input_ids.device if input_ids is not None else enc_ids.device
        lm_past_key_values = None if past_key_values is None else past_key_values[0]

        if input_ids is not None:
            embeddings = self.embeddings(input_ids)
        bos_shifted = False
        if lm_past_key_values is None:
            assert enc_ids.size(0) == batch_size

            enc_features = self.get_encoder_features(
                enc_ids, enc_mask)
            bottleneck_enc_features = self.bottelneck_alignment(enc_features)
            enc_features = self.alignment(enc_features)
            bottleneck_state = self._bottleneck(bottleneck_enc_features, enc_mask.bool(), return_encoder_states=True)
            enc_features += bottleneck_state["encoder_outputs"]
            enc_features = self._add_enc_eos(enc_features)
            
            if not self.training:
                del bottleneck_state['bottleneck_loss']
            
            # if self.training and src_enc_ids is not None:
            #     src_enc_features = self.get_encoder_features(
            #         src_enc_ids, src_enc_mask)
            #     src_bottleneck_state = self._bottleneck(src_enc_features, src_enc_mask.bool(), return_encoder_states=True)
            #     src_bottleneck_state.update({"bottleneck_mean": src_enc_features})
            #     tgt_enc_features = self.get_encoder_features(
            #         tgt_enc_ids, tgt_enc_mask)
            #     tgt_bottleneck_state = self._bottleneck(tgt_enc_features, tgt_enc_mask.bool(), return_encoder_states=True)
            #     tgt_bottleneck_state.update({"bottleneck_mean": tgt_enc_features})
            #     divergence_kernel_loss = self.compute_crosslingual_alignmenmt_divergence(src_bottleneck_state, tgt_bottleneck_state)

            if input_ids is not None:
                first_input_ids = input_ids[:, 0]
                if all(first_input_ids == self.lm.config.bos_token_id):
                    # move bos embedding to the front
                    bos_shifted = True
                    embeddings = torch.cat(
                        [embeddings[:, 0].unsqueeze(dim=1), enc_features, embeddings[:, 1:]], dim=1)
                else:
                    embeddings = torch.cat([enc_features, embeddings], dim=1)
            else:
                embeddings = enc_features
            enc_feature_length = enc_features.shape[1]
        else:
            enc_feature_length = past_key_values[1]

        if input_ids is not None:
            if self.config.alignments not in ['linear', 'ffn']:
                attn_mask = torch.cat(
                    [torch.ones((batch_size, enc_feature_length), device=device, dtype=torch.long), attention_mask], dim=1)
            else:
                # use the encoder masks for the soft prompts
                if bos_shifted:
                    # TODO: messy code
                    # torch.ones are there since the alignment adds a single learnable eos token at the enc
                    attn_mask = torch.cat(
                        [attention_mask[:, 0].unsqueeze(dim=1), enc_mask, torch.ones((batch_size, 1), device=device, dtype=torch.long), attention_mask[:, 1:]], dim=1)
                else:
                    attn_mask = torch.cat(
                        [enc_mask, torch.ones((batch_size, 1), device=device, dtype=torch.long), attention_mask], dim=1)
        else:
            attn_mask = enc_mask

        # pass through LM
        out: BaseModelOutputWithPast = self.lm(
            attention_mask=attn_mask,
            inputs_embeds=embeddings,
            use_cache=use_cache,
            past_key_values=lm_past_key_values,
            return_dict=True,
            **kwargs
        )

        logits: torch.Tensor = self.lm_head(out.last_hidden_state)

        loss = None
        if labels is not None:
            # no loss for soft prompts
            no_loss_labels = torch.zeros(
                (batch_size, enc_feature_length), device=device, dtype=torch.long) + -100

            if bos_shifted:
                full_labels = torch.cat(
                    [labels[:, 0].unsqueeze(dim=1), no_loss_labels, labels[:, 1:]], dim=1)
            else:
                full_labels = torch.cat(
                    [no_loss_labels, labels], dim=1)
            # logits shape (batch, seq_length, #words)
            shift_logits = logits[..., :-1, :].contiguous()
            # labels shape (batch, seq_length)
            shift_labels = full_labels[..., 1:].contiguous()

            # Flatten the tokens
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1), reduction=loss_reduction)
            if loss_reduction == 'none':
                # CrossEntropyLoss will flatten all dimensions by default
                loss = rearrange(loss, '(b s) -> b s', b=batch_size)
                # remove soft promtps from loss
                loss = loss[:, enc_feature_length:]

        # 如果 loss 超过阈值，打印调试信息
        # if loss.item() > 10.0:
        #     print(f'[DEBUG] High loss detected: lm_loss = {loss.item()}')
        # print(f'[DEBUG] lm_loss: {loss}')

        if self.training and "bottleneck_loss" in bottleneck_state:
            if bottleneck_state["bottleneck_loss"] != None:
                bottleneck_loss = bottleneck_state["bottleneck_loss"] * self.bottleneck_loss_weight
                # if bottleneck_loss > 10.0:
                #     print(f'[DEBUG] High loss detected: bottleneck_loss: {bottleneck_loss}')
                # print(f'[DEBUG] bottleneck_loss: {bottleneck_loss}')
                loss += bottleneck_loss
        
        # if self.training and not self.config.freeze_alignment:
        #     if divergence_kernel_loss is not None:
        #         if divergence_kernel_loss > 10.0:
        #             print(f'[DEBUG] High loss detected: divergence_kernel_loss: {divergence_kernel_loss}')
        #         print(f'[DEBUG] divergence_kernel_loss: {divergence_kernel_loss}')
        #         loss += divergence_kernel_loss
        #     else:
        #         raise ValueError
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=(out.past_key_values,
                             enc_feature_length) if use_cache else None,
            hidden_states=out.hidden_states,
            attentions=out.attentions,
        )


# used for debbuging with opt-125m
class VEDOPT(VEDBaseModel):
    config: VEDAlignConfig

    def __init__(self, config: VEDAlignConfig, random_init=True):
        from transformers import OPTForCausalLM, OPTModel
        super().__init__(config, random_init=random_init)

        if random_init:
            model_config = AutoConfig.from_pretrained(config.lm)
            base_lm: OPTForCausalLM = OPTForCausalLM(config=model_config)
        else:
            print('loading lm from pretrained')
            base_lm: OPTForCausalLM = OPTForCausalLM.from_pretrained(
                config.lm)
        assert self.config.dim_lm == base_lm.config.hidden_size, \
            f"specified {self.config.dim_lm} in VEDAlignConfig, but {config.lm} has hidden size={base_lm.config.hidden_size}"

        self.lm: OPTModel = base_lm.model
        self.lm_head = base_lm.lm_head
        self.embeddings = base_lm.get_input_embeddings()


class VEDLlama(VEDBaseModel):
    config: VEDAlignConfig

    def __init__(self, config: VEDAlignConfig, random_init=True):
        from transformers import LlamaForCausalLM, LlamaModel
        super().__init__(config, random_init=random_init)

        if random_init:
            model_config = AutoConfig.from_pretrained(config.lm)
            try:
                model_config.attn_implementation = 'flash_attention_2'
                base_lm: LlamaForCausalLM = LlamaForCausalLM(
                    config=model_config)
            except ImportError:
                print('Not using Flash Attention!')
                base_lm: LlamaForCausalLM = LlamaForCausalLM(
                    config=model_config)
        else:
            print('loading lm from pretrained')
            try:
                base_lm: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(
                    config.lm, use_flash_attention_2=True)
            except ImportError:
                print('Not using Flash Attention!')
                base_lm: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(
                    config.lm)

        assert self.config.dim_lm == base_lm.config.hidden_size, \
            f"specified {self.config.dim_lm} in VEDAlignConfig, but {config.lm} has hidden size={base_lm.config.hidden_size}"

        self.lm: LlamaModel = base_lm.model
        self.lm_head = base_lm.lm_head
        self.embeddings = base_lm.get_input_embeddings()


class VEDMistral(VEDBaseModel):
    config: VEDAlignConfig

    def __init__(self, config: VEDAlignConfig, random_init=True):
        from transformers import MistralForCausalLM, MistralModel
        super().__init__(config, random_init=random_init)

        if random_init:
            model_config = AutoConfig.from_pretrained(config.lm)
            try:
                model_config.attn_implementation = 'flash_attention_2'
                base_lm: MistralForCausalLM = MistralForCausalLM(
                    config=model_config)
            except ImportError:
                print('Not using Flash Attention!')
                base_lm: MistralForCausalLM = MistralForCausalLM(
                    config=model_config)
        else:
            try:
                base_lm: MistralForCausalLM = MistralForCausalLM.from_pretrained(
                    config.lm, use_flash_attention_2=True)
            except ImportError:
                print('Not using Flash Attention!')
                base_lm: MistralForCausalLM = MistralForCausalLM.from_pretrained(
                    config.lm)
        assert self.config.dim_lm == base_lm.config.hidden_size, \
            f"specified {self.config.dim_lm} in VEDAlignConfig, but {config.lm} has hidden size={base_lm.config.hidden_size}"

        self.lm: MistralModel = base_lm.model
        self.lm_head = base_lm.lm_head
        self.embeddings = base_lm.get_input_embeddings()


class VEDAlignModel(PreTrainedModel):
    config: VEDAlignConfig
    config_class = VEDAlignConfig

    _LANGUAGE_MODEL_VERSIONS = {
        'facebook/opt': VEDOPT,
        'EleutherAI/llemma': VEDLlama,
        'codellama/CodeLlama': VEDLlama,
        'microsoft/Orca-2': VEDLlama,
        'meta-math/MetaMath': VEDLlama,
        'meta-llama/Llama-2-7b-hf': VEDLlama,
        '/home/export/base/ycsc_chenkh/hitici_02/online1/data/pretrained-models/Llama-2-7b': VEDLlama,
        '/home/export/base/ycsc_chenkh/hitici_02/online1/data/pretrained-models/MetaMath-7B-V1.0': VEDLlama,
        'mistralai/Mistral-7B-v0.1': VEDMistral,
    }

    def __init__(self, config: VEDAlignConfig, random_init=True, model_class=None):
        super().__init__(config)

        if model_class is None:
            model_class = self._find_lm_class(config.lm)
        self.ved: VEDBaseModel = model_class(config, random_init=random_init)

        if config.freeze_language_model:
            self.freeze_lm()

        if config.freeze_encoder:
            self.freeze_encoder()
        
        if config.freeze_alignment:
            self.freeze_alignment()

    @classmethod
    def _find_lm_class(cls, language_model_id: str):
        for prefix, lm_class in cls._LANGUAGE_MODEL_VERSIONS.items():
            if language_model_id.startswith(prefix):
                return lm_class
        raise ValueError(f'unsupported language model {language_model_id}')

    def freeze_encoder(self):
        self.ved.freeze_encoder()

    def freeze_lm(self):
        self.ved.freeze_lm()

    def unfreeze_lm(self):
        self.ved.unfreeze_lm()
        
    def freeze_alignment(self):
        self.ved.freeze_alignment()

    def forward(
        self,
        enc_ids: torch.Tensor | None = None,
        enc_mask: torch.Tensor | None = None,
        src_enc_ids: torch.Tensor | None = None,
        src_enc_mask: torch.Tensor | None = None,
        tgt_enc_ids: torch.Tensor | None = None,
        tgt_enc_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = True,
        past_key_values: tuple | None = None,
        return_dict: bool = True,
        labels: torch.Tensor | None = None,
        loss_reduction: str = 'mean',
        **kwargs
    ) -> CausalLMOutputWithPast:

        return self.ved(
            enc_ids=enc_ids,
            enc_mask=enc_mask,
            src_enc_ids=src_enc_ids,
            src_enc_mask=src_enc_mask,
            tgt_enc_ids=tgt_enc_ids,
            tgt_enc_mask=tgt_enc_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            labels=labels,
            loss_reduction=loss_reduction,
            **kwargs
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        enc_ids: torch.Tensor | None = None,
        enc_mask: torch.Tensor | None = None,
        past=None,
        past_key_values=None,
        **kwargs
    ) -> Dict[str, Any]:
        """ hf specific function. Overridden from PreTrainedModel for text generation purposes.

        if use_cache is used, past is not None, then only the last column will be passed as input_ids.
        TODO was `past` renamed to `past_key_values` in transformers 4.26?
        """

        if past_key_values is not None or past is not None:
            input_ids = input_ids[:, -1:]

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            enc_ids=enc_ids,
            enc_mask=enc_mask,
            past_key_values=past_key_values if past_key_values is not None else past,
            **kwargs
        )

    def _reorder_cache(self, past, beam_idx):
        """ hf specific function. Overridden from PreTrainedModel.

        this is required for beam search in combination with use_cache.

        Args: 
            past is a tuple of past_key_values of the xattn layers, and of the LM layers.
            beam_idx: index of the beam
        """
        xattn_past, lm_past = past

        xattn_past_beam = tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device))
                  for past_state in layer_past)
            for layer_past in xattn_past
        )

        lm_past_beam = tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device))
                  for past_state in layer_past)
            for layer_past in lm_past
        )

        return xattn_past_beam, lm_past_beam

    # a simple function to test the model
    @torch.no_grad()
    def generate_from_prefix(
        self,
        enc_tokenizer: PreTrainedTokenizer,
        lm_tokenizer: PreTrainedTokenizer,
        prompts: List[str],
        **kwargs
    ):
        enc_input = enc_tokenizer(prompts, return_tensors='pt', padding=True)
        enc_ids = enc_input['input_ids'].to(self.device)
        enc_mask = enc_input['attention_mask'].to(self.device)

        input_ids = torch.LongTensor([lm_tokenizer.bos_token_id])
        input_ids = input_ids.repeat(enc_ids.shape[0], 1).to(self.device)
        attention_mask = torch.ones_like(input_ids)

        out_ids = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            enc_ids=enc_ids,
            enc_mask=enc_mask,
            early_stopping=True,
            use_cache=True,
            bos_token_id=lm_tokenizer.bos_token_id,
            eos_token_id=32002,  # <|im_end|>
            pad_token_id=lm_tokenizer.eos_token_id,
            **kwargs
        )

        completions = lm_tokenizer.batch_decode(
            out_ids, skip_special_tokens=True)
        # TODO: don't know why batch_decode doesn't remove <|im_end|>, since it's in the special tokens
        completions = [s.replace('<|im_end|>', '') for s in completions]
        return completions