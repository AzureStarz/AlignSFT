import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple, Union, List
import re
import os
import math
import contextlib
import logging

from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2PreTrainedModel, GPT2LMHeadModel
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoModel, GPTNeoPreTrainedModel, GPTNeoForCausalLM
from transformers.models.xglm.modeling_xglm import XGLMModel, XGLMPreTrainedModel, XGLMForCausalLM
from transformers.models.bloom.modeling_bloom import BloomModel, BloomPreTrainedModel, BloomForCausalLM
from transformers.models.llama.modeling_llama import LlamaModel, LlamaPreTrainedModel, LlamaForCausalLM, LlamaDecoderLayer, LlamaRMSNorm
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.mt5.modeling_mt5 import MT5Model, MT5PreTrainedModel, MT5EncoderModel
from transformers.models.mt5.configuration_mt5 import MT5Config
from transformers import Cache, DynamicCache, StaticCache, PreTrainedModel
from transformers import PreTrainedModel, AutoModel, AutoConfig, PreTrainedTokenizer, MT5EncoderModel, UMT5EncoderModel
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
    CausalLMOutputWithPast,
    CausalLMOutputWithCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    BaseModelOutputWithPast
)
from safetensors.torch import load_file  # Assuming you're using safetensors to load the projector weights

from .alignment_modules import Linear, LinearWithAddedEos, PerceiverResampler, FFNWithAddedEos, FFN
from ..hparams import get_infer_args, get_train_args
from .configuration_align import AlignConfig

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

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_bottom': average of the bottom layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_bottom2': average of the first two layers.
    'avg_first_last': average of the first and the last layers.
    'last': the hidden state of the last token in the last layers.
    'last_top2': average of the hidden state of the last token in the last two layers.
    'last_first_last': average of the hidden state of the last token in the first and last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        self.pooler_re = re.compile(r'^(avg|last|max)-[0-9]+$')
        assert pooler_type in [
            "cls", "cls_before_pooler", 
            "avg", "avg_top2", "avg_top1", "avg_first_last", "avg_bottom", "avg_bottom2", "avg_all",
            "max",
            "last", "last_top2", "last_first_last",
            ] or self.pooler_re.match(pooler_type), "unrecognized pooling type %s" % self.pooler_type

    def need_out_hiddens(self, pooler_type):
        assert pooler_type in [
            "cls", "cls_before_pooler", 
            "avg", "avg_top2", "avg_top1", "avg_first_last", "avg_bottom", "avg_bottom2", "avg_all", "avg_medium"
            "max",
            "last", "last_top2", "last_first_last",
            ] or self.pooler_re.match(pooler_type), "unrecognized pooling type %s" % self.pooler_type

        return pooler_type in [
            'avg_top1', 'avg_top2', 'avg_first_last', 
            "avg_bottom", "avg_bottom2", "avg_all",
            'last_top2', 'last_first_last'] or self.pooler_re.match(pooler_type)
        

    def average_embed(self, hidden_states:torch.tensor, attention_mask:torch.tensor):
        """
        hidden_states.shape: (bs, sent_len, hidden_size)
        attention_mask: (bs, sent_len) (1: attent, 0: pad)
        """
        return ((hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))

    def max_embed(self, hidden_states:torch.tensor, attention_mask:torch.tensor):
        """
        hidden_states.shape: (bs, sent_len, hidden_size)
        attention_mask: (bs, sent_len) (1: attent, 0: pad)
        """
        return torch.max((hidden_states * attention_mask.unsqueeze(-1)), dim=1)[0]

    def last_embed(self, hidden_states:torch.tensor, attention_mask:torch.tensor):
        """
        hidden_states.shape: (bs, sent_len, hidden_size)
        attention_mask: (bs, sent_len) (1: attent, 0: pad)
        """
        sequence_lengths = attention_mask.sum(-1) - 1
        batch_size = hidden_states.shape[0]
        return hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        # pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return self.average_embed(last_hidden, attention_mask)
        elif "avg-" in self.pooler_type:
            l_idx = int(self.pooler_type.split("-")[1])
            return self.average_embed(hidden_states[l_idx], attention_mask)
        elif "max-" in self.pooler_type:
            l_idx = int(self.pooler_type.split("-")[1])
            return self.max_embed(hidden_states[l_idx], attention_mask)
        elif "last-" in self.pooler_type:
            l_idx = int(self.pooler_type.split("-")[1])
            return self.last_embed(hidden_states[l_idx], attention_mask)
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            return self.average_embed((first_hidden + last_hidden) / 2.0, attention_mask)
        elif self.pooler_type == "avg_bottom":
            first_hidden = hidden_states[1]
            return self.average_embed(first_hidden, attention_mask)
        elif self.pooler_type == "avg_bottom2":
            first_hidden = hidden_states[1]
            second_hidden = hidden_states[2]
            return self.average_embed((first_hidden + second_hidden) / 2.0, attention_mask)
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            return self.average_embed((last_hidden + second_last_hidden) / 2.0, attention_mask)
        elif self.pooler_type == "avg_top1":
            # last_hidden = hidden_states[-1]
            return self.average_embed(last_hidden, attention_mask)
        elif self.pooler_type == "avg_all":
            num_layer = len(hidden_states)-1
            hidden_sum = 0
            for l in range(num_layer):
                hidden_sum += hidden_states[l+1]
            return self.average_embed(hidden_sum / float(num_layer), attention_mask)
        elif "avg_medium" in self.pooler_type:
            start_layer = int(self.pooler_type.split("-")[0])
            end_layer = int(self.pooler_type.split("-")[-1])
            return [self.average_embed(hidden_states[l], attention_mask) for l in range(start_layer, end_layer + 1)]
        elif self.pooler_type == "last":
            return self.last_embed(last_hidden, attention_mask)
        elif self.pooler_type == "last_first_last":
            first_layer_last = self.last_embed(hidden_states[1], attention_mask)
            last_layer_last = self.last_embed(hidden_states[-1], attention_mask)
            return (first_layer_last + last_layer_last) / 2.0
        elif self.pooler_type == "last_top2":
            second_last_layer_last = self.last_embed(hidden_states[-2], attention_mask)
            last_layer_last = self.last_embed(hidden_states[-1], attention_mask)
            return (second_last_layer_last + last_layer_last) / 2.0
        else:
            raise NotImplementedError

class CustomLlamaDecoderLayer(LlamaDecoderLayer):

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.align_projection = nn.Linear(2048, config.hidden_size, bias=False)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        sent_embed: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        # 将 sentence_embedding 投影到 hidden_states 维度
        sent_embed = self.align_projection(sent_embed)  # [batch_size, 1, hidden_size]
        hidden_states = hidden_states + sent_embed

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class CustomLlamaModel(LlamaModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, model_args):
        super().__init__(config)
        # Initialize weights and apply final processing
        self.post_init()
        self.init_weights()
        # Initialize the alignment modules
        self.align_init(config, model_args)
    
    def align_init(self, config, model_args):
        layers = []
        for layer_idx in range(config.num_hidden_layers):
            if layer_idx in model_args.trainable_layer_id:
                layers.append(CustomLlamaDecoderLayer(config, layer_idx))
            else:
                layers.append(LlamaDecoderLayer(config, layer_idx))
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        sent_embed: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            # logger.warning_once(
            #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            # )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            # logger.warning_once(
            #     "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
            #     "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            # )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    sent_embed,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    sent_embed=sent_embed,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class CustomMT5EncoderModel(MT5PreTrainedModel):
    _tied_weights_keys = ["model.encoder.embed_tokens.weight"]

    # Copied from transformers.models.t5.modeling_t5.T5ForTokenClassification.__init__ with T5->MT5
    def __init__(self, config: MT5Config, model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_args
        self.model = MT5EncoderModel(config)
        # Initialize weights and apply final processing
        # self.post_init()
        # Initialize pooler
        self.encoder_init(config)
    
    def encoder_init(self, config):
        """
        Contrastive learning class init function.
        """
        self.pooler_type = self.model_args.pooler_type
        self.pooler = Pooler(self.model_args.pooler_type)
        # if self.model_args.pooler_type == "cls" or self.model_args.cl_mlp:
        #     self.mlp = MLPLayer(config)
        # self.sim = Similarity(temp=self.model_args.temp)
        # self.init_weights()
        # self.model.requires_grad_(False)
        # for _, param in self.model.named_parameters():
        #     param.requires_grad_(False)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

class LlamaForALign(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_args
        self.model = CustomLlamaModel(config, model_args)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        # self.post_init()
        # self.lm_init()
        
    def lm_init(self):
        for name, param in self.model.named_parameters():
            if "align" in name or "post_ffn" in name:
                param.requires_grad_(True)

    def forward(self, *args, **kwargs):
        if "labels" in kwargs.keys():
            kwargs.pop("labels")
        return self.model(*args, **kwargs)

class AlignModel(PreTrainedModel):

    config: AlignConfig
    enc: PreTrainedModel
    lm: PreTrainedModel
    lm_head: nn.Linear
    embeddings: nn.Embedding
    alignment: nn.Linear

    config_class = AlignConfig

    def __init__(self, config: AlignConfig, model_args, random_init=True, suppress_warnings=True):
        super().__init__(config)
        # if 'umt5' in config.enc.lower():
        #     enc_class = UMT5EncoderModel
        # elif 'mt5' in config.enc.lower():
        #     enc_class = MT5EncoderModel
        # else:
        #     enc_class = MT5EncoderModel
        with suppress_model_loading_warnings(suppress_warnings):
            enc_config = MT5Config.from_pretrained(config.enc)
            model_config = LlamaConfig.from_pretrained(config.lm)
            if random_init:
                encoder = CustomMT5EncoderModel(config=enc_config, model_args=model_args)
                try:
                    model_config.attn_implementation = 'flash_attention_2'
                    lm = LlamaForALign(config=model_config, model_args=model_args)
                except ImportError:
                    print('Not using Flash Attention!')
                    lm = LlamaForALign(config=model_config, model_args=model_args)
            else:
                print('loading encoder from pretrained')
                encoder = CustomMT5EncoderModel.from_pretrained(config.enc, config=enc_config, model_args=model_args)
                print('loading lm from pretrained')
                try:
                    lm = LlamaForALign.from_pretrained(config.lm, config=model_config, model_args=model_args, use_flash_attention_2=True)
                except ImportError:
                    print('Not using Flash Attention!')
                    lm = LlamaForALign.from_pretrained(config.lm, config=model_config, model_args=model_args)

        assert self.config.dim_lm == lm.config.hidden_size, \
            f"specified {self.config.dim_lm=} in LangBridgeConfig, but {config.lm} has hidden size={lm.config.hidden_size}"
            
        self.lm = lm
        self.encoder = encoder

        # if config.alignments == 'linear':  # default
        #     self.alignment = Linear(
        #         dim=config.dim_enc, out_dim=config.dim_lm)
        # elif config.alignments == 'ffn':  # mlp
        #     self.alignment = FFN(
        #         dim=config.dim_enc, out_dim=config.dim_lm)
        # elif config.alignments == 'latent':
        #     self.alignment = PerceiverResampler(
        #         dim=config.dim_enc, out_dim=config.dim_lm, num_latents=config.num_latents)
        # else:
        #     raise ValueError(
        #         f'unknown alignment type {config.alignments}')
        
        # # Ensure alignment is on the same device as encoder and lm
        # self.alignment = self.alignment.to(self.encoder.device)

        # # Ensure alignment uses the same dtype as encoder and lm
        # alignment_dtype = self.encoder.dtype
        # self.alignment = self.alignment.to(dtype=alignment_dtype)
        
        if config.freeze_encoder:
            self.freeze_encoder()
        if config.freeze_language_model:
            self.freeze_lm()

    def freeze_encoder(self):
        """freeze vision model """
        for param in self.encoder.parameters():
            param.requires_grad = False

    def freeze_lm(self):
        for param in self.lm.parameters():
            param.requires_grad = False

    def unfreeze_lm(self):
        for param in self.lm.parameters():
            param.requires_grad = True

    def forward(self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        encoder_input_ids=None,
        encoder_attention_mask=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        encoder_input_dict = {
            "input_ids": encoder_input_ids,
            "attention_mask": encoder_attention_mask,
            "output_hidden_states": False,  # 设置为False，减少显存占用
            "return_dict": True,
        }

        # Get raw embeddings
        encoder_outputs = self.encoder(
            **encoder_input_dict
        )
        
        # get average pooling of encoder representation
        pooler_output = self.encoder.pooler(encoder_attention_mask, encoder_outputs).unsqueeze(1)
        # Linear projection module
        # encoder_hidden_states = self.alignment(pooler_output)
        encoder_hidden_states = pooler_output

        llm_input_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "inputs_embeds": inputs_embeds,
            "output_attentions": output_attentions,
            "output_hidden_states": False,  # 设置为False，减少显存占用
            "return_dict": return_dict,
            "sent_embed": encoder_hidden_states,
        }
        
        llm_input_dict.update(kwargs)

        outputs = self.lm(
            **llm_input_dict
        )

        # Calculate loss for CLM
        # labels = labels.view(-1, labels.size(-1))
        logits = self.lm.lm_head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=None,  # 设置为None，减少显存占用
            attentions=None,     # 设置为None，减少显存占用
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        encoder_input_ids=None,
        encoder_attention_mask=None,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "encoder_input_ids": encoder_input_ids,
                "encoder_attention_mask": encoder_attention_mask,
            }
        )
        return model_inputs
