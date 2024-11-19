# code adapted from flamingo-mini https://github.com/dhansmair/flamingo-mini

from __future__ import annotations
from abc import ABC
from typing import Any, Dict, List
import contextlib
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange, repeat

from transformers import PreTrainedModel, AutoModel, AutoConfig, PreTrainedTokenizer, MT5EncoderModel, UMT5EncoderModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast
)

from .configuration_langbridge import LangBridgeConfig
from .alignment_modules import LinearWithAddedEos, PerceiverResampler, FFNWithAddedEos, FeedForward
from .model_utils.aligner import build_aligner
import pickle

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


class Mapping(nn.Module):
    def __init__(self, mt_dim, llm_dim):
        super(Mapping, self).__init__()
        self.mlp = FeedForward(mt_dim, llm_dim)
        self.start_boundary = nn.Parameter(
            torch.zeros(1, 1, llm_dim), requires_grad=True
        )
        self.end_boundary = nn.Parameter(
            torch.zeros(1, 1, llm_dim), requires_grad=True
        )
    def forward(self, hidden_states):
        hidden_states = self.mlp(hidden_states)
        return hidden_states

    def get_embed(self):
        return self.start_boundary, self.end_boundary

class LBBaseModel(ABC, PreTrainedModel):

    config: LangBridgeConfig
    enc: PreTrainedModel
    lm: PreTrainedModel
    lm_head: nn.Linear
    embeddings: nn.Embedding

    config_class = LangBridgeConfig

    def __init__(self, config: LangBridgeConfig, random_init=True, suppress_warnings=True):
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

        # used for dimension mapping
        self.mapping = Mapping(config.dim_enc, config.dim_lm)
        # used for extracting language-agnostic information
        self.la_adapter = FeedForward(
                dim=config.dim_enc, out_dim=config.dim_enc)

        # create aligner
        # with open(config.lan_emb_path, 'rb') as pickle_file:
        # with open("/home/export/base/ycsc_chenkh/hitici_02/online1/language-agnostic-representation/wiki40b_hplt/mt5_repr_wiki40b_hplt.pkl", 'rb') as pickle_file:
        #     lan_emb = pickle.load(pickle_file)

        # self.aligner_func = build_aligner(config.align_method, lan_emb)
        # self.aligner_func = build_aligner('lir+2', lan_emb)
        
        # used for Contrastive Learning
        # self.sim = Similarity(temp=config.temp)
        self.sim = Similarity(temp=0.05)
        self.cl_loss_fct = nn.CrossEntropyLoss()
        # self.mse_loss_fct = nn.MSELoss()
        # self.coefficient_cl_loss = config.coefficient_cl_loss
        # self.coefficient_mse_loss = 0.5
        self.coefficient_cl_loss = 0.5
        self.language_mapping = {
            0: "en",
            1: "sw",
            2: "zh-cn",
            3: "bn",
            4: "de",
            5: "es",
            6: "fr",
            7: "ja",
            8: "ru",
            9: "th",
            10: "te"
        }

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
        for param in self.mapping.parameters():
            param.requires_grad = False
        for param in self.la_adapter.parameters():
            param.requires_grad = False

    # get soft prompts
    def get_encoder_features(self, enc_ids: torch.Tensor, enc_mask: torch.Tensor) -> torch.Tensor:
        if self.config.freeze_encoder:
            with torch.no_grad():
                enc_features = self.enc(
                    input_ids=enc_ids, attention_mask=enc_mask).last_hidden_state  # (b, s, d)
        else:
            enc_features = self.enc(
                input_ids=enc_ids, attention_mask=enc_mask).last_hidden_state
        return enc_features

    def _average_embed(self, hidden_states:torch.tensor, attention_mask:torch.tensor):
        """
        hidden_states.shape: (bs, sent_len, hidden_size)
        attention_mask: (bs, sent_len) (1: attent, 0: pad)
        """
        return ((hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))

    def _compute_cl_loss(self, embed_1, attn_mask_1, embed_2, attn_mask_2):
        # shape: [batch_size, bottleneck_dim]
        z1 = self._average_embed(embed_1, attn_mask_1).squeeze()
        z2 = self._average_embed(embed_2, attn_mask_2).squeeze()
        
        # Gather all embeddings if using distributed training
        if dist.is_initialized():
            # Dummy vectors for allgather
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            # Allgather
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            # Get full batch embeddings: (bs x N, hidden)
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)

        # Calculate the similariy of z1.unsqueeze(1) (bs, 1, hidden) and z2.unsqueeze(0) (1, bs, hidden)
        # Output: cos_sim (bs, bs)
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))

        labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)

        return self.cl_loss_fct(cos_sim, labels)

    def _compute_mse_loss(self, embed_1, attn_mask_1, embed_2, attn_mask_2):
        # shape: [batch_size, dim_enc]
        z1 = self._average_embed(embed_1, attn_mask_1).squeeze()
        z2 = self._average_embed(embed_2, attn_mask_2).squeeze()

        return self.mse_loss_fct(z1, z2)

    def _get_language_agnostic_repr(self, features, lang_list):
        if lang_list is not None:
            # Check if all elements in lang_list are the same
            if all(lang == lang_list[0] for lang in lang_list):
                # If all languages are the same, apply aligner_func once
                lang_agnostic_enc_features = self.aligner_func(features, self.language_mapping[lang_list[0].item()])
            else:
                # Handle different languages separately
                # Ensure features and aligned features are concatenated along the batch dimension
                aligned_features = []
                for lang, feature in zip(lang_list, features):
                    aligned_feature = self.aligner_func(feature, self.language_mapping[lang.item()])
                    aligned_features.append(aligned_feature)
                # Concatenate along the batch dimension, ensuring the final tensor matches features' shape
                lang_agnostic_enc_features = torch.stack(aligned_features, dim=0)
            
                # Ensure the returned tensor matches the dtype and device of the original features tensor
                lang_agnostic_enc_features = lang_agnostic_enc_features.to(features)
            
            return lang_agnostic_enc_features
        else:
            return self.aligner_func(features, 'en')

    def _find_special_token_positions(self, llm_input_ids, special_token_id=32000):
        """
        Find the positions of the special token in llm_input_ids.
        Args:
            llm_input_ids (torch.Tensor): Tensor of shape [bsz_size, seq_len_2]
            special_token_id (int): ID of the special token (default is 32000)
        
        Returns:
            torch.Tensor: A boolean mask indicating positions of the special token.
        """
        return (llm_input_ids == special_token_id).nonzero(as_tuple=True)[1]

    def _insert_enc_features_into_llm_embeddings(self, llm_embeddings, enc_features, special_token_positions, start_boundary, end_boundary):
        """
        Insert enc_features into llm_embeddings at positions where special_token_positions is True.
        
        Args:
            llm_embeddings (torch.Tensor): Tensor of shape [bsz_size, seq_len_2, hidden_dim]
            enc_features (torch.Tensor): Tensor of shape [bsz_size, seq_len_1, hidden_dim]
            special_token_positions (torch.Tensor): A boolean mask indicating special token positions.
        
        Returns:
            torch.Tensor: Updated embeddings with enc_features inserted.
        """
        bsz_size, seq_len_2, hidden_dim = llm_embeddings.shape
        seq_len_1 = enc_features.shape[1]
        new_seq_len = seq_len_2 - 1 + int(start_boundary is not None) + int(end_boundary is not None) + seq_len_1  # adding start_boundary, enc_features, and end_boundary

        # Create an empty tensor with the new shape
        updated_llm_embeddings = torch.zeros((bsz_size, new_seq_len, hidden_dim), dtype=llm_embeddings.dtype, device=llm_embeddings.device)

        add_start = add_end = False
        if start_boundary is not None:
            add_start = True
        if end_boundary is not None:
            add_end = True

        for i, insertion_index in enumerate(special_token_positions):
            # Split the embeddings at the insertion index
            prefix_embeddings = llm_embeddings[i, :insertion_index, :]
            postfix_embeddings = llm_embeddings[i, insertion_index + 1:, :]

            if add_start:
                prefix_embeddings = torch.cat([prefix_embeddings, start_boundary[i]], dim=0)
                
            if add_end:
                postfix_embeddings = torch.cat([end_boundary[i], postfix_embeddings], dim=0)

            # Insert enc_features at the position (the batch dimension should align)
            updated_llm_embeddings[i] = torch.cat(
                [prefix_embeddings, enc_features[i], postfix_embeddings], dim=0
            )

        return updated_llm_embeddings

    def _insert_enc_mask_into_llm_mask(self, llm_mask, enc_mask, special_token_positions, start_boundary, end_boundary):
        """
        Insert enc_mask into llm_mask at positions where special_token_positions is True.
        
        Args:
            llm_mask (torch.Tensor): Tensor of shape [bsz_size, seq_len_2]
            enc_mask (torch.Tensor): Tensor of shape [bsz_size, seq_len_1]
            special_token_positions (torch.Tensor): A boolean mask indicating special token positions.
        
        Returns:
            torch.Tensor: Updated embeddings with enc_mask inserted.
        """
        bsz_size, seq_len_2 = llm_mask.shape
        seq_len_1 = enc_mask.shape[1]
        new_seq_len = seq_len_2 - 1 + int(start_boundary is not None) + int(end_boundary is not None) + seq_len_1  # adding start_boundary, enc_features, and end_boundary

        # Create an empty tensor with the new shape
        updated_llm_mask = torch.zeros((bsz_size, new_seq_len), dtype=llm_mask.dtype, device=llm_mask.device)

        add_start = add_end = False
        if start_boundary is not None:
            add_start = True
        if end_boundary is not None:
            add_end = True

        for i, insertion_index in enumerate(special_token_positions):
            # Split the embeddings at the insertion index
            prefix_mask = llm_mask[i, :insertion_index]
            postfix_mask = llm_mask[i, insertion_index + 1:]
            
            if add_start:
                prefix_mask = torch.cat([prefix_mask, torch.ones((1), device=llm_mask.device, dtype=llm_mask.dtype)], dim=0)
            
            if add_end:
                postfix_mask = torch.cat([torch.ones((1), device=llm_mask.device, dtype=llm_mask.dtype), postfix_mask], dim=0)
            
            # Insert enc_mask at the position (the batch dimension should align)
            updated_llm_mask[i] = torch.cat(
                [prefix_mask, enc_mask[i], postfix_mask], dim=0
            )

        return updated_llm_mask

    # TODO: can be optimized via vectorization process
    def _move_padding_to_end_and_pad_to_multiple_8(self, input_embeddings, attention_mask):
        """
        将 input_embeddings 中的 padding 移动到每个样本的最长有效长度之后，更新 attention_mask。
        
        :param input_embeddings: 输入的 embedding，形状为 [batch_size, seq_len, hidden_size]
        :param attention_mask: 对应的 attention_mask，形状为 [batch_size, seq_len]
        :return: 移动后的 input_embeddings 和更新后的 attention_mask
        """
        batch_size, seq_len, _ = input_embeddings.size()
        
        # 计算每个样本的有效长度，即非padding的部分数量
        valid_lengths = attention_mask.sum(dim=1)  # [batch_size]，每个样本的有效长度
        
        # 创建一个新的 attention_mask
        new_attention_mask = attention_mask.clone()

        # 处理每个样本
        for i, valid_len in enumerate(valid_lengths):
            valid_len = valid_len.item()  # 当前样本的有效长度
            non_padding_indices = (attention_mask[i] == 1).nonzero(as_tuple=True)[0]  # 非padding部分
            padding_indices = (attention_mask[i] == 0).nonzero(as_tuple=True)[0]  # padding部分
            
            # 更新 input_embeddings：将非-padding部分移到前面，padding部分移到后面
            new_input_embeddings_i = torch.cat(
                (input_embeddings[i, non_padding_indices], input_embeddings[i, padding_indices]), dim=0
            )
            
            # 更新 attention_mask：非-padding部分为1，padding部分为0
            new_attention_mask_i = torch.cat(
                (torch.ones(valid_len, dtype=torch.long), torch.zeros(seq_len - valid_len, dtype=torch.long)), dim=0
            )

            # 更新 input_embeddings 和 attention_mask
            input_embeddings[i] = new_input_embeddings_i
            new_attention_mask[i] = new_attention_mask_i

        # pad to multiple 8
        # 计算每行的有效长度
        effective_length = new_attention_mask.sum(dim=1)

        # 找到最小的8的倍数
        target_length = (effective_length + 7) // 8 * 8

        # 创建一个新的mask张量，填充为0
        new_mask = torch.zeros(new_attention_mask.shape[0], target_length.max()).to(attention_mask)

        # 填充有效部分
        for i in range(new_attention_mask.shape[0]):
            new_mask[i, :effective_length[i]] = new_attention_mask[i, :effective_length[i]]
        
        # 创建一个新的input embedding张量，填充为0
        new_input_embeddings = torch.zeros(input_embeddings.shape[0], target_length.max(), input_embeddings.shape[-1], dtype=input_embeddings.dtype)
        # 填充有效部分
        for i in range(input_embeddings.shape[0]):
            new_input_embeddings[i, :effective_length[i], :] = input_embeddings[i, :effective_length[i], :]

        new_input_embeddings = new_input_embeddings.to(input_embeddings)

        return new_input_embeddings, new_mask

    def _insert_enc_embeddings(self, special_token_positions, enc_features, enc_mask, llm_embeddings, llm_mask, start_boundary, end_boundary):
        """
        Replace the embeddings and mask by inserting enc_features and enc_mask into the respective positions
        in llm_embeddings and llm_mask at positions marked by the special token id 32000.
        """
        # Step 2: Insert enc_features into llm_embeddings at the special token positions
        combined_embeddings = self._insert_enc_features_into_llm_embeddings(llm_embeddings, enc_features, special_token_positions, start_boundary, end_boundary)

        # Step 3: Insert enc_mask into llm_mask at the special token positions
        combined_mask = self._insert_enc_mask_into_llm_mask(llm_mask, enc_mask, special_token_positions, start_boundary, end_boundary)

        return combined_embeddings, combined_mask

    # TODO: can be optimized via vectorization process
    def _construct_labels(self, labels, total_seq_len, instruction_length):
        full_labels = []

        for i, label in enumerate(labels):
            # Create no-loss labels
            no_loss_label = torch.full(
                (instruction_length[i],), device=labels.device, dtype=labels.dtype, fill_value=-100)
            
            # Concatenate the no-loss label with the actual label
            # find the ending of the effective label index 
            last_index = (label == 2).nonzero(as_tuple=True)[0][-1].item()
            full_label = torch.cat([no_loss_label, label[:last_index + 1]], dim=0)
            # Pad to total_seq_len if needed
            if full_label.shape[0] < total_seq_len:
                padding = torch.full(
                    (total_seq_len - full_label.shape[0],), device=labels.device, dtype=labels.dtype, fill_value=-100)
                full_label = torch.cat([full_label, padding], dim=0)
            
            full_labels.append(full_label)

        # Stack full_labels and ensure they have the correct shape: (batch_size, total_seq_len)
        full_labels = torch.stack(full_labels).to(labels)  # Ensure it is on the correct device

        return full_labels

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
        ins_input_ids: torch.Tensor | None = None,
        ins_attention_mask: torch.Tensor | None = None,
        use_cache: bool = True,
        past_key_values: tuple | None = None,
        return_dict: bool = True,
        labels: torch.Tensor | None = None,
        loss_reduction: str = 'mean',
        src_lang: torch.Tensor | None = None,
        tgt_lang: torch.Tensor | None = None,
        **kwargs
    ) -> CausalLMOutputWithPast:
        # find the input shape
        batch_size = input_ids.shape[0]
        # llm_past_key_values = past_key_values[0] if past_key_values is not None else None
        
        if past_key_values is None:
            # get start and end embedding bouding the enc input features
            start_boundary, end_boundary = self.mapping.get_embed()
            start_boundary = start_boundary.expand([batch_size, 1, start_boundary.size(-1)])
            end_boundary = end_boundary.expand([batch_size, 1, end_boundary.size(-1)])

            # Step 1: Identify positions of special token (32000)
            special_token_positions = self._find_special_token_positions(ins_input_ids, special_token_id=32000)
            # replace special token prevent index error
            for i, pos in enumerate(special_token_positions):
                ins_input_ids[i, pos] = 2

            # Instruction Embeddings
            llm_ins_embeddings = self.embeddings(ins_input_ids)
            
            # mT5 forward input sentence
            # [bsz, seq_len_enc, hidden_dim_enc]
            enc_features = self.get_encoder_features(enc_ids, enc_mask)
            enc_features = self.la_adapter(enc_features)
            # using LIR to proj to language agnostic component
            # lang_agnostic_enc_features = self._get_language_agnostic_repr(enc_features, src_lang)
            # proj to llm dim
            # [bsz, seq_len_enc, hidden_dim_llm]
            enc_features = self.mapping(enc_features)

            # Insert input enc features into the llm instruction embeddings
            combined_llm_enc_embeddings, combined_llm_enc_attention_mask = self._insert_enc_embeddings(
                special_token_positions, enc_features, enc_mask, llm_ins_embeddings, ins_attention_mask, start_boundary, end_boundary
            )
            
            if self.training:
                # squeeze padding because of the enc padding
                squeeze_combined_llm_enc_embeddings, squeeze_llm_enc_attention_mask = self._move_padding_to_end_and_pad_to_multiple_8(combined_llm_enc_embeddings, combined_llm_enc_attention_mask)
                # if training concat the continuation
                continuation_embeddings = self.embeddings(input_ids)
                inputs_embeds = torch.cat([squeeze_combined_llm_enc_embeddings, continuation_embeddings], dim=1)
                attention_mask = torch.cat([squeeze_llm_enc_attention_mask, attention_mask], dim=1)
                # squeeze padding because of the instruction padding
                inputs_embeds, attention_mask = self._move_padding_to_end_and_pad_to_multiple_8(inputs_embeds, attention_mask)
            else:
                # first time generate
                inputs_embeds = combined_llm_enc_embeddings
                attention_mask = combined_llm_enc_attention_mask
                # past_instruction_len = combined_llm_enc_attention_mask.shape[1]
                # print(f"[DEBUG] past_instruction_len: {past_instruction_len}")
        else:
            # autoaggressive generate (only support for batch size 1)
            inputs_embeds = self.embeddings(input_ids)
            # past_instruction_len = past_key_values[1]
            # squeeze_llm_enc_attention_mask = torch.ones((batch_size, past_instruction_len)).to(attention_mask)
            # attention_mask = torch.cat([squeeze_llm_enc_attention_mask, attention_mask], dim=1)

        # pass through LM
        out: BaseModelOutputWithPast = self.lm(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=True,
            **kwargs
        )

        logits: torch.Tensor = self.lm_head(out.last_hidden_state)

        loss = None
        if labels is not None:
            # Get total sequence length
            total_seq_len = inputs_embeds.shape[1]
            # record the instruction length
            instruction_length = squeeze_llm_enc_attention_mask.sum(dim=1)
            
            full_labels = self._construct_labels(labels, total_seq_len, instruction_length)
            
            # logits shape (batch, seq_length, #words)
            shift_logits = logits[..., :-1, :].contiguous()
            # labels shape (batch, seq_length)
            shift_labels = full_labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction=loss_reduction)

        # print(f"[DEBUG] lm_loss: {loss}")

        # contrastive learning to improve semantic alignment
        # if self.training and src_enc_ids is not None and tgt_enc_ids is not None:
        #     src_features = self.get_encoder_features(src_enc_ids, src_enc_mask)
        #     lang_agnostic_src_features = self.la_adapter(src_features)
        #     # lang_agnostic_src_features = self.la_adapter(self._get_language_agnostic_repr(src_features, src_lang))
            
        #     tgt_features = self.get_encoder_features(tgt_enc_ids, tgt_enc_mask)
        #     lang_agnostic_tgt_features = self.la_adapter(tgt_features)
        #     # lang_agnostic_tgt_features = self.la_adapter(self._get_language_agnostic_repr(tgt_features, tgt_lang))
            
        #     cl_loss = self._compute_cl_loss(lang_agnostic_src_features, src_enc_mask, lang_agnostic_tgt_features, tgt_enc_mask)
        #     loss += self.coefficient_cl_loss * cl_loss
        # print(f"[DEBUG] cl_loss: {cl_loss}")
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=out.past_key_values if use_cache else None,
            hidden_states=out.hidden_states,
            attentions=out.attentions,
        )


# used for debbuging with opt-125m
class LBOPT(LBBaseModel):
    config: LangBridgeConfig

    def __init__(self, config: LangBridgeConfig, random_init=True):
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
            f"specified {self.config.dim_lm} in LangBridgeConfig, but {config.lm} has hidden size={base_lm.config.hidden_size}"

        self.lm: OPTModel = base_lm.model
        self.lm_head = base_lm.lm_head
        self.embeddings = base_lm.get_input_embeddings()


class LBLlama(LBBaseModel):
    config: LangBridgeConfig

    def __init__(self, config: LangBridgeConfig, random_init=True):
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
            f"specified {self.config.dim_lm} in LangBridgeConfig, but {config.lm} has hidden size={base_lm.config.hidden_size}"

        self.lm: LlamaModel = base_lm.model
        self.lm_head = base_lm.lm_head
        self.embeddings = base_lm.get_input_embeddings()


class LBMistral(LBBaseModel):
    config: LangBridgeConfig

    def __init__(self, config: LangBridgeConfig, random_init=True):
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
            f"specified {self.config.dim_lm} in LangBridgeConfig, but {config.lm} has hidden size={base_lm.config.hidden_size}"

        self.lm: MistralModel = base_lm.model
        self.lm_head = base_lm.lm_head
        self.embeddings = base_lm.get_input_embeddings()


class LangBridgeModel(PreTrainedModel):
    config: LangBridgeConfig
    config_class = LangBridgeConfig

    _LANGUAGE_MODEL_VERSIONS = {
        'facebook/opt': LBOPT,
        'EleutherAI/llemma': LBLlama,
        'codellama/CodeLlama': LBLlama,
        'microsoft/Orca-2': LBLlama,
        'meta-math/MetaMath': LBLlama,
        'meta-llama/Llama-2-7b-hf': LBLlama,
        '/home/export/base/ycsc_chenkh/hitici_02/online1/data/pretrained-models/Llama-2-7b': LBLlama,
        '/home/export/base/ycsc_chenkh/hitici_02/online1/data/pretrained-models/MetaMath-7B-V1.0': LBLlama,
        'mistralai/Mistral-7B-v0.1': LBMistral,
    }

    def __init__(self, config: LangBridgeConfig, random_init=True, model_class=None):
        super().__init__(config)

        if model_class is None:
            model_class = self._find_lm_class(config.lm)
        self.lb: LBBaseModel = model_class(config, random_init=random_init)

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
        self.lb.freeze_encoder()

    def freeze_lm(self):
        self.lb.freeze_lm()

    def unfreeze_lm(self):
        self.lb.unfreeze_lm()
        
    def freeze_alignment(self):
        self.lb.freeze_alignment()
        
    # def get_input_embeddings(self):
    #     return self.lb.lm.model.embed_tokens

    # def set_input_embeddings(self, value):
    #     self.lb.lm.model.embed_tokens = value

    # def get_output_embeddings(self):
    #     return self.lb.lm.lm_head

    # def set_output_embeddings(self, new_embeddings):
    #     self.lb.lm.lm_head = new_embeddings

    def forward(
        self,
        enc_ids: torch.Tensor | None = None,
        enc_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        ins_input_ids: torch.Tensor | None = None,
        ins_attention_mask: torch.Tensor | None = None,
        use_cache: bool = True,
        past_key_values: tuple | None = None,
        return_dict: bool = True,
        labels: torch.Tensor | None = None,
        loss_reduction: str = 'mean',
        src_lang: torch.Tensor | None = None,
        tgt_lang: torch.Tensor | None = None,
        src_enc_ids: torch.Tensor | None = None,
        src_enc_mask: torch.Tensor | None = None,
        tgt_enc_ids: torch.Tensor | None = None,
        tgt_enc_mask: torch.Tensor | None = None,
        **kwargs
    ) -> CausalLMOutputWithPast:

        return self.lb(
            enc_ids=enc_ids,
            enc_mask=enc_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            ins_input_ids=ins_input_ids,
            ins_attention_mask=ins_attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            labels=labels,
            loss_reduction=loss_reduction,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            src_enc_ids=src_enc_ids,
            src_enc_mask=src_enc_mask,
            tgt_enc_ids=tgt_enc_ids,
            tgt_enc_mask=tgt_enc_mask,
            **kwargs
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        enc_ids: torch.Tensor | None = None,
        enc_mask: torch.Tensor | None = None,
        ins_input_ids: torch.Tensor | None = None,
        ins_attention_mask: torch.Tensor | None = None,
        src_lang: torch.Tensor | None = None,
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
            ins_input_ids=ins_input_ids,
            ins_attention_mask=ins_attention_mask,
            src_lang=src_lang,
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