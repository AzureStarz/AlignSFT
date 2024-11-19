# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import os
sys.path.append(os.path.abspath('/home/export/base/ycsc_chenkh/hitici_02/online1/AlignSFT'))
import logging
import string
from typing import TYPE_CHECKING, List, Optional, Dict

import torch
import torch.distributed as dist

from ...data import VEDAlignDataCollatorForSeq2Seq, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.ploting import plot_loss
from ...model.adapter import init_adapter
from ...model.configuration_ved_align import VEDAlignConfig
from ...model.modeling_ved_align import VEDAlignModel
from ...extras.misc import count_parameters
from ..trainer_utils import create_modelcard_and_push
from .trainer import CustomSeq2SeqTrainer

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from transformers.utils import PaddingStrategy

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments

logger = logging.getLogger(__name__)

def _load_model(
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
):
    config = VEDAlignConfig(
        enc=model_args.enc_name_or_path,
        lm=model_args.lm_name_or_path,
        alignments=model_args.alignments,
        dim_enc=model_args.enc_hidden_size,
        dim_lm=model_args.lm_hidden_size,
        freeze_language_model=model_args.freeze_language_model,
        freeze_encoder=model_args.freeze_encoder,
        freeze_alignment=model_args.freeze_alignment,
        bottleneck_num_attention_heads=model_args.bottleneck_num_attention_heads,
        bottleneck_model_dim=model_args.bottleneck_model_dim,
        bottleneck_beta_individual=model_args.bottleneck_beta_individual,
        bottleneck_alpha_aggregate=model_args.bottleneck_alpha_aggregate,
        bottleneck_individual_posterior_kernel=model_args.bottleneck_individual_posterior_kernel,
        bottleneck_loss_weight=model_args.bottleneck_loss_weight,
        bottleneck_use_fina_linear=model_args.bottleneck_use_fina_linear,
        divergence_kernel_individual_posterior_kernel=model_args.divergence_kernel_individual_posterior_kernel,
        divergence_kernel_scaler=model_args.divergence_kernel_scaler,
    )

    model_class = VEDAlignModel
    if model_args.model_name_or_path:
        logger.info('loading from HF checkpoint...')
        model = model_class.from_pretrained(
            model_args.model_name_or_path, config=config)
    else:
        model = model_class(config, random_init=False)

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                param.data = param.data.to(model_args.compute_dtype)

        model.eval()
    else:
        model.train()

    return model

from transformers import AutoTokenizer

def _load_tokenizer(model_args):
    def get_tokenizer(path, use_fast, add_eos_token=False):
        try:
            return AutoTokenizer.from_pretrained(path, use_fast=use_fast, add_eos_token=add_eos_token)
        except:
            return AutoTokenizer.from_pretrained(path, add_eos_token=add_eos_token)

    encoder_tokenizer_path = f'{model_args.model_name_or_path}/encoder_tokenizer' if model_args.model_name_or_path else model_args.enc_name_or_path
    lm_tokenizer_path = model_args.model_name_or_path or model_args.lm_name_or_path

    encoder_tokenizer = get_tokenizer(encoder_tokenizer_path, use_fast=True)
    lm_tokenizer = get_tokenizer(lm_tokenizer_path, use_fast=False, add_eos_token=True)

    lm_tokenizer.padding_side = 'right'

    if not encoder_tokenizer.pad_token:
        encoder_tokenizer.pad_token = encoder_tokenizer.eos_token
    if not lm_tokenizer.pad_token:
        lm_tokenizer.pad_token = lm_tokenizer.eos_token

    return {'tokenizer': lm_tokenizer, 'encoder_tokenizer': encoder_tokenizer, "processor": None}

def run_ved_align(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    logging.basicConfig(
        format=f'%(asctime)s {model_args.exp_run_name} %(message)s',
        datefmt='%H:%M:%S',
        force=True,
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
        ]
    )

    # Load Tokenzier
    logger.info('loading tokenizer...')
    tokenizer_module = _load_tokenizer(model_args)
    
    # Load Model
    logger.info('loading model...')
    model = _load_model(model_args, finetuning_args, training_args.do_train)

    # this is true for all our experiments, explained in section D.1
    if model_args.add_new_lines_to_enc:
        logger.info('Adding whitespaces to encoder tokenizer')
        whitespace = list(string.whitespace)[1:]  # exclude single space
        whitespace = whitespace + ['  ', '   ', '    ']  # add multispace
        tokenizer_module['encoder_tokenizer'].add_special_tokens(
            {'additional_special_tokens': whitespace})

        if model_args.freeze_encoder:
            model.ved.enc.get_input_embeddings().weight.requires_grad = True
            logger.warning(
                'Unfreezing encoder embedding layer since new tokens were added')
    
    template = get_template_and_fix_tokenizer(tokenizer_module['tokenizer'], data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="ved_align", **tokenizer_module)

    data_collator = VEDAlignDataCollatorForSeq2Seq(
        template=template,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        max_length=data_args.cutoff_len,
        max_length_enc=data_args.max_length_enc,
        padding=PaddingStrategy.LONGEST,
        **tokenizer_module,
    )

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
    )

    # Log model information
    trainable_params, all_param = count_parameters(model)
    if training_args.do_train:
        param_stats = "trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    else:
        param_stats = "all params: {:,}".format(all_param)

    logger.info(param_stats)

    # Print Trainable Params
    if model_args.print_param_status:
        if dist.get_rank() == 0:
            for name, param in model.named_parameters():
                print(
                    "name: {}, dtype: {}, device: {}, trainable: {}".format(
                        name, param.dtype, param.device, param.requires_grad
                    )
                )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)