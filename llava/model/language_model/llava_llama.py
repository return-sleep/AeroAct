#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# This file is modified from https://github.com/haotian-liu/LLaVA/


import inspect
import os
from typing import List, Optional, Tuple, Union

import torch
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from llava.model.loss import soft_cross_entropy, reweight_cross_entropy

from ...train.utils import calculate_loss_weight
from ..configuration_llava import LlavaConfig
from ..llava_arch import LlavaMetaForCausalLM, LlavaMetaModel


class LlavaLlamaConfig(LlavaConfig):
    model_type = "llava_llama"


## FIXME we will follow the convention to add a new class for CausalLM in the future
class LlavaLlamaModel(LlavaMetaModel, LlavaMetaForCausalLM, PreTrainedModel):
    config_class = LlavaLlamaConfig
    main_input_name = "input_embeds"
    supports_gradient_checkpointing = True

    def __init__(self, config: LlavaLlamaConfig = None, *args, **kwargs) -> None:
        super().__init__(config)
        self.init_vlm(config=config, *args, **kwargs)  # init modules by config and load pretrained weights if needed

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        if hasattr(cls, "load_pretrained"):
            return cls.load_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                use_safetensors=use_safetensors,
                **kwargs,
            )
        return super(LlavaLlamaModel).from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        images: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        sample_weight: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        seqlens_in_batch: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        dpo_forward: bool = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        self.freezed_module_patch()  # freeze module
        # image tokenizer -> insert visual token -> padding to max_len
        if inputs_embeds is None:  # True
            (
                input_ids,  # None
                position_ids,  #  (batch_size, max_len)
                attention_mask,  #  (batch_size, max_len)
                past_key_values,  # same input
                inputs_embeds,  #  (batch_size, max_len, embedding_size)
                labels,  #  (batch_size, max_len)
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images
            )

        support_packing = "seqlens_in_batch" in inspect.signature(self.llm.forward).parameters

        # navigation label cannot use repack, lead to #lables != #sample_weight
        if self.training and support_packing and not dpo_forward and sample_weight is None:
            (
                _,
                new_position_ids,
                new_attention_mask,
                _,
                new_inputs_embeds,
                new_labels,
                sorted_seqlens_in_batch,
            ) = self.repack_multimodal_data(
                input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels
            )
            if sorted_seqlens_in_batch is None:
                sorted_seqlens_in_batch = seqlens_in_batch
            new_input_ids = None
            past_key_values = None
        else:
            new_attention_mask = attention_mask
            new_position_ids = position_ids
            new_inputs_embeds = inputs_embeds
            new_labels = labels
            sorted_seqlens_in_batch = attention_mask.sum(-1).int() # len(context)
            new_input_ids = input_ids

        # type=ModelOutput, outputs["loss"] or outputs.loss or outputs[0]
        # add aux loss, outputs.aux_loss = aux_loss add new key
        if support_packing:
            outputs = self.llm.forward(
                input_ids=new_input_ids,
                attention_mask=new_attention_mask,
                position_ids=new_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=new_inputs_embeds,
                labels=new_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                seqlens_in_batch=sorted_seqlens_in_batch,
            )
        else:
            # loss, logits
            outputs = self.llm.forward(
                input_ids=new_input_ids,
                attention_mask=new_attention_mask,
                position_ids=new_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=new_inputs_embeds,
                labels=new_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        if self.training and self.config.time_token_ids:
            outputs.loss = soft_cross_entropy(
                outputs.logits,
                new_labels,
                soft_tokens=self.config.time_token_ids,  # soft_tokens
                std=self.config.soft_ce_std,
            )

        if self.training and sample_weight is not None and not dpo_forward:  # reweight by sample_weight
            outputs.loss = reweight_cross_entropy(
                outputs.logits,
                new_labels,
                sample_weight,
            )
            
        # Loss rescale for SP & DP loss match
        loss_weight = calculate_loss_weight(new_labels) # reweight by  num_active_elements token of each gpu
        outputs.loss = outputs.loss * loss_weight

        if dpo_forward:
            return outputs.logits, new_labels

        return outputs


AutoConfig.register("llava_llama", LlavaLlamaConfig)
AutoModel.register(LlavaLlamaConfig, LlavaLlamaModel)
