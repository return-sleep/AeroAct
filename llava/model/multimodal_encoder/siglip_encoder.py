# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0

import torch

from torch import nn
from transformers import PretrainedConfig, SiglipImageProcessor, SiglipVisionModel
import torch.distributed as dist
from llava.model.multimodal_encoder.vision_encoder import VisionTower, VisionTowerS2

from transformers.models.siglip import modeling_siglip as ms

def _safe_init_weights(self, module):
    if isinstance(module, nn.Linear):
        if module.weight is not None:
            if module.weight.ndim >= 2:
                nn.init.xavier_uniform_(module.weight)
            else:
                nn.init.ones_(module.weight)  # 
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        if module.weight is not None and module.weight.ndim >= 2:
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


ms.SiglipVisionModel._init_weights = _safe_init_weights

class SiglipVisionTower(VisionTower):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig, state_dict=None):
        super().__init__(model_name_or_path, config)
        # print('siglip vision tower init image_processor', model_name_or_path, flush=True)
        self.image_processor = SiglipImageProcessor.from_pretrained(model_name_or_path)
        # print('siglip vision tower init vision_tower', model_name_or_path, flush=True)
        
        # self.vision_tower = SiglipVisionModel.from_pretrained(
        #     # TODO(ligeng): why pass config here leading to errors?
        #     model_name_or_path,
        #     torch_dtype=eval(config.model_dtype),
        #     state_dict=state_dict,
        # )
        import deepspeed,os

        dtype = eval(config.model_dtype)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        with deepspeed.zero.Init(enabled=False):
            self.vision_tower = SiglipVisionModel.from_pretrained(
                model_name_or_path,
                torch_dtype=dtype,
            )

        if state_dict is not None:
            self.vision_tower.load_state_dict(state_dict, strict=False)

        self.vision_tower.to(device=f"cuda:{local_rank}", dtype=dtype)
            
        # print('siglip vision tower init vision_tower done', model_name_or_path, flush=True)
        self.is_loaded = True


class SiglipVisionTowerS2(VisionTowerS2):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig):
        super().__init__(model_name_or_path, config)
        
        self.image_processor = SiglipImageProcessor.from_pretrained(model_name_or_path)
        
        self.vision_tower = SiglipVisionModel.from_pretrained(model_name_or_path, torch_dtype=eval(config.model_dtype))
        
        # Make sure it crops/resizes the image to the largest scale in self.scales to maintain high-res information
        self.image_processor.size["height"] = self.image_processor.size["width"] = self.scales[-1]

        self.is_loaded = True
