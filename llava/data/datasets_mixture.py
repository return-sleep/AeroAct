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

import warnings
from dataclasses import dataclass, field


@dataclass
class Dataset:
    dataset_name: str
    dataset_type: str = field(default="torch")
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    meta_path: str = field(default=None, metadata={"help": "Path to the meta data for webdataset."})
    image_path: str = field(default=None, metadata={"help": "Path to the training image data."})
    caption_choice: str = field(default=None, metadata={"help": "Path to the caption directory for recaption."})
    description: str = field(
        default=None,
        metadata={
            "help": "Detailed desciption of where the data is from, how it is labelled, intended use case and the size of the dataset."
        },
    )
    test_script: str = (None,)
    maintainer: str = (None,)
    ############## ############## ############## ############## ############## ##############
    caption_choice: str = field(default=None, metadata={"help": "Path to the captions for webdataset."})
    caption_choice_2: str = field(default=None, metadata={"help": "Path to the captions for webdataset."})
    start_idx: float = field(default=-1, metadata={"help": "Start index of the dataset."})
    end_idx: float = field(default=-1, metadata={"help": "Start index of the dataset."})


DATASETS_LEGACY = {}


def add_dataset(dataset):
    if dataset.dataset_name in DATASETS_LEGACY:
        # make sure the data_name is unique
        warnings.warn(f"{dataset.dataset_name} already existed in DATASETS. Make sure the name is unique.")
    assert "+" not in dataset.dataset_name, "Dataset name cannot include symbol '+'."
    DATASETS_LEGACY.update({dataset.dataset_name: dataset})


def register_datasets_mixtures():


    # aerial vln data
    airvln = Dataset(
        dataset_name="airvln_merge_triple_updated",
        dataset_type="UpdateTripleMergeAirVLN",
        data_path="Dataset/AerialVLN-Dataset/data/aerialvln-s/train_merged_triple.json",  # annotations info including action and language annatiion
        image_path="Dataset/AerialVLN-Dataset/Raw_data/aerialvln-s",  # path to raw frames
        description="airvln_merge_triple_updated",
    )
    add_dataset(airvln)

    # airvln_subtraj_summary data
    airvln_subtraj_sum_updated = Dataset(
        dataset_name="airvln_subtraj_sum_updated",
        dataset_type="UpdateTripleMergeAirVLNSubTrajectorySummary",
        data_path="Dataset/AerialVLN-Dataset/data/aerialvln-s/subgoal_pointer_list.json",  # annotations info including action and language annatiion
        image_path="Dataset/AerialVLN-Dataset/Raw_data/aerialvln-s",  # path to raw frames
        description="airvln_subtraj_sum_updated",
    )
    add_dataset(airvln_subtraj_sum_updated)
    
    # VQA data
    gqa_spatial = Dataset(
        dataset_name="gqa_spatial",
        dataset_type="torch",
        data_path="Dataset/VQA-Dataset/ShareGPT4V-SFT/sharegpt4v_gqa_spatial_outdoor.json", # add vg & gqa
        image_path="Dataset/VQA-Dataset/ShareGPT4V-SFT",
        description="sharegpt4v_sft spatial subset of gqa.",
    )
    add_dataset(gqa_spatial)
    
    open3dvqa_embodiedcity_wuhan = Dataset(
        dataset_name="open3dvqa_embodiedcity_wuhan",
        dataset_type="Open3DVQA",
        data_path="Dataset/VQA-Dataset/Open3DVQA/O3DVQA/EmbodiedCity/Wuhan/merged_qa.json",
        image_path="Dataset/VQA-Dataset/Open3DVQA/O3DVQA/EmbodiedCity/Wuhan/rgb",
        description="Open3DVQA EmbodiedCity Wuhan training set.",
    )
    add_dataset(open3dvqa_embodiedcity_wuhan)

    open3dvqa_urban_scene_campus = Dataset(
        dataset_name="open3dvqa_urban_scene_campus",
        dataset_type="Open3DVQA",
        data_path="Dataset/VQA-Dataset/Open3DVQA/O3DVQA/UrbanScene/Campus/merged_qa.json",
        image_path="Dataset/VQA-Dataset/Open3DVQA/O3DVQA/UrbanScene/Campus/rgb",
        description="Open3DVQA UrbanScene Campusn training set.",
    )
    add_dataset(open3dvqa_urban_scene_campus)

    open3dvqa_urban_scene_residence = Dataset(
        dataset_name="open3dvqa_urban_scene_residence",
        dataset_type="Open3DVQA",
        data_path="Dataset/VQA-Dataset/Open3DVQA/O3DVQA/UrbanScene/Residence/merged_qa.json",
        image_path="Dataset/VQA-Dataset/Open3DVQA/O3DVQA/UrbanScene/Residence/rgb",
        description="Open3DVQA UrbanScene_Residence training set.",
    )
    add_dataset(open3dvqa_urban_scene_residence)

