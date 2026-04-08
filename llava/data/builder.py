import os
from typing import Any, Optional

from hydra.utils import instantiate
from torch.utils.data import ConcatDataset, Dataset
from transformers import PreTrainedTokenizer

from llava.data.datasets_mixture import DATASETS_LEGACY
from llava.train.args import DataArguments, TrainingArguments
from llava.utils import io
from llava.utils.logging import logger

__all__ = ["DATASETS", "register_datasets", "build_dataset"]


def register_datasets(name: Optional[str] = None):
    print(f"llava/data/builder.py Registering datasets with name: {name}")
    if name is None:
        name = os.environ.get("VILA_DATASETS", "default")
        logger.info(f"Registering datasets from `{name}`.")
    return io.load(os.path.join(os.path.dirname(__file__), "registry", f"{name}.yaml"))


DATASETS = register_datasets()


class RepeatedDataset(Dataset):
    def __init__(self, dataset: Dataset, times: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.times = times

    def __len__(self) -> int:
        return len(self.dataset) * self.times

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index % len(self.dataset)]


def build_dataset(
    mixture: str,
    data_args: DataArguments,
    training_args: TrainingArguments,
    tokenizer: PreTrainedTokenizer,
) -> Dataset:
    datasets = []
    print(f"llava/data/builder.py Building dataset from mixture: {mixture}")
    for name in mixture.strip().lower().split("+"):
        if "*" in name:
            name, times = name.split("*")
            times = int(times)
        else:
            times = 1
        if name in DATASETS_LEGACY:  # dataset_name in registered dataset
            logger.warning(f"Dataset {name} is registered under legacy mode.")
            dataset = build_dataset_legacy(
                name,  # r2r airvln_merge_triple_updated
                data_args=data_args,
                training_args=training_args,
                tokenizer=tokenizer,
            )
        else:
            raise ValueError(f"Dataset {name} is not registered.")

        if times > 1:  # weight add frequency
            dataset = RepeatedDataset(dataset, times)
        datasets.append(dataset)
    return ConcatDataset(datasets)


def build_dataset_legacy(
    name: str,
    data_args: DataArguments,
    training_args: TrainingArguments,
    tokenizer: PreTrainedTokenizer,
) -> Dataset:
    from llava.data.dataset import (
        DummyDataset,
        LazyEnvDropDataset,
        LazySupervisedDataset,
        LazyVLNCEDataset,
        LazyUpdateTripleMergeAirVLNDataset,
        LazyUpdateTripleMergeAirVLNSubTrajectorySummaryDataset,
        LazyOpen3DVQADataset,
        
    )

    dataset = DATASETS_LEGACY[name]  # refer to datasets_mixture Line99
    dataset_type = dataset.dataset_type
    if dataset_type == "torch":
        dataset_cls = LazySupervisedDataset  # for vqa dataset "scanqa" "sharegpt4v_sft"
    elif dataset_type == "envdrop":
        dataset_cls = LazyEnvDropDataset
    elif dataset_type == "vlnce":  # for navigation dataset
        dataset_cls = LazyVLNCEDataset
    elif dataset_type == "UpdateTripleMergeAirVLN":
        dataset_cls = LazyUpdateTripleMergeAirVLNDataset
    elif dataset_type == "UpdateTripleMergeAirVLNSubTrajectorySummary":
        dataset_cls = LazyUpdateTripleMergeAirVLNSubTrajectorySummaryDataset #airvln_subtraj_sum_updated
    elif dataset_type == 'Open3DVQA': # Open3DVQA_EmbodiedCity_Wuhan / Open3DVQA_UrbanScene_Campus / Open3DVQA_UrbanScene_Residence
        dataset_cls = LazyOpen3DVQADataset
    else:
        raise NotImplementedError(f"{dataset_type} is not supported.")

    data_args.meta_path = getattr(dataset, "meta_path", None)  # None
    data_args.caption_choice = getattr(dataset, "caption_choice", None)  # None
    data_args.caption_choice_2 = getattr(dataset, "caption_choice_2", None)  # None
    data_args.start_idx = getattr(dataset, "start_idx", None)  # -1
    data_args.end_idx = getattr(dataset, "end_idx", None)  # -1

    return dataset_cls(
        tokenizer=tokenizer,
        data_path=dataset.data_path,
        image_folder=getattr(dataset, "image_path"),
        data_args=data_args,
        training_args=training_args,
    )
