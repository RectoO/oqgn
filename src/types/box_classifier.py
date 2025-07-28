from typing import TypedDict, List
import torch


class ProcessedBoxClassifierInput(TypedDict):
    layoutlm_vectors: List[List[float]]
    tags: List[List[int]]
    bbox_ids: List[str]
    page_index: int


class TrainingProcessedBoxClassifierInput(ProcessedBoxClassifierInput, total=False):
    labels: List[int]
    task_id: str


class TorchProcessedBoxClassifierInput(TypedDict):
    layoutlm_vectors: torch.Tensor
    tags: torch.Tensor
    bbox_ids: List[str]


class TorchTrainingProcessedBoxClassifierInput(TorchProcessedBoxClassifierInput, total=False):
    labels: torch.Tensor
