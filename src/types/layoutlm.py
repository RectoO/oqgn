from typing import TypedDict, List, Dict
import torch


class RawProcessedLayoutLmInput(TypedDict):
    input_ids: List[int]
    attention_mask: List[int]
    bbox: List[List[int]]
    labels: List[List[int]] | None
    tags: List[List[int]] | None
    pixel_values: List[List[List[List[int]]]]


class ProcessedLayoutLMInputWithoutClassesRequired(TypedDict):
    input_ids: List[int]
    attention_mask: List[int]
    bbox: List[List[int]]
    labels: List[List[int]] | None
    tags: List[List[int]] | None


class ProcessedLayoutLMInputWithoutClassesNotRequired(TypedDict, total=False):
    pixel_values: List[List[List[int]]]


class ProcessedLayoutLMInputWithoutClasses(
    ProcessedLayoutLMInputWithoutClassesRequired,
    ProcessedLayoutLMInputWithoutClassesNotRequired,
):
    pass


class ProcessedLayoutLMInput(ProcessedLayoutLMInputWithoutClasses):
    class_labels: List[int] | None


class ProcessedLayoutLMInputWithTaskMeta(ProcessedLayoutLMInput):
    task_id: str
    page_index: int
    window_index: int


class TorchProcessedLayoutLMInput(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    bbox: torch.Tensor
    pixel_values: torch.Tensor
    labels: torch.Tensor
    tags: torch.Tensor
    class_labels: torch.Tensor


class ProcessedLineSplitterInput(TypedDict):
    bb_groups: Dict[str, List[str]]
    attention_mask: List[int]
    bbox: List[List[int]]
    pixel_values: List[List[List[int]]]
    labels: List[List[int]] | None
    tags: List[List[int]] | None
    class_labels: List[int] | None
