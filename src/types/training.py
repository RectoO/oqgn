from typing import TypedDict, List, Optional, Dict, Literal
from enum import Enum


class TrainingType(Enum):
    EXTRACTION = "extraction"
    CLASSIFICATION = "classification"


class SaveStrategy(Enum):
    NO = "no"
    EPOCH = "epoch"
    STEPS = "steps"


class LRSchedulerType(Enum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    INVERSE_SQRT = "inverse_sqrt"
    REDUCE_LR_ON_PLATEAU = "reduce_lr_on_plateau"


class EarlyStoppingConfig(TypedDict, total=False):
    patience: Optional[int]
    warmup: Optional[int]


class CallbacksConfig(TypedDict, total=False):
    early_stopping: Optional[EarlyStoppingConfig]


class LayoutLMTrainingConfigSchema(TypedDict, total=False):
    batch_size: Optional[int]
    epochs: Optional[int]
    lr: Optional[float]
    eval_strategy: Optional[str]
    eval_steps: Optional[int]
    load_best_model_at_end: Optional[bool]
    save_strategy: SaveStrategy
    save_total_limit: Optional[int]
    metric_for_best_model: Optional[str]
    greater_is_better: Optional[bool]
    lr_scheduler_type: LRSchedulerType
    callbacks: Optional[CallbacksConfig]


class LayoutLMv3Schema(TypedDict, total=False):
    vocab_size: Optional[int]
    hidden_size: Optional[int]
    num_hidden_layers: Optional[int]
    num_attention_heads: Optional[int]
    intermediate_size: Optional[int]
    hidden_act: Optional[str]
    hidden_dropout_prob: Optional[float]
    attention_probs_dropout_prob: Optional[float]
    max_position_embeddings: Optional[int]
    type_vocab_size: Optional[int]
    initializer_range: Optional[float]
    layer_norm_eps: Optional[float]
    max_2d_position_embeddings: Optional[int]
    coordinate_size: Optional[int]
    shape_size: Optional[int]
    has_relative_attention_bias: Optional[bool]
    rel_pos_bins: Optional[int]
    max_rel_pos: Optional[int]
    max_rel_2d_pos: Optional[int]
    rel_2d_pos_bins: Optional[int]
    has_spatial_attention_bias: Optional[bool]
    visual_embed: Optional[bool]
    input_size: Optional[int]
    num_channels: Optional[int]
    patch_size: Optional[int]
    classifier_dropout: Optional[float]


DatasetType = Literal["train", "val", "test"]


class TrainingConfig(TypedDict):
    type: TrainingType
    taskIds: Dict[DatasetType, List[str] | Dict[str, List[str]]]
    featureEngineering: List[str]
    initialModelPath: str | None
    layoutlmv3: LayoutLMv3Schema
    train: LayoutLMTrainingConfigSchema
