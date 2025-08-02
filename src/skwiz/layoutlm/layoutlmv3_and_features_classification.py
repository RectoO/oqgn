from copy import deepcopy
from typing import Dict
import torch
from transformers import LayoutLMv3Config, LayoutLMv3Model  # type: ignore[import-untyped]

from src.types.training import TrainingConfig
from src.skwiz.layoutlm.default_layoutlm_config import DEFAULT_LAYOUTLM_CONFIG
from src.skwiz.layoutlm.layoutlmv3_classification_head import (
    LayoutLMv3ClassificationHead,
)
from src.skwiz.layoutlm.layoutlmv3_and_features_head_classification import (
    LayoutLMv3AndFeaturesHeadClassification,
)


class LayoutLMv3AndFeaturesClassification(torch.nn.Module):
    def __init__(
        self,
        config: TrainingConfig,
        extraction_label2id: Dict[str, Dict[str, int]],
        classification_label2id: Dict[str, Dict[str, int]],
    ):
        super().__init__()
        layoutlm_base_config = deepcopy(
            {**DEFAULT_LAYOUTLM_CONFIG, **(config.get("layoutlm", {}).get('architectureConfig', {}) or {})}
        )

        layoutlm_config = LayoutLMv3Config(**layoutlm_base_config)
        self.layoutlmv3 = LayoutLMv3Model(layoutlm_config)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.ModuleDict()
        self.num_labels = {}

        for key, labels in extraction_label2id.items():
            self.num_labels[key] = len(labels)
            ocr_tags = config.get('tagging', {}).get('ocrTags', [])
            self.classifier[key] = LayoutLMv3AndFeaturesHeadClassification(
                config=layoutlm_config,
                num_labels=self.num_labels[key],
                n_tags=len(ocr_tags),
            )

        self.doc_classifier = torch.nn.ModuleDict()
        self.doc_num_labels = {}
        for key, labels in classification_label2id.items():
            self.doc_num_labels[key] = len(labels)
            self.doc_classifier[key] = LayoutLMv3ClassificationHead(
                layoutlm_config, self.doc_num_labels[key]
            )

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        tags=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pixel_values=None,
    ):
        self.eval()
        outputs = self.layoutlmv3(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
        )

        # 0 is the index of the observation (size of index 0 is batch_size)
        # 1 is the number of tokens
        number_of_tokens = input_ids.size()[1]

        # We take the last hidden layer of layoutlmv3
        # the sequence length of hidden layer is: number of text token + number of image token + 1 (cls)
        layoutlmv3_output = outputs[0][:, :number_of_tokens]

        layoutlmv3_cls_output = outputs[0][:, 0, :]
        logits = {}
        doc_logits = {}

        for key, module in self.classifier.items():
            dropped_out_layoutlmv3_output = self.dropout(layoutlmv3_output)
            logits[key] = module(dropped_out_layoutlmv3_output, tags)

        for key, module in self.doc_classifier.items():
            dropped_out_cls_layoutlmv3_output = self.dropout(layoutlmv3_cls_output)
            doc_logits[key] = module(dropped_out_cls_layoutlmv3_output)

        return (
            {key: logit for key, logit in logits.items()},
            {key: logit for key, logit in doc_logits.items()},
            outputs[0]
        )
