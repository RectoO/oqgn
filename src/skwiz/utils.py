from src.types.box_classifier import ProcessedBoxClassifierInput, TorchProcessedBoxClassifierInput
import torch
from typing import List, Any, Tuple, Dict, cast
import numpy as np
from PIL import Image
from transformers import AutoProcessor  # type: ignore[import-untyped]

from src.skwiz.bounding_boxes import normalise_ocr_bounding_box
from src.skwiz.constants import NONE_LABEL
from src.skwiz.layoutlm.layoutlmv3_and_features_classification import (
    LayoutLMv3AndFeaturesClassification,
)
from src.types.training import TrainingConfig
from src.types.ocr import PageInfo, Tags
from src.types.inference import InferenceBoundingBox, InferencePageClassification
from src.types.layoutlm import (
    ProcessedLayoutLMInput,
    TorchProcessedLayoutLMInput,
    RawProcessedLayoutLmInput,
)


def get_processed_input_tags(
    processed_input, bounding_boxes_coordinates, bounding_boxes_tags, number_of_tags
):
    coordinates_to_tags = {
        tuple(coordinates): tags
        for coordinates, tags in zip(bounding_boxes_coordinates, bounding_boxes_tags)
    }
    processed_input_tags = []
    for preprocess_bbox in processed_input["bbox"]:
        if tuple(preprocess_bbox) in coordinates_to_tags:
            processed_input_tags.append(coordinates_to_tags[tuple(preprocess_bbox)])
        else:
            processed_input_tags.append([0] * number_of_tags)
    return processed_input_tags


def get_page_preprocess_input_splited_by_windows(
    image: Image.Image,
    bounding_boxes_text: List[str],
    bounding_boxes_coordinates: List[List[int]],
    bounding_boxes_tags: List[List[int]],
    processor: AutoProcessor,
    training_config: TrainingConfig,
) -> List[ProcessedLayoutLMInput]:
    use_image: bool = training_config.get("layoutlm", {}).get("architectureConfig", {}).get("visual_embed") or False

    window_size = processor.tokenizer.model_max_length
    overlap_ratio = 0.2
    overlap_size = int(window_size * overlap_ratio)

    number_of_tags = len(training_config.get('tagging', {}).get('ocrTags', []))

    processed_input: RawProcessedLayoutLmInput = processor(
        image,
        bounding_boxes_text,
        boxes=bounding_boxes_coordinates,
        word_labels=None,
        padding=False,
    )

    processed_input["tags"] = get_processed_input_tags(
        processed_input=processed_input,
        bounding_boxes_coordinates=bounding_boxes_coordinates,
        bounding_boxes_tags=bounding_boxes_tags,
        number_of_tags=number_of_tags,
    )

    # Remove first and last token (<s> and </s>) before splitting
    processed_input["input_ids"] = processed_input["input_ids"][1:-1]
    processed_input["attention_mask"] = processed_input["attention_mask"][1:-1]
    processed_input["bbox"] = processed_input["bbox"][1:-1]
    processed_input["tags"] = cast(List[List[int]], processed_input["tags"])[1:-1]

    windowed_preprocessed_input = []
    # Split preprocessed input by window max length
    for i in range(
        0, len(processed_input["input_ids"]), (window_size - overlap_size) - 2
    ):
        window_start_index = i
        window_end_index = (i + window_size) - 2

        # 0 is the start (<s>) token that starts all windows and 2 is the end (</s>) token that ends all windows
        windowed_input_ids = (
            [0]
            + processed_input["input_ids"][window_start_index:window_end_index]
            + [2]
        )
        windowed_attention_mask = (
            [1]
            + processed_input["attention_mask"][window_start_index:window_end_index]
            + [1]
        )
        windowed_bbox = (
            [[0, 0, 0, 0]]
            + processed_input["bbox"][window_start_index:window_end_index]
            + [[0, 0, 0, 0]]
        )
        windowed_tags = (
            [[0] * number_of_tags]
            + cast(List[List[int]], processed_input["tags"])[
                window_start_index:window_end_index
            ]
            + [[0] * number_of_tags]
        )

        if len(windowed_input_ids) < window_size:
            size_to_pad = window_size - len(windowed_input_ids)

            # padding for input_ids is done with 1 (<pad> token)
            windowed_input_ids = windowed_input_ids + [1 for _ in range(size_to_pad)]
            windowed_attention_mask = windowed_attention_mask + [
                0 for _ in range(size_to_pad)
            ]
            windowed_bbox = windowed_bbox + [[0, 0, 0, 0] for _ in range(size_to_pad)]
            windowed_tags = windowed_tags + [
                ([0] * number_of_tags) for _ in range(size_to_pad)
            ]

        # We include pixel_values to the frame only if use_image
        preprocessed_window: ProcessedLayoutLMInput = {
            "input_ids": windowed_input_ids,
            "attention_mask": windowed_attention_mask,
            "bbox": windowed_bbox,
            "tags": windowed_tags,
            "labels": None,
            "class_labels": None,
        }
        if use_image:
            preprocessed_window["pixel_values"] = processed_input["pixel_values"][0]

        windowed_preprocessed_input.append(preprocessed_window)

    return windowed_preprocessed_input


attributes_to_vectorize = [
    "input_ids",
    "attention_mask",
    "bbox",
    "tags",
    "pixel_values",
]


def get_torch_tensor(feature_name: str, vector: Any) -> torch.Tensor | None:
    if feature_name not in attributes_to_vectorize:
        return None
    elif vector is None:
        return torch.tensor(False)
    elif feature_name == "tags":
        return torch.tensor(vector, dtype=torch.float)
    else:
        return torch.tensor(vector)


def get_layoutlm_inference_batches(
    preprocess_input_windows: List[ProcessedLayoutLMInput],
    batch_size: int,
    device: torch.device,
) -> List[TorchProcessedLayoutLMInput]:

    return [
        cast(
            TorchProcessedLayoutLMInput,
            {
                key: torch.stack(
                    [
                        get_torch_tensor(key, item[key])  # type: ignore
                        for item in preprocess_input_windows[i : i + batch_size]
                    ]
                ).to(device)
                for key in preprocess_input_windows[0].keys()
                if key in attributes_to_vectorize
            },
        )
        for i in range(0, len(preprocess_input_windows), batch_size)
    ]


def get_layoutlm_inference_predictions(
    model: LayoutLMv3AndFeaturesClassification,
    batches: list[TorchProcessedLayoutLMInput],
) -> Tuple[
    Dict[str, np.ndarray[Tuple[int, int, int], np.dtype[np.float32]]],
    Dict[str, np.ndarray[Tuple[int, int], np.dtype[np.float32]]],
    np.ndarray[
        Tuple[int, int, int],
        np.dtype[np.float32]
    ]
]:
    outputs_extraction: Dict[
        str, List[np.ndarray[Tuple[int, int, int], np.dtype[np.float32]]]
    ] = {}
    outputs_classifier: Dict[
        str, List[np.ndarray[Tuple[int, int], np.dtype[np.float32]]]
    ] = {}
    vector_outputs: List[
        np.ndarray[
            Tuple[int, int, int],
            np.dtype[np.float32]
        ]
    ] = []

    with torch.no_grad():
        for batch in batches:
            model_output_extraction: Dict[str, torch.Tensor]
            model_output_classification: Dict[str, torch.Tensor]

            model_output_extraction, model_output_classification, model_vector_outputs = model(**batch)

            for task, logit in model_output_extraction.items():
                if task not in outputs_extraction:
                    outputs_extraction[task] = []
                outputs_extraction[task].append(
                    torch.nn.Softmax(dim=2)(logit).cpu().detach().numpy()
                )

            for key, logit in model_output_classification.items():
                if key not in outputs_classifier:
                    outputs_classifier[key] = []
                outputs_classifier[key].append(
                    torch.nn.Softmax(dim=1)(logit).cpu().detach().numpy()
                )

            vector_outputs.append(model_vector_outputs.cpu().detach().numpy())

    concat_outputs_extraction: Dict[
        str, np.ndarray[Tuple[int, int, int], np.dtype[np.float32]]
    ] = {
        task: np.concatenate(output_task, axis=0)
        for task, output_task in outputs_extraction.items()
    }

    concat_outputs_classifier: Dict[
        str, np.ndarray[Tuple[int, int], np.dtype[np.float32]]
    ] = {
        key: np.concatenate(output_class, axis=0)
        for key, output_class in outputs_classifier.items()
    }

    concat_vector_outputs: np.ndarray[
        Tuple[int, int, int],
        np.dtype[np.float32]
    ] = np.concatenate(vector_outputs, axis=0)

    return concat_outputs_extraction, concat_outputs_classifier, concat_vector_outputs



def post_processing_vectors(
    vectors_output: np.ndarray[
        Tuple[int, int, int],
        np.dtype[np.float32]
    ],
    ocr_pages: List[PageInfo],
    page_indexes: List[int],
    window_page_indexes: List[int],
    document_preprocess_input_windows: List[ProcessedLayoutLMInput]
) -> Dict[str, List[float]]:

    layoutlm_vector_by_bbox: Dict[str, List[float]] = {}
    for index, page_index in enumerate(page_indexes):
        page = ocr_pages[index]

        for bounding_box in page['boundingBoxes']:
            normalised_bounding_box_coordinates = normalise_ocr_bounding_box(
                bounding_box['coordinates'])

            # We look for vector for this bounding box on all windows
            # predicted
            for window_index, window_page_index in enumerate(
                    window_page_indexes):
                # if the prediction window is not on the same page we skip
                if window_page_index != page_index:
                    continue

                # If the bounding box is not in this window we skip
                if normalised_bounding_box_coordinates not in document_preprocess_input_windows[
                        window_index]['bbox']:
                    continue

                bounding_box_index = document_preprocess_input_windows[window_index]['bbox'].index(
                    normalised_bounding_box_coordinates)

                bounding_box_vector = vectors_output[window_index][bounding_box_index]

                if bounding_box['id'] not in layoutlm_vector_by_bbox:
                    layoutlm_vector_by_bbox[bounding_box['id']
                                            ] = bounding_box_vector.tolist()
                else:
                    layoutlm_vector_by_bbox[bounding_box['id']] = np.mean([
                        np.array(
                            layoutlm_vector_by_bbox[bounding_box['id']]),
                        bounding_box_vector
                    ], axis=0).tolist()

    return layoutlm_vector_by_bbox

def post_processing_classification(
    outputs_classifier: Dict[str, np.ndarray[Tuple[int, int], np.dtype[np.float32]]],
    classification_id2label: Dict[str, Dict[int, str]],
    page_indexes: List[int],
    document_preprocess_input_windows_page_indexes: List[int],
) -> List[InferencePageClassification]:
    classification_prediction_pages: List[InferencePageClassification] = [
        {
            "pageNumber": int(page_index + 1),
            "rawPredictions": {
                class_name: {
                    label: float(1) if label == NONE_LABEL else float(0)
                    for label in labels.values()
                }
                for class_name, labels in classification_id2label.items()
            },
        }
        for page_index in page_indexes
    ]

    # Iterate over all predictions windows to aggregate classification results
    for window_index, window_page_index in enumerate(
        document_preprocess_input_windows_page_indexes
    ):
        current_page_pred = next(
            (
                pred
                for pred in classification_prediction_pages
                if pred["pageNumber"] == window_page_index + 1
            ),
            None,
        )

        assert current_page_pred is not None, "Page not found"

        for (
            classification_head_name,
            classification_header_id2label,
        ) in classification_id2label.items():
            window_head_prediction = outputs_classifier[classification_head_name][
                window_index
            ]

            max_confidence_previous_prediction = max(
                [
                    label_confidence
                    for label, label_confidence in current_page_pred["rawPredictions"]
                    .get(classification_head_name, {})
                    .items()
                    if label != NONE_LABEL
                ],
                default=0,
            )

            if window_head_prediction.max() > max_confidence_previous_prediction:
                current_page_pred["rawPredictions"][classification_head_name] = {
                    classification_header_id2label[prediction_id]: float(confidence)
                    for prediction_id, confidence in enumerate(window_head_prediction)
                }

    return classification_prediction_pages

def get_tags_vector(tags: Tags, training_config: TrainingConfig) -> List[int]:
    tag_vector = []
    for tag in training_config.get('tagging', {}).get('ocrTags', []):
        if (tags.get(tag) or None) is not None or \
            (tags.get(f"{tag}B") or None) is not None or \
                (tags.get(f"{tag}I") or None) is not None:
            tag_vector.append(1)
        else:
            tag_vector.append(0)

    return tag_vector

def get_page_bounding_boxes_tags(page: PageInfo, training_config: TrainingConfig) -> List[List[int]]:
    return [
        get_tags_vector(bounding_box['tags'], training_config)
        for bounding_box in page['boundingBoxes']
    ]
def preprocess_single_task_box_classifier_for_inference(
    training_config: TrainingConfig,
    input_size: int,
    layoutlm_vectors_by_bbox: Dict[str, List[float]],
    ocr_pages: List[PageInfo],
    page_indexes: List[int]
) -> Tuple[List[ProcessedBoxClassifierInput], List[int]]:
    ocr_tags = training_config.get('tagging', {}).get('ocrTags', [])
    num_tags = len(ocr_tags)
    box_classifier_config = training_config.get('boxClassifier', {})
    window_size = box_classifier_config.get(
        'model', {}).get('window_size', 1024) or 1024

    overlap_ratio = 0.2
    overlap_size = int(window_size * overlap_ratio)

    window_page_indexes: List[int] = []
    preprocessed_windows: List[ProcessedBoxClassifierInput] = []

    for index, page_index in enumerate(page_indexes):
        page = ocr_pages[index]

        if len(page['boundingBoxes']) == 0:
            continue

        page_bounding_boxes_layoutlm_vectors = [
            layoutlm_vectors_by_bbox[bbox['id']
                                     ] if bbox['id'] in layoutlm_vectors_by_bbox else None
            for bbox in page['boundingBoxes']
        ]

        page_bounding_boxes_tags = get_page_bounding_boxes_tags(
            page, training_config)

        page_bounding_boxes_ids = [bbox['id']
                                   for bbox in page['boundingBoxes']]

        # Remove empty text bbox
        mask = [vec is not None for vec in page_bounding_boxes_layoutlm_vectors]
        filtered_page_bounding_boxes_layoutlm_vectors = cast(List[List[float]], [vec for vec, keep in zip(
            page_bounding_boxes_layoutlm_vectors, mask) if keep])
        filtered_page_bounding_boxes_tags = [tag for tag, keep in zip(
            page_bounding_boxes_tags, mask) if keep]
        filtered_page_bounding_boxes_ids = [id for id, keep in zip(
            page_bounding_boxes_ids, mask) if keep]

        for i in range(0, len(filtered_page_bounding_boxes_layoutlm_vectors), (window_size - overlap_size)):
            window_start_index = i
            window_end_index = (i + window_size)

            windowed_bouding_boxes_layoutlm_vectors = filtered_page_bounding_boxes_layoutlm_vectors[
                window_start_index:window_end_index]
            windowed_bounding_boxes_tags = filtered_page_bounding_boxes_tags[
                window_start_index:window_end_index]
            windowed_bounding_boxes_ids = filtered_page_bounding_boxes_ids[
                window_start_index:window_end_index]

            # Padding
            if len(windowed_bouding_boxes_layoutlm_vectors) < window_size:
                pad_len = window_size - \
                    len(windowed_bouding_boxes_layoutlm_vectors)
                windowed_bouding_boxes_layoutlm_vectors += [[0.0 for _ in range(input_size)]
                                                            for _ in range(pad_len)]
                windowed_bounding_boxes_tags += [[0 for _ in range(num_tags)]
                                                 for _ in range(pad_len)]

            window_preprocessed_data: ProcessedBoxClassifierInput = {
                "layoutlm_vectors": windowed_bouding_boxes_layoutlm_vectors,
                "bbox_ids": windowed_bounding_boxes_ids,
                "tags": windowed_bounding_boxes_tags,
                "page_index": page_index
            }

            preprocessed_windows.append(window_preprocessed_data)
            window_page_indexes.append(page_index)

    return preprocessed_windows, window_page_indexes

def get_box_classifier_inference_batches(
    preprocess_box_classifier_windows: List[ProcessedBoxClassifierInput],
    batch_size: int,
    device: torch.device
) -> List[TorchProcessedBoxClassifierInput]:
    return [
        cast(TorchProcessedBoxClassifierInput, {
            "layoutlm_vectors": torch.stack([
                torch.tensor(item['layoutlm_vectors'])
                for item in preprocess_box_classifier_windows[i:i + batch_size]
            ]).to(device),
            "tags": torch.stack([
                torch.tensor(item['tags'], dtype=torch.float32)
                for item in preprocess_box_classifier_windows[i:i + batch_size]
            ]).to(device)
        })
        for i in range(0, len(preprocess_box_classifier_windows), batch_size)
    ]

def get_box_classifier_inference_predictions(
    inference_model: torch.nn.Module,
    batches: List[TorchProcessedBoxClassifierInput]
) -> Dict[str, np.ndarray[
    Tuple[int, int, int],
    np.dtype[np.float32]
]]:
    inference_model.eval()
    output: Dict[str, List[np.ndarray[
        Tuple[int, int, int],
        np.dtype[np.float32]
    ]]] = {}
    with torch.no_grad():
        for batch in batches:
            box_classifier_inference = inference_model(
                batch['layoutlm_vectors'], batch['tags'])

            for task, logit in box_classifier_inference.items():
                if task not in output:
                    output[task] = []
                output[task].append(torch.nn.Softmax(
                    dim=2)(logit).cpu().detach().numpy())

    concat_outputs: Dict[str, np.ndarray[
        Tuple[int, int, int],
        np.dtype[np.float32]
    ]] = {
        task: np.concatenate(output_task, axis=0)
        for task, output_task in output.items()
    }

    return concat_outputs

def box_classifier_post_processing(
    box_classifier_predictions: Dict[
        str,
        np.ndarray[
            Tuple[int, int, int],
            np.dtype[np.float32]
        ]
    ],
    extraction_id2label: Dict[str, Dict[int, str]],
    ocr_pages: List[PageInfo],
    page_indexes: List[int],
    window_page_indexes: List[int],
    preprocessed_box_classifier_windows: List[ProcessedBoxClassifierInput],
) -> List[InferenceBoundingBox]:
    extraction_prediction_heads = extraction_id2label.keys()

    bounding_boxes_with_predictions: List[InferenceBoundingBox] = []
    for index, page_index in enumerate(page_indexes):
        page = ocr_pages[index]

        for bounding_box in page['boundingBoxes']:
            # Initialize bounding box with predictions (with empty predictions)
            bounding_box_with_predictions: InferenceBoundingBox = {
                'id': bounding_box['id'],
                'rawPredictions': {
                    task: {}
                    for task in extraction_prediction_heads
                }
            }

            # We look for prediction for this bounding box on all windows
            # predicted
            for window_index, window_page_index in enumerate(window_page_indexes):
                # if the prediction window is not on the same page we skip
                if window_page_index != page_index:
                    continue

                # If the bounding box is not in this window we skip
                if bounding_box['id'] not in preprocessed_box_classifier_windows[window_index]['bbox_ids']:
                    continue

                bounding_box_index = preprocessed_box_classifier_windows[window_index]['bbox_ids'].index(
                    bounding_box['id'])

                for extraction_prediction_head in extraction_prediction_heads:
                    head_window_prediction = box_classifier_predictions[
                        extraction_prediction_head][window_index]
                    predictions = head_window_prediction[bounding_box_index]

                    previous_bounding_box_raw_predictions = bounding_box_with_predictions[
                        "rawPredictions"]

                    max_confidence_previous_prediction = max(
                        [
                            label_confidence
                            for label, label_confidence in (
                                previous_bounding_box_raw_predictions[extraction_prediction_head]
                            ).items()
                            if label != NONE_LABEL
                        ],
                        default=0,
                    )

                    # We replace the prediction results if we have a higher confidence for any class except NONE_LABEL
                    # Since max_confidence_previous_prediction default to 0, if
                    # we don't have any results yet, we just add them
                    if predictions.max() > max_confidence_previous_prediction:
                        bounding_box_with_predictions['rawPredictions'][extraction_prediction_head] = {
                            extraction_id2label[extraction_prediction_head][prediction_id]: float(
                                confidence)
                            for prediction_id, confidence in enumerate(predictions)
                        }

            bounding_boxes_with_predictions.append(
                bounding_box_with_predictions)
    return bounding_boxes_with_predictions
