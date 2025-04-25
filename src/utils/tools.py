import numpy as np

from scipy.spatial.distance import cdist

from datasets import Dataset, concatenate_datasets

from dataclasses import dataclass
from typing import List, Dict, Callable, Tuple, Hashable, Any

from tqdm import tqdm

from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate

from sentence_transformers.evaluation import SentenceEvaluator


@dataclass
class Mappings:
    label_id_to_positive_ids: Dict[int, List[int]]
    
    positive2id: Dict[str, int]
    id2positive: Dict[int, str]
    
    label2id: Dict[str, int]
    id2label: Dict[int, str]


def cos_sim(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return 1 - cdist(u, v, "cosine")


def masked_semantic_search(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    candidates_ids: List[List[int]],
    query_chunk_size: int = 5096,
    topk: int = 3,
    similiarity_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = cos_sim,
    verbose: bool = False
):
    
    n_query, n_corpus = len(query_embeddings), len(corpus_embeddings)
    
    if n_query != len(query_embeddings):
        raise ValueError("Mismatched lengths!")
    
    if len(max(candidates_ids, key=lambda x: len(x))) > n_corpus:
        raise ValueError("Mismatched lengths!")

    similarities = []
    for k in tqdm(range(0, n_query, query_chunk_size), disable=(not verbose)):
        similarities.append(similiarity_func(query_embeddings[k : k + query_chunk_size], corpus_embeddings))
    
    similarities = np.concatenate(similarities)
    similarities = similarities + _get_mask(candidates_ids, shape=similarities.shape)

    return _postprocess_semantic_search_results(*_get_topk(similarities, topk))


def _get_mask(indexes: List[List[int]], shape: Tuple[int, int]) -> np.ndarray:
    mask = np.full(shape=shape, fill_value=-np.inf)

    for k, ids in enumerate(indexes):
        mask[k, list(ids)] = 0

    return mask


def _get_topk(similarities: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    scores = np.sort(similarities, axis=1)[:, ::-1][:, :topk]
    indices = np.argsort(similarities, axis=1)[:, ::-1][:, :topk]

    return indices, scores


def _postprocess_semantic_search_results(indices: np.array, scores: np.ndarray) -> List[List[Dict[str, float]]]:
    search_results: List[List[Dict[str, float]]] = []
    
    for ids, scrs in zip(indices, scores):
        query_results: List[Dict[str, float]] = []

        for _id, _score in zip(ids, scrs):
            if _score != -np.inf:
                query_results.append({
                    "corpus_id": _id,
                    "score": _score
                })

        search_results.append(query_results)

    return search_results


def get_label_positives_mappings(labels: List[str], positives: List[str]) -> Tuple[List[List[int]], Mappings]:
    label2id = __convert_to_ids(set(labels))
    positive2id = __convert_to_ids(set(positives))

    id2positive = _reverse_dictionary(positive2id)
    id2label = _reverse_dictionary(label2id)

    label_ids = _apply_dictionary_to_list(labels, label2id)
    positive_ids = _apply_dictionary_to_list(positives, positive2id)

    label_id_to_positive_ids = _map_labels_to_unique_positives(label_ids, positive_ids)
    label_id_to_positive_ids = {i: list(values) for i, values in label_id_to_positive_ids.items()}

    return Mappings(label_id_to_positive_ids, positive2id, id2positive, label2id, id2label)


def _apply_dictionary_to_list(values: List[Hashable], value2id: Dict[Hashable, Any]) -> List[Any]:
    return list(map(lambda label: value2id[label], values))


def _reverse_dictionary(dictionary: Dict[str, int]) -> Dict[int, str]:
    return {value: key for key, value in dictionary.items()}


def __convert_to_ids(values: List[str]) -> Dict[str, int]:
    return {value: i for i, value in enumerate(values)}


def _map_labels_to_unique_positives(labels: List[str | int], positives: List[str | int]) -> Dict[str | int, List[str | int]]:
    if len(labels) != len(positives):
        raise ValueError("Length of labels must be equal to lenght of positives.")
    
    mapping: Dict[str, List[str]] = {}
    
    for label, positive in zip(labels, positives):
        if label not in mapping:
            mapping[label] = set()

        if positive not in mapping[label]:
            mapping[label].add(positive)

    return mapping


def instantiate_evaluators(config: DictConfig, dataset: Dataset, split_type: str = "validation") -> List[SentenceEvaluator] | SentenceEvaluator:
    full_ds = concatenate_datasets([ds for _, ds in dataset.items()])
    
    evaluators: List[SentenceEvaluator] = []
    
    for split_name, ds in dataset.items():
        if split_type in split_name:
            name = split_name.replace(split_type, "").replace("_", "")
            labels, positives = full_ds["label"], full_ds["positive"]
            
            evaluators.append(
                instantiate(
                    config.train.evaluator, 
                    anchors=ds["anchor"], 
                    positives=ds["positive"], 
                    labels=ds["label"],
                    name=name,
                    _convert_="all"
                ).initialize_mappings(labels, positives)
            )

    if len(evaluators) == 1:
        evaluators = evaluators[-1]

    return evaluators


def instantiate_eval_datasets(dataset: Dataset) -> Dict[str, Dataset] | Dataset:
    eval_datasets = {
        key.replace("validation", ""): ds for key, ds in dataset.items() if "validation" in key
    }
        
    if len(eval_datasets) == 1:
        eval_datasets = list(eval_datasets.values())[-1]

    return eval_datasets