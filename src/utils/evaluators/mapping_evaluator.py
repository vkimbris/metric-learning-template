import json
import os
import numpy as np

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from typing import List, Dict, Callable, Tuple, Any

from ..tools import get_label_positives_mappings, masked_semantic_search
from ..tools import _apply_dictionary_to_list, _reverse_dictionary


def compute_precision_recall_f_beta_score(beta: float):
    def tmp(elements: List[Dict[str, str | float]], threshold: float) -> Dict[str, int | float]:
        tp, fp, fn = 0, 0, 0

        for element in elements:
            if element["score"] < threshold:
                fn += 1
            
            elif element["true_positive"] == element["pred_positive"]:
                tp += 1

            elif element["true_positive"] != element["pred_positive"]:
                fp += 1

            else:
                raise ValueError("Unexcpected case.")

        precision = tp / (tp + fp) if tp + fp != 0 else 0
        recall = tp / (tp + fn) if tp + fn != 0 else 0

        nominator = (1 + beta ** 2) * (precision * recall)
        denominator = (precision * beta ** 2 + recall)

        f_beta_score = 0 if denominator == 0 else nominator / denominator

        return {"precision": precision, "recall": recall, "support": len(elements), "f_score": f_beta_score}

    return tmp


def compute_weighted_averages(metrics: List[Dict[str, int | float]]) -> Dict[str, int | float]:
    aggregated_metrics = {"recall": 0, "precision": 0, "f_score": 0}

    supports = 0
    for metric in metrics:
        support = metric["support"]
        supports += support
        
        for metric_name in aggregated_metrics.keys():
            aggregated_metrics[metric_name] += metric[metric_name] * support
        
    for metric_name in aggregated_metrics.keys():
        aggregated_metrics[metric_name] = aggregated_metrics[metric_name] / supports

    return aggregated_metrics | {"support": supports}



class MappingEvaluator(SentenceEvaluator):

    def __init__(
        self,
        anchors: List[str],
        positives: List[str],
        labels: List[str],
        compute_metrics_fn: Callable[[List[Dict[str, str | float]], float], Dict[str, int | float]] = compute_precision_recall_f_beta_score(beta=1.0),
        compute_aggregated_metrics_fn: Callable[[List[Dict[str, int | float]]], Dict[str, int | float]] = compute_weighted_averages,
        positive2id: Dict[str, int] | None = None,
        label2id: Dict[str, int] | None = None,
        label_id_to_positive_ids: Dict[int, List[int]] | None = None,
        name: str = "",
        write_json: bool = True,
        primary_metric: str = "f_score",
        encoding_kwargs: Dict[str, Any] | None = None
    ):
        super().__init__()

        self.anchors = anchors
        self.positives = positives
        self.labels = labels

        self.compute_metrics_fn = compute_metrics_fn
        self.compute_aggregated_metrics_fn = compute_aggregated_metrics_fn

        self.positive2id = positive2id
        self.label2id = label2id
        self.label_id_to_positive_ids = label_id_to_positive_ids

        if self.positive2id is not None:
            self.id2positive = _reverse_dictionary(self.positive2id)
            self.unique_positives = list(self.positive2id.keys())
        else:
            self.id2positive = None
            self.unique_positives = None
        
        if self.label2id is not None and self.label_id_to_positive_ids is not None:
            self.candidates_ids = self.__get_candidates_ids()
        else:
            self.candidates_ids = None

        if self.label2id is not None:
            self.id2label = _reverse_dictionary(self.label2id)
        else:
            self.id2label = None

        self.encoding_kwargs = encoding_kwargs if encoding_kwargs is not None else {}
        
        self.name = name
        self.write_json = write_json
        self.json_file = "evaluation" + ("_" + name if name else "") + "_results.json"

        self.primary_metric = primary_metric

    def initialize_mappings(self, labels: List[str], positives: List[str]):
        mappings = get_label_positives_mappings(labels=labels, positives=positives)

        self.label_id_to_positive_ids = mappings.label_id_to_positive_ids
        
        self.positive2id = mappings.positive2id
        self.unique_positives = list(self.positive2id.keys())
        self.id2positive = mappings.id2positive
        
        self.label2id = mappings.label2id
        self.id2label = mappings.id2label

        self.candidates_ids = self.__get_candidates_ids()

        return self

    def __call__(self, model, output_path = None, epoch = -1, steps = -1):
        metrics_per_label, aggregated_metrics = self.compute_metrics(model)

        if output_path is not None and self.write_json:
            json_path = os.path.join(output_path, self.json_file)
            
            # first time saving
            if not os.path.isfile(json_path):
                current_metrics_per_label = []
            else:
                # load current file first if not the first time saving
                with open(json_path, "r") as f:
                    current_metrics_per_label = json.load(f)

            current_metrics_per_label.append(
                {"epoch": epoch, "step": steps, "metrics": {"per_label": metrics_per_label, "aggregated": aggregated_metrics}}
            )

            with open(json_path, "w") as f:
                json.dump(current_metrics_per_label, f)
        
        return aggregated_metrics

    def compute_metrics(self, model: SentenceTransformer, thresholds: Dict[str, float] | float | None = None) -> Tuple[Dict[str, Dict[str, float]], Dict[str, int | float]]:
        if self.positive2id is None or self.label2id is None:
            raise ValueError("You need to specify positive2id and label2id either directly to constructor or with source labels and positives via initialize_mappings() method: evaluator.initialize_mappings(labels, positives)")
        
        search_results = self._fetch_search_results(self._do_semantic_search(model))
        
        metrics = {}
        if thresholds is None:
            # find optimal thresholds per label
            for label in set(self.labels):
                search_results_per_label = [res for res in search_results if res["label"] == label]

                best_threshold, best_metrics = self._find_best_threshold_and_metrics(
                    search_results_per_label
                )

                metrics[label] = best_metrics | {"threshold": best_threshold}

        else:
            # use predefined threshold
            for label in set(self.labels):
                search_results_per_label = [res for res in search_results if res["label"] == label]

                if isinstance(thresholds, dict):
                    threshold = thresholds.get(label)

                elif isinstance(thresholds, float):
                    threshold = thresholds

                else:
                    raise ValueError("Unexpected thresholds type.")

                if threshold is not None:
                    metrics[label] = self.compute_metrics_fn(
                        search_results_per_label, threshold
                    )

                else:
                    raise ValueError(f"There is no threshold for label {label}.")
                
        aggregated_metrics = self.compute_aggregated_metrics_fn(metrics.values())
        
        return metrics, aggregated_metrics
        
    def _find_best_threshold_and_metrics(self, elements: List[Dict[str, str | float]]) -> Tuple[float, Dict[str, int | float]]:
        min_score = min(elements, key=lambda x: x["score"])["score"]
        max_score = max(elements, key=lambda x: x["score"])["score"]

        if len(elements) == 1:
            best_thd = min_score

            best_metrics = self.compute_metrics_fn(elements, best_thd)

            return best_thd, best_metrics

        thresholds_range = np.arange(min_score, max_score, 0.01)
        
        best_thd, best_target_metric = 0.0, 0.0
        best_metrics = {}
        for thd in thresholds_range:
            metrics = self.compute_metrics_fn(
                elements, thd
            )

            target_metric = metrics.get(self.primary_metric)
            if target_metric >= best_target_metric:
                best_thd, best_target_metric = thd, target_metric

                best_metrics = metrics

        return best_thd, best_metrics


    def _do_semantic_search(self, model: SentenceTransformer) -> List[List[Dict[str, float]]]:
        query_embeddings = model.encode(
            self.anchors, **self.encoding_kwargs
        )

        corpus_embeddings = model.encode(
            self.unique_positives, **self.encoding_kwargs
        )

        semantic_search_results = masked_semantic_search(
            query_embeddings, corpus_embeddings, self.candidates_ids, topk=1
        )

        return semantic_search_results
    
    def _fetch_search_results(self, search_results: List[List[Dict[str, float]]]):
        rows: List[Dict[str, str | float]] = []

        for k, results in enumerate(search_results):
            result = results[0]
            
            rows.append({
                "anchor": self.anchors[k], 
                "label": self.labels[k],
                "true_positive": self.positives[k],
                "pred_positive": self.id2positive[result["corpus_id"]],
                "score": result["score"]
            })

        return rows
    
    def _add_prefix_to_metrics_name(self, metrics: Dict[str, int | float]) -> Dict[str, int | float]:
        return {self.name + "_" + key: value for key, value in metrics.items()}

    def __get_candidates_ids(self) -> List[List[int]]:
        return _apply_dictionary_to_list(_apply_dictionary_to_list(self.labels, self.label2id), self.label_id_to_positive_ids)
