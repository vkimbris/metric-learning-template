import logging

from sentence_transformers import SentenceTransformer
from datasets import Dataset
from typing import List, Dict, Any

from .negative_sampler import NegativeSampler
from .negative_sampler import NotEnoughCanidadatesError

from ..tools import masked_semantic_search, get_label_positives_mappings, _apply_dictionary_to_list


class MaskedHardNegativeSampler(NegativeSampler):

    def __init__(
            self, 
            model: SentenceTransformer, 
            n_negatives: int, 
            as_triplets: bool = True,
            anchor_column_name: str = "anchor",
            positive_column_name: str = "positive",
            label_column_name: str = "label",
            model_encoding_params: Dict[str, Any] | None = None,
            masked_semantic_search_params: Dict[str, Any] | None = None,
        ):
        
        super().__init__(n_negatives, as_triplets)

        self.model = model
        self.n_negatives = n_negatives
        self.as_triplets = as_triplets

        self.anchor_column_name = anchor_column_name
        self.positive_column_name = positive_column_name
        self.label_column_name = label_column_name

        self.model_encoding_params = {} if model_encoding_params is None else model_encoding_params
        self.masked_semantic_search_params = {} if masked_semantic_search_params is None else masked_semantic_search_params

    def sample(self, dataset: Dataset) -> Dataset:
        anchors = dataset[self.anchor_column_name]
        labels = dataset[self.label_column_name]
        positives = dataset[self.positive_column_name]
        
        # get valid positives ids per each element in dataset base on label
        mappings = get_label_positives_mappings(labels, positives)

        id2positive = mappings.id2positive

        candidates_ids = _apply_dictionary_to_list(
            _apply_dictionary_to_list(labels, mappings.label2id), mappings.label_id_to_positive_ids
        )

        # delete real positive value from valid positives to not include it in negatives
        for k in range(len(candidates_ids)):
            candidates_ids[k] = [pos for pos in candidates_ids[k] if pos != mappings.positive2id[positives[k]]]

        logging.info("Generating embeddings for anchors")
        query_embeddings = self.model.encode(
            anchors, **self.model_encoding_params
        )

        logging.info("Generating embeddings for unique positives")
        corpus_embeddings = self.model.encode(
            list(mappings.positive2id.keys()), **self.model_encoding_params
        )

        logging.info("Masked semantic searching")
        search_results = masked_semantic_search(
            query_embeddings, corpus_embeddings, candidates_ids, topk=self.n_negatives, **self.masked_semantic_search_params
        )

        logging.info("Generating a dataset with negatives")
        processed_results: List[Dict[str, str]] = []
        for k, negatives in enumerate(search_results):            
            result = {"anchor": dataset[k]["anchor"], "positive": dataset[k]["positive"]}

            if self.as_triplets:
                for negative in negatives:
                    processed_results.append(result | {"negative": id2positive[negative["corpus_id"]]})

            else:
                if len(negatives) < self.n_negatives:
                    raise NotEnoughCanidadatesError(f'Cannot create a dataset of "negative_1", ..., "negative_{self.n_negatives}" because label {dataset[k]["label"]} has less that {self.n_negatives} unique positives. Try to use as_triplets=True.')
                
                for i, negative in enumerate(negatives):
                    result[f"negative_{i}"] = id2positive[negative["corpus_id"]]

                processed_results.append(result)

        return Dataset.from_list(processed_results)