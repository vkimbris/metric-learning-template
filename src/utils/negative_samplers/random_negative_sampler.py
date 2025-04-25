import random

from typing import List, Set

from .negative_sampler import NegativeSampler, NotEnoughCanidadatesError
from datasets import Dataset


class RandomNegativeSampler(NegativeSampler):

    def __init__(self, n_negatives: int, as_triplets: bool = True, random_state: int = 42):
        super().__init__(n_negatives, as_triplets)

        self.random_state = random_state

    def sample(self, dataset: Dataset) -> Dataset:
        random.seed(self.random_state)

        candidates = set(dataset["positive"])
        
        elements_with_negatives = []
        for el in dataset:
            anchor, positive = el["anchor"], el["positive"]
            
            candidates_for_element = candidates.copy()

            negatives = self._get_negatives(
                anchor=anchor, positive=positive, candidates=candidates_for_element
            )

            if self.as_triplets:
                for negative in negatives:
                    element_with_negative = {
                        "anchor": anchor, "positive": positive, "negative": negative
                    }

            else:
                if len(candidates_for_element) < self.n_negatives:
                    raise NotEnoughCanidadatesError("Number of candidates less than a requested number of negatives. Try to use as_triplets=True.")
                
                element_with_negative = {"anchor": anchor, "positive": positive}
                for k, negative in enumerate(negatives):
                    element_with_negative[f"negative_{k + 1}"] = negative

            elements_with_negatives.append(element_with_negative)

        return Dataset.from_list(elements_with_negatives)

    def _get_negatives(self, anchor: str, positive: str, candidates: Set[str]) -> List[str]:        
        candidates.remove(positive)
        
        return random.sample(list(candidates), k=min(self.n_negatives, len(candidates)))
