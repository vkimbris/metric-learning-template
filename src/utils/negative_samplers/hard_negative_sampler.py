from sentence_transformers import SentenceTransformer
from sentence_transformers.util import mine_hard_negatives

from .negative_sampler import NegativeSampler, NotEnoughCanidadatesError
from datasets import Dataset


class HardNegativeSampler(NegativeSampler):

    def __init__(self, 
                 model: SentenceTransformer, 
                 n_negatives: int, 
                 as_triplets: bool = True,
                 **params):
        
        super().__init__(n_negatives, as_triplets)

        self.model = model
        self.params = params

    def sample(self, dataset: Dataset) -> Dataset:
        if len(set(dataset["positive"])) < self.n_negatives:
            raise NotEnoughCanidadatesError("Number of candidates less than a requested number of negatives.")
        
        dataset_with_negatives = mine_hard_negatives(
            dataset=dataset, 
            model=self.model, 
            num_negatives=self.n_negatives,
            as_triplets=self.as_triplets,
            **self.params
        )

        return dataset_with_negatives