from abc import ABC, abstractmethod
from datasets import Dataset


class NotEnoughCanidadatesError(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class NegativeSampler(ABC):

    def __init__(self, n_negatives: int, as_triplets: bool = True):
        super().__init__()

        self.n_negatives = n_negatives
        self.as_triplets = as_triplets

    @abstractmethod
    def sample(self, dataset: Dataset) -> Dataset:
        raise NotImplementedError
