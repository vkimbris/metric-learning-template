from .negative_sampler import NegativeSampler

from datasets import Dataset
from datasets import concatenate_datasets

from tqdm import tqdm
from typing import List


class GroupByNegativeSampler(NegativeSampler):

    def __init__(self, sampler: NegativeSampler, by: str = "label", verbose: bool = True):
        super().__init__(sampler.n_negatives, sampler.as_triplets)

        self.sampler = sampler
        self.by = by
        self.verbose = verbose

    def sample(self, dataset: Dataset) -> Dataset:
        if self.by not in dataset.column_names:
            raise ValueError(f"There is no column {self.by} in dataset.")
        
        datasets: List[Dataset] = []
        
        for label in tqdm(set(dataset[self.by])):
            filtered_dataset = dataset.filter(lambda el: el[self.by] == label)

            datasets.append(self.sampler.sample(filtered_dataset))

        return concatenate_datasets(datasets)
