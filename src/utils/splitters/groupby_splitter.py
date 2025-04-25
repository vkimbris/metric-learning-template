import pandas as pd

from datasets import DatasetDict
from datasets import concatenate_datasets

from .splitter import Splitter, SplittingError
from .random_splitter import RandomSplitter

from typing import List, Callable


class GroupBySplitter(Splitter):

    def __init__(self, splitter: Splitter, by: str = "label"):
        super().__init__(splitter.add_validation)

        self.by = by
        self.splitter = splitter

    def _split_on_train_test(self, data: pd.DataFrame) -> DatasetDict:
        if self.by not in data.columns:
            raise ValueError(f"There is no column {self.by} in data.")
        
        datasets: List[DatasetDict] = []

        for label in data[self.by].unique():
            samples = data[data[self.by] == label]

            dataset = self.splitter._split_on_train_test(samples)
            
            datasets.append(dataset)

        dataset = {}
        for ds in datasets:
            for key, val in ds.items():
                if key not in dataset:
                    dataset[key] = []
                
                dataset[key].append(val)

        for key in dataset.keys():
            dataset[key] = concatenate_datasets(dataset[key])

        return DatasetDict(dataset)