import pandas as pd

from datasets import DatasetDict, Dataset
from abc import ABC, abstractmethod


class SplittingError(Exception):

    def __init__(self, *args):
        super().__init__(*args)


class Splitter(ABC):

    _preserve_index: bool = False

    def __init__(self, add_validation: bool = True, name: str | None = None):
        super().__init__()

        self.add_validation = add_validation
        self.name = name

    def split(self, data: pd.DataFrame) -> DatasetDict:
        dataset = self._split_on_train_test(data)
        
        if self.add_validation:
            new_dataset = self._split_on_train_test(dataset["train"].to_pandas())

            dataset["train"] = new_dataset["train"]
            
            for key in new_dataset.keys():
                if "test" in key:
                    dataset[key.replace("test", "validation")] = new_dataset[key]

        return dataset
    
    def _dataframes_to_datasetdict(self, train: pd.DataFrame, test: pd.DataFrame) -> DatasetDict:
        postfix = "" if self.name is None else "_" + self.name

        return DatasetDict({
            "train": Dataset.from_pandas(train, preserve_index=Splitter._preserve_index), 
            "test" + postfix: Dataset.from_pandas(test, preserve_index=Splitter._preserve_index)
        })

    @abstractmethod
    def _split_on_train_test(self, data: pd.DataFrame) -> DatasetDict:
        raise NotImplementedError