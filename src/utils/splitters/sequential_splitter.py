import pandas as pd

from datasets import DatasetDict

from .splitter import Splitter, SplittingError
from typing import List

class SequentialSplitter(Splitter):

    def __init__(self, splitters: List[Splitter], add_validation: bool = True):
        super().__init__(add_validation)

        self.splitters = splitters

    def _split_on_train_test(self, data: pd.DataFrame) -> DatasetDict:
        dataset = DatasetDict()

        for splitter in self.splitters:
            splitter_result = splitter._split_on_train_test(data)

            for key in splitter_result.keys():
                if "test" in key:
                    dataset[key] = splitter_result[key]

            data = splitter_result["train"].to_pandas()

        dataset["train"] = splitter_result["train"]
        
        return dataset