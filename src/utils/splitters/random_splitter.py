import pandas as pd

from datasets import DatasetDict
from sklearn.model_selection import train_test_split

from .splitter import Splitter, SplittingError

class RandomSplitter(Splitter):

    def __init__(self, add_validation = True, stratify: str | None = None, name: str | None = None, **params):
        super().__init__(add_validation, name)

        self.params = params
        self.stratify = stratify

    def _split_on_train_test(self, data: pd.DataFrame) -> DatasetDict:
        stratify = data[self.stratify] if self.stratify is not None else None
        
        train, test = train_test_split(data, stratify=stratify, **self.params)

        if len(train) == 0:
            raise SplittingError("There are no examples in train set")
        
        if len(test) == 0:
            raise SplittingError("There are no examples in validation or test set")

        return self._dataframes_to_datasetdict(train, test)