import pandas as pd

from datasets import DatasetDict

from .splitter import Splitter, SplittingError


class LabelsSplitter(Splitter):
    
    def __init__(self, add_validation = True, label_column: str = "label", test_size: float = 0.1, name: str | None = None):
        super().__init__(add_validation, name)

        self.label_column = label_column
        self.test_size = test_size

    def _split_on_train_test(self, data: pd.DataFrame) -> DatasetDict:
        train_labels, test_labels = self._split_on_train_test_labels(data)

        train = data[data[self.label_column].isin(train_labels)]
        test = data[data[self.label_column].isin(test_labels)]

        if len(train) == 0:
            raise SplittingError("There are no examples in train set")
        
        if len(test) == 0:
            raise SplittingError("There are no examples in validation or test set")

        return self._dataframes_to_datasetdict(train, test)

    def _split_on_train_test_labels(self, data: pd.DataFrame) -> DatasetDict:
        labels_counts = data[self.label_column].value_counts(normalize=True, ascending=True)
        
        train_labels, test_labels = [], []

        current_test_size = 0
        for label_name, label_frequency in zip(labels_counts.index, labels_counts.values):
            if current_test_size < self.test_size:
                test_labels.append(label_name)

                current_test_size += label_frequency

            else:
                train_labels.append(label_name)

        return train_labels, test_labels