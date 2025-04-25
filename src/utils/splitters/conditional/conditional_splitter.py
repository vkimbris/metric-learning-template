import pandas as pd

from datasets import DatasetDict

from ..splitter import Splitter, SplittingError
from .selectors import Selector, ThresholdSelector
from typing import List, Callable


class ConditionalSplitter(Splitter):

    def __init__(self,
                 major_splitter: Splitter, 
                 minor_splitter: Splitter,
                 selector: Selector,
                 add_validation: bool = True):
        
        super().__init__(add_validation)

        self.major_splitter = major_splitter
        self.minor_splitter = minor_splitter

        self.selector = selector
    
    def _split_on_train_test(self, data: pd.DataFrame) -> DatasetDict:
        if self.selector.select(data):
            return self.major_splitter._split_on_train_test(data)

        return self.minor_splitter._split_on_train_test(data)