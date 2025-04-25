import pandas as pd

from abc import ABC, abstractmethod


class Selector(ABC):
    
    def __init__(self):
        super().__init__()

    @abstractmethod
    def select(self, data: pd.DataFrame) -> bool:
        raise NotImplementedError


class ThresholdSelector(Selector):

    def __init__(self, threshold: int, column: str):
        self.threshold = threshold
        self.column = column

    def select(self, data: pd.DataFrame) -> bool:
        return data[self.column].nunique() >= self.threshold