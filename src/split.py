import logging
import pandas as pd

from omegaconf import OmegaConf
from hydra.utils import instantiate

from datasets import disable_caching
disable_caching()


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main() -> None:
    config = OmegaConf.load("params.yaml")

    logging.info("Starting data loading")
    data = pd.read_parquet(config.constant.path.data)
    
    splitter = instantiate(config.split)

    logging.info("Starting split data")
    dataset = splitter.split(data)

    dataset_columns_mapping = config.constant.dataset_columns_mapping

    dataset = dataset.rename_columns(dataset_columns_mapping)

    for column in dataset.column_names["train"]:
        if column not in dataset_columns_mapping.values():
            dataset = dataset.remove_columns(column)

    logging.info("Dataset saving")
    dataset.save_to_disk(config.constant.path.artifacts.dataset)

if __name__ == "__main__":
    main()