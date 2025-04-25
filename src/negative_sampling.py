import logging
import pandas as pd

from datasets import load_from_disk, DatasetDict, disable_caching
disable_caching()

from omegaconf import OmegaConf
from hydra.utils import instantiate


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main() -> None:
    config = OmegaConf.load("params.yaml")

    logging.info("Loading train dataset")
    dataset = load_from_disk(config.constant.path.artifacts.dataset)

    sampler = instantiate(config.negative_sampling)

    logging.info(f"Starting negative mining")
    dataset_with_negatives = DatasetDict()
    for split, ds in dataset.items():
        dataset_with_negatives[split] = sampler.sample(ds)
    
    logging.info("Dataset with negatives saving")
    dataset_with_negatives.save_to_disk(config.constant.path.artifacts.dataset_with_negatives)

if __name__ == "__main__":
    main()