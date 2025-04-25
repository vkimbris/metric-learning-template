import json
import logging

from datasets import load_from_disk

from omegaconf import OmegaConf
from hydra.utils import instantiate

from sentence_transformers import SentenceTransformer

from utils.tools import instantiate_eval_datasets, instantiate_evaluators

from typing import Dict, Any


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_thresholds(path: str, step: int) -> Dict[str, float]:
    with open(path, "r") as f:
        evaluation_results = json.load(f)

    for result in evaluation_results:
        if result["step"] == step:
            metrics = result["metrics"]["per_label"]
            break

    return {label: metric["threshold"] for label, metric in metrics.items()}


def save_as_json(obj: Any, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f)


def main() -> None:
    config = OmegaConf.load("params.yaml")

    logging.info("Loading test dataset")
    dataset = load_from_disk(config.constant.path.artifacts.dataset)

    logging.info(f"Loadinig best model")
    model = SentenceTransformer(model_name_or_path=config.constant.path.model)

    logging.info(f"Instantiating evaluators")
    evaluator = instantiate_evaluators(config, dataset, "test")    
   
    logging.info(f"Loading optimal thresholds")
    
    best_checkpoint = model._model_config.get("best_checkpoint") or 0
    best_checkpoint = 0 if isinstance(best_checkpoint, int) else int(best_checkpoint.replace("checkpoint-", ""))

    thresholds = load_thresholds(
        path=config.constant.path.artifacts.trainer + "/eval/evaluation_results.json",
        step=best_checkpoint
    )

    logging.info(f"Starting evaluating test dataset")
    metrics, aggregated_metrics = evaluator.compute_metrics(model, thresholds)

    logging.info(f"Saving metrics")
    save_as_json(metrics, config.constant.path.metrics.per_label)
    save_as_json(aggregated_metrics, config.constant.path.metrics.aggregated)

if __name__ == "__main__":
    main()