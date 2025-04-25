import logging

from datasets import load_from_disk

from omegaconf import OmegaConf
from hydra.utils import instantiate

from sentence_transformers import SentenceTransformerTrainer

from utils.tools import instantiate_eval_datasets, instantiate_evaluators


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main() -> None:
    config = OmegaConf.load("params.yaml")

    logging.info("Loading dataset")
    dataset = load_from_disk(config.constant.path.artifacts.dataset)
    
    logging.info("Loading dataset with negatives")
    dataset_with_negatives = load_from_disk(config.constant.path.artifacts.dataset_with_negatives)

    logging.info(f"Loadinig model")
    model = instantiate(config.train.model, _convert_="all")

    logging.info(f"Instantiating evaluators")
    evaluators = instantiate_evaluators(config, dataset, "validation")    

    logging.info("Instantiating loss function, train and eval datasets, training arguments")
    loss_fn = instantiate(config.train.loss, model=model)
    
    train_dataset = dataset_with_negatives["train"]
    eval_datasets = instantiate_eval_datasets(dataset_with_negatives)
    training_args = instantiate(
        config.train.args, 
        overwrite_output_dir=True, 
        output_dir=config.constant.path.artifacts.trainer,
        do_eval=True,
        load_best_model_at_end=True
    )
    
    trainer = SentenceTransformerTrainer(
        model=model, 
        args=training_args,
        loss=loss_fn, 
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        evaluator=evaluators
    )
    
    if trainer.args.do_train:
        logging.info("Starting training")
        trainer.train()
    else:
        logging.info("No training due to do_train=False. Starting evaluating model")
        trainer.evaluate()

    logging.info("Saving the best model")        
    best_model_checkpoint = trainer.state.best_model_checkpoint
    
    if best_model_checkpoint is not None:
        best_model_checkpoint = best_model_checkpoint.split("/")[-1]
    
        trainer.model._model_config["best_checkpoint"] = best_model_checkpoint
    
    trainer.model.save_pretrained(config.constant.path.model)        

if __name__ == "__main__":
    main()