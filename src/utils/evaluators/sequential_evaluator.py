import json
import os
import numpy as np

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from typing import List, Dict, Callable, Tuple


class SequentialEvaluator(SentenceEvaluator):

    def __init__(self, evaluators: List[SentenceEvaluator], main_score_function=lambda scores: scores[-1]):
        super().__init__()

        self.evaluators = evaluators
        self.main_score_function = main_score_function

    def __call__(self, model, output_path = None, epoch = -1, steps = -1):
        evaluations = []
        scores = []
        for evaluator_idx, evaluator in enumerate(self.evaluators):
            evaluation = evaluator(model, output_path, epoch, steps)

            if not isinstance(evaluation, dict):
                scores.append(evaluation)
                evaluation = {f"evaluator_{evaluator_idx}": evaluation}
            else:
                if hasattr(evaluator, "primary_metric"):
                    scores.append(evaluation[evaluator.primary_metric])
                else:
                    scores.append(evaluation[list(evaluation.keys())[0]])

            evaluations.append(evaluation)

        results = {}
        for evaluator_idx, evaluation in enumerate(evaluations):
            evaluator = self.evaluators[evaluator_idx]
            
            evaluator_name = evaluator.name if evaluator.name != "" else str(evaluator_idx)
            
            for key, value in evaluation.items():
                results[evaluator_name + "_" + key] = value
        
        self.primary_metric = "sequential_score"
        main_score = self.main_score_function(scores)

        results["sequential_score"] = main_score

        return results
