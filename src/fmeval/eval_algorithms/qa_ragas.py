import logging

from typing import List, Optional

from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.embeddings import BaseRagasEmbeddings
from ragas.llms import BaseRagasLLM
from ragas.metrics import faithfulness

from fmeval.data_loaders.hf_data_loader_utils import convert_ray_dataset_to_hf
from fmeval.data_loaders.util import get_dataset
from fmeval.data_loaders.data_config import DataConfig
from fmeval.eval_algorithms.util import (
    generate_output_dataset_path,
)
from fmeval.eval_algorithms.eval_algorithm import (
    EvalAlgorithmInterface,
)
from fmeval.eval_algorithms import (
    EvalOutput,
    EvalScore,
)
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.perf_util import timed_block
from fmeval.ragas.util import get_bedrock_model, get_bedrock_embedding


logger = logging.getLogger(__name__)


class RagasFaithfulness(EvalAlgorithmInterface):
    """
    RagasFaithfulness Eval algorithm
    """

    eval_name = "Ragas_faithfulness_poc"

    def __init__(self):
        super().__init__(eval_algorithm_config=None)

    def evaluate(
        self,
        dataset_config: Optional[DataConfig] = None,
        prompt_template: Optional[str] = None,
        # save: bool = False,
        num_records=100,
        llm: Optional[BaseRagasLLM] = None,
        embeddings: Optional[BaseRagasEmbeddings] = None,
    ) -> List[EvalOutput]:
        """
        Ragas Faithfulness evaluate
        """
        if dataset_config:
            dataset_configs = [dataset_config]
        else:
            dataset_configs = [
                DataConfig(
                    dataset_name="fiqa_sample",
                    dataset_uri="fiqa_sample.jsonl",
                    dataset_mime_type="application/jsonlines",
                    model_input_location="question",
                    model_output_location="answer",
                    contexts_location="contexts",
                    target_output_location="ground_truths",
                )
            ]

        if llm is None:
            llm = get_bedrock_model()

        if embeddings is None:
            embeddings = get_bedrock_embedding()

        eval_outputs: List[EvalOutput] = []
        for dataset_config in dataset_configs:
            dataset = get_dataset(dataset_config, num_records)
            hf_dataset = convert_ray_dataset_to_hf(dataset)
            metrics = [
                faithfulness,
            ]
            with timed_block(f"Computing score and aggregation on dataset {dataset_config.dataset_name}", logger):

                result = ragas_evaluate(hf_dataset, metrics=metrics, llm=llm, embeddings=embeddings)

                scores = result.copy()
                dataset_scores = [EvalScore(name=k, value=v) for k, v in scores.items()]

                eval_outputs.append(
                    EvalOutput(
                        eval_name=self.eval_name,
                        dataset_name=dataset_config.dataset_name,
                        dataset_scores=dataset_scores,
                        output_path=generate_output_dataset_path(
                            path_to_parent_dir=self._eval_results_path,
                            eval_name=self.eval_name,
                            dataset_name=dataset_config.dataset_name,
                        ),
                    )
                )

        return eval_outputs

    def evaluate_sample(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        llm: Optional[BaseRagasLLM] = None,
        embeddings: Optional[BaseRagasEmbeddings] = None,
    ) -> List[EvalScore]:  # type: ignore[override]
        """
        Evaluate a single record.
        """
        if question is None:
            raise EvalAlgorithmClientError("Missing required input: question, for Faithfulness evaluate_sample")
        if answer is None:
            raise EvalAlgorithmClientError("Missing required input: answer, for Faithfulness evaluate_sample")
        if contexts is None or len(contexts) == 0:
            raise EvalAlgorithmClientError("Missing required input: contexts, for Faithfulness evaluate_sample")

        if llm is None:
            llm = get_bedrock_model()

        if embeddings is None:
            embeddings = get_bedrock_embedding()

        hf_ds = Dataset.from_dict(
            {"question": [question], "answer": [answer], "contexts": [[context] for context in contexts]}
        )
        result = ragas_evaluate(
            hf_ds,
            metrics=[faithfulness],
            llm=llm,
            embeddings=embeddings,
        )
        scores = result.copy()
        return [EvalScore(name=k, value=v) for k, v in scores.items()]
