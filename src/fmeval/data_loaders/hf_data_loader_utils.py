import datasets
import ray.data


def convert_ray_dataset_to_hf(dataset: ray.data.Dataset) -> datasets.Dataset:
    df = dataset.to_pandas()
    hf_ds = datasets.Dataset.from_pandas(
        df.rename(
            columns={
                "model_input": "question",
                "model_output": "answer",
                "context": "contexts",
                "target_output": "ground_truth",
            }  # OSS ragas use ground_truth, RagasBedrock use ground_truths
        )
    )

    def cast_to_sequence(hf_dataset: datasets.Dataset):
        hf_dataset["contexts"] = [hf_dataset["contexts"]]
        return hf_dataset

    return hf_ds.map(cast_to_sequence)


"""
import os
import ray
import datasets
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    faithfulness,
    context_recall,
    answer_relevancy,
)
from fmeval.data_loaders.data_config import DataConfig
from fmeval.data_loaders.util import get_dataset

dataset_config = DataConfig(
    dataset_name="fiqa_sample",
    dataset_uri="/Users/xiayche/workplace3/fmeval/examples/fiqa_sample.jsonl",
    dataset_mime_type="application/jsonlines",
    model_input_location="question",
    model_output_location="answer",
    contexts_location="contexts",
    target_output_location="ground_truths"
)

ray_dataset = get_dataset(dataset_config)

hf_dataset = convert_ray_dataset_to_hf(ray_dataset)

metrics = [
    faithfulness,
]

os.environ["OPENAI_API_KEY"] = ""

result = evaluate(hf_dataset, metrics=metrics)

"""
