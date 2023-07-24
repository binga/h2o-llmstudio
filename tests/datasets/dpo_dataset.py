import os

import pandas as pd

from llm_studio.python_configs.text_dpo_language_modeling_config import (
    ConfigNLPDPOLMDataset,
    ConfigProblemBase,
)
from llm_studio.src.datasets.text_dpo_language_modeling_ds import CustomDataset

if __name__ == "__main__":
    filename = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data/user/hh/train.pq")
    )
    df = pd.read_parquet(filename)
    df.head()

    cfg = ConfigProblemBase(
        dataset=ConfigNLPDPOLMDataset(
            prompt_column=("instruction",),
            answer_column="output",
            parent_id_column="parent_id",
        )
    )

    dataset = CustomDataset(df, cfg, mode="train")
    print(len(dataset))
    sample = dataset[0]
    print(sample)

    for key in [
        "labels",
        "chosen_labels",
        "rejected_labels",
        "prompt_input_ids",
        "input_ids",
        "chosen_input_ids",
        "rejected_input_ids",
    ]:
        sample[key][sample[key] == -100] = 0
        print(
            key,
            dataset.tokenizer.decode(sample[key], skip_special_tokens=True),
        )
        print("*" * 80)
