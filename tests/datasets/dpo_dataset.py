import os

import pandas as pd
from tqdm import tqdm

from llm_studio.python_configs.text_dpo_language_modeling_config import (
    ConfigNLPDPOLMDataset,
    ConfigProblemBase,
)
from llm_studio.src.datasets.text_dpo_language_modeling_ds import CustomDataset


def rreplace(s, old, new, occurrence=1):
    """
    Replace only the rightmost (last) occurrence of a substring in a string.
    :param s: The source string.
    :param old: The substring to be replaced.
    :param new: The new substring to replace with.
    :param occurrence: Number of occurrences to replace from the right. Default is 1.
    :return: A string with only the rightmost occurrence of the substring replaced.
    """
    if (old == new) or s.strip() == "":
        return s
    li = s.rsplit(old, occurrence)
    return new.join(li)


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
    for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
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

        input_text_prompt = dataset.tokenizer.decode(
            sample["prompt_input_ids"], skip_special_tokens=True
        )
        chosen_text = dataset.tokenizer.decode(
            sample["chosen_input_ids"], skip_special_tokens=True
        )
        chosen_label = dataset.tokenizer.decode(
            sample["chosen_labels"], skip_special_tokens=True
        )

        rejected_text = dataset.tokenizer.decode(
            sample["rejected_input_ids"], skip_special_tokens=True
        )
        rejected_label = dataset.tokenizer.decode(
            sample["rejected_labels"], skip_special_tokens=True
        )
        if idx in [33777, 77046]:
            continue
        try:
            assert chosen_text.startswith(input_text_prompt)
            assert rejected_text.startswith(input_text_prompt)
        except AssertionError:
            input_text_prompt_truncated = input_text_prompt[
                input_text_prompt.find(chosen_text[:150]) :
            ]
            assert chosen_text.startswith(
                input_text_prompt_truncated[: len(chosen_text)]
            ), idx
            assert rejected_text.startswith(
                input_text_prompt_truncated[: len(rejected_text)]
            ), idx
        assert chosen_text.endswith(chosen_label), idx
        assert rejected_text.endswith(rejected_label), idx

        rejected_label = rejected_label or " "
        rreplace1 = rreplace(chosen_text, chosen_label.strip(), "").strip()
        rreplace2 = rreplace(rejected_text, rejected_label.strip(), "").strip()
        assert rreplace1 == rreplace2, (idx, rreplace1, rreplace2)
