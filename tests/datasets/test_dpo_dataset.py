import os
from typing import Dict

import pandas as pd
import torch
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


def test_sample_is_correct():
    filename = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data/user/hh/train.pq")
    )
    df = pd.read_parquet(filename).iloc[:5000]
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
            assert len(sample[key].shape) == 1, (
                key,
                sample[key].shape,
            )  # Check sample shape is correct
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
        if idx in [
            33777,
            77046,
            88047,
            88476,
            89090,
            92121,
            95371,
            95606,
            96376,
            99342,
            99918,
        ]:
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


def test_dataloader():
    filename = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data/user/hh/train.pq")
    )
    df = pd.read_parquet(filename).iloc[: (5000 // 16) * 16]
    df.head()

    cfg = ConfigProblemBase(
        dataset=ConfigNLPDPOLMDataset(
            prompt_column=("instruction",),
            answer_column="output",
            parent_id_column="parent_id",
        )
    )

    dataset = CustomDataset(df, cfg, mode="train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        for key in batch:
            if idx != len(dataloader) - 1:
                assert batch[key].size(0) == 16, (
                    key,
                    batch[key].shape,
                )

            if key in [
                "labels",
                "chosen_labels",
                "rejected_labels",
                "prompt_input_ids",
                "input_ids",
                "chosen_input_ids",
                "rejected_input_ids",
            ]:
                pass
