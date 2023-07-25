import os

import pandas as pd
import torch

from llm_studio.python_configs.text_dpo_language_modeling_config import (
    ConfigProblemBase,
    ConfigNLPDPOLMDataset,
)
from llm_studio.src.datasets.text_dpo_language_modeling_ds import CustomDataset
from llm_studio.src.models.text_dpo_language_modeling_model import get_batch_logps


def test_logps_mask_is_correct():
    labels = torch.tensor(
        [-100] * 202
        + [
            309,
            6468,
            626,
            1014,
            1869,
            670,
            352,
            15,
            0,
        ]
        + [-100] * 301
    )
    assert len(labels) == 512
    labels = labels[None]
    logits = torch.randn((1, labels.shape[1], 50265))
    logps = get_batch_logps(logits=logits, labels=labels)

    r = list(range(200)) + list(range(250, 512))
    for i in r:
        old_logits = torch.clone(logits)
        logits[0, i] = 100
        logps_1 = get_batch_logps(logits=logits, labels=labels)
        assert logps_1 == logps, (i, logps_1, logps)
        logits = old_logits


def test_batch_padding_works():
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
