from unittest import mock

import pandas as pd
from torch.utils.data import DataLoader

from llm_studio.python_configs.text_dpo_language_modeling_config import (
    ConfigNLPDPOLMDataset,
    ConfigProblemBase,
)
from llm_studio.src.datasets.text_dpo_language_modeling_ds import CustomDataset
from llm_studio.src.plots.text_dpo_language_modeling_plots import Plots


def test_can_plot_batch():
    df = pd.DataFrame(
        {
            "prompt": ["prompt 1", "prompt 2", "prompt 3"],
            "answer": ["answer 1", "answer 2", "answer 3"],
            "chosen_answer": ["chosen 1", "chosen 2", "chosen 3"],
            "rejected_answer": ["rejected 1", "rejected 2", "rejected 3"],
            "id": [1, 2, 0],
        }
    )

    cfg = ConfigProblemBase(
        dataset=ConfigNLPDPOLMDataset(
            prompt_column=("prompt",),
            answer_column="answer",
            chosen_response_column="chosen_answer",
            rejected_response_column="rejected_answer",
        )
    )
    dataset = CustomDataset(df, cfg)
    dataloder = DataLoader(
        dataset, batch_size=1, collate_fn=dataset.get_train_collate_fn()
    )
    batch = next(iter(dataloder))
    plot = Plots.plot_batch(batch, cfg)

    plot = Plots.plot_validation_predictions(val_outputs=None,
                                             cfg=cfg,
                                             val_df=df)

