import os
from typing import Any, Dict, List, Tuple

import pandas as pd

from llm_studio.src.datasets.text_causal_language_modeling_ds import CustomDataset
from llm_studio.src.datasets.text_utils import get_tokenizer
from llm_studio.src.utils.data_utils import (
    read_dataframe_drop_missing_labels,
    sample_indices,
)
from llm_studio.src.utils.plot_utils import (
    PlotData,
    format_for_markdown_visualization,
    list_to_markdown_representation,
)


class Plots:
    NUM_TEXTS: int = 20

    @classmethod
    def plot_batch(cls, batch, cfg) -> PlotData:
        tokenizer = get_tokenizer(cfg)

        df = pd.DataFrame(
            {
                "Prompt Text": [
                    tokenizer.decode(input_ids, skip_special_tokens=True)
                    for input_ids in batch["prompt_input_ids"].detach().cpu().numpy()
                ]
            }
        )
        df["Prompt Text"] = df["Prompt Text"].apply(format_for_markdown_visualization)
        if "labels" in batch.keys():
            df["Answer Text"] = [
                tokenizer.decode(
                    [label for label in labels if label != -100],
                    skip_special_tokens=True,
                )
                for labels in batch.get("labels", batch["input_ids"])
                .detach()
                .cpu()
                .numpy()
            ]
        tokens_list = [
            tokenizer.convert_ids_to_tokens(input_ids)
            for input_ids in batch["input_ids"].detach().cpu().numpy()
        ]
        masks_list = [
            [label != -100 for label in labels]
            for labels in batch.get("labels", batch["input_ids"]).detach().cpu().numpy()
        ]
        df["Tokenized Text"] = [
            list_to_markdown_representation(
                tokens, masks, pad_token=tokenizer.pad_token, num_chars=100
            )
            for tokens, masks in zip(tokens_list, masks_list)
        ]
        # limit to 2000 rows, still renders fast in wave
        df = df.iloc[:2000]

        # Convert into a scrollable table by transposing the dataframe
        df_transposed = pd.DataFrame(columns=["Sample Number", "Field", "Content"])
        for i, row in df.iterrows():
            offset = 3 if "Answer Text" in df.columns else 2
            df_transposed.loc[i * offset] = [
                i,
                "Prompt Text",
                row["Prompt Text"],
            ]
            if "Answer Text" in df.columns:
                df_transposed.loc[i * offset + 1] = [
                    i,
                    "Answer Text",
                    row["Answer Text"],
                ]
            df_transposed.loc[i * offset + 2] = [
                i,
                "Tokenized Text",
                row["Tokenized Text"],
            ]
        df_transposed["Content"] = df_transposed["Content"].apply(
            format_for_markdown_visualization
        )

        path = os.path.join(cfg.output_directory, "batch_viz.parquet")
        df_transposed.to_parquet(path)

        return PlotData(path, encoding="df")

    @classmethod
    def plot_data(cls, cfg) -> PlotData:
        df = read_dataframe_drop_missing_labels(cfg.dataset.train_dataframe, cfg)
        input_text_lists, target_texts = cls.get_chained_conversations(df, cfg, True)

        idxs = sample_indices(len(input_text_lists), Plots.NUM_TEXTS)
        input_text_lists = [input_text_lists[i] for i in idxs]
        target_texts = [target_texts[i] for i in idxs]

        df = pd.DataFrame(
            {
                "Input Text List": input_text_lists,
                "Target Text": target_texts,
            }
        )
        # Convert into a scrollable table by transposing the dataframe
        df_transposed = pd.DataFrame(columns=["Sample Number", "Field", "Content"])

        i = 0
        for sample_number, row in df.iterrows():
            input_text_lists = row["Input Text List"]
            for j, input_text in enumerate(input_text_lists):
                suffix = "- Prompt" if j % 2 == 0 else "- Answer "
                df_transposed.loc[i] = [
                    sample_number,
                    f"Input Text {suffix}",
                    input_text,
                ]
                i += 1
            df_transposed.loc[i] = [sample_number, "Target Text", row["Target Text"]]
            i += 1

        df_transposed["Content"] = df_transposed["Content"].apply(
            format_for_markdown_visualization
        )
        path = os.path.join(
            os.path.dirname(cfg.dataset.train_dataframe), "data_viz.parquet"
        )
        df_transposed.to_parquet(path)

        return PlotData(path, encoding="df")

    @classmethod
    def get_chained_conversations(
        cls, df: pd.DataFrame, cfg, limit_chained_samples=True
    ) -> Tuple[List[List[str]], List[str]]:
        limit_chained_samples_default = cfg.dataset.limit_chained_samples
        if limit_chained_samples:
            cfg.dataset.limit_chained_samples = True
        dataset = CustomDataset(df, cfg, mode="validation")
        input_text_lists = [
            dataset.get_chained_prompt_text_list(i) for i in dataset.indices
        ]
        target_texts = [dataset.answers[i] for i in dataset.indices]
        cfg.dataset.limit_chained_samples = limit_chained_samples_default
        return input_text_lists, target_texts

    @classmethod
    def plot_validation_predictions(
        cls, val_outputs: Dict, cfg: Any, val_df: pd.DataFrame, mode: str
    ) -> PlotData:
        assert mode in ["validation"]
        input_text_lists, target_texts = cls.get_chained_conversations(
            val_df, cfg, limit_chained_samples=False
        )

        if "predicted_text" in val_outputs.keys():
            predicted_texts = val_outputs["predicted_text"]
        else:
            predicted_texts = [
                "No predictions are generated for the selected metric"
            ] * len(target_texts)

        df = pd.DataFrame(
            {
                "Input Text": [
                    "\n".join(input_text_list) for input_text_list in input_text_lists
                ],
                "Target Text": target_texts,
                "Predicted Text": predicted_texts,
            }
        )
        df["Input Text"] = df["Input Text"].apply(format_for_markdown_visualization)
        df["Target Text"] = df["Target Text"].apply(format_for_markdown_visualization)
        df["Predicted Text"] = df["Predicted Text"].apply(
            format_for_markdown_visualization
        )

        if val_outputs.get("metrics") is not None:
            df[f"Metric ({cfg.prediction.metric})"] = val_outputs["metrics"]
            df[f"Metric ({cfg.prediction.metric})"] = df[
                f"Metric ({cfg.prediction.metric})"
            ].round(decimals=3)
        if val_outputs.get("explanations") is not None:
            df["Explanation"] = val_outputs["explanations"]

        path = os.path.join(cfg.output_directory, f"{mode}_viz.parquet")
        df.to_parquet(path)
        return PlotData(data=path, encoding="df")

    @staticmethod
    def plot_empty(cfg, error="Not yet implemented.") -> PlotData:
        """Plots an empty default plot.

        Args:
            cfg: config

        Returns:
            The default plot as `PlotData`.
        """

        return PlotData(f"<h2>{error}</h2>", encoding="html")
