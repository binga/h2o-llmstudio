from typing import Any, Dict

import pandas as pd
from bokeh.models import Div, Panel, Tabs

from llm_studio.src.datasets.text_utils import get_texts, get_tokenizer
from llm_studio.src.utils.data_utils import (
    read_dataframe_drop_missing_labels,
    sample_indices,
)
from llm_studio.src.utils.plot_utils import (
    PlotData,
    color_code_tokenized_text,
    get_best_and_worst_sample_idxs,
    get_line_separator_html,
    text_to_html,
    to_html,
)


class Plots:
    NUM_TEXTS: int = 20

    @classmethod
    def plot_batch(cls, batch, cfg) -> PlotData:
        tokenizer = get_tokenizer(cfg)

        texts = [
            tokenizer.decode(input_ids, skip_special_tokens=True)
            for input_ids in batch["input_ids"].detach().cpu().numpy()
        ]

        texts = [text_to_html(text) for text in texts]

        tokenized_texts = [
            color_code_tokenized_text(
                tokenizer.convert_ids_to_tokens(input_ids), tokenizer
            )
            for input_ids in batch["input_ids"].detach().cpu().numpy()
        ]

        target_texts = {"chosen": [], "rejected": []}
        tokenized_target_texts = {"chosen": [], "rejected": []}
        for type in ["chosen", "rejected"]:
            if f"{type}_labels" in batch.keys():
                labels = [
                    [input_id for input_id in input_ids if input_id != -100]
                    for input_ids in batch[f"{type}_labels"].detach().cpu().numpy()
                ]

                target_texts[type] = [
                    tokenizer.decode(input_ids, skip_special_tokens=False)
                    for input_ids in labels
                ]

                tokenized_target_texts[type] = [
                    color_code_tokenized_text(
                        tokenizer.convert_ids_to_tokens(input_ids),
                        tokenizer,
                    )
                    for input_ids in labels
                ]

        markup = ""

        for i in range(len(tokenized_texts)):
            markup += f"<p><strong>Input Text: </strong>{texts[i]}</p>\n"
            markup += (
                "<p><strong>Tokenized Input Text: "
                f"</strong>{tokenized_texts[i]}</p>\n"
            )
            if len(target_texts["chosen"]) > 0:
                chosen_target_text = target_texts["chosen"][i]
                tokenized_chosen_target_text = tokenized_target_texts["chosen"][i]
                markup += (
                    "<p><strong>Chosen Target Text: "
                    f"</strong>{chosen_target_text}</p>\n"
                )
                markup += (
                    "<p><strong>Tokenized Chosen Target Text:"
                    f" </strong>{tokenized_chosen_target_text}</p>\n"
                )
            if len(target_texts["rejected"]) > 0:
                rejected_target_text = target_texts["rejected"][i]
                tokenized_rejected_target_text = tokenized_target_texts["rejected"][i]
                markup += (
                    "<p><strong>Rejected Target Text: "
                    f"</strong>{rejected_target_text}</p>\n"
                )
                markup += (
                    "<p><strong>Tokenized Rejected Target Text:"
                    f" </strong>{tokenized_rejected_target_text}</p>\n"
                )
            markup += get_line_separator_html()
        return PlotData(markup, encoding="html")

    @classmethod
    def plot_data(cls, cfg) -> PlotData:
        df = read_dataframe_drop_missing_labels(cfg.dataset.train_dataframe, cfg)
        df = df.iloc[sample_indices(len(df), Plots.NUM_TEXTS)]

        input_texts = get_texts(df, cfg, separator="")

        if cfg.dataset.answer_column in df.columns:
            target_texts = df[cfg.dataset.answer_column].values
        else:
            target_texts = ""

        markup = ""
        for input_text, target_text in zip(input_texts, target_texts):
            markup += (
                f"<p><strong>Input Text: </strong>{text_to_html(input_text)}</p>\n"
            )
            markup += "<br/>"
            markup += (
                f"<p><strong>Target Text: </strong>{text_to_html(target_text)}</p>\n"
            )
            markup += "<br/>"
            markup += get_line_separator_html()
        return PlotData(markup, encoding="html")

    @classmethod
    def selection_validation_predictions(
        cls,
        val_outputs: Dict,
        cfg: Any,
        val_df: pd.DataFrame,
        metrics: Any,
        sample_idx: Any,
    ) -> str:
        input_texts = get_texts(val_df, cfg, separator="")
        markup = ""

        true_labels = val_outputs["target_text"]
        if "predicted_text" in val_outputs.keys():
            pred_labels = val_outputs["predicted_text"]
        else:
            pred_labels = [
                "No predictions are generated for the selected metric"
            ] * len(true_labels)

        for idx in sample_idx:
            input_text = input_texts[idx]
            markup += (
                f"<p><strong>Input Text: </strong>{text_to_html(input_text)}</p>\n"
            )

            if true_labels is not None:
                target_text = true_labels[idx]
                markup += "<br/>"
                markup += (
                    f"<p><strong>Target Text: "
                    f"</strong>{text_to_html(target_text)}</p>\n"
                )

            predicted_text = pred_labels[idx]
            markup += "<br/>"
            markup += (
                f"<p><strong>Predicted Text: </strong>"
                f"{text_to_html(predicted_text)}</p>\n"
            )

            if metrics is not None:
                markup += "<br/>"
                markup += (
                    f"<p><strong>{cfg.prediction.metric} Score: </strong>"
                    f"{metrics[idx]:.3f}"
                )

            if "explanations" in val_outputs:
                markup += "<br/>"
                markup += (
                    f"<p><strong>Explanation: </strong>"
                    f"{val_outputs['explanations'][idx]}"
                )

            if idx != sample_idx[-1]:
                markup += get_line_separator_html()

        return markup

    @classmethod
    def plot_validation_predictions(
        cls, val_outputs: Dict, cfg: Any, val_df: pd.DataFrame, mode: str
    ) -> PlotData:
        assert mode in ["validation"]

        metrics = val_outputs["metrics"]
        best_samples, worst_samples = get_best_and_worst_sample_idxs(
            cfg, metrics, n_plots=min(cfg.logging.number_of_texts, len(val_df))
        )
        random_samples = sample_indices(len(val_df), len(best_samples))
        selection_plots = {
            title: cls.selection_validation_predictions(
                val_outputs=val_outputs,
                cfg=cfg,
                val_df=val_df,
                metrics=metrics,
                sample_idx=indices,
            )
            for (indices, title) in [
                (random_samples, f"Random {mode} samples"),
                (best_samples, f"Best {mode} samples"),
                (worst_samples, f"Worst {mode} samples"),
            ]
        }

        tabs = [
            Panel(
                child=Div(
                    text=markup, sizing_mode="scale_width", style={"font-size": "105%"}
                ),
                title=title,
            )
            for title, markup in selection_plots.items()
        ]
        return PlotData(to_html(Tabs(tabs=tabs)), encoding="html")
