"""Evaluation helpers for Hebrew G2P training."""

from __future__ import annotations

import numpy as np
from jiwer import cer, wer

from tokenization import decode_ctc, decode_ipa


def build_compute_metrics():
    def compute_metrics(eval_pred):
        if hasattr(eval_pred, "predictions") and hasattr(eval_pred, "label_ids"):
            predictions = eval_pred.predictions
            labels = eval_pred.label_ids
        else:
            predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        pred_ids = np.asarray(predictions).argmax(axis=-1)
        label_ids = np.asarray(labels)

        pred_texts = [decode_ctc(row.tolist()) for row in pred_ids]
        label_texts = [
            decode_ipa([int(token_id) for token_id in row if int(token_id) != -100])
            for row in label_ids
        ]

        return {
            "cer": cer(label_texts, pred_texts),
            "wer": wer(label_texts, pred_texts),
        }

    return compute_metrics
