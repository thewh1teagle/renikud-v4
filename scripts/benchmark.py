"""
Inference script for Hebrew G2P model with benchmark evaluation.

Download benchmark data first:
wget https://raw.githubusercontent.com/thewh1teagle/heb-g2p-benchmark/refs/heads/main/gt.tsv

Usage:
    uv run scripts/eval.py --checkpoint checkpoints/step_2500.pt --gt gt.tsv
"""

import csv
import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm
import jiwer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from constants import ENCODER_MODEL, MAX_LEN
from tokenization import decode_ipa, load_encoder_tokenizer
from checkpoint import load_checkpoint
from evaluate import ctc_greedy_decode
from transformers import AutoModel


def load_gt(filepath: str):
    """Load ground truth TSV: Sentence, Phonemes, Field"""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            data.append(
                {
                    "sentence": row["Sentence"],
                    "phonemes": row["Phonemes"],
                    "field": row.get("Field", ""),
                }
            )
    return data


def benchmark(model, encoder, tokenizer, gt_data, device):
    """Run inference and compute WER/CER against ground truth phonemes."""
    wer_scores = []
    cer_scores = []
    examples = []

    print(f"\nEvaluating {len(gt_data)} samples...")

    for item in tqdm(gt_data):
        sentence = item["sentence"]
        gt_phonemes = item["phonemes"]

        enc = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)
        with torch.no_grad():
            encoder_out = encoder(**enc).last_hidden_state
            log_probs = model(encoder_out, enc["attention_mask"])
            input_lengths = enc["attention_mask"].sum(dim=1).long() * 2
            decoded = ctc_greedy_decode(log_probs, input_lengths)

        pred_phonemes = decode_ipa(decoded[0])

        w = jiwer.wer(gt_phonemes, pred_phonemes)
        c = jiwer.cer(gt_phonemes, pred_phonemes)

        wer_scores.append(w)
        cer_scores.append(c)

        if len(examples) < 5:
            examples.append(
                {
                    "sentence": sentence,
                    "gt": gt_phonemes,
                    "pred": pred_phonemes,
                }
            )

    mean_wer = sum(wer_scores) / len(wer_scores)
    mean_cer = sum(cer_scores) / len(cer_scores)

    print(f"\n{'=' * 80}")
    print("Sample Predictions (first 5):")
    print(f"{'=' * 80}")
    for i, ex in enumerate(examples, 1):
        print(f"\n{i}. Input:    {ex['sentence']}")
        print(f"   GT:       {ex['gt']}")
        print(f"   Pred:     {ex['pred']}")

    print(f"\n{'=' * 80}")
    print("Results:")
    print(f"  Mean WER: {mean_wer:.4f}")
    print(f"  Mean CER: {mean_cer:.4f}")
    print(f"  Samples:  {len(gt_data)}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Hebrew G2P model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--gt", type=str, default="gt.tsv", help="Ground truth TSV file"
    )
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    if not Path(args.gt).exists():
        print(f"Error: {args.gt} not found. Download it with:")
        print(
            "wget https://raw.githubusercontent.com/thewh1teagle/heb-g2p-benchmark/refs/heads/main/gt.tsv"
        )
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    tokenizer = load_encoder_tokenizer()
    encoder = AutoModel.from_pretrained(ENCODER_MODEL, trust_remote_code=True).to(device)
    if hasattr(encoder, 'bert'):
        encoder = encoder.bert
    encoder.eval()

    model, _ = load_checkpoint(args.checkpoint, device)
    model.eval()

    gt_data = load_gt(args.gt)
    benchmark(model, encoder, tokenizer, gt_data, device)


if __name__ == "__main__":
    main()