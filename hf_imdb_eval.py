"""
Evaluate Hugging Face sentiment models on imdb_top_500.csv (same split file as Task 3).

HF Hub env vars must be set **before** importing ``transformers`` (this file does that).

Mirror & SSL (Windows / campus network):
  set HF_ENDPOINT=https://hf-mirror.com
  If you still see CERTIFICATE_VERIFY_FAILED:
    set HF_HUB_DISABLE_SSL_VERIFICATION=1
  Or: python hf_imdb_eval.py --insecure

Install deps (Tsinghua PyPI mirror example):
  pip install pandas -r requirements_hf.txt \\
      -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

Run:
  python hf_imdb_eval.py
  python hf_imdb_eval.py --models textattack/bert-base-uncased-imdb textattack/roberta-base-imdb
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# --- Hugging Face Hub: env MUST be set before importing transformers/huggingface_hub ---
_hf_pre = argparse.ArgumentParser(add_help=False)
_hf_pre.add_argument(
    "--hf-endpoint",
    default=None,
    metavar="URL",
    help="HF Hub API endpoint (default: https://hf-mirror.com if unset)",
)
_hf_pre.add_argument(
    "--insecure",
    action="store_true",
    help="Set HF_HUB_DISABLE_SSL_VERIFICATION=1 (only if you trust the network)",
)
_hf_args, _argv_rest = _hf_pre.parse_known_args()
sys.argv = [sys.argv[0]] + _argv_rest

if _hf_args.hf_endpoint:
    os.environ["HF_ENDPOINT"] = _hf_args.hf_endpoint.rstrip("/")
elif not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

if _hf_args.insecure:
    os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

import pandas as pd
from transformers import pipeline

ROOT = Path(__file__).resolve().parent
CSV_CANDIDATES = (ROOT / "imdb_top_500.csv", ROOT / "datasets" / "imdb_top_500.csv")


def find_csv() -> Path:
    for p in CSV_CANDIDATES:
        if p.is_file():
            return p
    raise FileNotFoundError(f"Missing imdb_top_500.csv (looked in {CSV_CANDIDATES})")


def pred_to_binary(p: dict, id2label: dict[int, str]) -> int:
    """Map pipeline output to 1=positive, 0=negative to match CSV label."""
    lab = p["label"]
    s = str(lab)
    if lab == "LABEL_1" or s.upper() == "POSITIVE":
        return 1
    if lab == "LABEL_0" or s.upper() == "NEGATIVE":
        return 0
    # Fallback: infer from config id2label
    for i, name in id2label.items():
        if name == lab:
            name_u = str(name).upper()
            if "POS" in name_u or name.endswith("1") or name == "LABEL_1":
                return 1
            return 0
    raise ValueError(f"Unknown label: {lab}, id2label={id2label}")


def evaluate_model(model_name: str, reviews: list[str], true_labels: list[int]) -> tuple[list[int], list[dict]]:
    print(f"\nHF_ENDPOINT={os.environ.get('HF_ENDPOINT')}")
    print(f"HF_HUB_DISABLE_SSL_VERIFICATION={os.environ.get('HF_HUB_DISABLE_SSL_VERIFICATION', '0')}")
    print(f"\nLoading model: {model_name} ...")
    t0 = time.time()
    clf = pipeline(
        task="sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        framework="pt",
    )
    id2label = dict(clf.model.config.id2label.items())
    print(f"Model loaded in {time.time() - t0:.1f}s  id2label={id2label}")

    print("Running inference ...")
    t1 = time.time()
    preds_raw = clf(reviews, truncation=True, batch_size=16)
    print(f"Inference done in {time.time() - t1:.1f}s")

    pred_labels = [pred_to_binary(p, id2label) for p in preds_raw]
    return pred_labels, preds_raw


def report(
    model_name: str,
    reviews: list[str],
    true_labels: list[int],
    pred_labels: list[int],
    preds_raw: list[dict],
    out_txt: Path,
) -> None:
    n = len(true_labels)
    correct_all = sum(p == t for p, t in zip(pred_labels, true_labels))
    acc_all = correct_all / n

    correct_50 = sum(p == t for p, t in zip(pred_labels[:50], true_labels[:50]))
    acc_50 = correct_50 / min(50, n)

    wrong_indices = [i for i in range(n) if pred_labels[i] != true_labels[i]]

    lines = [
        "=" * 60,
        f"Model: {model_name}",
        "=" * 60,
        f"Accuracy on first 50 reviews: {acc_50:.2%} ({correct_50}/{min(50, n)})",
        f"Accuracy on ALL {n} reviews: {acc_all:.2%} ({correct_all}/{n})",
        f"Wrong predictions: {len(wrong_indices)}/{n}",
        "=" * 60,
        "",
    ]

    print("\n".join(lines))

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n\nAll wrong predictions:\n")
        for k, idx in enumerate(wrong_indices):
            snippet = reviews[idx].replace("\n", " ")[:220]
            f.write(
                f"\n[{k + 1}] Review #{idx + 1} True={true_labels[idx]} Pred={pred_labels[idx]} "
                f"Score={preds_raw[idx]['score']:.4f}\n    {snippet}...\n"
            )

    print(f"Details saved to {out_txt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="HF sentiment models on IMDB top 500 CSV")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "textattack/bert-base-uncased-imdb",
            "textattack/roberta-base-imdb",
        ],
        help="Hugging Face model ids",
    )
    args = parser.parse_args()

    csv_path = find_csv()
    df = pd.read_csv(csv_path)
    print(f"CSV: {csv_path}")
    print(f"Dataset size: {len(df)}")

    reviews = df["text"].tolist()
    true_labels = [int(x) for x in df["label"].tolist()]

    for model_name in args.models:
        safe = model_name.replace("/", "_")
        out_txt = ROOT / f"{safe}_imdb_results.txt"
        pred_labels, preds_raw = evaluate_model(model_name, reviews, true_labels)
        report(model_name, reviews, true_labels, pred_labels, preds_raw, out_txt)


if __name__ == "__main__":
    main()
