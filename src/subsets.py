"""Σταθερά υποσύνολα queries για τα ακριβά μοντέλα (LLMs).

Ένα LLM 7B δεν προλαβαίνει όλο το test στο δωρεάν Colab, οπότε αξιολογείται σε ένα σταθερό,
τυχαίο (seed 42) υποσύνολο queries. Όλα τα υπόλοιπα μοντέλα ξαναβαθμολογούνται στο ίδιο
υποσύνολο από τα predictions τους (src/report.py --subset), ώστε η σύγκριση να είναι δίκαιη.

Χρήση:
    python -m src.subsets            # γράφει configs/subsets/test_q500.json και val_q150.json
"""
import argparse
import json

import numpy as np
import pandas as pd

from src.baselines import ROOT

DEFAULTS = [("test", 500), ("val", 150)]


def make_subset(split, n, seed=42, data_dir="data/processed/esci_small_us"):
    qids = np.sort(pd.read_parquet(ROOT / data_dir / f"{split}.parquet",
                                   columns=["query_id"])["query_id"].unique())
    chosen = np.sort(np.random.default_rng(seed).choice(qids, size=n, replace=False))
    return {"split": split, "n_queries": n, "seed": seed,
            "query_ids": [int(q) for q in chosen]}


def load_subset(path):
    with open(ROOT / path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = ROOT / "configs" / "subsets"
    out.mkdir(parents=True, exist_ok=True)
    for split, n in DEFAULTS:
        sub = make_subset(split, n, args.seed)
        (out / f"{split}_q{n}.json").write_text(json.dumps(sub), encoding="utf-8")
        print(f"{split}_q{n}: {n} queries")


if __name__ == "__main__":
    main()
