"""Paired bootstrap για τη διαφορά δύο μοντέλων σε μετρική ranking ανά query.

Ξαναδειγματίζει (με επανάθεση) τα queries B φορές και μετρά πόσο σταθερή είναι η μέση διαφορά
A − B. Δίνει 95% διάστημα εμπιστοσύνης και αμφίπλευρο p-value. Οι μετρικές ανά query βγαίνουν
από το src/evaluate.py (ίδιο tie-break με seed).

Χρήση:
    python -m src.significance --split test --subset configs/subsets/test_q500.json \
        --pairs llm_zeroshot_qwen7b:cross_encoder_minilm llm_zeroshot_qwen7b:llm_rag_qwen7b
Έξοδος: results/significance_<split>[_<tag>].csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import attach_labels, load_split, per_query_ranking, rank_within_query
from src.subsets import load_subset

ROOT = Path(__file__).resolve().parents[1]


def per_query(exp, split, gold, qids, metric, seed):
    preds = pd.read_parquet(ROOT / "results" / exp / f"predictions_{split}.parquet")
    if qids is not None:
        preds = preds[preds["query_id"].isin(qids)]
    df = attach_labels(preds, gold)
    return per_query_ranking(rank_within_query(df, seed))[metric]


def paired_bootstrap(a, b, n_boot=10000, seed=0):
    d = (a - b).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean()))
    return {"delta": d.mean(), "ci_low": lo, "ci_high": hi, "p_value": p, "n_queries": len(d),
            "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--subset", default=None)
    parser.add_argument("--pairs", nargs="+", required=True, help="A:B, π.χ. llm_zeroshot_qwen7b:llm_rag_qwen7b")
    parser.add_argument("--metric", default="ndcg@10")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0, help="seed του bootstrap")
    args = parser.parse_args()

    gold = load_split(args.split)
    qids, key = None, args.split
    if args.subset:
        sub = load_subset(args.subset)
        assert sub["split"] == args.split, (sub["split"], args.split)
        qids = set(sub["query_ids"])
        gold = gold[gold["query_id"].isin(qids)]
        key += "_" + Path(args.subset).stem.split("_")[-1]

    cache, rows = {}, []
    for pair in args.pairs:
        a, b = pair.split(":")
        for e in (a, b):
            if e not in cache:
                cache[e] = per_query(e, args.split, gold, qids, args.metric, seed=42)
        res = paired_bootstrap(cache[a], cache[b], args.n_boot, args.seed)
        rows.append({"A": a, "B": b, "metric": args.metric, **res})
    df = pd.DataFrame(rows).round(4)
    df.to_csv(ROOT / "results" / f"significance_{key}.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
