"""Συγκεντρώνει τα results/<exp>/metrics_<split>.json σε έναν πίνακα.

Χρήση:
    python -m src.report --split test
Έξοδος: results/summary_<split>.csv (και markdown στην οθόνη).
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLS = ["ndcg@10", "ndcg@5", "mrr", "recall@5", "recall@10", "accuracy", "macro_f1"]


def collect(split):
    rows = []
    for path in sorted((ROOT / "results").glob(f"*/metrics_{split}.json")):
        res = json.loads(path.read_text(encoding="utf-8"))
        row = {"experiment": path.parent.name, **res["ranking"]}
        if "classification" in res:
            row["accuracy"] = res["classification"]["accuracy"]
            row["macro_f1"] = res["classification"]["macro_f1"]
        rows.append(row)
    df = pd.DataFrame(rows).set_index("experiment")
    return df.reindex(columns=[c for c in COLS if c in df]).sort_values("ndcg@10", ascending=False)


def evaluate_on_subset(split, subset_path):
    """Ξαναβαθμολογεί κάθε results/<exp>/predictions_<split>.parquet μόνο στα queries του υποσυνόλου.

    Γράφει results/<exp>/metrics_<split>_<tag>.json (tag = όνομα αρχείου υποσυνόλου, π.χ. test_q500 → q500).
    """
    from src.evaluate import evaluate, load_split
    from src.subsets import load_subset

    sub = load_subset(subset_path)
    assert sub["split"] == split, (sub["split"], split)
    tag = Path(subset_path).stem.split("_")[-1]
    qids = set(sub["query_ids"])
    gold = load_split(split)
    gold = gold[gold["query_id"].isin(qids)]
    for path in sorted((ROOT / "results").glob(f"*/predictions_{split}.parquet")):
        preds = pd.read_parquet(path)
        preds = preds[preds["query_id"].isin(qids)]
        if preds["query_id"].nunique() != len(qids):
            print(f"skip {path.parent.name}: δεν έχει predictions για όλο το υποσύνολο")
            continue
        res = {"experiment": path.parent.name, "subset": subset_path, **evaluate(preds, gold, seed=42)}
        (path.parent / f"metrics_{split}_{tag}.json").write_text(
            json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return tag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--subset", default=None,
                        help="π.χ. configs/subsets/test_q500.json: βαθμολόγηση μόνο σε αυτά τα queries")
    args = parser.parse_args()
    key = args.split
    if args.subset:
        key = f"{args.split}_{evaluate_on_subset(args.split, args.subset)}"
    df = collect(key)
    df.to_csv(ROOT / "results" / f"summary_{key}.csv")
    print(df.to_markdown(floatfmt=".4f") if _has_tabulate() else df.to_string())


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
