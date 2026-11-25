"""Dense (bi-encoder) re-ranking χωρίς εκπαίδευση: score = cosine(embedding query, embedding product).

Κάθε μοναδικό query και product κωδικοποιείται μία φορά με sentence-transformers.
Δεν χρησιμοποιεί ετικέτες, άρα δεν υπάρχει εκπαίδευση: μετράμε πόσο βοηθά το «νόημα»
σε σχέση με τις κοινές λέξεις (BM25).

Χρήση (θέλει GPU για λογικό χρόνο· τρέχει στο Colab):
    python -m src.dense --config configs/dense_minilm.yaml [--max-queries 200]
"""
import argparse
import json
import time

import numpy as np
import yaml

from src.baselines import ROOT, load_config, load_splits, product_text
from src.classifiers import GOLD_COLS
from src.evaluate import evaluate


def subsample(df, max_queries, seed):
    """Για γρήγορο smoke test: κρατά max_queries τυχαία queries (ίδια για κάθε τρέξιμο)."""
    if not max_queries:
        return df
    qids = np.sort(df["query_id"].unique())
    keep = np.random.default_rng(seed).choice(qids, size=min(max_queries, len(qids)), replace=False)
    return df[df["query_id"].isin(keep)].reset_index(drop=True)


def encode(model, texts, cfg, prefix=""):
    return model.encode([prefix + t for t in texts], batch_size=cfg["batch_size"],
                        normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=True)


def run(cfg, max_queries=None):
    from sentence_transformers import SentenceTransformer

    seed = cfg["seed"]
    splits = load_splits(cfg["data_dir"], cfg["eval_splits"])
    name = cfg["name"] + ("_smoke" if max_queries else "")
    out_dir = ROOT / "results" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(cfg["model"])
    model.max_seq_length = cfg["max_seq_length"]
    print(f"model {cfg['model']} on {model.device}")

    for split in cfg["eval_splits"]:
        df = subsample(splits[split], max_queries, seed)
        t0 = time.time()
        queries = df["query"].drop_duplicates()
        products = df.drop_duplicates("product_id")
        q_emb = encode(model, queries.tolist(), cfg, cfg.get("query_prefix", ""))
        p_emb = encode(model, product_text(products, cfg["fields"]).tolist(), cfg,
                       cfg.get("doc_prefix", ""))
        q_pos = dict(zip(queries, range(len(queries))))
        p_pos = dict(zip(products["product_id"], range(len(products))))
        qi = df["query"].map(q_pos).values
        pi = df["product_id"].map(p_pos).values
        score = np.einsum("ij,ij->i", q_emb[qi], p_emb[pi])  # normalised → cosine

        preds = df[["query_id", "product_id"]].assign(score=score)
        preds.to_parquet(out_dir / f"predictions_{split}.parquet", index=False)
        res = {"experiment": name, "seconds": round(time.time() - t0, 1),
               **evaluate(preds, df[GOLD_COLS], seed=seed)}
        (out_dir / f"metrics_{split}.json").write_text(
            json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(split, json.dumps(res["ranking"]))

    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-queries", type=int, default=None, help="smoke test σε λίγα queries")
    args = parser.parse_args()
    run(load_config(args.config), args.max_queries)


if __name__ == "__main__":
    main()
