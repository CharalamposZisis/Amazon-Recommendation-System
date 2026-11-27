"""LLM με Retrieval-Augmented Generation για ταξινόμηση E/S/C/I (χωρίς εκπαίδευση του LLM).

Για κάθε ζεύγος (query, product) το prompt περιέχει:
1. τους ορισμούς των 4 ετικετών του ESCI,
2. **retrieved παραδείγματα** (k_examples > 0): λυμένα ζεύγη από το TRAIN με παρόμοιο query
   (BM25 πάνω στα train queries) και παρόμοιο τίτλο προϊόντος (TF-IDF cosine),
3. τα στοιχεία του προϊόντος από τον κατάλογο (title, brand, color, bullets, description).
Με k_examples = 0 είναι το zero-shot LLM (ablation: πόσο βοηθά το retrieval).

Το LLM δεν παράγει ελεύθερο κείμενο: διαβάζουμε την πιθανότητα του επόμενου token για τα
γράμματα E/S/C/I (ένα forward pass ανά ζεύγος) → P(E), P(S), P(C), P(I).
- pred_label = argmax, score (ranking) = P(E) + 0.1 P(S) + 0.01 P(C).
- Δεύτερη εκδοχή «_calibrated»: σταθερές μετατοπίσεις στα log-probs ανά κλάση, που επιλέγονται
  ώστε να μεγιστοποιούν το macro-F1 στο VAL υποσύνολο και μετά εφαρμόζονται στο test.

Αξιολόγηση σε σταθερά υποσύνολα (src/subsets.py) γιατί ένα 7B LLM είναι αργό.

Χρήση (GPU· Colab):
    python -m src.llm_rag --config configs/llm_rag_qwen7b.yaml [--max-queries 5]
"""
import argparse
import json
import re
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

from src.baselines import BM25, HTML_TAG, ROOT, TOKEN_PATTERN, load_config, load_splits
from src.classifiers import GOLD_COLS, to_predictions
from src.evaluate import LABELS, evaluate
from src.subsets import load_subset

SYSTEM = "You are an expert annotator of Amazon product search relevance."
DEFINITIONS = """Classify how well the product matches the shopping query, using exactly one label:
E (Exact): the product is relevant for the query and satisfies all the query specifications.
S (Substitute): the product is somewhat relevant; it fails some aspects of the query but can be used as a functional substitute.
C (Complement): the product does not fulfil the query, but it could be used in combination with an exact item.
I (Irrelevant): the product is irrelevant to the query or fails a central aspect of it."""
WS = re.compile(r"\s+")


def clip(value, n):
    if not isinstance(value, str):
        return ""
    return WS.sub(" ", HTML_TAG.sub(" ", value)).strip()[:n]


def product_block(row, max_chars):
    parts = [("Title", row["product_title"], max_chars["title"]),
             ("Brand", row["product_brand"], 60),
             ("Color", row["product_color"], 60),
             ("Bullet points", row["product_bullet_point"], max_chars["bullets"]),
             ("Description", row["product_description"], max_chars["description"])]
    return "\n".join(f"{k}: {clip(v, n)}" for k, v, n in parts if clip(v, n))


class ExampleRetriever:
    """Λυμένα παραδείγματα από το train: πρώτα παρόμοια queries (BM25), μετά παρόμοιοι τίτλοι."""

    def __init__(self, train, pool_queries):
        self.train = train.reset_index(drop=True)
        self.pool_queries = pool_queries
        uq = self.train.drop_duplicates("query_id")[["query_id", "query"]]
        self.q_ids = uq["query_id"].values
        self.bm25 = BM25().fit(uq["query"].values)
        self.q_doc_T = self.bm25.doc_matrix.T.tocsr()
        self.by_query = self.train.groupby("query_id").indices
        titles = self.train["product_title"].fillna("")
        # απλοί numpy πίνακες: η επιλογή γραμμών σε arrow string στήλες του pandas είναι πολύ αργή
        self.ex_cols = {c: self.train[c].fillna("").to_numpy(dtype=object)
                        for c in ["query", "product_title", "esci_label"]}
        self.title_vec = TfidfVectorizer(token_pattern=TOKEN_PATTERN, sublinear_tf=True, min_df=2)
        self.train_titles = self.title_vec.fit_transform(titles).tocsr()  # μία φορά για όλο το train

    def examples_for(self, df, k):
        """dict: index γραμμής του df → λίστα από k (query, product_title, esci_label) του train."""
        out = {}
        for query, group in df.groupby("query"):
            sims = (self.bm25.query_matrix([query]) @ self.q_doc_T).toarray().ravel()
            top_q = self.q_ids[np.argsort(-sims)[:self.pool_queries]]
            pool = np.concatenate([self.by_query[q] for q in top_q])
            C = self.title_vec.transform(group["product_title"].fillna(""))
            S = (C @ self.train_titles[pool].T).toarray()
            best = pool[np.argsort(-S, axis=1)[:, :k]]
            for row_idx, rows in zip(group.index, best):
                out[row_idx] = list(zip(*(self.ex_cols[c][rows] for c in ["query", "product_title", "esci_label"])))
        return out


def build_prompt(tok, row, examples, cfg):
    text = DEFINITIONS + "\n\n"
    if examples:
        text += "Here are labelled examples of similar queries and products:\n\n"
        for ex_query, ex_title, ex_label in examples:
            text += f"Query: {ex_query}\nProduct title: {clip(ex_title, 150)}\nLabel: {ex_label}\n\n"
        text += "Now classify this pair.\n\n"
    text += (f"Query: {row['query']}\nProduct:\n{product_block(row, cfg['max_chars'])}\n\n"
             "Answer with a single letter: E, S, C or I.")
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def label_probs(model, tok, prompts, cfg):
    """P(E,S,C,I) από τα logits του επόμενου token· batches ταξινομημένα κατά μήκος."""
    import torch

    label_ids = []
    for lab in LABELS:
        ids = tok.encode(lab, add_special_tokens=False)
        assert len(ids) == 1, (lab, ids)
        label_ids.append(ids[0])
    lengths = np.array([len(p) for p in prompts])
    order = np.argsort(-lengths)
    out = np.zeros((len(prompts), len(LABELS)), dtype=np.float32)
    t0, done = time.time(), 0
    bs = cfg["batch_size"]
    for start in range(0, len(order), bs):
        idx = order[start:start + bs]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            # μόνο η τελευταία θέση: τα logits όλου του λεξιλογίου (~152k) για κάθε token δεν χωράνε
            last = model.model(**enc).last_hidden_state[:, -1]
            logits = model.lm_head(last)[:, label_ids].float()
        out[idx] = torch.log_softmax(logits, dim=-1).cpu().numpy()
        done += len(idx)
        if start // bs % 50 == 0:
            rate = done / (time.time() - t0)
            print(f"  {done}/{len(prompts)} ζεύγη · {rate:.1f}/s · ETA {(len(prompts) - done) / rate / 60:.0f} λεπτά",
                  flush=True)
    return out  # log-probs


def fit_offsets(logp, gold_labels, rounds=3, grid=np.arange(-4, 4.01, 0.25)):
    """Μετατοπίσεις ανά κλάση (E σταθερό = 0) που μεγιστοποιούν το macro-F1 (μόνο στο val)."""
    from sklearn.metrics import f1_score

    y = np.asarray(gold_labels)
    b = np.zeros(len(LABELS))
    def f1(bb):
        pred = np.asarray(LABELS)[(logp + bb).argmax(1)]
        return f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)
    best = f1(b)
    for _ in range(rounds):
        for j in range(1, len(LABELS)):
            for v in grid:
                cand = b.copy()
                cand[j] = v
                score = f1(cand)
                if score > best + 1e-9:
                    best, b = score, cand
    return b, best


def run(cfg, max_queries=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    torch.manual_seed(cfg["seed"])
    name = cfg["name"] + ("_smoke" if max_queries else "")
    out_dir = ROOT / "results" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = load_splits(cfg["data_dir"], ["train", "val", "test"])

    eval_sets, tags = {}, {}
    for split in ["val", "test"]:
        sub = load_subset(cfg[f"{split}_subset"])
        qids = sub["query_ids"][:max_queries] if max_queries else sub["query_ids"]
        df = splits[split]
        eval_sets[split] = df[df["query_id"].isin(set(qids))].reset_index(drop=True)
        tags[split] = cfg[f"{split}_subset"].split("_")[-1].removesuffix(".json")  # π.χ. q500

    retriever = ExampleRetriever(splits["train"], cfg["example_pool_queries"]) if cfg["k_examples"] else None

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    tok.padding_side = "left"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16) if cfg["load_in_4bit"] else None
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], quantization_config=quant,
                                                 torch_dtype=torch.float16, device_map="auto")
    model.eval()

    logps = {}
    for split, df in eval_sets.items():
        t0 = time.time()
        exs = retriever.examples_for(df, cfg["k_examples"]) if retriever else {}
        prompts = [build_prompt(tok, row, exs.get(i), cfg) for i, row in df.iterrows()]
        if split == "test":
            (out_dir / "prompt_examples.txt").write_text("\n\n=====\n\n".join(prompts[:3]), encoding="utf-8")
        n_tok = [len(tok(p).input_ids) for p in prompts[:200]]
        print(f"{split}: {len(df)} ζεύγη, {df['query_id'].nunique()} queries, ~{np.mean(n_tok):.0f} tokens/prompt")
        logps[split] = label_probs(model, tok, prompts, cfg)
        print(f"{split}: {time.time() - t0:.0f}s")

    offsets, val_f1 = fit_offsets(logps["val"], eval_sets["val"]["esci_label"])
    print(f"calibration offsets {dict(zip(LABELS, offsets.round(2)))} · val macro-F1 {val_f1:.4f}")

    for variant, b in [("", np.zeros(len(LABELS))), ("_calibrated", offsets)]:
        vdir = ROOT / "results" / (name + variant)
        vdir.mkdir(parents=True, exist_ok=True)
        for split, df in eval_sets.items():
            lp = logps[split] + b
            proba = np.exp(lp - lp.max(1, keepdims=True))
            proba /= proba.sum(1, keepdims=True)
            preds = to_predictions(df, proba, LABELS)
            preds.to_parquet(vdir / f"predictions_{split}.parquet", index=False)
            res = {"experiment": name + variant, "subset": cfg[f"{split}_subset"],
                   "offsets": dict(zip(LABELS, b.round(3).tolist())),
                   **evaluate(preds, df[GOLD_COLS], seed=cfg["seed"])}
            (vdir / f"metrics_{split}_{tags[split]}.json").write_text(
                json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
            c = res["classification"]
            print(name + variant, split, f"acc {c['accuracy']} macro-F1 {c['macro_f1']}",
                  f"F1 {c['f1_per_class']} nDCG@10 {res['ranking']['ndcg@10']}")
        with open(vdir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-queries", type=int, default=None, help="smoke test σε λίγα queries")
    args = parser.parse_args()
    run(load_config(args.config), args.max_queries)


if __name__ == "__main__":
    main()
