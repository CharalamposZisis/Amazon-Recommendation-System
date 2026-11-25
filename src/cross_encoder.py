"""Fine-tuned cross-encoder για E/S/C/I: το query και το product μπαίνουν μαζί στο ίδιο transformer.

- Είσοδος: "query" [SEP] "title | brand | color" (κομμένο στα max_length tokens).
- Έξοδος: 4 κλάσεις (E/S/C/I), cross-entropy, προαιρετικά με class weights.
- Εκπαίδευση μόνο στο train· επιλογή checkpoint (ανά epoch) με macro-F1 στο val.
- Ranking score = expected gain P(E) + 0.1 P(S) + 0.01 P(C), όπως στο src/classifiers.py.

Χρήση (GPU· τρέχει στο Colab):
    python -m src.cross_encoder --config configs/cross_encoder_minilm.yaml [--max-queries 200]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import yaml

from src.baselines import ROOT, load_config, load_splits, product_text
from src.classifiers import GOLD_COLS, to_predictions
from src.dense import subsample
from src.evaluate import LABELS, evaluate

LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}


def make_batches(df, cfg, shuffle, rng):
    idx = rng.permutation(len(df)) if shuffle else np.arange(len(df))
    bs = cfg["batch_size"] if shuffle else cfg["eval_batch_size"]
    for start in range(0, len(idx), bs):
        yield idx[start:start + bs]


def predict_proba(model, tok, df, texts, cfg, device):
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for b in make_batches(df, cfg, shuffle=False, rng=None):
            enc = tok(df["query"].values[b].tolist(), texts[b].tolist(), truncation="only_second",
                      max_length=cfg["max_length"], padding=True, return_tensors="pt").to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(**enc).logits
            out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def run(cfg, max_queries=None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    seed = cfg["seed"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = cfg["name"] + ("_smoke" if max_queries else "")
    out_dir = ROOT / "results" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_splits(cfg["data_dir"], ["train", "val", "test"])
    splits = {k: subsample(v, max_queries, seed) for k, v in splits.items()}
    if cfg.get("train_max_queries") and not max_queries:
        splits["train"] = subsample(splits["train"], cfg["train_max_queries"], seed)
    texts = {k: product_text(v, cfg["fields"]).str.slice(0, 1000).values for k, v in splits.items()}

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"], num_labels=len(LABELS), ignore_mismatched_sizes=True).to(device)

    train = splits["train"]
    y = torch.tensor(train["esci_label"].map(LABEL2ID).values)
    weight = None
    if cfg.get("class_weight") == "balanced":
        counts = np.bincount(y.numpy(), minlength=len(LABELS))
        weight = torch.tensor(len(y) / (len(LABELS) * counts), dtype=torch.float).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    steps = cfg["epochs"] * int(np.ceil(len(train) / cfg["batch_size"]))
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    print(f"device {device} · train pairs {len(train)} · steps {steps}")

    best = {"macro_f1": -1}
    history = []
    for epoch in range(cfg["epochs"]):
        model.train()
        t0 = time.time()
        for step, b in enumerate(make_batches(train, cfg, shuffle=True, rng=rng)):
            enc = tok(train["query"].values[b].tolist(), texts["train"][b].tolist(),
                      truncation="only_second", max_length=cfg["max_length"],
                      padding=True, return_tensors="pt").to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(**enc).logits
            loss = loss_fn(logits.float(), y[b].to(device))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if step % 500 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f} ({time.time() - t0:.0f}s)")

        val = splits["val"]
        pv = to_predictions(val, predict_proba(model, tok, val, texts["val"], cfg, device), LABELS)
        f1 = evaluate(pv, val[GOLD_COLS], seed=seed, by_source=False)["classification"]["macro_f1"]
        history.append({"epoch": epoch, "val_macro_f1": f1, "seconds": round(time.time() - t0)})
        print(history[-1])
        if f1 > best["macro_f1"]:
            best = {"macro_f1": f1, "epoch": epoch}
            torch.save(model.state_dict(), out_dir / "best.pt")

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    (out_dir / "best.pt").unlink()  # μεγάλο αρχείο· δεν το κρατάμε
    for split in cfg["eval_splits"]:
        df = splits[split]
        t0 = time.time()
        preds = to_predictions(df, predict_proba(model, tok, df, texts[split], cfg, device), LABELS)
        preds.to_parquet(out_dir / f"predictions_{split}.parquet", index=False)
        res = {"experiment": name, "best_epoch": best["epoch"], "history": history,
               "predict_seconds": round(time.time() - t0, 1),
               **evaluate(preds, df[GOLD_COLS], seed=seed)}
        (out_dir / f"metrics_{split}.json").write_text(
            json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        c = res["classification"]
        print(split, f"acc {c['accuracy']} macro-F1 {c['macro_f1']} F1 {c['f1_per_class']}",
              f"nDCG@10 {res['ranking']['ndcg@10']}")

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
