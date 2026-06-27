"""Evaluate a prediction file against gold labels.

Usage:
  python evaluate.py --pred results/dev_pred.json --gold data/train.json \
      --dev_split data/dev_split.json

  python evaluate.py --pred results/dev_pred.json --gold data/train.json
"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tools import evaluate, format_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred",      required=True, help="Prediction JSON file")
    p.add_argument("--gold",      required=True, help="Gold JSON file (train.json)")
    p.add_argument("--dev_split", default="",    help="dev_split.json to select dev subset")
    p.add_argument("--limit",     type=int, default=0)
    args = p.parse_args()

    preds = json.load(open(args.pred))
    gold_all = json.load(open(args.gold))

    if args.dev_split:
        split = json.load(open(args.dev_split))
        gold = [gold_all[i] for i in split["dev_idx"]]
    else:
        gold = gold_all

    n = min(len(preds), len(gold))
    if args.limit: n = min(n, args.limit)
    preds, gold = preds[:n], gold[:n]

    mismatched = sum(1 for p, g in zip(preds, gold) if p.get("text") != g.get("text"))
    if mismatched:
        print(f"WARN: {mismatched}/{n} docs have mismatched text")

    m = evaluate(preds, gold)
    print(format_metrics(m))


if __name__ == "__main__":
    main()
