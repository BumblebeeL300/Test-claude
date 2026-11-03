# Distill examples

* `docs/` - two short documents for `qa_from_docs` (`--docs examples/distill/docs`).
* `recipe_faq_strict.json` - a custom recipe (any JSON with the recipe schema works).
* `eval_holdout.jsonl` - a held-out eval set; some questions overlap what the simulator generates, so
  decontamination visibly drops samples.
* `quickstart.py` - the Python API end to end.

```bash
python -m nimlab.distill.cli run instruction_diversity --eval-set examples/distill/eval_holdout.jsonl --n 80 --out data --mock
python -m nimlab.distill.cli run examples/distill/recipe_faq_strict.json --docs examples/distill/docs --n 40 --mock
```

See [docs/distill.md](../../docs/distill.md).
