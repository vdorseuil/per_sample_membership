"""LaTeX (booktabs) table of the top-10 SST2 sentences by |leverage score|
for each BERT variant (model_210 where available; model_209 fallback for
the linear-probe variant since model_210 is not in its used_models).

Two side-by-side columns per BERT variant: Newton and Influence rankings.
Sentences are LaTeX-escaped and truncated to fit a NeurIPS-width table.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from datasets import load_dataset as load_hf_dataset

HERE = Path(__file__).resolve().parent
FIG = HERE.parent
REPO = FIG.parent

TGT = REPO / "experiments" / "target_models"

BERT_VARIANTS = [
    ("Bert",        r"BERT (full ft.)",     "model_210"),
    ("Bert_freeze", r"BERT (linear probe)", "model_209"),  # 210 not converged
]

TOP_K = 10
MAX_LEN = 70  # characters per sentence after truncation

_LATEX_ESCAPE = {
    "\\": r"\textbackslash{}",
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    "<": r"\textless{}", ">": r"\textgreater{}",
}


def latex_escape(s: str) -> str:
    return "".join(_LATEX_ESCAPE.get(c, c) for c in s)


def latest_run(model_dir: Path) -> Path:
    base = model_dir / "leverage_score"
    cands = [r for r in base.iterdir()
             if r.is_dir() and (r / "leverage_newton.pkl").exists()
             and (r / "leverage_influence.pkl").exists()]
    return sorted(cands)[-1]


def top_indices(scores: dict[int, float], k: int) -> list[int]:
    items = [(i, abs(v)) for i, v in scores.items() if np.isfinite(v)]
    items.sort(key=lambda t: -t[1])
    return [i for i, _ in items[:k]]


def short(sent: str) -> str:
    sent = sent.strip().replace("`", "'")
    if len(sent) > MAX_LEN:
        sent = sent[: MAX_LEN - 3] + "..."
    return latex_escape(sent)


def main() -> None:
    sst2 = load_hf_dataset(
        "glue", "sst2",
        cache_dir=str(REPO / "cache" / "huggingface" / "datasets"),
        split="train",
    )

    for ds_key, label, model in BERT_VARIANTS:
        run = latest_run(TGT / ds_key / "outputs" / model)
        with open(run / "leverage_newton.pkl", "rb") as fh:
            newton = pickle.load(fh)
        with open(run / "leverage_influence.pkl", "rb") as fh:
            influence = pickle.load(fh)

        top_n = top_indices(newton, TOP_K)
        top_i = top_indices(influence, TOP_K)
        sents_n = [short(sst2[int(i)]["sentence"]) for i in top_n]
        sents_i = [short(sst2[int(i)]["sentence"]) for i in top_i]

        print(r"\begin{table}[t]")
        print(r"\centering")
        print(r"\scriptsize")
        print(r"\setlength{\tabcolsep}{4pt}")
        print(r"\renewcommand{\arraystretch}{1.15}")
        print(rf"\caption{{Top-{TOP_K} highest-$|$leverage$|$ training sentences "
              rf"for {label} on SST2, ranked by Newton (left) and Influence (right) "
              rf"surrogates ({model}).}}")
        print(rf"\label{{tab:bert_top_sents_{ds_key.lower()}}}")
        print(r"\begin{tabular}{@{}c p{0.42\linewidth} p{0.42\linewidth}@{}}")
        print(r"\toprule")
        print(r"Rank & Newton & Influence \\")
        print(r"\midrule")
        for r in range(TOP_K):
            print(rf"{r + 1} & {sents_n[r]} & {sents_i[r]} \\")
        print(r"\bottomrule")
        print(r"\end{tabular}")
        print(r"\end{table}")
        print()


if __name__ == "__main__":
    main()
