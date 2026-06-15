"""Assemble results/{zeroshot,aligned,transfer,analysis}.json into a markdown
report with the tables from EXPERIMENT_LOG (Table 1, transfer matrix, RSA,
hardest categories).

Run:  python src/make_report.py        # -> results/REPORT.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402

FAMILY = {k: v["family"] for k, v in C.MODEL_ZOO.items()}


def _load(name):
    p = C.RESULTS_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


def _fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def table1(zs, al) -> str:
    rows = ["| Model | Family | d | Zero-shot | Aligned | Δ | % ceiling |",
            "|---|---|---|---|---|---|---|"]
    models = sorted(set((zs or {})) | set((al or {})))
    for m in models:
        z = (zs or {}).get(m, {})
        a = (al or {}).get(m, {})
        zsa = z.get("acc")
        ali = a.get("aligned_test")
        delta = (ali - zsa) if (ali is not None and zsa is not None) else None
        pct = a.get("pct_ceiling")
        d = a.get("dim") or z.get("dim")
        rows.append(f"| {m} | {FAMILY.get(m,'?')} | {d or '—'} | {_fmt(zsa)} | "
                    f"{_fmt(ali)} | {_fmt(delta) if delta is not None else '—'} | "
                    f"{_fmt(pct,1)+'%' if pct else '—'} |")
    return "\n".join(rows)


def transfer_table(tr) -> str:
    if not tr:
        return "_no transfer.json yet_"
    names = tr["names"]
    M = tr["transfer_matrix"]
    head = "| W ↓ \\ target → | " + " | ".join(names) + " |"
    sep = "|" + "---|" * (len(names) + 1)
    rows = [head, sep]
    for i, src in enumerate(names):
        rows.append(f"| **{src}** | " + " | ".join(_fmt(M[i][j], 3)
                    for j in range(len(names))) + " |")
    base = tr.get("baseline", {})
    rows.append("")
    rows.append("Baseline (W=I): " + ", ".join(f"{k} {_fmt(v,3)}"
                for k, v in base.items()))
    rows.append(f"\n_shared_dim = {tr.get('shared_dim')}_")
    return "\n".join(rows)


def rsa_table(an) -> str:
    if not an or "rsa" not in an:
        return "_no analysis.json yet_"
    rows = ["| Model | RSA baseline | RSA aligned |", "|---|---|---|"]
    for m, v in an["rsa"].items():
        rows.append(f"| {m} | {_fmt(v.get('baseline'))} | {_fmt(v.get('aligned'))} |")
    return "\n".join(rows)


def hardest_categories(an, top=8) -> str:
    if not an or "category" not in an:
        return "_no analysis.json yet_"
    out = []
    for m, rec in an["category"].items():
        al = rec.get("aligned", {}).get("per_category", {})
        if not al:
            continue
        worst = sorted(al.items(), key=lambda kv: kv[1])[:top]
        out.append(f"**{m}** (lowest aligned acc): " +
                   ", ".join(f"{c} {v:.2f}" for c, v in worst))
    return "\n\n".join(out) if out else "_n/a_"


def robustness_table(rb) -> str:
    if not rb:
        return "_no robustness.json yet_"
    rows = ["| Model | seeds (mean±std) | best λ | image-disjoint | matched triplet | leakage gap |",
            "|---|---|---|---|---|---|"]
    for m, r in rb.items():
        s = r.get("seeds", {})
        ls = r.get("lambda_sweep", {})
        im = r.get("image_split", {})
        mean = s.get("mean")
        std = s.get("std")
        seedcell = f"{mean:.4f}±{std:.4f}" if mean is not None else "—"
        rows.append(f"| {m} | {seedcell} | {ls.get('best_lambda','—')} | "
                    f"{_fmt(im.get('aligned_test'))} | "
                    f"{_fmt(im.get('triplet_matched_test'))} | "
                    f"{_fmt(im.get('leakage_gap'))} |")
    return "\n".join(rows)


def main():
    zs, al = _load("zeroshot.json"), _load("aligned.json")
    tr, an = _load("transfer.json"), _load("analysis.json")
    rb = _load("robustness.json")

    md = f"""# Results Report

Human noise ceiling = {C.HUMAN_NOISE_CEILING:.3f} · chance = {C.CHANCE:.3f}

## Table 1 — Zero-shot vs aligned odd-one-out (test set)

{table1(zs, al)}

## Table 2 — Cross-model transfer (test acc; row = source W, col = target features)

{transfer_table(tr)}

## RSA vs human SPoSE embedding

{rsa_table(an)}

## Hardest categories after alignment

{hardest_categories(an)}

## Robustness (seeds / λ sweep / image-disjoint split)

{robustness_table(rb)}
"""
    out = C.RESULTS_DIR / "REPORT.md"
    out.write_text(md)
    print(f"wrote -> {out}")
    print(md)


if __name__ == "__main__":
    main()
