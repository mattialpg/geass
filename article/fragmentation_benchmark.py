#!/usr/bin/env python3
"""Reproduce the Geass/BRICS/MetaDi benchmark and its publication figures.

The method calls, input filtering, sampling and palette follow
article/benchmark.ipynb. Unlike that notebook, this script keeps the
run data, numerical summaries and figures in a single output directory.

From the repository root::

    python article/fragmentation_benchmark.py
    python article/fragmentation_benchmark.py --sample-size 10000 --repeats 1
    python article/fragmentation_benchmark.py --from-csv article/fragmentation_benchmark.csv

Default settings reproduce the *configured notebook experiment* (300 sampled
molecules, seed 42, five repeats), not an independently verified 10,000-
compound benchmark. In the notebook, Geass is run with annotation disabled;
this remains the default and can be overridden with --geass-annotate.

Outputs under article/fragmentation_benchmark/ (unless overridden):
    runs.csv                      Per-execution outcomes, including failures
    fragments.jsonl               Per-execution fragment SMILES (optional)
    granularity.csv               One row per compound/method, first replicate
    runtime_summary.csv           Timings from all executions
    granularity_summary.csv       Fragment metrics from successful first runs
    metadata.json                 Sampling and method configuration
    fig3a.{svg,png}, fig3b.{svg,png}, fig3.{svg,png}

When using --from-csv, only summaries/figures are produced. The supplied CSV
must contain the columns emitted by the notebook or this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rdkit import Chem
from rdkit.Chem import BRICS, Descriptors

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTICLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

METHODS = ("GEASS", "BRICS", "MetaDi")
PALETTE = {
    "GEASS": ("#0D4A70", "#6189A2"),
    "BRICS": ("#A9475B", "#BD7281"),
    "MetaDi": ("#CF9B26", "#DAB359"),
}
RUN_COLUMNS = (
    "replicate", "ID", "method", "setup_s", "runtime_s", "total_s",
    "ok", "error", "n_frags", "mean_atoms_per_frag", "heavy_atoms",
    "mol_wt",
)


def _load_dataset(path: Path, n: int, seed: int) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"NP Atlas input not found: {path}")
    df = pd.read_csv(path, sep="\t", usecols=["npaid", "compound_smiles"])
    df = df.dropna(subset=["npaid", "compound_smiles"])
    # The notebook excludes multi-component SMILES before sampling.
    df = df[~df["compound_smiles"].str.contains(r"\.", regex=True)].reset_index(drop=True)
    df = df.rename(columns={"npaid": "ID", "compound_smiles": "ISOSMILES"})
    if n > len(df):
        raise ValueError(f"Requested {n} compounds; only {len(df)} remain after filtering")
    sample = df.sample(n, random_state=seed).reset_index(drop=True)
    if sample["ID"].duplicated().any():
        raise ValueError("Duplicate NPA IDs in the sampled set; cannot safely resume")
    mols = [Chem.MolFromSmiles(smi) for smi in sample["ISOSMILES"]]
    invalid = [sample.loc[i, "ID"] for i, mol in enumerate(mols) if mol is None]
    if invalid:
        raise ValueError(f"Sample includes invalid SMILES for NPA IDs: {invalid[:10]}")
    sample["ROMol"] = mols
    return sample


def _run_once(method: str, mol: Chem.Mol, smi: str, *, annotate: bool, metadi_limit: float):
    """Return fragments, setup seconds, processing seconds, and failure detail.

    Instantiation is outside runtime_s, as in the original notebook. Setup
    errors are recorded without discarding the sample or aborting all runs.
    """
    setup_start = time.perf_counter()
    try:
        if method == "GEASS":
            from new_geass import Geass
            worker = Geass(annotate=annotate)
        elif method == "BRICS":
            worker = None
        elif method == "MetaDi":
            from metadisassembler import MetaDisassembler
            worker = MetaDisassembler()
            worker.time_limit = metadi_limit
        else:
            raise ValueError(f"Unrecognised method: {method}")
    except Exception as exc:
        return [], time.perf_counter() - setup_start, 0.0, f"setup: {type(exc).__name__}: {exc}"
    setup_s = time.perf_counter() - setup_start

    start = time.perf_counter()
    try:
        if method == "GEASS":
            frags = worker.get_fragments(Chem.Mol(mol))
        elif method == "BRICS":
            # BRICS returns *distinct terminal fragment types*, not occurrences.
            frags = list(BRICS.BRICSDecompose(Chem.Mol(mol), returnMols=True))
        else:
            if not worker.input_query(smi):
                raise ValueError("MetaDi rejected the query SMILES")
            inchis = worker.disassemble(return_inchis=True, result_id=0, write_output=False) or []
            # Preserve the notebook's convention of discarding unparsable InChIs.
            frags = [frag for inchi in inchis
                     if isinstance(inchi, str) and inchi.strip()
                     and (frag := Chem.MolFromInchi(inchi)) is not None]
        frags = list(frags or [])
        error = "" if frags else "no_output (possibly a MetaDi time limit)"
    except Exception as exc:
        frags, error = [], f"{type(exc).__name__}: {exc}"
    return frags, setup_s, time.perf_counter() - start, error


def _run_benchmark(args: argparse.Namespace, output: Path) -> pd.DataFrame:
    sample = _load_dataset(args.input, args.sample_size, args.seed)
    output.mkdir(parents=True, exist_ok=True)
    run_path = output / "runs.csv"
    fragments_path = output / "fragments.jsonl"
    metadata_path = output / "metadata.json"
    expected = {
        "input": str(args.input.resolve()),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "repeats": args.repeats,
        "methods": list(args.methods),
        "geass_annotate": args.geass_annotate,
        "metadi_time_limit_s": None if args.metadi_time_limit == 0 else args.metadi_time_limit,
        "save_fragments": not args.no_save_fragments,
        "runtime_definition": "method processing; object setup excluded",
        "brics_fragment_definition": "distinct terminal types (BRICSDecompose)",
        "granularity_definition": "first replicate for each compound/method, valid outputs only",
    }
    if args.overwrite:
        for file in (run_path, fragments_path, metadata_path):
            file.unlink(missing_ok=True)
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if any(existing.get(k) != v for k, v in expected.items()):
            raise ValueError("Existing run metadata differ. Choose another --output-dir or use --overwrite.")
    elif run_path.exists():
        raise ValueError("runs.csv exists without metadata.json. Choose another --output-dir or use --overwrite.")
    else:
        metadata_path.write_text(
            json.dumps({**expected, "created_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n",
            encoding="utf-8",
        )
    if run_path.exists():
        old = pd.read_csv(run_path)
        completed = {(int(row.replicate), str(row.ID), str(row.method)) for row in old.itertuples()}
    else:
        completed = set()
    required = args.sample_size * args.repeats * len(args.methods)
    print(f"Sample: {args.sample_size} compounds; methods: {', '.join(args.methods)}; "
          f"repeats: {args.repeats}; completed: {len(completed)}/{required}", flush=True)

    with run_path.open("a", newline="", encoding="utf-8") as out_csv:
        writer = csv.DictWriter(out_csv, fieldnames=RUN_COLUMNS)
        if out_csv.tell() == 0:
            writer.writeheader()
        with fragments_path.open("a", encoding="utf-8") if not args.no_save_fragments else _NullContext() as out_jsonl:
            for repeat in range(1, args.repeats + 1):
                for r in sample.itertuples(index=False):
                    for method in args.methods:
                        key = (repeat, str(r.ID), method)
                        if key in completed:
                            continue
                        frags, setup_s, runtime_s, error = _run_once(
                            method, r.ROMol, r.ISOSMILES,
                            annotate=args.geass_annotate,
                            metadi_limit=(math.inf if args.metadi_time_limit == 0 else args.metadi_time_limit),
                        )
                        counts = [frag.GetNumAtoms() for frag in frags]
                        row = {
                            "replicate": repeat,
                            "ID": r.ID,
                            "method": method,
                            "setup_s": setup_s,
                            "runtime_s": runtime_s,
                            "total_s": setup_s + runtime_s,
                            "ok": bool(frags),
                            "error": error,
                            "n_frags": len(frags),
                            "mean_atoms_per_frag": float(np.mean(counts)) if counts else "",
                            "heavy_atoms": r.ROMol.GetNumHeavyAtoms(),
                            "mol_wt": Descriptors.MolWt(r.ROMol),
                        }
                        writer.writerow(row)
                        out_csv.flush()  # Preserve progress through long MetaDi searches.
                        if out_jsonl is not None:
                            out_jsonl.write(json.dumps({
                                "replicate": repeat, "ID": str(r.ID), "method": method,
                                "ok": bool(frags), "error": error,
                                "fragment_smiles": [Chem.MolToSmiles(f, canonical=True, isomericSmiles=True)
                                                    for f in frags],
                            }) + "\n")
                            out_jsonl.flush()
                        completed.add(key)
                        print(f"{repeat}/{args.repeats} {r.ID:<12} {method:<6} "
                              f"{'ok' if frags else 'no output':<9} "
                              f"{len(frags):>3} fragments in {runtime_s:.2f}s", flush=True)
    return pd.read_csv(run_path)


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *_):
        return False


def _clean_runs(raw: pd.DataFrame) -> pd.DataFrame:
    mandatory = {"ID", "method", "runtime_s", "ok", "n_frags", "mean_atoms_per_frag"}
    missing = mandatory.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing benchmark CSV columns: {sorted(missing)}")
    df = raw.copy()
    df["ID"] = df["ID"].astype(str)
    df["method"] = df["method"].astype(str)
    df["ok"] = df["ok"].astype(str).str.lower().eq("true")
    if "replicate" not in df.columns:
        df["replicate"] = 1
    for col in ("replicate", "runtime_s", "n_frags", "mean_atoms_per_frag"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["replicate", "runtime_s", "n_frags"]].isna().any().any():
        raise ValueError("Invalid replicate, runtime or fragment-count values in benchmark results")
    if df.duplicated(["ID", "method", "replicate"]).any():
        raise ValueError("Duplicate ID/method/replicate rows: data must be unambiguous")
    return df


def _summary(runs: pd.DataFrame, output: Path):
    """Separate all-attempt runtime from first-attempt successful granularity."""
    first = runs.sort_values("replicate", kind="stable").drop_duplicates(["ID", "method"], keep="first")
    valid = first.loc[first["ok"] & first["n_frags"].gt(0)].copy()
    first.to_csv(output / "granularity.csv", index=False)

    runtime = runs.groupby("method", sort=False).agg(
        n_executions=("runtime_s", "size"),
        median_runtime_s=("runtime_s", "median"),
        mean_runtime_s=("runtime_s", "mean"),
        sd_runtime_s=("runtime_s", "std"),
        p95_runtime_s=("runtime_s", lambda x: x.quantile(0.95)),
        max_runtime_s=("runtime_s", "max"),
    )
    coverage = first.groupby("method", sort=False).agg(
        benchmark_size=("ID", "size"),
        valid_count=("ok", "sum"),
        valid_fraction=("ok", "mean"),
    )
    runtime = coverage.join(runtime)
    runtime.to_csv(output / "runtime_summary.csv")

    granularity = valid.groupby("method", sort=False).agg(
        valid_molecules=("ID", "size"),
        mean_fragments_per_molecule=("n_frags", "mean"),
        median_fragments_per_molecule=("n_frags", "median"),
        mean_atoms_per_fragment=("mean_atoms_per_frag", "mean"),
        median_atoms_per_fragment=("mean_atoms_per_frag", "median"),
    )
    granularity.to_csv(output / "granularity_summary.csv")
    return first, valid, runtime, granularity


def _style(fig: go.Figure, *, width: int, height: int):
    fig.update_layout(
        width=width, height=height, template="plotly_white",
        font=dict(family="Arial", color="black", size=17),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=75, r=25, t=45, b=70),
        legend=dict(font=dict(size=15), bgcolor="rgba(255,255,255,0.85)"),
    )
    fig.update_xaxes(showline=True, linecolor="black", linewidth=1.5,
                     showgrid=True, gridcolor="#D3D3D3", zeroline=False)
    fig.update_yaxes(showline=True, linecolor="black", linewidth=1.5,
                     showgrid=True, gridcolor="#D3D3D3", zeroline=False)
    return fig


def _runtime_traces(fig: go.Figure, runs: pd.DataFrame, *, subplot: bool = False):
    for method in METHODS:
        vals = runs.loc[runs["method"].eq(method), "runtime_s"].dropna().to_numpy(dtype=float)
        vals = np.sort(vals[vals > 0])
        if not len(vals):
            continue
        trace = go.Scatter(
            x=vals, y=np.arange(1, len(vals) + 1) / len(vals),
            mode="lines", line=dict(color=PALETTE[method][0], width=2.5),
            name="Geass" if method == "GEASS" else method,
        )
        if subplot:
            fig.add_trace(trace, row=1, col=1)
        else:
            fig.add_trace(trace)
    axis = dict(type="log", title_text="Runtime per molecule (s, log scale)",
                tickvals=[1e-3, 1e-2, 1e-1, 1, 10, 100, 1000])
    if subplot:
        fig.update_xaxes(**axis, row=1, col=1)
        fig.update_yaxes(title_text="ECDF", range=[0, 1.01], row=1, col=1)
    else:
        fig.update_xaxes(**axis)
        fig.update_yaxes(title_text="ECDF", range=[0, 1.01])


def _granularity_traces(fig: go.Figure, valid: pd.DataFrame, *, subplot: bool = False):
    categories = []
    for method in METHODS:
        for idx, (col, label) in enumerate((
            ("n_frags", "fragments"),
            ("mean_atoms_per_frag", "atoms/frag"),
        )):
            cat = f"{'Geass' if method == 'GEASS' else method}<br>{label}"
            categories.append(cat)
            vals = valid.loc[valid["method"].eq(method), col].dropna()
            if vals.empty:
                continue
            trace = go.Violin(
                x=[cat] * len(vals), y=vals,
                box_visible=True, meanline_visible=True,
                line_color=PALETTE[method][idx],
                fillcolor=PALETTE[method][idx],
                showlegend=False, points=False,
                name=cat,
            )
            if subplot:
                fig.add_trace(trace, row=1, col=2)
            else:
                fig.add_trace(trace)
    axis = dict(categoryorder="array", categoryarray=categories, tickangle=0)
    if subplot:
        fig.update_xaxes(**axis, row=1, col=2)
        fig.update_yaxes(title_text="Fragments per molecule / mean atoms per fragment",
                         rangemode="tozero", row=1, col=2)
    else:
        fig.update_xaxes(**axis)
        fig.update_yaxes(title_text="Fragments per molecule / mean atoms per fragment", rangemode="tozero")


def _save_figure(fig: go.Figure, stem: Path, formats: tuple[str, ...]):
    for fmt in formats:
        path = stem.with_suffix("." + fmt)
        try:
            if fmt == "html":
                fig.write_html(str(path), include_plotlyjs="cdn")
            else:
                fig.write_image(str(path))  # Requires Kaleido (as in notebook).
        except Exception as exc:
            raise RuntimeError(f"Cannot save {path}. Install a compatible Plotly/Kaleido "
                               f"renderer or use --formats html. Original error: {exc}") from exc
        print(f"Saved {path}")


def _figures(runs: pd.DataFrame, valid: pd.DataFrame, output: Path, formats: tuple[str, ...]):
    fig_a = _style(go.Figure(), width=750, height=470)
    _runtime_traces(fig_a, runs)
    fig_a.update_layout(legend=dict(x=0.98, y=0.02, xanchor="right", yanchor="bottom"))

    fig_b = _style(go.Figure(), width=850, height=470)
    _granularity_traces(fig_b, valid)

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09, column_widths=[0.48, 0.52])
    _style(fig, width=1650, height=530)
    _runtime_traces(fig, runs, subplot=True)
    _granularity_traces(fig, valid, subplot=True)
    fig.add_annotation(text="(a)", xref="paper", yref="paper", x=0, y=1.11,
                       showarrow=False, font=dict(size=20), xanchor="left")
    fig.add_annotation(text="(b)", xref="paper", yref="paper", x=0.54, y=1.11,
                       showarrow=False, font=dict(size=20), xanchor="left")
    fig.update_layout(legend=dict(x=0.45, y=0.02, xanchor="right", yanchor="bottom"))

    _save_figure(fig_a, output / "fig3a", formats)
    _save_figure(fig_b, output / "fig3b", formats)
    _save_figure(fig, output / "fig3", formats)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data/NPAtlas_download_2024_09.tsv")
    parser.add_argument("--output-dir", type=Path, default=ARTICLE_DIR / "fragmentation_benchmark")
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    parser.add_argument("--geass-annotate", action="store_true",
                        help="Enable MRS annotation; notebook defaults to disabled")
    parser.add_argument("--metadi-time-limit", type=float, default=20,
                        help="MetaDi internal limit in seconds; use 0 for unlimited")
    parser.add_argument("--no-save-fragments", action="store_true",
                        help="Skip the optional fragments.jsonl export")
    parser.add_argument("--from-csv", type=Path,
                        help="Rebuild summaries and figures from existing notebook/script CSV; skip benchmarking")
    parser.add_argument("--overwrite", action="store_true",
                        help="Discard previous benchmark results in output directory and start again")
    parser.add_argument("--formats", choices=("svg", "png", "html"), nargs="+",
                        default=["svg", "png"], help="Figure export formats")
    args = parser.parse_args(argv)
    if args.sample_size <= 0 or args.repeats <= 0 or args.metadi_time_limit < 0:
        parser.error("--sample-size and --repeats must be positive; MetaDi time limit cannot be negative")
    if args.from_csv and args.overwrite:
        parser.error("--from-csv and --overwrite cannot be combined")
    if len(set(args.methods)) != len(args.methods) or len(set(args.formats)) != len(args.formats):
        parser.error("Duplicate methods or formats are not supported")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.from_csv:
        print(f"Loading existing benchmark data from {args.from_csv}")
        runs = pd.read_csv(args.from_csv)
    else:
        runs = _run_benchmark(args, output)
    runs = _clean_runs(runs)
    first, valid, runtime, granularity = _summary(runs, output)
    print("\nRuntime summary (all executions):\n", runtime.round(4).to_string())
    print("\nGranularity summary (valid first attempts only):\n", granularity.round(3).to_string())
    _figures(runs, valid, output, tuple(args.formats))
    print(f"\nBenchmark outputs in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
