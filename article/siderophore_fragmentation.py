#!/usr/bin/env python3
"""Compare Geass, BRICS and MetaDi fragmentation of the four case-study siderophores.

Run from any working directory::

    python article/siderophore_fragmentation.py

Outputs are written to article/siderophore_fragmentation/ by default:
    comparison.png, comparison.svg       4-by-3 RDKit fragment-grid figure
    fragments.json, fragments.csv         machine-readable products and status
    sdf/<compound>_<method>.sdf           per-method fragment structures

The structures and method calls follow article/siderophors.ipynb and
article/benchmark.ipynb. BRICSDecompose returns unique terminal fragment
*types*, not necessarily every repeated fragment instance in the parent.
MetaDi displays its highest-ranked candidate (result_id=0); alternative
candidate counts are recorded in the output when available.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import BRICS, Draw, rdDepictor

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTICLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

# Exact case-study structures from article/siderophors.ipynb.
COMPOUNDS = {
    "enterobactin": "C1[C@@H](C(=O)OC[C@@H](C(=O)OC[C@@H](C(=O)O1)NC(=O)C2=C(C(=CC=C2)O)O)NC(=O)C3=C(C(=CC=C3)O)O)NC(=O)C4=C(C(=CC=C4)O)O",
    "salmochelin_s4": "C1[C@@H](C(=O)OC[C@@H](C(=O)OC[C@@H](C(=O)O1)NC(=O)C2=C(C(=CC(=C2)[C@H]3[C@@H]([C@H]([C@@H]([C@H](O3)CO)O)O)O)O)O)NC(=O)C4=C(C(=CC(=C4)[C@H]5[C@@H]([C@H]([C@@H]([C@H](O5)CO)O)O)O)O)O)NC(=O)C6=C(C(=CC=C6)O)O",
    "cyclic_trichrysobactin": "C1[C@@H](C(=O)OC[C@@H](C(=O)OC[C@@H](C(=O)O1)NC(=O)[C@@H](CCCCN)NC(=O)C2=C(C(=CC=C2)O)O)NC(=O)[C@@H](CCCCN)NC(=O)C3=C(C(=CC=C3)O)O)NC(=O)[C@@H](CCCCN)NC(=O)C4=C(C(=CC=C4)O)O",
    "mirubactin": "NC(=NCCC[C@@H](NC(=O)C1=C(O)C(O)=CC=C1)C(=O)N[C@H](CCCN(O)C=O)C(=O)O)NC(=O)C1=C(O)C(O)=CC=C1",
}
METHODS = ("Geass", "BRICS", "MetaDi")
DISPLAY_NAMES = {
    "enterobactin": "Enterobactin",
    "salmochelin_s4": "Salmochelin S4",
    "cyclic_trichrysobactin": "Cyclic trichrysobactin",
    "mirubactin": "Mirubactin",
}


def decompose(method: str, mol: Chem.Mol, smiles: str):
    """Return (molecules, metadata) using the existing benchmark APIs."""
    if method == "Geass":
        from new_geass import Geass

        fragments = Geass(annotate=False).get_fragments(Chem.Mol(mol))
        metadata = {"annotation_enabled": False}
    elif method == "BRICS":
        fragments = list(BRICS.BRICSDecompose(Chem.Mol(mol), returnMols=True))
        metadata = {"distinct_fragment_types": True}
    elif method == "MetaDi":
        from metadisassembler import MetaDisassembler

        md = MetaDisassembler()
        md.time_limit = math.inf
        if not md.input_query(smiles):
            raise ValueError("MetaDi rejected the input SMILES")
        inchis = md.disassemble(return_inchis=True, result_id=0, write_output=False) or []
        fragments = []
        for inchi in inchis:
            fragment = Chem.MolFromInchi(inchi) if isinstance(inchi, str) and inchi else None
            if fragment is None:
                raise ValueError("MetaDi returned an invalid fragment InChI")
            fragments.append(fragment)
        metadata = {
            "candidate_count": len(md.result),
            "selected_candidate": 0 if md.result else None,
        }
    else:
        raise ValueError(f"Unknown method: {method}")

    fragments = [Chem.Mol(frag) for frag in fragments if frag is not None]
    # Stable display order; repeated Geass/MetaDi occurrences are NOT deduplicated.
    fragments.sort(key=lambda m: (-m.GetNumHeavyAtoms(), Chem.MolToSmiles(m, isomericSmiles=True)))
    return fragments, metadata


def _fragment_record(mol: Chem.Mol, index: int) -> dict:
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return {
        "index": index,
        "smiles": smiles,
        "inchi_key": Chem.MolToInchiKey(mol) if all(a.GetAtomicNum() > 0 for a in mol.GetAtoms()) else None,
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "total_atoms": mol.GetNumAtoms(),
    }


def _write_sdf(path: Path, fragments: list[Chem.Mol], compound: str, method: str):
    with Chem.SDWriter(str(path)) as writer:
        for i, mol in enumerate(fragments, start=1):
            copy = Chem.Mol(mol)
            copy.SetProp("_Name", f"{compound}_{method}_{i:02d}")
            copy.SetProp("compound", compound)
            copy.SetProp("method", method)
            copy.SetProp("fragment_index", str(i))
            writer.write(copy)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _grid_images(fragments: list[Chem.Mol], per_row: int, tile: tuple[int, int]):
    if not fragments:
        return None, None
    legends = [f"{i + 1}  ({m.GetNumHeavyAtoms()} atoms)" for i, m in enumerate(fragments)]
    options = dict(molsPerRow=per_row, subImgSize=tile, legends=legends)
    pil = Draw.MolsToGridImage(fragments, useSVG=False, **options).convert("RGB")
    vector = Draw.MolsToGridImage(fragments, useSVG=True, **options)
    return pil, vector


def _svg_embedded(grid_svg: str, x: int, y: int) -> str:
    # Include RDKit's paths directly, retaining vector molecular structures.
    start = grid_svg.find("<svg")
    if start == -1:
        raise ValueError("Unexpected RDKit SVG grid output")
    return re.sub(r"<svg\b", f'<svg x="{x}" y="{y}"', grid_svg[start:], count=1)


def make_figure(results: dict, output: Path, per_row: int, tile: tuple[int, int]):
    rdDepictor.SetPreferCoordGen(True)
    label_w = 235
    col_w = per_row * tile[0] + 28
    header_h = 74
    cell_top = 43
    width = label_w + len(METHODS) * col_w
    row_heights = []
    for compound in COMPOUNDS:
        max_fragments = max(len(results[compound][method]["mols"]) for method in METHODS)
        row_heights.append(cell_top + max(1, math.ceil(max_fragments / per_row)) * tile[1] + 14)
    height = header_h + sum(row_heights)

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    font_big, font_medium, font_small = _font(24), _font(19), _font(15)

    def rect(box, fill, outline="#D4DDE5"):
        draw.rectangle(box, fill=fill, outline=outline, width=1)
        x1, y1, x2, y2 = box
        svg.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="{fill}" stroke="{outline}" stroke-width="1"/>')

    def text(x, y, label, font, size, colour="#203040"):
        draw.text((x, y), label, fill=colour, font=font)
        svg.append(f'<text x="{x}" y="{y + size}" fill="{colour}" font-family="DejaVu Sans, Arial, sans-serif" font-size="{size}">{html.escape(label)}</text>')

    rect((0, 0, width, header_h), "#EAF0F5")
    text(20, 23, "Compound", font_big, 24)
    for col, method in enumerate(METHODS):
        text(label_w + col * col_w + 18, 23, method, font_big, 24)

    row_y = header_h
    for row, compound in enumerate(COMPOUNDS):
        rh = row_heights[row]
        rect((0, row_y, label_w, row_y + rh), "#F4F7F9")
        label = DISPLAY_NAMES[compound]
        labels = ["Cyclic", "trichrysobactin"] if compound == "cyclic_trichrysobactin" else [label]
        for n, line in enumerate(labels):
            text(17, row_y + 24 + n * 27, line, font_medium, 19)

        for col, method in enumerate(METHODS):
            x = label_w + col * col_w
            rect((x, row_y, x + col_w, row_y + rh), "#FFFFFF")
            entry = results[compound][method]
            mols = entry["mols"]
            status = entry["status"]
            note = f"{len(mols)} fragments" if status == "ok" else ("Not run" if status == "skipped" else "No valid output")
            if method == "BRICS" and status == "ok":
                note = f"{len(mols)} distinct fragments"
            text(x + 14, row_y + 9, note, font_small, 15, "#596879")
            if mols:
                pil, native_svg = _grid_images(mols, per_row, tile)
                canvas.paste(pil, (x + 14, row_y + cell_top))
                svg.append(_svg_embedded(native_svg, x + 14, row_y + cell_top))
            elif status == "error":
                text(x + 14, row_y + cell_top + 10, "Method failed; see fragments.json", font_small, 15, "#994747")
        row_y += rh
    svg.append("</svg>")
    canvas.save(output / "comparison.png")
    (output / "comparison.svg").write_text("\n".join(svg), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ARTICLE_DIR / "siderophore_fragmentation")
    parser.add_argument("--mols-per-row", type=int, default=3, help="RDKit fragment tiles per cell row")
    parser.add_argument("--tile-width", type=int, default=200)
    parser.add_argument("--tile-height", type=int, default=160)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS), help="Run subset of methods for diagnostics; other figure cells say Not run")
    args = parser.parse_args(argv)
    if args.mols_per_row < 1 or min(args.tile_width, args.tile_height) < 80:
        parser.error("Mols-per-row must be positive; tile dimensions must be at least 80 pixels")

    output = args.output_dir.expanduser().resolve()
    sdf_dir = output / "sdf"
    sdf_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    summary = []
    for compound, smiles in COMPOUNDS.items():
        parent = Chem.MolFromSmiles(smiles)
        if parent is None:
            raise ValueError(f"Invalid case-study SMILES for {compound}")
        all_results[compound] = {}
        for method in METHODS:
            if method not in args.methods:
                mols, metadata, status, error, runtime = [], {}, "skipped", "", 0.0
            else:
                start = time.perf_counter()
                try:
                    mols, metadata = decompose(method, parent, smiles)
                    status, error = ("ok", "") if mols else ("no_output", "No fragments returned")
                except Exception as exc:
                    mols, metadata, status, error = [], {}, "error", f"{type(exc).__name__}: {exc}"
                runtime = round(time.perf_counter() - start, 4)
            print(f"{compound:24} {method:6} {status:10} {len(mols):>3} fragments ({runtime:.2f}s)", flush=True)
            if error:
                print(f"  {error}", file=sys.stderr, flush=True)
            if mols:
                _write_sdf(sdf_dir / f"{compound}_{method.lower()}.sdf", mols, compound, method)
            fragments = [_fragment_record(m, i) for i, m in enumerate(mols, start=1)]
            all_results[compound][method] = {
                "status": status, "runtime_s": runtime, "error": error,
                "fragments": fragments, "metadata": metadata, "mols": mols,
            }
            summary.append({
                "compound": compound, "parent_smiles": Chem.MolToSmiles(parent, isomericSmiles=True),
                "method": method, "status": status, "runtime_s": runtime,
                "n_fragments": len(mols), "error": error,
                "candidate_count": metadata.get("candidate_count", ""),
                "fragments": fragments,
            })

    json_out = []
    for entry in summary:
        json_out.append({**{k: v for k, v in entry.items() if k != "fragments"}, "fragments": entry["fragments"]})
    (output / "fragments.json").write_text(json.dumps(json_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    csv_columns = ["compound", "method", "status", "runtime_s", "n_fragments", "candidate_count", "fragment_index", "fragment_smiles", "inchi_key", "heavy_atoms", "error"]
    with (output / "fragments.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=csv_columns)
        writer.writeheader()
        for entry in summary:
            for frag in entry["fragments"] or [None]:
                writer.writerow({
                    "compound": entry["compound"], "method": entry["method"], "status": entry["status"],
                    "runtime_s": entry["runtime_s"], "n_fragments": entry["n_fragments"],
                    "candidate_count": entry["candidate_count"], "error": entry["error"],
                    "fragment_index": frag["index"] if frag else "",
                    "fragment_smiles": frag["smiles"] if frag else "",
                    "inchi_key": frag["inchi_key"] if frag else "",
                    "heavy_atoms": frag["heavy_atoms"] if frag else "",
                })
    make_figure(all_results, output, args.mols_per_row, (args.tile_width, args.tile_height))
    print(f"Saved fragment data, SDFs and comparison figures to: {output}")
    return 0 if all(row["status"] in {"ok", "skipped", "no_output"} for row in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
