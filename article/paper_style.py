from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET

import cairosvg
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import plotly.graph_objects as go
import plotly.io as pio

FONT_FAMILY = 'Arial'


def _ungroup_svg(infile, outfile):
    tree = ET.parse(infile)
    root = tree.getroot()

    def flatten(parent):
        for child in list(parent):
            flatten(child)
            if child.tag.endswith('g') and len(child):
                child_style = child.attrib.get('style', '')
                child_transform = child.attrib.get('transform', '')
                idx = list(parent).index(child)
                for grandchild in list(child):
                    grandchild = deepcopy(grandchild)
                    if child_style:
                        grandchild.attrib['style'] = child_style + ';' + grandchild.attrib.get('style', '')
                    if child_transform and 'transform' not in grandchild.attrib:
                        grandchild.attrib['transform'] = child_transform
                    parent.insert(idx, grandchild)
                    idx += 1
                parent.remove(child)

    flatten(root)
    tree.write(outfile, encoding='utf-8', xml_declaration=True)


def apply_paper_style(fig, rows=1, cols=1, zoom=1):
    paper_style = go.layout.Template(
            layout=dict(width=750, height=470,
            margin=dict(t=40, b=55 * zoom, l=72 * zoom, r=40),
            font=dict(family=FONT_FAMILY, color='black', size=16 * zoom),
            legend=dict(
                font=dict(family=FONT_FAMILY, color='black', size=16 * zoom),
                tracegroupgap=0,
            ),
            plot_bgcolor='white', paper_bgcolor='white', 
            xaxis=dict(
                showgrid=True, gridcolor='lightgrey',
                showline=True, linewidth=2 * zoom, linecolor='black',
                zeroline=False,
                tickfont=dict(family=FONT_FAMILY, size=16 * zoom, color='black'),
                title_font=dict(family=FONT_FAMILY, size=18 * zoom, color='black'),
                title_standoff=12 * zoom
            ),
            yaxis=dict(
                showgrid=True, gridcolor='lightgrey',
                showline=True, linewidth=zoom, linecolor='lightgrey',
                mirror=True,
                zeroline=False,
                tickfont=dict(family=FONT_FAMILY, size=16 * zoom, color='black'),
                title_font=dict(family=FONT_FAMILY, size=18 * zoom, color='black'),
                title_standoff=10 * zoom)))

    pio.templates['paper_style'] = paper_style
    fig.update_layout(template='paper_style')

    if rows == 1 and cols == 1:
        fig.add_shape(
            type='line',
            x0=0, x1=1, y0=1, y1=1,
            xref='paper', yref='paper',
            line=dict(color='lightgrey', width=1 * zoom),
            layer='below'
        )
    else:
        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                fig.add_shape(
                    type='line',
                    x0=0, x1=1, y0=1, y1=1,
                    xref='x domain', yref='y domain',
                    line=dict(color='lightgrey', width=1 * zoom),
                    layer='below',
                    row=r, col=c
                )
    return fig


def save_paper_style(fig, path, rows=1, cols=1, zoom=1):
    fig = apply_paper_style(fig, rows=rows, cols=cols, zoom=zoom)
    path = Path(path)
    fig.write_image(str(path))
    if path.suffix.lower() == '.svg':
        _ungroup_svg(path, path)
    return fig


def export_consistent_mol_figures(
    mols,
    out_dir='figures',
    prefix='',
    canvas_w=300,
    padding=0.02,
    fixed_font_size=None,
    save_png=False,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(mols, dict):
        items = list(mols.items())
    else:
        items = [(None, mol) for mol in mols]

    def _safe_name(name):
        text = str(name).strip()
        if not text:
            return 'mol'
        safe = []
        for ch in text:
            if ch.isalnum() or ch in {'-', '_'}:
                safe.append(ch)
            else:
                safe.append('_')
        return ''.join(safe).strip('_') or 'mol'

    valid = []
    for idx, (name, mol) in enumerate(items, start=1):
        if mol is None:
            continue

        m = Chem.Mol(mol)
        rdDepictor.Compute2DCoords(m)
        conf = m.GetConformer()

        bl = []
        for b in m.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            pi = conf.GetAtomPosition(i)
            pj = conf.GetAtomPosition(j)
            bl.append(((pi.x - pj.x) ** 2 + (pi.y - pj.y) ** 2) ** 0.5)

        mean_bl = float(np.mean(bl)) if bl else 1.0
        xs = [conf.GetAtomPosition(i).x for i in range(m.GetNumAtoms())]
        ys = [conf.GetAtomPosition(i).y for i in range(m.GetNumAtoms())]
        span = max((max(xs) - min(xs)), (max(ys) - min(ys)), 1e-6)
        label = _safe_name(name) if name is not None else chr(ord('a') + idx - 1)
        valid.append((idx, label, m, mean_bl, span))

    if not valid:
        raise ValueError('No valid molecules to draw')

    target_bl = float(np.median([x[3] for x in valid]))
    scaled_spans = []
    for _, _, _, mean_bl, span in valid:
        factor = target_bl / max(mean_bl, 1e-6)
        scaled_spans.append(span * factor)

    max_scaled_span = max(scaled_spans)
    fixed_scale = (1.0 - 2.0 * padding * canvas_w / 160) / max_scaled_span

    for idx, label, mol, mean_bl, _ in valid:
        suffix = label if label is not None else chr(ord('a') + idx - 1)
        stem = f'{prefix}_{suffix}' if prefix else suffix
        svg_path = out_dir / f'{stem}.svg'

        m = Chem.Mol(mol)
        conf = m.GetConformer()
        factor = target_bl / max(mean_bl, 1e-6)
        for a in range(m.GetNumAtoms()):
            p3 = conf.GetAtomPosition(a)
            conf.SetAtomPosition(a, (p3.x * factor, p3.y * factor, 0.0))

        drawer = rdMolDraw2D.MolDraw2DSVG(canvas_w, canvas_w)
        draw_opts = drawer.drawOptions()
        draw_opts.useBWAtomPalette()
        draw_opts.addAtomIndices = False
        draw_opts.fixedBondLength = 22
        draw_opts.multipleBondOffset = 0.2
        draw_opts.bondLineWidth = 0.12 * fixed_font_size
        draw_opts.padding = padding * canvas_w / 160
        draw_opts.additionalAtomLabelPadding = 0.05
        draw_opts.clearBackground = True
        draw_opts.fixedFontSize = fixed_font_size
        draw_opts.fixedScale = fixed_scale

        drawer.DrawMolecule(m)
        drawer.FinishDrawing()
        svg_text = drawer.GetDrawingText()

        svg_path.write_text(svg_text, encoding='utf-8')
        print(f'Saved {str(svg_path)}')

        if save_png:
            png_path = out_dir / f'{stem}.png'
            cairosvg.svg2png(bytestring=svg_text.encode('utf-8'), write_to=str(png_path))
            print(f'Saved {str(png_path)}')
