from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET

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
