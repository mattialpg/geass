import os, sys, re, io, html
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
from urllib import parse

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point2D

workdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(workdir)
import utils

from annotator import AnnotationEngine
from fragmenter import FragmentationEngine

#* Nomenclature
#* fragment        >  mol with dummy atom(s)
#* building unit   >  fragment whose dummy atoms (*) have been replaced with real atoms 
#* building block  >  molecule formed by two or more building units


def reset_view(mol):
    """ Reset coordinates for display """
    Chem.Kekulize(mol)
    Chem.rdDepictor.SetPreferCoordGen(True)
    Chem.rdDepictor.Compute2DCoords(mol)
    Chem.rdDepictor.StraightenDepiction(mol)
    return


def mol2svg(mol):
    try:
        # Handle special cases
        if mol.HasProp('Annotation') and mol.GetProp('Annotation') == 'Special':
            atoms = list(mol.GetAtoms())
            if mol.GetProp('MUID') == 'C00019':
                atom_C = next(a for a in atoms if a.GetAtomicNum() == 6)
                atom_C.SetProp("atomLabel", "RCH<sub>3</sub>")
            else:
                atom_R = next(a for a in atoms if a.GetAtomicNum() == 0)
                atom_R.SetProp('atomLabel', 'R')

        reset_view(mol)
        AllChem.Compute2DCoords(mol)

        # Draw SVG
        drawer = rdMolDraw2D.MolDraw2DSVG(150, 150)
        opts = drawer.drawOptions()
        opts.bondLineWidth = 1.0
        opts.useFractionalBondWidths = True
        opts.clearBackground = False
        opts.includeRadicals = False
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        
        # Minify SVG payload for smaller data URIs.
        svg_string = drawer.GetDrawingText()
        svg_string = re.sub(R"<\\?xml[^>]*\\?>\\s*", "", svg_string, count=1)
        svg_string = re.sub(R">\\s+<", "><", svg_string)
        svg_string = re.sub(R"\\s+", " ", svg_string).strip()

        return 'data:image/svg+xml;charset=utf-8,' + parse.quote(svg_string, safe="")

    except Exception as exc:
        print(exc)
        return ''


class NoBreakableBondError(ValueError):
    pass


class Geass:
    _resources_loaded = False

    def __init__(self, fp="map4", annotate=True, sim_threshold=None, verbose=False, as_dict=False, method=None):
        self.annotate = annotate
        self.as_dict = as_dict
        if method is not None:
            fp = method
        if sim_threshold is None:
            sim_threshold = 0.35 if fp == "map4" else 0.76
        self.sim_threshold = sim_threshold
        self.verbose = verbose
        self.method = fp
        self.mol = None
        self.bonds_to_break = {}
        self.frag_mols = []
        self.atom_frag = {}

        # Initialise engines
        # Reload annotation resources if method changed across instances.
        if getattr(AnnotationEngine, "_loaded_method", None) != self.method:
            AnnotationEngine.df_MUDB = None
            AnnotationEngine._muid_by_inchikey = None
            AnnotationEngine._muid_ids = None
        AnnotationEngine.method = self.method
        AnnotationEngine.initialize_resources()
        self.annotator = AnnotationEngine()
        self.fragmenter = FragmentationEngine(workdir)


    def get_fragments(self, query_mol, as_smiles=False, check_unknowns=False):
        """
        Fragment a molecule into potential metabolic units (MUs).
        Cleans, fragments, deduplicates, and classifies fragments.
        """
        # --- Reset transient state for a new molecule ---
        self.mol = None
        self.bonds_to_break.clear()
        self.frag_mols.clear()
        self.atom_frag.clear()

        try:
            # --- Step 1: Prepare molecule ---
            self.mol = utils.standardize_mol(Chem.Mol(query_mol))

            # --- Step 2: Determine which reaction applies ---
            self.bonds_to_break = self.fragmenter.get_bonds_to_break(self.mol)
            
            # --- Step 3: Run fragmentation ---
            note = None
            if self.bonds_to_break:
                self.frag_mols, self.atom_frag = self.fragmenter.fragment_molecule(self.mol)
            else:
                self.frag_mols = [self.mol]
                note = "No breakable bonds found"

            # # --- Step 4: Combine if isotopic coverage incomplete ---
            # total_isotopes = {atom.GetIsotope() for atom in self.mol.GetAtoms() if atom.GetIsotope()}
            # covered_isotopes = set().union(*[{atom.GetIsotope() for atom in f.GetAtoms()
            #     if atom.GetIsotope()} for f in self.frag_mols])

            # if covered_isotopes != total_isotopes:
            #     print("Warning: Incomplete atom coverage; some fragments may be missing.")
            #     combined = self.combine_frags_with_timeout(unique_frags, timeout=10)
            #     if combined:
            #         self.frag_mols = combined

            # --- Step 5: Annotate ---
            frag_dicts = []
            for frag_mol in self.frag_mols:
                if self.annotate:
                    frag_props = self.annotator.annotate_fragment(frag_mol, self.sim_threshold)
                else:
                    frag_props = {
                        "ISOSMILES": Chem.MolToSmiles(frag_mol, canonical=True, isomericSmiles=True),
                        "InChIKey": Chem.MolToInchiKey(frag_mol),
                        "MUID": Chem.MolToInchiKey(frag_mol),
                        "isNP": "",
                        "Annotation": "",
                        "Similarity": "0.0",
                    }
                frag_props['Note'] = note

                if self.as_dict:
                    frag_dicts.append(frag_props)
                    continue
                
                for key, value in frag_props.items():
                    frag_mol.SetProp(key, str(value))

            if self.as_dict:
                return frag_dicts
            if as_smiles:
                return [Chem.MolToSmiles(m, canonical=True, isomericSmiles=True) for m in self.frag_mols]
            return self.frag_mols

        except Exception as e:
            self.bonds_to_break.clear()
            self.atom_frag.clear()
            if self.mol is None:
                return []

            frag_mol = Chem.Mol(self.mol)
            frag_dict = None
            if self.annotate:
                frag_dict = self.annotator.annotate_fragment(frag_mol, self.sim_threshold)
            else:
                frag_dict = {
                    "ISOSMILES": Chem.MolToSmiles(frag_mol, canonical=True, isomericSmiles=True),
                    "InChIKey": Chem.MolToInchiKey(frag_mol),
                    "MUID": Chem.MolToInchiKey(frag_mol),
                    "isNP": "",
                    "Annotation": "",
                    "Similarity": "0.0",
                }

            if self.as_dict:
                return [frag_dict]

            for key, value in frag_dict.items():
                frag_mol.SetProp(key, str(value))
            self.frag_mols = [frag_mol]

            if as_smiles:
                return [Chem.MolToSmiles(frag_mol, canonical=True, isomericSmiles=True)]
            return self.frag_mols


    def create_connection_graph(self, img=True):
        G = nx.Graph()

        # Create nodes
        for i, frag_mol in enumerate(self.frag_mols):
            node_data = {
                'SMILES': frag_mol.GetProp('ISOSMILES'),
                'MUID': frag_mol.GetProp('MUID'),
                'Similarity': float(frag_mol.GetProp('Similarity'))
                    if frag_mol.HasProp('Similarity') else 0.0,
                'Annotation': frag_mol.GetProp('Annotation')
                    if frag_mol.HasProp('Annotation') else '',
                'isNP': frag_mol.GetProp('isNP')
                    if frag_mol.HasProp('isNP') else ''}
            if img:
                node_data['img'] = mol2svg(frag_mol)
            G.add_nodes_from([(i, node_data)])

            # # Overwrite annotation if sim = 1
            # if float(frag_mol.GetProp('Similarity')) >= 0.99:
            #     node_data['Annotation'] = 'Direct'
            #     print(node_data)
        
        # Create edge/attributes dictionary 
        edge_dict = {}
        for k, v in self.bonds_to_break.items():
            if k[0] in self.atom_frag and k[1] in self.atom_frag:
                f1 = self.atom_frag[k[0]]
                f2 = self.atom_frag[k[1]]
                edge_dict[(f1, f2)] = v
                #<-- This last condition arises because, for symmetrical envs,
                #<-- only one of the two possible breakage contained in bonds_to_break
                #<-- is actually carried out.
        #TODO Fantastic-biofrag should overcome this fundamental limitation by creating a branching point
        #TODO the fragmentation tree when symmetrical patterns produce asymmetrical blocks.

        for edge, attribute in edge_dict.items():
            G.add_edge(*edge, reaction=attribute)
        
        return G


    def show_connection_graph(self, G=None, method='cytoscape'):
        if not G:
            G = self.create_connection_graph()
        if G:
            for _, data in G.nodes(data=True):
                if not 'img' in data.keys() or data.get('img') == '':
                    data['img'] = mol2svg(Chem.MolFromSmiles(data['SMILES']))
        
        #! ----------- Plot with Cytoscape -----------!#
        if method  == 'cytoscape':
            #* get ideas from https://giter.vip/cytoscape/ipycytoscape
            #* and https://dash.plotly.com/cytoscape/layout
            from ipycytoscape import CytoscapeWidget

            cyto = CytoscapeWidget()
            cyto.layout.height = '600px'
            cyto.layout.width = '1400px'
            cyto.set_layout(name='cola',
                nodeSpacing=20, avoidOverlap=True,
                padding=50, maxSimulationTime=6000)
            cyto.graph.add_graph_from_networkx(G)

            # Update node labels to use the 'MUID' attribute
            for node in cyto.graph.nodes:
                node.data['label'] = node.data.get('MUID', '')

            cyto.set_style([
                # base node style
                {
                    'selector': 'node',
                    'css': {
                        'content': 'data(MUID)',
                        'width': 50,
                        'height': 50,
                        'border-width': 1.5,
                        'border-color': "#989898",
                        'background-color': '#FFFFFF',
                        'background-width': '80%',
                        'background-height': '80%',
                        'background-image': 'data(img)',
                        'font-family': 'helvetica',
                        'font-size': '10px',
                        'text-margin-y': '8px',
                        'text-halign': 'center',
                        'text-valign': 'bottom',
                        'text-background-color': '#FFFFFF',
                        'text-background-shape': 'roundrectangle',
                        'text-background-opacity': 1,
                        'text-background-padding': 2
                    }
                },

                # Override border colour
                {
                    'selector': "node[Similarity > 0.99]",
                    'css': {'border-color': '#009b73'}
                },
                {
                    'selector': "node[Similarity >= 0.01][Similarity <= 0.99]",
                    'css': {'border-color': '#ffc32d'}
                },
                {
                    'selector': "node[Annotation = 'None']",
                    'css': {'border-color': '#d10808'}
                },
                {
                    'selector': "node[Annotation = 'Special']",
                    'css': {'border-color': '#009b73'}
                    # 'css': {'border-color': "#0a59c1"}
                },
                {
                    'selector': "node[is_support = 'Yes']",
                    'css': {'border-color': '#0a59c1'}
                },

                # base edge style
                {
                    'selector': 'edge',
                    'css': {
                        'curve-style': 'bezier',
                        'line-color': '#009b73',
                        'width': 1.5,
                        'label': 'data(reaction)',
                        'font-family': 'helvetica',
                        'font-size': '10px',
                        'text-valign': 'center',
                        'text-background-color': '#ffffff',
                        'text-background-shape': 'roundrectangle',
                        'text-background-opacity': 1,
                        'text-background-padding': 2
                    }
                },

                # edge variant overrides only colour
                {
                    'selector': "edge[reaction = '']",
                    'css': {'line-color': '#d10808'}
                }


                            # {'selector': 'node:hover',
                            #  'style': {'label': 'data(atoms)',
                            #            'overlay-opacity': 0.1,
                            #            'overlay-color': 'black'}}
                            ])

            return cyto

    def export_connection_graph(self, output_path, scale=2):
        """
        Export connection graph using a pure-Python static renderer.
        Supports: .svg, .pdf, .png
        """
        output_path = Path(output_path)
        fmt = output_path.suffix.lower().lstrip(".")

        G = self.create_connection_graph()

        import matplotlib.pyplot as plt
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        plt.rcParams["font.family"] = "DejaVu Sans"

        n_nodes = max(1, G.number_of_nodes())
        k = 1.25 / np.sqrt(n_nodes)
        pos = nx.spring_layout(G, seed=7, k=k)

        # Normalize layout to fill the canvas consistently.
        p = np.array([pos[n] for n in G.nodes()], dtype=float)
        if len(p) > 0:
            mins = p.min(axis=0)
            maxs = p.max(axis=0)
            span = np.maximum(maxs - mins, 1e-6)
            p = (p - mins) / span
            p = 0.1 + 0.8 * p  # inner padding
            for i, n in enumerate(G.nodes()):
                pos[n] = p[i]

        # True vector export path: write pure SVG with vector molecule drawings.
        if fmt == "svg":
            x_min, x_max = -0.14, 1.14
            y_min, y_max = -0.28, 1.14
            W, H = 1800, 1050

            def to_px(xy):
                x, y = xy
                px = (x - x_min) / (x_max - x_min) * W
                py = H - (y - y_min) / (y_max - y_min) * H
                return px, py

            def add_label_with_box(lines_out, x, y, text, font_size):
                txt = html.escape(str(text))
                # Approximate text metrics for centered label background box.
                box_pad = 4
                text_w = max(1.0, 0.62 * font_size * len(str(text)))
                text_h = 1.15 * font_size
                x0 = x - (text_w / 2.0) - box_pad
                y0 = y - (text_h / 2.0) - box_pad
                bw = text_w + 2 * box_pad
                bh = text_h + 2 * box_pad
                lines_out.append(
                    f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{bw:.2f}" height="{bh:.2f}" '
                    f'rx="5" ry="5" fill="#FFFFFF" fill-opacity="1"/>'
                )
                lines_out.append(
                    f'<text x="{x:.2f}" y="{y:.2f}" font-size="{font_size}" fill="#111" '
                    f'text-anchor="middle" dominant-baseline="middle">{txt}</text>'
                )

            # Node border colour logic from show_connection_graph()
            border_by_node = {}
            for n, d in G.nodes(data=True):
                sim = float(d.get("Similarity", 0.0))
                ann = d.get("Annotation", "")
                is_support = d.get("is_support", "")
                if ann == "None":
                    col = "#d10808"
                elif ann == "Special":
                    col = "#009b73"
                elif is_support == "Yes":
                    col = "#0a59c1"
                elif sim > 0.99:
                    col = "#009b73"
                elif 0.01 <= sim <= 0.99:
                    col = "#ffc32d"
                else:
                    col = "#989898"
                border_by_node[n] = col

            lines = []
            lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
            lines.append('<rect width="100%" height="100%" fill="white"/>')
            lines.append('<g font-family="DejaVu Sans, Arial, sans-serif">')

            # Edges + labels
            for u, v, d in G.edges(data=True):
                x1, y1 = to_px(pos[u])
                x2, y2 = to_px(pos[v])
                ecol = "#d10808" if d.get("reaction", "") == "" else "#009b73"
                lines.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{ecol}" stroke-width="3"/>')
                lbl = str(d.get("reaction", ""))
                if lbl:
                    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    add_label_with_box(lines, mx, my, lbl, 22)

            # Nodes + molecule drawings + node labels
            node_r = 74
            mol_box = 98
            for n, d in G.nodes(data=True):
                cx, cy = to_px(pos[n])
                lines.append(
                    f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{node_r}" fill="white" '
                    f'stroke="{border_by_node[n]}" stroke-width="3"/>'
                )

                smi = d.get("SMILES", "")
                mol = Chem.MolFromSmiles(smi) if smi else None
                if mol is not None:
                    drawer = rdMolDraw2D.MolDraw2DSVG(520, 520)
                    dopts = drawer.drawOptions()
                    dopts.addAtomIndices = False
                    dopts.clearBackground = False
                    dopts.bondLineWidth = 1.6
                    dopts.padding = 0.20
                    if hasattr(dopts, "fixedFontSize"):
                        dopts.fixedFontSize = 30
                    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
                    drawer.FinishDrawing()
                    mol_svg = drawer.GetDrawingText()
                    mol_svg = re.sub(r"<\?xml[^>]*\?>\s*", "", mol_svg, count=1)
                    inner = re.search(r"<svg[^>]*>(.*)</svg>", mol_svg, flags=re.S)
                    if inner:
                        mol_svg = inner.group(1)
                    scale_m = (2.0 * mol_box) / 520.0
                    tx = cx - mol_box
                    ty = cy - mol_box
                    lines.append(f'<g transform="translate({tx:.2f},{ty:.2f}) scale({scale_m:.6f})">{mol_svg}</g>')

                muid = str(d.get("MUID", ""))
                if muid:
                    add_label_with_box(lines, cx, cy + 116, muid, 24)

            lines.append("</g>")
            lines.append("</svg>")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("\n".join(lines), encoding="utf-8")
            return output_path

        fig_w, fig_h = 12, 7
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_facecolor("white")

        # Edge styling (same semantics as show_connection_graph)
        default_edge_color = "#009b73"
        nx.draw_networkx_edges(
            G, pos, width=1.5, edge_color=default_edge_color,
            connectionstyle="arc3,rad=0.08", ax=ax
        )
        empty_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("reaction", "") == ""]
        if empty_edges:
            nx.draw_networkx_edges(
                G, pos, edgelist=empty_edges, width=1.5, edge_color="#d10808",
                connectionstyle="arc3,rad=0.08", ax=ax
            )

        edge_labels = {(u, v): d.get("reaction", "") for u, v, d in G.edges(data=True)}
        if edge_labels:
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels=edge_labels, font_size=10, font_family="DejaVu Sans",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2}, ax=ax
            )

        # Node border colour logic from show_connection_graph()
        border_cols = []
        for _, d in G.nodes(data=True):
            sim = float(d.get("Similarity", 0.0))
            ann = d.get("Annotation", "")
            is_support = d.get("is_support", "")
            if ann == "None":
                border_cols.append("#d10808")
            elif ann == "Special":
                border_cols.append("#009b73")
            elif is_support == "Yes":
                border_cols.append("#0a59c1")
            elif sim > 0.99:
                border_cols.append("#009b73")
            elif 0.01 <= sim <= 0.99:
                border_cols.append("#ffc32d")
            else:
                border_cols.append("#989898")

        nx.draw_networkx_nodes(
            G, pos, node_size=6200, node_color="white",
            edgecolors=border_cols, linewidths=1.5, ax=ax
        )

        # Overlay fragment drawings on nodes with native RDKit PNG rendering.
        # This avoids SVG conversion/runtime dependencies and makes sizing reliable.
        for n, d in G.nodes(data=True):
            smi = d.get("SMILES", "")
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            drawer = rdMolDraw2D.MolDraw2DCairo(520, 520)
            dopts = drawer.drawOptions()
            dopts.addAtomIndices = False
            dopts.clearBackground = False
            dopts.bondLineWidth = 1.6
            dopts.padding = 0.20
            if hasattr(dopts, "fixedFontSize"):
                dopts.fixedFontSize = 30
            rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
            drawer.FinishDrawing()
            arr = plt.imread(io.BytesIO(drawer.GetDrawingText()), format="png")
            ab = AnnotationBbox(
                OffsetImage(arr, zoom=0.24, interpolation="lanczos"),
                pos[n],
                frameon=False,
                box_alignment=(0.5, 0.5),
                zorder=3,
                annotation_clip=False,
            )
            ax.add_artist(ab)

        node_labels = {n: d.get("MUID", "") for n, d in G.nodes(data=True)}
        if node_labels:
            label_pos = {n: (xy[0], xy[1] - 0.15) for n, xy in pos.items()}
            nx.draw_networkx_labels(
                G, label_pos, labels=node_labels, font_size=10, font_family="DejaVu Sans",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2}, ax=ax
            )

        # Keep full graph contents inside the exported figure (nodes + labels).
        ax.set_xlim(-0.14, 1.14)
        ax.set_ylim(-0.28, 1.14)
        ax.set_axis_off()
        fig.tight_layout(pad=0.2)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if fmt == "png":
            save_kwargs["dpi"] = int(300 * max(1, float(scale)))
        fig.savefig(str(output_path), **save_kwargs)
        plt.close(fig)
        return output_path

        # elif method == 'yEd':
        #     import random
        #     from yfiles_jupyter_graphs import GraphWidget
        #     import networkx as nx

        #     # --- Preprocess graph: random jitter to break symmetry ---
        #     for node in G.nodes():
        #         G.nodes[node]["pos"] = (random.random() * 0.1, random.random() * 0.1)

        #     # Optional: reorder nodes by degree to space hubs
        #     G = nx.Graph(sorted(G.degree, key=lambda x: -x[1]))
            
        #     def custom_images(node):
        #         props = node.get('properties', {})
        #         img = props.get('img')
        #         print(img)
        #         style = {'fill': 'transparent', 'stroke': 'transparent'}
        #         style['image'] = img
        #         return style

        #     w = GraphWidget(graph=G)
        #     w.node_label_mapping = ''  # Hide labels
        #     w.set_node_styles_mapping(custom_images)
        #     w.set_overview(enabled=False)
        #     w.set_sidebar(enabled=False)

        #     # --- Enhanced organic layout configuration ---
        #     w.organic_layout(
        #         # preferred_edge_length=250,
        #         # node_overlaps_allowed=False,
        #         # node_size_consideration=True,
        #         # edge_crossing_avoidance=True,
        #         # deterministic=True,
        #         # consider_node_labels=True,
        #         # act_on_selection=False,
        #         # minimum_node_distance=40
        #     )

        #     # --- Optional: rerun layout twice to further relax edges ---
        #     # w.organic_layout(preferred_edge_length=150)

        #     w.show()

        # #! ------------- Plot with Plotly ------------!#
        # if method == 'plotly':
        #     # --- Compute RDKit 2D coordinates ---
        #     AllChem.Compute2DCoords(mol)
        #     coords = mol.GetConformer().GetPositions()  # (n_atoms, 3)

        #     # --- Compute node positions as barycenters of their atoms ---
        #     raw_pos = {}; scale = 40
        #     for node in G.nodes():
        #         atoms = G.nodes[node].get('atoms', [])
        #         if atoms:
        #             atom_coords = np.array([coords[i][:2] for i in atoms])
        #             barycenter = atom_coords.mean(axis=0)
        #             raw_pos[node] = barycenter * scale
        #         else:
        #             raw_pos[node] = np.array([1, 1])  # fallback

        #     # --- Collect ranges for normalisation ---
        #     xs = np.array([raw_pos[n][0] for n in G.nodes()])
        #     ys = np.array([raw_pos[n][1] for n in G.nodes()])

        #     xmin, xmax = xs.min(), xs.max()
        #     ymin, ymax = ys.min(), ys.max()

        #     # Avoid zero-division; if molecule collapsed to a point, widen artificially
        #     if xmax == xmin:
        #         xmax = xmin + 1.0
        #     if ymax == ymin:
        #         ymax = ymin + 1.0

        #     # --- Normalise into [0,1] ---
        #     norm_pos = {}
        #     for n in G.nodes():
        #         x, y = raw_pos[n]
        #         nx = (x - xmin) / (xmax - xmin)
        #         ny = (y - ymin) / (ymax - ymin)
        #         norm_pos[n] = np.array([nx, ny])

        #     # Optional: scale to plot-friendly size
        #     plot_scale = 500
        #     pos = {n: norm_pos[n] * plot_scale for n in G.nodes()}

        #     # --- Extract coordinates for plotting ---
        #     x_vals = [pos[n][0] for n in G.nodes()]
        #     y_vals = [pos[n][1] for n in G.nodes()]

        #     # --- Edges ---
        #     edge_x = []
        #     edge_y = []
        #     for e in G.edges():
        #         x0, y0 = pos[e[0]]
        #         x1, y1 = pos[e[1]]
        #         edge_x += [x0, x1, None]
        #         edge_y += [y0, y1, None]

        #     edge_trace = go.Scatter(
        #         x=edge_x, y=edge_y,
        #         line=dict(width=1.7, color='black'),
        #         mode='lines',
        #         hoverinfo='none'
        #     )

        #     # --- Nodes ---
        #     node_colors = [
        #         'green' if G.nodes[n].get('isNP') == 'Yes' else 'red'
        #         for n in G.nodes()
        #     ]

        #     node_trace = go.Scatter(
        #         x=x_vals, y=y_vals,
        #         mode='markers',
        #         marker=dict(
        #             size=100,
        #             color='white',
        #             line=dict(width=1.7, color=node_colors)
        #         ),
        #         text=[
        #             f"Node {n}<br>isNP: {G.nodes[n].get('isNP')}<br>"
        #             f"Atoms: {G.nodes[n].get('atoms')}"
        #             for n in G.nodes()
        #         ],
        #         hoverinfo='text'
        #     )

        #     # --- Overlay images ---
        #     image_annotations = []
        #     for node, data in G.nodes(data=True):
        #         x, y = pos[node]
        #         if 'img' in data and data['img']:
        #             image_annotations.append(
        #                 dict(
        #                     source=data['img'],
        #                     xref="x", yref="y",
        #                     x=x, y=y,
        #                     sizex=200, sizey=200,
        #                     xanchor="center", yanchor="middle",
        #                     layer="above"
        #                 )
        #             )

        #     # --- Build figure ---
        #     fig = go.Figure(data=[edge_trace, node_trace])
        #     fig.update_layout(
        #         images=image_annotations,
        #         showlegend=False,
        #         xaxis=dict(showgrid=False, zeroline=False, visible=False),
        #         yaxis=dict(showgrid=False, zeroline=False, visible=False),
        #         paper_bgcolor="white",
        #         plot_bgcolor="white",
        #         # margin=dict(t=0, b=0, l=0, r=0),
        #         width=700, height=500)

        #     fig.update_xaxes(constrain='domain', fixedrange=True)
        #     fig.update_yaxes(scaleanchor="x", scaleratio=1)
        #     fig.update_yaxes(fixedrange=True)
        #     fig.update_layout(autosize=False)
        #     fig.show()
