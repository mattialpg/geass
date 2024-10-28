# import necessary modules


from rdkit.Chem.Draw import rdMolDraw2D
import networkx as nx
from urllib import parse
from rdkit import Chem
from rdkit.Chem import rdChemReactions as Reactions


class Node(object):

    def __init__(self, mol, typ):
        """

        :param mol: the rdkit molecule to be encoded in the node
        :param typ: the type of node: {'root', 'child', 'leaf'}
        """
        self.mol = mol
        self.smiles = Chem.MolToSmiles(mol)
        self.type = typ
        self.img = self.smi2image()

    def nx_attributes(self):
        '''

        :return: the attributes of the node in a dictionary, so that the can be set by Networkx module.
        '''

        return {self.nx_node(): {'type': self.type, 'img': self.img}}

    def nx_node(self):
        '''

        :return: the smiles of the node.
        '''

        return self.smiles

    def smi2svg(self):
        """

        :return: an svg draw encoding the molecule of the node.
        """

        # mol = Chem.MolFromSmiles(self.smiles)
        mol = self.mol
        '''
        try:
            Chem.rdmolops.Kekulize(mol)
        except:
            pass
        '''
        drawer = rdMolDraw2D.MolDraw2DSVG(350, 300)
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()

        return svg

    def smi2image(self):
        '''

        :return: the image to be shown in the node.
        '''

        svg_string = self.smi2svg()
        impath = 'data:image/svg+xml;charset=utf-8,' + parse.quote(svg_string, safe="")

        return impath


class Edge(object):
    def __init__(self, nodeOut, nodeIn, reaction, otherProds):
        """

        :param nodeIn: the Node in which the edge ends
        :param nodeOut: The Node from which the edge begins
        :param reaction: The reaction name
        :param otherProds: a tuple containing the SMILES generate products others than nodeIn
        """
        self.Out = nodeOut
        self.In = nodeIn
        self.reaction = reaction
        self.otherProds = tuple([Chem.MolToSmiles(x) for x in otherProds])

    def nx_edge(self):
        '''

        :return: the edge in Networkx format
        '''

        return self.Out.nx_node(), self.In.nx_node()

    def nx_attributes(self):
        return {self.nx_edge(): {'reaction': self.reaction, 'otherProds': self.otherProds}}


def sort_prodseq(prodseq):
    """
    Sorts a sequence of products by number of atoms
    :param prodseq: the sequence of products
    :return: the sequence of products sorted by number of atoms
    """

    tSeq = [(prod.GetNumAtoms(), idx) for idx, prod in
            enumerate(prodseq)]  # Sort by number of atoms
    tSeq.sort()
    ts = [(x, prodseq[y]) for x, y in tSeq]

    return [_ for _ in ts]


class FragmentNet(object):
    def __init__(self, mol, reactions, bul):
        """

        :param mol: the molecule to be analyzed
        :param smarts: dictionary with names as keys, rdkit reactions as values
        :param bul: the natural products database
        """
        self.mol = mol  # The query mol
        self.smiles = Chem.MolToSmiles(self.mol)
        self.graph = nx.DiGraph()  # The fragmentation graph
        # self.reactions = {k: Reactions.ReactionFromSmarts(v) for k, v in smarts.items()}
        self.reactions = reactions
        self.bul = bul

    def add_node_nx(self, node):
        '''
        Adds a node to the graph
        :param node: the node to be added
        '''
        self.graph.add_node(node.nx_node())
        nx.set_node_attributes(self.graph, node.nx_attributes())

    def add_edge_nx(self, edge):
        '''
        Adds an edge to the graph
        :param edge: the egde to be added
        '''
        self.graph.add_edge(edge.Out.nx_node(), edge.In.nx_node())
        nx.set_edge_attributes(self.graph, edge.nx_attributes())

    def chek_is_bul(self, mol):
        """
        Checks if a molecule is in the natural products database
        :param mol: the rdkit molecule to be checked
        :return: True if mol is found in the database, False otherwise
        """
        for name, molBUL in self.bul.items():
            # iterate over the Natural products library
            if len(mol.GetSubstructMatch(molBUL, )) == mol.GetNumAtoms():
                # Match
                return True

        return False

    def get_leaves(self):
        """
        :return: the smiles of the nodes that are leaves
        """

        types = nx.get_node_attributes(self.graph, 'type')
        return [smiles for smiles, typ in types.items() if typ == 'leave']

    def set_leaves(self):
        """
        Changes the attribute typ to those nodes whose out degree is 0, and sets it as 'leave'.
        """

        leaves = [smiles for smiles, out_degree in self.graph.out_degree if out_degree == 0]
        attrs = {smile: {'type': 'leave'} for smile in leaves}
        nx.set_node_attributes(self.graph, attrs)

    def force_NP_leaves(self):
        """
        Forces that the leaves of the graph were contained in the natural products library.
        :return:
        """
        delta = 1
        while delta != 0:
            nNodes = len(self.graph.nodes)
            leaves = self.get_leaves()
            for leaf in leaves:
                if not self.chek_is_bul(Chem.MolFromSmiles(leaf)):
                    self.graph.remove_node(leaf)
            self.set_leaves()
            delta = nNodes - len(self.graph.nodes)

    def run_recap(self, minFragmentSize=1, core=None, useHs=False, sanitizeFlag=False, keggAlign=True, forceNPLeaves=False,
                  verbose=False, maxNodes=100):
        """
        Executes the recap algorithm.
        :param minFragmentSize: if a fragment is below this size, it is removed from the pool
        :param useHs: if the reaction rules have been defined using explicit hydrogens, set this parameter to true
        :param sanitizeFlag: whether the molecules need to be sanitized
        :param keggAlign: if this argument is set to True, only those nodes with a molecule in the natural product database are considered
        :param forceNPLeaves: forces the leaves to be in the natural product database
        :param verbose: whether to show messages during the execution
        :param maxNodes: the maximum number of nodes in the graph
        :return: True if the fragmentation was successful, False otherwise.
        """
        node = Node(self.mol, 'root')
        self.add_node_nx(node)
        nodesPool = {Chem.MolToSmiles(self.mol): node}  # Includes the first molecule in the pool
        while nodesPool:

            if len(self.graph.nodes) > maxNodes:
                self.set_leaves()
                print(f'The Maximum number of nodes ({maxNodes}) was achieved')
                return False

            # Get the next smiles in the nodesPool
            nSmi = next(iter(nodesPool))
            # Access the corresponding  node and remove it from the Pool
            node = nodesPool.pop(nSmi)

            if not node.mol:
                # Checks the node contains a molecule
                continue

            for rxnIdx, (reactionName, reaction) in enumerate(self.reactions.items()):
                if verbose:
                    print('Nodes in Pool ' + str(len(nodesPool)), end='\r')
                # Iterate over the reactions

                if useHs:
                    # Consider the case in which the rules are defined with explicit Hydrogens
                    reactant = Chem.rdmolops.AddHs(node.mol)
                    Chem.Kekulize(reactant)
                else:
                    reactant = node.mol
                    Chem.Kekulize(reactant)

                reactionResults = reaction.RunReactants((reactant,))
                if reactionResults:
                    # If the reaction applies

                    for prodSeq in reactionResults:
                        # Iterate over all the possible products

                        seqOk = True  # The sequence is valid
                        prodSeq = sort_prodseq(prodSeq)

                        if useHs:
                            # Remove the hydrogens in the products
                            prodSeq = [(nats, Chem.MolFromSmiles(Chem.MolToSmiles(prod))) for nats, prod in prodSeq if
                                       '.' not in Chem.MolToSmiles(prod, 1)]
                        else:
                            prodSeq = [(nats, prod) for nats, prod in prodSeq if '.' not in Chem.MolToSmiles(prod, 1)]
                        for nats, prod in prodSeq:

                            # Iterate over the products (molecules) in the products sequence

                            if sanitizeFlag:
                                try:
                                    # Molecules must be sanitize to avoid invalid valences
                                    Chem.SanitizeMol(prod)
                                except Exception:
                                    continue

                            # Get the SMILES of the product
                            pSmi = Chem.MolToSmiles(prod, 1)

                            if minFragmentSize > 0:
                                # Check the fragment size
                                nDummies = pSmi.count('*')
                                if nats - nDummies < minFragmentSize:
                                    seqOk = False
                                    # If a fragment is below the allowed size the complete sequence is dicarded
                                    break

                                elif pSmi.replace('*', '').replace('()', '') in ('', 'C', 'CC', 'CCC'):
                                    seqOk = False
                                    break

                            prod.pSmi = pSmi

                        if seqOk:
                            # If no problems were found in the sequence add the products to the network
                            for i, (nats, prod) in enumerate(prodSeq):
                                otherProds = tuple([y for x, y in prodSeq[:i] + prodSeq[i + 1:]])  # The
                                # other products of the reaction
                                if core is not None:
                                    if prod.HasSubstructMatch(core):
                                        pSmi = prod.pSmi
                                        pNode = Node(prod, 'children')  # All are children initially
                                        edge = Edge(node, pNode, reactionName, otherProds)

                                        if pSmi not in self.graph.nodes:
                                            # Include the new node in the graph and in the Pool
                                            self.add_node_nx(pNode)
                                            nodesPool[pSmi] = pNode

                                        self.add_edge_nx(edge)

                                if keggAlign:
                                    # Adds nodes ony if they are in the natural products database
                                    if len(otherProds) != 0:
                                        if self.chek_is_bul(otherProds[0]):

                                            # Consider only the case in which the other products are in NP database

                                            pSmi = prod.pSmi
                                            pNode = Node(prod, 'children')  # All are children initially
                                            edge = Edge(node, pNode, reactionName, otherProds)

                                            if pSmi not in self.graph.nodes:
                                                # Include the new node in the graph and in the Pool
                                                self.add_node_nx(pNode)
                                                nodesPool[pSmi] = pNode

                                            self.add_edge_nx(edge)

                                    if len(otherProds) == 0:
                                        # Considers the case of ring opening or molecule modification

                                        pSmi = prod.pSmi
                                        pNode = Node(prod, 'children')  # All are children initially
                                        edge = Edge(node, pNode, reactionName, otherProds)

                                        if pSmi not in self.graph.nodes:
                                            # Include the new node in the graph and in the Pool
                                            self.add_node_nx(pNode)
                                            nodesPool[pSmi] = pNode

                                        self.add_edge_nx(edge)
                                if core is None and not keggAlign:
                                    # If keggAlign is False, includes the node always
                                    pSmi = prod.pSmi
                                    pNode = Node(prod, 'children')  # All are children initially
                                    edge = Edge(node, pNode, reactionName, otherProds)

                                    if pSmi not in self.graph.nodes:
                                        # Include the new node in the graph and in the Pool
                                        self.add_node_nx(pNode)
                                        nodesPool[pSmi] = pNode

                                    self.add_edge_nx(edge)

        self.set_leaves()
        if forceNPLeaves:
            self.force_NP_leaves()

        if len(self.graph.nodes) < 2:
            # The fragmentation was not successful
            return False

        return True
