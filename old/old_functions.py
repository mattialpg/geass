    def replace_BUs(self, query_mol, bu_to_replace, new_bu, bu_class):
        
        def prepare_BUs(bu):
            replacements = {'aas': (Chem.MolFromSmiles('CC(N)C(=O)O'), Chem.MolFromSmiles('C')),  # Working from C-beta onwards
                            'fas': (Chem.MolFromSmarts('C(=O)O'), Chem.MolFromSmiles('[14C](=O)'))}  # Labelling to keep track during replacement
            replacement_centers = {'aas': Chem.MolFromSmarts('[CX3;$(CC(N)C(=O))]'),  # C-beta
                                   'fas': Chem.MolFromSmarts('[14C;$(C(=O))]')}  # If not labelled, the first matching carbonyl group is chosen
            
            # Prepare BUs to be replaced
            bu = Chem.ReplaceSubstructs(bu, replacements[bu_class][0], replacements[bu_class][1])[0]

            # Renumber atoms such that atom 0 is the one involved in bond formation
            patt = bu.GetSubstructMatch(replacement_centers[bu_class])
            new_order = list(patt) + [x for x in range(bu.GetNumAtoms()) if x not in patt][::-1]
            bu = Chem.ReplaceSubstructs(bu, Chem.MolFromSmiles('[14C](=O)'), Chem.MolFromSmiles('C(=O)'))[0]  # Removing labelling
            Chem.SanitizeMol(bu)

            return AllChem.RenumberAtoms(bu, new_order)

        bu_to_replace, new_bu = [prepare_BUs(bu) for bu in [bu_to_replace, new_bu]]

        # Replace BU
        chimera = AllChem.ReplaceSubstructs(query_mol, bu_to_replace, new_bu, useChirality=True)[0]

        #! BUG: ReplaceSubstructs automatically invert chirality even with useChirality=True.
        #! Revert to originl chirality
        # C_beta = query_mol.GetAtomWithIdx(query_mol.GetSubstructMatch(bu_to_replace)[0])
        # C_alpha_idx = C_beta.GetNeighbors()[0].GetIdx()
        # chimera.GetAtomWithIdx(C_alpha_idx).InvertChirality()
        
        chimera = Chem.RemoveHs(chimera)
        Chem.SanitizeMol(chimera)

        return chimera
        # return reset_view(chimera)


def recompose_molecule(self, mol):
    # * MWE of reversing molecular fragmentation
    # It lacks recognition of atoms forming bonds  
    tree = []
    for edge in g.edges:
        node1, node2 = edge
        smiles1 = g.nodes[node1]['SMILES']
        smiles2 = g.nodes[node2]['SMILES']
        tree.append((smiles1, smiles2, g.edges[edge]['reaction']))

    reactions = {'amide': AllChem.ReactionFromSmarts("[NX3;$(N([#6])[#6]):0]-[C:1]=[O:2]>>[#7:0].[C:1](=[O:2])-O")}

    rev_reactions = {}
    for name, rxn in reactions.items():
        rxn_rev = AllChem.ChemicalReaction()
        for i in range(rxn.GetNumReactantTemplates()):
            rxn_rev.AddProductTemplate(rxn.GetReactantTemplate(i))
        for i in range(rxn.GetNumProductTemplates()):
            rxn_rev.AddReactantTemplate(rxn.GetProductTemplate(i))
        rxn_rev.Initialize()
        rev_reactions[name] = rxn_rev

    There shuold be a way to enumerate atoms linked in the original molecule
    So that they can be rejoined again 
    new_prods = rxn_rev.RunReactants('.....')[0][0]
    


    def fragment_molecule(self, reactant_list, reactions_involved, max_depth=10, max_frags=30):
        """
        Fragment the molecule iteratively using the provided reactions.
        Optimised to prevent infinite recursion and redundant recalculations.
        """

        final_prods = reactant_list[:]

        for rxn_name, rxn in list(reactions_involved.items()):
            waiting_list = []

            # Limit recursive fragmentation per reaction
            for depth in range(max_depth):
                intermediate_prods = []
                seen_smiles = set()  # cache SMILES to avoid redundant RunReactants calls

                # Pre-cache SMILES of reactants to avoid recomputation later
                reactant_smiles = {}
                for r in final_prods:
                    try:
                        reactant_smiles[r] = Chem.MolToSmiles(r)
                    except Exception:
                        reactant_smiles[r] = None

                for reactant in final_prods:
                    smi_r = reactant_smiles[reactant]
                    if smi_r in seen_smiles:
                        continue
                    seen_smiles.add(smi_r)

                    try:
                        prod_Mol = rxn.RunReactants((reactant,), maxProducts=1)
                    except Exception:
                        waiting_list.append(reactant)
                        continue

                    if not prod_Mol:
                        waiting_list.append(reactant)
                        continue

                    prod_Mol = prod_Mol[0][0]
                    prod_mols = Chem.GetMolFrags(prod_Mol, asMols=True)
                    intermediate_prods.extend(prod_mols)

                if not intermediate_prods:
                    break  # exit early if nothing new produced

                # Merge waiting and new products
                final_prods = waiting_list + intermediate_prods

                # Deduplicate by SMILES efficiently
                seen = set()
                unique = []
                for r in final_prods:
                    try:
                        smi = Chem.MolToSmiles(r)
                    except Exception:
                        continue
                    if smi not in seen:
                        seen.add(smi)
                        unique.append(r)
                final_prods = unique

                # Safety: prevent runaway growth
                if len(final_prods) > max_frags:
                    print(f"Stopping early: exceeded {max_frags} fragments at depth {depth+1}")
                    break

        # Create a dictionary mapping query_mol atoms to fragment molecules
        for i, prod_mol in enumerate(final_prods):
            for atom in prod_mol.GetAtoms():
                isotope = atom.GetIsotope()
                if isotope:
                    self.atom_to_frag[isotope - 1] = i
        self.frag_mols = final_prods
        return self.frag_mols