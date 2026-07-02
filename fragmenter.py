from pathlib import Path
import yaml, re
from rdkit import Chem
from rdkit.Chem import AllChem

#* For consistency, reaction products must be written in parentheses and separated by a dot.
#* Molecules in the resulting Mol object are then separated by Chem.GetMolFrags.
#* In the reaction smarts, the breaking bond must always be between atom 0 and 1.


class FragmentationEngine:
    _reactions = None

    def __init__(self, workdir, verbose=False):
        self.verbose = verbose
        self.workdir = Path(workdir)
        if FragmentationEngine._reactions is None:
            FragmentationEngine._reactions = self._load_reactions()
        self.reactions = FragmentationEngine._reactions
        self.bonds_to_break = {}
        self.frag_mols = []

    def _load_reactions(self):
        path = self.workdir / "reactions/reactions.yml"
        if not path.exists():
            raise FileNotFoundError(f"Reaction file not found: {path}")
        with open(path, "r") as f:
            reaction_smarts = yaml.safe_load(f)
        reactions = {k: AllChem.ReactionFromSmarts(v) for k, v in reaction_smarts.items()}
        return reactions

    def get_bonds_to_break(self, mol):
        """Identify bonds that match any reaction SMARTS."""
        self.bonds_to_break.clear()
        for rxn_name, rxn in self.reactions.items():
            matches = mol.GetSubstructMatches(Chem.Mol(rxn.GetReactantTemplate(0).ToBinary()))
            for match in matches:
                self.bonds_to_break[match] = rxn_name
        if self.verbose:
            print(f"Bonds to break: {len(self.bonds_to_break)} found.")
        return self.bonds_to_break


    def fragment_molecule(self, mol):
        """Break bonds and produce fragments based on reactions."""

        # Label atoms with isotopes for tracking
        for idx, atom in enumerate(mol.GetAtoms()):
            atom.SetIsotope(idx + 1)
            atom.SetProp('_protected', '1')

        final_prods = [mol]
        skipped_bonds = []
        for bond, rxn_name in self.bonds_to_break.items():
            rxn = self.reactions[rxn_name]
            bond_broken = False
            new_prods = []

            for prod in final_prods:
                if prod.HasProp('_freeze_frag') and prod.GetProp('_freeze_frag') == '1':
                    new_prods.append(prod)
                    continue

                if bond_broken:
                    new_prods.append(prod)
                    continue

                # Check presence of bond in the molecule
                isotopes = [a.GetIsotope() for a in prod.GetAtoms()]
                if not all((i + 1) in isotopes for i in bond):
                    new_prods.append(prod)
                    continue

                # Unprotect atoms in the bond to be broken
                for a in prod.GetAtoms():
                    if (a.GetIsotope() - 1) in bond:
                        a.ClearProp('_protected')

                try:
                    Chem.SanitizeMol(prod, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
                except Exception:
                    Chem.SanitizeMol(prod, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ADJUSTHS)

                inter = rxn.RunReactants((prod,))
                if len(inter) == 0:
                    new_prods.append(prod)
                    continue

                prod_new = inter[0][0]
                frags = Chem.GetMolFrags(prod_new, asMols=True, sanitizeFrags=False)
                malformed = False
                for frag in frags:
                    if frag is None:
                        malformed = True
                        break
                    try:
                        Chem.MolToSmiles(frag, kekuleSmiles=True)
                    except Exception:
                        malformed = True
                        break

                if malformed:
                    prod.SetProp('_freeze_frag', '1')
                    new_prods.append(prod)
                    if self.verbose:
                        print(f"Skipping malformed split for bond {bond} ({rxn_name}); keeping parent fragment.")
                    continue

                new_prods.extend(frags)
                bond_broken = True

            final_prods = new_prods
            if not bond_broken:
                skipped_bonds.append(bond)
        
        # Remove skipped bonds to avoid confusion in the connection graph
        if skipped_bonds:
            for b in skipped_bonds:
                if b in self.bonds_to_break:
                    del self.bonds_to_break[b]

        # Keep only fragments that can be serialized safely.
        valid_frags = []
        for frag_mol in final_prods:
            if frag_mol is None:
                continue
            try:
                Chem.MolToSmiles(frag_mol, kekuleSmiles=True)
                valid_frags.append(frag_mol)
            except Exception:
                if self.verbose:
                    print("Skipping malformed fragment during serialization.")

        # Rebuild atom->fragment mapping against the filtered list.
        self.atom_frag = {}
        for i, frag in enumerate(valid_frags):
            for atom in frag.GetAtoms():
                iso = atom.GetIsotope()
                if iso:
                    self.atom_frag[iso - 1] = i

        self.frag_mols = valid_frags

        return self.frag_mols, self.atom_frag
