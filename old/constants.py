from rdkit import Chem
from rdkit.Chem import rdChemReactions as Reactions

def create_reactions():
    '''
    Creates the reactions

    Note:
        These are the definitions that will be applied to fragment molecules:

        They include the originally defined reaction rules in RECAP and
        the modified version with the openning ring capability. 

        They DO NOT keep the dummy atoms.
    '''
    reactionDefs = (
    "[#7;+0;D2,D3:1]!@C(!@=O)!@[#7;+0;D2,D3:2]>>[#7:1].[#7:2]", # urea + bicarbonato CHECK
    "[#7;+0;D2,D3:1]C(!@=O)[#7;+0;D2,D3:2]>>([#7:1].[#7:2])", # urea ring

    "[C;!$(C([#7])[#7]):1](=!@[O:2])!@[#7;+0;!D1:3]>>O[C:1]=[O:2].[#7:3]",  # amide not cyclic
    "[C;!$(C([#7])[#7]):1](=[O:2])[#7;+0;!D1:3]>>(O[C:1]=[O:2].[#7:3])", # amide macrocycles

    "[*:4][C:1](=!@[O:2])!@-[O;+0:3][*:5]>>[*:4][C:1](O)=[O:2].[O:3][*:5]", # ester !!!!!!!!!!!!
    #"[*:4][C:1](=!@[O:2])[O;+0:3]>>([*:4][C:1](O)=[O:2].[O:3])", # ester ring

    "[N;!D1;+0;!$(N-C=[#7,#8,#15,#16])](-!@[*:1])-!@[*:2]>>[*:1].[*:2]", # amines + amoniaco(?) CHECK
    "[N;!D1;+0;!$(N-C=[#7,#8,#15,#16])](-[*:1])-[*:2]>>([*:1].[*:2])", # amines ring
    #"[N;!D1](!@[*:1])!@[*:2]>>[*][*:1].[*:2][*]", # amines

    # again: what about aromatics?
    "[#7;R;D3;+0:1]-!@[*:2]>>[#7:1].[*:2][O]", # cyclic amines CHECK
    #"[#7;R;D3;+0:1]-[*:2]>>([#7:1].[*:2][O])", # cyclic amines ring # Seems too general?

    "[#6:1]-!@[O;+0]-!@[#6:2]>>[#6:1][O].[O][#6:2]", # ether CHECK
    "[#6:1]-[O;+0;!r6;!r5]-[#6:2]>>([#6:1][O].[O][#6:2])", # ether ring

    "[C:1]=!@[C:2]>>[C:1].[C:2]", # olefin
    #"[C:1]=[C:2]>>([C:1][*].[*][C:2])", # olefin ring

    "[n;+0:1]-!@[C:2]>>[n:1].[C:2]", # aromatic nitrogen - aliphatic carbon
    #"[n;+0:1]-[C:2]>>([n:1][*].[C:2][*])", # aromatic nitrogen - aliphatic carbon ring

    "[O:3]=[C:4]-@[N;+0:1]-!@[C:2]>>[O:3]=[C:4]-[N:1].[C:2]", # lactam nitrogen - aliphatic carbon
    #"[O:3]=[C:4]-@[N;+0:1]-[C:2]>>([O:3]=[C:4]-[N:1][*].[C:2][*])", # lactam nitrogen - aliphatic carbon ring

    "[c:1]-!@[c:2]>>[c:1].[c:2]", # aromatic carbon - aromatic carbon
    "[c:1]-[c:2]>>([c:1].[c:2])", # aromatic carbon - aromatic carbon ring

    "[n;+0:1]-!@[c:2]>>[n:1].[c:2]", # aromatic nitrogen - aromatic carbon *NOTE* this is not part of the standard recap set.
    "[n;+0:1]-[c:2]>>([n:1].[c:2])", # aromatic nitrogen - aromatic carbon ring *NOTE* this is not part of the standard recap set.

    "[#7;+0;D2,D3:1]-!@[S:2](=[O:3])=[O:4]>>[#7:1].[S:2](=[O:3])=[O:4]", # sulphonamide
    #"[#7;+0;D2,D3:1]-[S:2](=[O:3])=[O:4]>>([#7:1][*].[*][S:2](=[O:3])=[O:4])", # sulphonamide ring
    )

    names = ['urea', 'ureaR', 'amide', 'amideR', 'ester', 'amines', 'aminesR', 'cyclicAmines',
            'ether', 'etherR', 'olefin', 'arNalC', 'laNalC', 'arCarC', 'arCarCR', 'arNarC',
            'arNarCR', 'sulphonamide']
    reactions = dict(zip(names, reactionDefs))
    reactionsMol2 = {k: Reactions.ReactionFromSmarts(v) for k, v in reactions.items()}
    return reactionsMol2

def create_reactants(reactionsMol2):
    '''
    Creates the reactants
    '''
    reactants = {k:v.GetReactants()[0] for k, v in reactionsMol2.items()}
    for i, (k, v) in enumerate(reactants.items()):
        Chem.GetSymmSSSR(v)
        reactants[k] = v
    return reactants

def create_bonds():
    '''
    Defines the bonds that will be broken in each reaction
    '''

    bondsMols = {
        'urea': [Chem.MolFromSmarts('[#7;+0;D2,D3:1]!@C'), Chem.MolFromSmarts('C!@[#7;+0;D2,D3:2]')],
        'ureaR': [Chem.MolFromSmarts('[#7;+0;D2,D3:1]C'), Chem.MolFromSmarts('C[#7;+0;D2,D3:2]')],
        'amide': [Chem.MolFromSmarts('[C;!$(C([#7])[#7]):1]!@[#7;+0;!D1:3]')],
        'amideR': [Chem.MolFromSmarts('[C;!$(C([#7])[#7]):1][#7;+0;!D1:3]')],
        'ether': [Chem.MolFromSmarts('[O;+0]-!@[#6:2]')],
        'etherR': [Chem.MolFromSmarts('[O;+0;!r6;!r5]-[#6:2]')],
        'ester': [Chem.MolFromSmarts('[C:1]!@-[O;+0:3]')],
        #esterR': [Chem.MolFromSmarts('[C:1][O;+0:3]')],
        'amines': [Chem.MolFromSmarts('[N;!D1;+0;!$(N-C=[#7,#8,#15,#16])]-!@[*:1]'), Chem.MolFromSmarts('[N;!D1;+0;!$(N-C=[#7,#8,#15,#16])]-!@[*:2]')],
        'aminesR': [Chem.MolFromSmarts('[N;!D1;+0;!$(N-C=[#7,#8,#15,#16])]-[*:1]'), Chem.MolFromSmarts('[N;!D1;+0;!$(N-C=[#7,#8,#15,#16])]-[*:2]')],
        'cyclicAmines': [Chem.MolFromSmarts('[#7;R;D3;+0:1]-!@[*:2]')],
        #'cyclicAminesR': [Chem.MolFromSmarts('[#7;R;D3;+0:1]-[*:2]')],
        'olefin': [Chem.MolFromSmarts('[C:1]=!@[C:2]')],
        'arNalC': [Chem.MolFromSmarts('[c:1]-!@[c:2]')],
        'laNalC': [Chem.MolFromSmarts('[N;+0:1]-!@[C:2]')],
        'arCarC': [Chem.MolFromSmarts('[c:1]-!@[c:2]')],
        'arCarCR': [Chem.MolFromSmarts('[c:1]-[c:2]')],
        'arNarC': [Chem.MolFromSmarts('[n;+0:1]-!@[c:2]')],
        'arNarCR': [Chem.MolFromSmarts('[n;+0:1]-[c:2]')],
        'sulphonamide': [Chem.MolFromSmarts('[#7;+0;D2,D3:1]-!@[S:2]')]
        
    }


    for k, v in bondsMols.items():
        lst = []
        for molecule in v:
            Chem.GetSymmSSSR(molecule)
            lst.append(molecule)
        bondsMols[k] = lst
        
    brokenBondsDefs = {k: [molecule.GetBonds()[0] for molecule in v] for k, v in bondsMols.items()}
    return brokenBondsDefs

def create_lipid_bonds():
    '''
    Defines the bonds that will be broken in each reaction
    '''

    bondsMols = {
        # 'ether': [Chem.MolFromSmarts('[O;+0]-!@[#6:2]')],
        # 'etherR': [Chem.MolFromSmarts('[O;+0;!r6;!r5]-[#6:2]')],
        'ester': [Chem.MolFromSmarts('[C:1]!@-[O;+0:3]')]
    }


    for k, v in bondsMols.items():
        lst = []
        for molecule in v:
            Chem.GetSymmSSSR(molecule)
            lst.append(molecule)
        bondsMols[k] = lst
        
    brokenBondsDefs = {k: [molecule.GetBonds()[0] for molecule in v] for k, v in bondsMols.items()}
    return brokenBondsDefs

def create_atom_mapping_rules():
    # Atom mapping rules according to the neighbor of the dummy atom. 
    # allChars is employed when any atom is replaced.
    # '' wil be replaced by hidrogens so that the valences match

    allChars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    aux = {ch:'O' for ch in allChars}
    aux.update({'N': ''})
    fillingAtoms = {'ureaR':{'N':'', 'C':'O'},
                    'urea':{'N':'', 'C':'O'},
                    'amide': {'N': '', 'C': 'O'},
                    'amideR': {'N': '', 'C': 'O'},
                    'ester': {'O': '', 'C': 'O'},
                    #'esterR': {'O': '', 'C': 'O'},
                    'ether': {'O': '', 'C': 'O', 'c':'O'},
                    'etherR': {'O': '', 'C': 'O', 'c':'O', 'o':''},
                    'amines': {ch:'' for ch in allChars},
                    'aminesR': {ch:'' for ch in allChars},
                    'cyclicAmines': aux,
                    #'cyclicAminesR': {ch:'O' for ch in allChars}.update({'N': ''}),
                    'olefin': {'C': ''},
                    'arNalC': {'N': '', 'C': '', 'n': '', 'c': ''},
                    'laNalC': {'N': '', 'C': '', 'n': '', 'c': ''},
                    'arCarC': {'C': '', 'c': ''},
                    'arCarCR': {'C': '', 'c': ''},
                        'arNarC': {'C': '', 'c': '', 'N': '', 'n': ''},
                    'arNarCR': {'C': '', 'c': '', 'N': '', 'n': ''},
                    'sulphonamide': {'N': '', 'S': ''}}
    return fillingAtoms

REACTIONS = create_reactions()
REACTANTS = create_reactants(REACTIONS)
BROKEN_BONDS = create_bonds()
BROKEN_BONDS_LIPIDS = create_lipid_bonds()
FILLING_ATOMS = create_atom_mapping_rules()
