import numpy as np
import pandas as pd
import itertools
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem.MolStandardize import rdMolStandardize
import swifter

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def standardize_mol(mol):
    _normalizer = rdMolStandardize.Normalizer()
    _uncharger = rdMolStandardize.Uncharger()
    if mol is None:
        return None
    mol = _normalizer.normalize(mol)
    mol = _uncharger.uncharge(mol)
    mol = rdMolStandardize.Cleanup(mol)
    return mol


#!------ Load KEGG reactions ------#
# Load and filter KEGG compounds
df_cpd = pd.read_parquet('KEGG_cpd.gzip', engine='fastparquet')
df_cpd = df_cpd.iloc[20:].dropna(subset=['FORMULA', 'INCHI', 'REACTION'])
df_cpd = df_cpd[df_cpd['MOL_WEIGHT'] > 50]

allowed_elements = R'(C|H|N|O|F|Cl|Br|P|S)'
pattern = rf'^({allowed_elements}[0-9]*)+$'
df_cpd = df_cpd[
    df_cpd['FORMULA'].str.fullmatch(pattern, case=False, na=False) &
    df_cpd['FORMULA'].str.contains('C', case=False, na=False) &
    df_cpd['FORMULA'].str.contains('H', case=False, na=False)]

df1 = pd.read_parquet('KEGG_rn.gzip', engine='fastparquet')
df1 = df1.explode('RPAIR')
df1['RPAIR'] = df1['RPAIR'].apply(lambda x: x.split('_'))
df1['RPAIR'] = df1['RPAIR'].apply(lambda x: [df_cpd.loc[df_cpd['ENTRY'] == r, 'ISOSMILES'].values[0]
    if not df_cpd.loc[df_cpd['ENTRY'] == r, 'ISOSMILES'].empty else None for r in x])
df1 = df1[df1['RPAIR'].apply(lambda x: all(x))]


#!------ Biosynfoni reactions ------#
df2 = pd.read_csv('Biosynfoni_biosynthetic_chains.tsv', sep='\t', header=None)
pairs = []
for _, row in df2.iterrows():
    smiles_chain = [s for s in row if pd.notna(s) and str(s).strip() != ""]
    for i in range(len(smiles_chain) - 1):
        pairs.append((smiles_chain[i], smiles_chain[i+1]))
df2 = pd.DataFrame(pairs, columns=["Reactant", "Product"])


#!------ Rhea reactions ------#
df_pairs = pd.read_csv("Rhea_reactions.tsv", sep='\t').astype(str)
df_pairs = df_pairs[~df_pairs.apply(lambda row: row.str.contains(R'\*', na=False)).any(axis=1)]

# Generate all RP pairs
df_pairs['Reactants'] = df_pairs['Reactants'].str.split('.')
df_pairs['Products'] = df_pairs['Products'].str.split('.')
df_pairs['Pairs'] = df_pairs.apply(lambda row: list(
    itertools.product( row['Reactants'], row['Products'])), axis=1)
df_pairs = df_pairs.explode('Pairs', ignore_index=True)

# Split RP pairs into two columns
df_pairs['Reactant'] = df_pairs['Pairs'].str[0]
df_pairs['Product'] = df_pairs['Pairs'].str[1]
df_pairs = df_pairs.drop(columns=['Pairs'])
PandasTools.AddMoleculeColumnToFrame(df_pairs, smilesCol='Reactant', molCol='ROMol_R')
PandasTools.AddMoleculeColumnToFrame(df_pairs, smilesCol='Product', molCol='ROMol_P')

# Remove duplicates and trivial/inorganic reactions
df_pairs = df_pairs.drop_duplicates(subset=['Reactant', 'Product']).reset_index(drop=True)
df_pairs = df_pairs[df_pairs['Reactant'] != df_pairs['Product']]
excluded = ['C', 'CO', 'O=C([O-])O', 'CN', 'CNC', 'CS', '[Cu+2]',
    'Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O',
    'Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O']
df_pairs = df_pairs[~df_pairs['Reactant'].isin(excluded) | df_pairs['Product'].isin(excluded)]
df_pairs = df_pairs[df_pairs['Reactant'].str.contains('C') & df_pairs['Product'].str.contains('C')]
df_pairs = df_pairs[['Reactant', 'Product']].reset_index(drop=True)


#!------ Final dataframe ------#
df_pairs = pd.DataFrame(df1['RPAIR'].tolist(), columns=['Reactant', 'Product'])
df_pairs = pd.concat([df_pairs, df2, df_pairs], ignore_index=True).drop_duplicates()
df_pairs['Reactant'] = df_pairs['Reactant'].str.replace(R'@+', '', regex=True)
df_pairs['Product'] = df_pairs['Product'].str.replace(R'@+', '', regex=True)
PandasTools.AddMoleculeColumnToFrame(df_pairs, smilesCol='Reactant', molCol='ROMol_R')
PandasTools.AddMoleculeColumnToFrame(df_pairs, smilesCol='Product', molCol='ROMol_P')
df_pairs = df_pairs.dropna(subset=['ROMol_R', 'ROMol_P'])

# Standardize molecules
df_pairs['ROMol_R'] = df_pairs['ROMol_R'].swifter.apply(standardize_mol)
df_pairs['ROMol_P'] = df_pairs['ROMol_P'].swifter.apply(standardize_mol)
df_pairs['Reactant'] = df_pairs['ROMol_R'].swifter.apply(Chem.MolToSmiles)
df_pairs['Product'] = df_pairs['ROMol_P'].swifter.apply(Chem.MolToSmiles)
df_pairs = df_pairs.dropna(subset=['ROMol_R', 'ROMol_P'])

# Filter by atom count difference (<6)
df_pairs['ATOMS_R'] = df_pairs['ROMol_R'].apply(lambda x: x.GetNumHeavyAtoms())
df_pairs['ATOMS_P'] = df_pairs['ROMol_P'].apply(lambda x: x.GetNumHeavyAtoms())
df_pairs['ATOM_DIFF'] = abs(df_pairs['ATOMS_R'] - df_pairs['ATOMS_P'])
df_pairs = df_pairs[df_pairs['ATOM_DIFF'] < 6]

# # Calculate feature vectors
# df_pairs['Feature_vec_R'] = df_pairs['ROMol_R'].swifter.apply(calc_feature_vec)
# df_pairs['Feature_vec_P'] = df_pairs['ROMol_P'].swifter.apply(calc_feature_vec)

# Final cleanup and save
# df_pairs = df_pairs[['Reactant', 'Product', 'ROMol_R', 
#     'ROMol_P', 'Feature_vec_R', 'Feature_vec_P']]
# df_pairs.to_pickle('xgb_reaction_pairs.pkl')

df_pairs = df_pairs[['Reactant', 'Product', 'ROMol_R', 'ROMol_P']].reset_index(drop=True)
df_pairs.to_pickle('reaction_pairs.pkl')