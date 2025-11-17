import os, sys
import numpy as np
import pandas as pd
import itertools
from rdkit import Chem
from rdkit.Chem import PandasTools, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
import swifter

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

sys.path.append(os.path.abspath(R"C:\\Users\\Idener\\Sync\\CODE"))
from aiddtools import fingerprint_tools as fpt

normalizer = rdMolStandardize.Normalizer()
uncharger = rdMolStandardize.Uncharger()
def standardize_mol(mol):
    if mol is None:
        return None
    mol = normalizer.normalize(mol)
    mol = uncharger.uncharge(mol)
    mol = rdMolStandardize.Cleanup(mol)
    return mol

def calc_feature_vec(mol, fp='atompair'):
    fp_vec = fpt.calculate_fp(mol, method=fp, as_numpy=True)
    desc_vec = np.array([rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6 and a.GetIsAromatic()),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6 and not a.GetIsAromatic())])
    feature_vec = np.concatenate([desc_vec, fp_vec])
    return feature_vec


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
#!---------------------------------#

#!------ Biosynfoni reactions ------#
df2 = pd.read_csv('Biosynfoni_biosynthetic_chains.tsv', sep='\t', header=None)
pairs = []
for _, row in df2.iterrows():
    smiles_chain = [s for s in row if pd.notna(s) and str(s).strip() != ""]
    for i in range(len(smiles_chain) - 1):
        pairs.append((smiles_chain[i], smiles_chain[i+1]))
df2 = pd.DataFrame(pairs, columns=["Reactant", "Product"])
#!---------------------------------#

#!------ Rhea reactions ------#
df3 = pd.read_csv("Rhea_reactions.tsv", sep='\t').astype(str)
df3 = df3[~df3.apply(lambda row: row.str.contains(R'\*', na=False)).any(axis=1)]

# Generate all RP pairs
df3['Reactants'] = df3['Reactants'].str.split('.')
df3['Products'] = df3['Products'].str.split('.')
df3['Pairs'] = df3.apply(lambda row: list(
    itertools.product( row['Reactants'], row['Products'])), axis=1)
df3 = df3.explode('Pairs', ignore_index=True)

# Split RP pairs into two columns
df3['Reactant'] = df3['Pairs'].str[0]
df3['Product'] = df3['Pairs'].str[1]
df3 = df3.drop(columns=['Pairs'])
PandasTools.AddMoleculeColumnToFrame(df3, smilesCol='Reactant', molCol='ROMol_R')
PandasTools.AddMoleculeColumnToFrame(df3, smilesCol='Product', molCol='ROMol_P')

# Filter by atom count difference (<6)
df3 = df3[df3.swifter.apply(lambda row: abs(
    rdMolDescriptors.CalcNumAtoms(row['ROMol_R']) -
    rdMolDescriptors.CalcNumAtoms(row['ROMol_P'])) < 6, axis=1)]

# Standardize molecules
df3['ROMol_R'] = df3['ROMol_R'].swifter.apply(standardize_mol)
df3['ROMol_P'] = df3['ROMol_P'].swifter.apply(standardize_mol)
df3['Reactant'] = df3['ROMol_R'].swifter.apply(Chem.MolToSmiles)
df3['Product'] = df3['ROMol_P'].swifter.apply(Chem.MolToSmiles)

# Remove duplicates and trivial/inorganic reactions
df3 = df3.drop_duplicates(subset=['Reactant', 'Product']).reset_index(drop=True)
df3 = df3[df3['Reactant'] != df3['Product']]
excluded = ['C', 'CO', 'O=C([O-])O', 'CN', 'CNC', 'CS', '[Cu+2]',
    'Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O',
    'Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O']
df3 = df3[~df3['Reactant'].isin(excluded) | df3['Product'].isin(excluded)]
df3 = df3[df3['Reactant'].str.contains('C') & df3['Product'].str.contains('C')]
df3 = df3[['Reactant', 'Product']].reset_index(drop=True)
#!---------------------------------#

# Create final dataframe of reaction pairs
df_pairs = pd.DataFrame(df1['RPAIR'].tolist(), columns=['Reactant', 'Product'])
df_pairs = pd.concat([df_pairs, df2, df3], ignore_index=True).drop_duplicates().reset_index(drop=True)
df_pairs['Reactant'] = df_pairs['Reactant'].str.replace(R'@+', '', regex=True)
df_pairs['Product'] = df_pairs['Product'].str.replace(R'@+', '', regex=True)
PandasTools.AddMoleculeColumnToFrame(df_pairs, smilesCol='Reactant', molCol='ROMol_R')
PandasTools.AddMoleculeColumnToFrame(df_pairs, smilesCol='Product', molCol='ROMol_P')
df_pairs = df_pairs.dropna(subset=['ROMol_R', 'ROMol_P']).reset_index(drop=True)

# Filter by atom count difference (<6)
df_pairs['ATOMS_R'] = df_pairs['ROMol_R'].apply(lambda x: x.GetNumHeavyAtoms() if x else None)
df_pairs['ATOMS_P'] = df_pairs['ROMol_P'].apply(lambda x: x.GetNumHeavyAtoms() if x else None)
df_pairs['ATOM_DIFF'] = abs(df_pairs['ATOMS_R'] - df_pairs['ATOMS_P'])
df_pairs = df_pairs[df_pairs['ATOM_DIFF'] < 6]

# Calculate feature vectors
df_pairs['Feature_vec_R'] = df_pairs['ROMol_R'].swifter.apply(calc_feature_vec)
df_pairs['Feature_vec_P'] = df_pairs['ROMol_P'].swifter.apply(calc_feature_vec)

# Final cleanup and save
df_pairs = df_pairs[['Reactant', 'Product', 'ROMol_R',
    'ROMol_P', 'Feature_vec_R', 'Feature_vec_P']].reset_index(drop=True)
df_pairs.to_pickle('xgb_reaction_pairs.pkl')
