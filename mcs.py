import pandas as pd
import rdkit
import sys
from rdkit import Chem
from rdkit.Chem import rdFMCS

def aromaticity_as_isotope(mol):
     for atom in mol.GetAtoms():
        value = 500*atom.GetIsAromatic() + atom.GetAtomicNum()
        atom.SetIsotope(value)
     return mol

def similarity_mcs(query_smiles, smiles):
    query_mol = aromaticity_as_isotope(Chem.MolFromSmiles(query_smiles))
    mols = [aromaticity_as_isotope(Chem.MolFromSmiles(s)) for s in smiles]
    params = rdFMCS.MCSParameters()
    params.AtomTyper = rdFMCS.AtomCompare.CompareIsotopes
    params.BondTyper = rdFMCS.BondCompare.CompareOrder
    params.BondCompareParameters.RingMatchesRingOnly = True
    params.BondCompareParameters.CompleteRingsOnly = False
    params.matchValences = True
    mcs1 = [rdFMCS.FindMCS([query_mol,mol_i],params) for mol_i in mols]
    numCommonAtomsBonds = [(mcs11.numAtoms + mcs11.numBonds) for mcs11 in mcs1]
    numAtomsBondsR = [(mol_i.GetNumAtoms() + mol_i.GetNumBonds()) for mol_i in mols]
    numAtomsBondsQ = query_mol.GetNumAtoms() + query_mol.GetNumBonds()
    similarities = [numCommonAtomsBonds[i]/(numAtomsBondsR[i]+numAtomsBondsQ-numCommonAtomsBonds[i])  for i in range(0,len(mols))]
    return similarities

lines= open("tmpForRDKit_" + sys.argv[1] + ".txt",'r').read().splitlines()

similarities = []

i = 0
for line in lines:
      smiles = line.split()
      query_smiles = line.split()[0]
      _ = smiles.pop(0)
      try:
         ds = similarity_mcs(query_smiles, smiles)
         similarities.append(ds)
      except:
         ds = [-1.0 for i in range(len(smiles))]
         similarities.append(ds)
      if i % 1000 == 0:
          print('MSC Similarity: '+str(i))
      i = i+1

df = pd.DataFrame(similarities)
df.to_csv("tmpForRDKit_out_" + sys.argv[1] + ".txt", index=False)

print(rdkit.__version__)
