import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs


# ---------------------------------------------------------
# SMILES → RDKit Mol
# ---------------------------------------------------------

def smiles_to_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol


# ---------------------------------------------------------
# BASIC PHYSICOCHEMICAL DESCRIPTORS
# ---------------------------------------------------------

def compute_physchem_descriptors(mol):
    if mol is None:
        return None

    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "RotatableBonds": Descriptors.NumRotatableBonds(mol),
        "AromaticRings": Descriptors.NumAromaticRings(mol)
    }


def add_physchem_features(df, smiles_col="smiles"):
    descriptor_list = []

    for smi in df[smiles_col]:
        mol = smiles_to_mol(smi)
        desc = compute_physchem_descriptors(mol)
        descriptor_list.append(desc)

    desc_df = pd.DataFrame(descriptor_list)
    return pd.concat([df.reset_index(drop=True), desc_df], axis=1)


# ---------------------------------------------------------
# MORGAN (ECFP) FINGERPRINTS
# ---------------------------------------------------------

def compute_morgan_fingerprint(mol, radius=2, n_bits=2048):
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius,
        nBits=n_bits
    )
    arr = np.zeros((1,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def add_morgan_features(df, smiles_col="smiles", radius=2, n_bits=2048):
    fingerprint_matrix = []

    for smi in df[smiles_col]:
        mol = smiles_to_mol(smi)
        if mol:
            fp = compute_morgan_fingerprint(mol, radius, n_bits)
        else:
            fp = np.zeros(n_bits)

        fingerprint_matrix.append(fp)

    fp_df = pd.DataFrame(
        fingerprint_matrix,
        columns=[f"ECFP_{i}" for i in range(n_bits)]
    )

    return pd.concat([df.reset_index(drop=True), fp_df], axis=1)


# ---------------------------------------------------------
# MACCS FINGERPRINTS
# ---------------------------------------------------------

def compute_maccs_fingerprint(mol):
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((1,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def add_maccs_features(df, smiles_col="smiles"):
    maccs_matrix = []

    for smi in df[smiles_col]:
        mol = smiles_to_mol(smi)
        if mol:
            fp = compute_maccs_fingerprint(mol)
        else:
            fp = np.zeros(167)

        maccs_matrix.append(fp)

    maccs_df = pd.DataFrame(
        maccs_matrix,
        columns=[f"MACCS_{i}" for i in range(167)]
    )

    return pd.concat([df.reset_index(drop=True), maccs_df], axis=1)


# ---------------------------------------------------------
# BEMIS–MURCKO SCAFFOLD
# ---------------------------------------------------------

def compute_scaffold(smiles):
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def add_scaffold_column(df, smiles_col="smiles"):
    df["scaffold"] = df[smiles_col].apply(compute_scaffold)
    return df


# ---------------------------------------------------------
# LIPINSKI RULE OF FIVE
# ---------------------------------------------------------

def lipinski_rule_of_five(mol):
    return (
        Descriptors.MolWt(mol) <= 500 and
        Descriptors.MolLogP(mol) <= 5 and
        Descriptors.NumHDonors(mol) <= 5 and
        Descriptors.NumHAcceptors(mol) <= 10
    )


def add_lipinski_flag(df, smiles_col="smiles"):
    flags = []

    for smi in df[smiles_col]:
        mol = smiles_to_mol(smi)
        if mol:
            flags.append(int(lipinski_rule_of_five(mol)))
        else:
            flags.append(0)

    df["lipinski_pass"] = flags
    return df


# ---------------------------------------------------------
# FINAL ML FEATURE MATRIX
# ---------------------------------------------------------

def prepare_ml_features(
    df,
    smiles_col="smiles",
    use_physchem=True,
    use_morgan=True,
    use_maccs=False
):
    df_processed = df.copy()

    if use_physchem:
        df_processed = add_physchem_features(df_processed, smiles_col)

    if use_morgan:
        df_processed = add_morgan_features(df_processed, smiles_col)

    if use_maccs:
        df_processed = add_maccs_features(df_processed, smiles_col)

    df_processed = add_lipinski_flag(df_processed, smiles_col)

    # Drop non-numeric columns except target
    numeric_df = df_processed.select_dtypes(include=[np.number])

    return numeric_df
