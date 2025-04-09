#!/usr/bin/env python
import pandas as pd
import random
from tqdm import tqdm
from Bio import pairwise2
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np

def partition_by_sequence_identity(df):
    df['sequence'] = df['sequence'].astype(str)
    sequences = df['sequence'].tolist()
    groups = df['mapped_rxn_new'].tolist()
    group_to_sequences = df.groupby('mapped_rxn_new')['sequence'].apply(list).to_dict()

    def sequence_identity(seq1, seq2):
        alignments = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)
        seqA, seqB, score, start, end = alignments[0]
        matches = sum(res1 == res2 for res1, res2 in zip(seqA, seqB))
        return matches / len(seqA)

    max_identities = []
    sample_limit = 100
    for idx in tqdm(range(len(df))):
        seq = sequences[idx]
        group = groups[idx]
        other_sequences = []
        for other_group, seqs in group_to_sequences.items():
            if other_group != group:
                other_sequences.extend(seqs)
        sampled_sequences = random.sample(other_sequences, sample_limit) if len(other_sequences) > sample_limit else other_sequences
        max_identity = 0.0
        for other_seq in sampled_sequences:
            identity = sequence_identity(seq, other_seq)
            if identity > max_identity:
                max_identity = identity
            if max_identity >= 0.8:
                break
        max_identities.append(max_identity)

    categories = []
    for identity in max_identities:
        if identity <= 0.4:
            categories.append('0-40%')
        elif identity <= 0.6:
            categories.append('40-60%')
        elif identity <= 0.8:
            categories.append('60-80%')
        else:
            categories.append('>80%')
    
    df['max_identity'] = max_identities
    df['category'] = categories
    df['set'] = ''
    for cat in ['0-40%', '40-60%', '60-80%', '>80%']:
        cat_indices = df[df['category'] == cat].index.tolist()
        n_samples = len(cat_indices)
        n_test = max(1, int(n_samples * 0.1))
        test_indices = random.sample(cat_indices, n_test)
        train_indices = list(set(cat_indices) - set(test_indices))
        df.loc[train_indices, 'set'] = 'train'
        df.loc[test_indices, 'set'] = 'test'
    return df

def partition_by_molecular_similarity(df):
    smiles_list = df['mapped_rxn_new'].tolist()

    def smiles_to_fp(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    
    fps = []
    valid_indices = []
    for idx, smi in enumerate(tqdm(smiles_list)):
        fp = smiles_to_fp(smi)
        if fp is not None:
            fps.append(fp)
            valid_indices.append(idx)
    invalid_indices = list(set(range(len(df))) - set(valid_indices))
    num_mols = len(fps)
    similarity_matrix = np.zeros((num_mols, num_mols))
    for i in tqdm(range(num_mols)):
        fp_i = fps[i]
        similarities = DataStructs.BulkTanimotoSimilarity(fp_i, fps)
        similarity_matrix[i, :] = similarities
    np.fill_diagonal(similarity_matrix, 0)
    max_similarities = similarity_matrix.max(axis=1)
    categories = []
    for sim in max_similarities:
        if sim <= 0.7:
            categories.append('0-70%')
        elif sim <= 0.85:
            categories.append('70-85%')
        elif sim <= 0.95:
            categories.append('85-95%')
        else:
            categories.append('>95%')
    
    df_valid = df.iloc[valid_indices].copy()
    df_valid['max_similarity'] = max_similarities
    df_valid['category'] = categories
    df_valid['set'] = ''
    for cat in ['0-70%', '70-85%', '85-95%', '>95%']:
        cat_indices = df_valid[df_valid['category'] == cat].index.tolist()
        n_samples = len(cat_indices)
        n_test = max(1, int(n_samples * 0.1))
        test_indices = random.sample(cat_indices, n_test)
        train_indices = list(set(cat_indices) - set(test_indices))
        df_valid.loc[train_indices, 'set'] = 'train'
        df_valid.loc[test_indices, 'set'] = 'test'
    
    df_invalid = df.iloc[invalid_indices].copy()
    df_invalid['max_similarity'] = None
    df_invalid['category'] = None
    df_invalid['set'] = 'invalid'
    
    df_result = pd.concat([df_valid, df_invalid], axis=0)
    return df_result

def partition_by_combined_similarity(df_seq, df_mol):
    df_seq = df_seq.sort_index()
    df_mol = df_mol.sort_index()
    df_merged = df_seq.copy()
    df_merged['max_mol_similarity'] = df_mol['max_mol_similarity']
    if 'max_seq_identity' not in df_merged.columns or 'max_mol_similarity' not in df_merged.columns:
        raise ValueError("Missing required columns: 'max_seq_identity' or 'max_mol_similarity'")
    condition = (df_merged['max_seq_identity'] < 0.7) & (df_merged['max_mol_similarity'] < 0.9)
    eligible_indices = df_merged[condition].index.tolist()
    num_total = len(df_merged)
    desired_test_size = max(1, int(num_total * 0.1))
    if len(eligible_indices) >= desired_test_size:
        test_indices = random.sample(eligible_indices, desired_test_size)
    else:
        test_indices = eligible_indices.copy()
        remaining_needed = desired_test_size - len(test_indices)
        remaining_indices = list(set(df_merged.index) - set(eligible_indices))
        additional_test_indices = random.sample(remaining_indices, remaining_needed)
        test_indices.extend(additional_test_indices)
    df_merged['set'] = 'train'
    df_merged.loc[test_indices, 'set'] = 'test'
    return df_merged

def main():
    # Example:
    # df_combined = pd.read_pickle('combined_data.pkl')
    # df_seq = partition_by_sequence_identity(df_combined.copy())
    # df_mol = partition_by_molecular_similarity(df_combined.copy())
    # df_combined_final = partition_by_combined_similarity(df_seq, df_mol)
    pass

if __name__ == "__main__":
    main()
