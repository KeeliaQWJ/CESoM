#!/usr/bin/env python
import os
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import torch
from torch.utils.data import DataLoader
import dgl
from dgllife.utils import (
    smiles_to_bigraph,
    CanonicalAtomFeaturizer,
    CanonicalBondFeaturizer,
    mol_to_bigraph,
    mol_to_complete_graph
)
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from model_triatt import AtomProteinAttentionGNN 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
edge_featurizer = CanonicalBondFeaturizer(bond_data_field='e')

def canonicalizatonsmi(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    mol = Chem.RemoveHs(mol)
    newsmi = Chem.MolToSmiles(mol, canonical=True)
    return newsmi

def collate_fn(samples):
    g_list = [sample[0] for sample in samples]
    g_complete_list = [sample[1] for sample in samples]
    atom_feats_list = [sample[2] for sample in samples]
    protein_feats_list = [sample[3] for sample in samples]
    atom_labels_list = [sample[4] for sample in samples]

    batched_graph = dgl.batch(g_list)
    batched_graph_c = dgl.batch(g_complete_list)

    return batched_graph, batched_graph_c, atom_feats_list, protein_feats_list, atom_labels_list

def get_test_dataloader(test_data_path, limit=None):
    """
    Load test data and construct a DataLoader.
    """
    with open(test_data_path, 'rb') as f:
        df_test = pickle.load(f)
    if limit is not None:
        df_test = df_test[:limit]

    smiles_list = list(df_test['smiles_list'])
    test_smiles_list = [canonicalizatonsmi(smi) for smi in smiles_list]
    test_protein_features_list = list(df_test['active_site_features'])

    dataset = []
    valid_samples_idx = []
    for idx, smiles in enumerate(test_smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        g = mol_to_bigraph(
            mol=mol, 
            node_featurizer=node_featurizer,
            edge_featurizer=edge_featurizer,
            canonical_atom_order=False
        )
        g_c = mol_to_complete_graph(mol)

        g_label = []
        for atom in mol.GetAtoms():
            is_reaction_site = 1 if atom.GetAtomMapNum() == 1 else 0
            g_label.append([is_reaction_site])
        g_label = torch.Tensor(g_label)
        g.ndata['atom label'] = g_label

        g_mask = torch.Tensor([[1] for _ in mol.GetAtoms()])
        g.ndata['mask'] = g_mask

        atom_feats = g.ndata['h'].float()

        protein_features = test_protein_features_list[idx]
        if protein_features is None or not isinstance(protein_features, (np.ndarray, list, torch.Tensor)) or len(protein_features) == 0:
            continue

        if isinstance(protein_features, torch.Tensor):
            protein_feats = protein_features.float()
        else:
            protein_feats = torch.tensor(protein_features, dtype=torch.float32)

        dataset.append((g, g_c, atom_feats, protein_feats, g_label))
        valid_samples_idx.append(idx)

    test_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False
    )

    return test_loader, valid_samples_idx

def test_model(model, test_loader, test_data_path, valid_samples_idx, output_path):
    """
    Evaluate the model and save predictions.
    """
    model.eval()
    all_atom_predictions = []

    with torch.no_grad():
        for batch_data in test_loader:
            batched_graph, batched_graph_c, atom_feats_batch, protein_feats_batch, _ = batch_data
            batched_graph = batched_graph.to(device)
            batched_graph_c = batched_graph_c.to(device)
            atom_feats_batch = [feat.to(device) for feat in atom_feats_batch]
            protein_feats_batch = [feat.to(device) for feat in protein_feats_batch]

            predictions = model(
                batched_graph,
                batched_graph_c,
                atom_feats_batch,
                protein_feats_batch
            )
            predictions = predictions.squeeze(-1).detach().cpu().numpy()

            start = 0
            for feats in atom_feats_batch:
                n_atoms = feats.shape[0]
                atom_prediction_array = predictions[start:start+n_atoms]
                all_atom_predictions.append(atom_prediction_array)
                start += n_atoms

    df_test = pd.read_pickle(test_data_path)
    valid_smiles = [df_test.loc[idx, 'smiles_list'] for idx in valid_samples_idx]
    valid_rxn = [df_test.loc[idx, 'mapped_rxn_new'] for idx in valid_samples_idx]
    valid_r = [df_test.loc[idx, 'rxn_smiles'] for idx in valid_samples_idx]
    valid_uniprot_id = [df_test.loc[idx, 'uniprot_id'] for idx in valid_samples_idx]
    valid_ec = [df_test.loc[idx, 'ec'] for idx in valid_samples_idx]
    valid_sequence = [df_test.loc[idx, 'sequence'] for idx in valid_samples_idx]
    valid_first_digit = [df_test.loc[idx, 'first_digit'] for idx in valid_samples_idx]

    result_df = pd.DataFrame({
        'smiles_list': valid_smiles,
        'uniprot_id': valid_uniprot_id,
        'smiles_nosplit': valid_rxn,
        'mapped_rxn': valid_r,
        'ec': valid_ec,
        'sequence': valid_sequence,
        'first_digit': valid_first_digit,
        'Atom_Predictions': all_atom_predictions
    })

    result_df.to_pickle(output_path)

def main():
    test_data_path = 'test_data.pkl'
    model_path = 'model.pth'
    output_path = 'output_predictions.pkl'
    
    test_loader, valid_samples_idx = get_test_dataloader(test_data_path, limit=1000)
    
    model = AtomProteinAttentionGNN(
        node_feat_size=74,
        edge_feat_size=12,
        protein_feat_size=512,
        graph_feat_size=512,
        num_layers=6,
        output_size=1,
        dropout=0.2
    ).to(device)
    
    model.load_state_dict(torch.load(model_path))
    test_model(model, test_loader, test_data_path, valid_samples_idx, output_path)

if __name__ == "__main__":
    main()