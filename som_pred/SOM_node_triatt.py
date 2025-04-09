import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"  
from torch.utils.data import DataLoader
import dgl
import torch.nn.functional as F
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer,mol_to_bigraph,mol_to_complete_graph
import torch.optim as optim
import ast
import torch
from torch.utils.data import DataLoader
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer,mol_to_bigraph,mol_to_complete_graph
import torch.optim as optim
import numpy as np
import random 
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score
from sklearn.metrics import recall_score
from model_triatt import AtomProteinAttentionGNN
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 0
random.seed(seed) 
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  
np.random.seed(seed)  
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
edge_featurizer = CanonicalBondFeaturizer(bond_data_field='e')

df = pd.read_pickle('data/concat/final_train_set_ec_smi_split.pkl')
print(df.columns)
df = df.dropna(subset=['active_site_features']).reset_index(drop=True)
print(df['active_site_features'].apply(type).value_counts())
df = df.drop_duplicates(subset=['smiles_list'], keep='first')

def collate_fn(samples):
    g_lis = [sample[0] for sample in samples]
    g_c_lis = [sample[1] for sample in samples]
    atom_feats_list = [sample[2] for sample in samples]
    protein_feats_list = [sample[3] for sample in samples]
    atom_labels_list = [sample[4] for sample in samples]

    batched_graph = dgl.batch(g_lis)
    batched_graph_c = dgl.batch(g_c_lis)

    return batched_graph, batched_graph_c, atom_feats_list, protein_feats_list, atom_labels_list

def get_train_val_dataloader(val_fold, df):
    fold_lis = [f'fold_{i}' for i in range(1, 11)]
    train_fold = [i for i in fold_lis if i != val_fold]

    df_train = df[df['fold'].isin(train_fold)].dropna(subset=['active_site_features']).reset_index(drop=True)
    df_val = df[df['fold'] == val_fold].dropna(subset=['active_site_features']).reset_index(drop=True)

    train_smiles_lis = list(df_train['smiles_list'])
    val_smiles_lis = list(df_val['smiles_list'])

    train_protein_features = list(df_train['active_site_features'])
    val_protein_features = list(df_val['active_site_features'])

    print(f"train set: {len(train_smiles_lis)}")
    print(f"val set: {len(val_smiles_lis)}")

    batch_size = 128

    train_dataset = []
    for idx, smiles in enumerate(train_smiles_lis):
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
        if g.num_nodes() != g_c.num_nodes() or g.num_nodes() == 0 or g_c.num_nodes() == 0:
            continue

        g_label = torch.tensor([[1 if atom.GetAtomMapNum() == 1 else 0] for atom in mol.GetAtoms()])
        g_mask = torch.tensor([[1 if atom.GetAtomMapNum() == 1 else 0] for atom in mol.GetAtoms()])

        g.ndata['atom label'] = g_label
        g.ndata['mask'] = g_mask

        protein_features = train_protein_features[idx]
        if not isinstance(protein_features, (np.ndarray, list, torch.Tensor)) or len(protein_features) == 0:
            continue

        protein_feats = torch.tensor(protein_features, dtype=torch.float32) if not isinstance(protein_features, torch.Tensor) else protein_features.float()
        atom_feats = g.ndata['h'].float()
        atom_labels = g.ndata['atom label'].float()
        train_dataset.append((g, g_c, atom_feats, protein_feats, atom_labels))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False)
    train_loader_ = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn,drop_last = False)

    # 构建验证集 DataLoader
    val_dataset = []
    for idx, smiles in enumerate(val_smiles_lis):
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
        if g.num_nodes() != g_c.num_nodes() or g.num_nodes() == 0 or g_c.num_nodes() == 0:
            continue

        g_label = torch.tensor([[1 if atom.GetAtomMapNum() == 1 else 0] for atom in mol.GetAtoms()])
        g_mask = torch.tensor([[1 if atom.GetAtomMapNum() == 1 else 0] for atom in mol.GetAtoms()])

        g.ndata['atom label'] = g_label
        g.ndata['mask'] = g_mask

        protein_features = val_protein_features[idx]
        if not isinstance(protein_features, (np.ndarray, list, torch.Tensor)) or len(protein_features) == 0:
            continue

        protein_feats = torch.tensor(protein_features, dtype=torch.float32) if not isinstance(protein_features, torch.Tensor) else protein_features.float()
        atom_feats = g.ndata['h'].float()
        atom_labels = g.ndata['atom label'].float()

        val_dataset.append((g, g_c, atom_feats, protein_feats, atom_labels))

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)

    return train_loader,train_loader_,val_loader

def train_5_fold(df, path, is_atom_level_loss, learning_rate, num_epochs):
    num_folds = 10
    best_sensitivity_atom = 0.0
    best_epoch = 0

    result_roc_auc_atom = 0
    result_accuracy_atom = 0
    result_precision_atom = 0
    result_sensitivity_atom = 0

    fold_lis = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5',
                'fold_6', 'fold_7', 'fold_8', 'fold_9', 'fold_10']
    all_true_labels = []
    all_predictions = []

    node_feat_size = 74
    edge_feat_size = 12
    protein_feat_size = 512
    graph_feat_size = 512 
    num_layers = 6 
    output_size = 1  
    dropout = 0.2
    scaler = torch.cuda.amp.GradScaler() 

    for val_fold in fold_lis:
        print(f"Starting fold: {val_fold}")

        train_loader, train_loader_, val_loader = get_train_val_dataloader(val_fold, df)

        batch_size = 1024 

        model = AtomProteinAttentionGNN(
            node_feat_size=node_feat_size,
            edge_feat_size=edge_feat_size,
            protein_feat_size=protein_feat_size,
            graph_feat_size=graph_feat_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout
        ).to(device)

        if path is not None:
            model.load_state_dict(torch.load(path))

        loss_func = torch.nn.BCELoss(reduction='sum') 
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.00005)
        best_loss = float('inf')

        for epoch in range(1, num_epochs + 1):
            model.train()
            running_loss = 0.0

            with tqdm(total=len(train_loader), desc=f'Fold {val_fold} Epoch {epoch}/{num_epochs}', unit='batch') as pbar:
                for iter, batch_data in enumerate(train_loader):
                
                    batched_graph, batched_graph_c, atom_feats_batch, protein_feats_batch, atom_labels_batch = batch_data
                    batched_graph = batched_graph.to(device)
                    batched_graph_c = batched_graph_c.to(device)
                    atom_feats_batch = [feat.to(device) for feat in atom_feats_batch]
                    protein_feats_batch = [feat.to(device) for feat in protein_feats_batch]
                    atom_labels_batch = [al.to(device) for al in atom_labels_batch]
                    optimizer.zero_grad()

                    predictions = model(
                        batched_graph, 
                        batched_graph_c, 
                        atom_feats_batch, 
                        protein_feats_batch
                    )

                    atom_labels = torch.cat(atom_labels_batch, dim=0).squeeze(-1)  # [total_num_atoms]
                    
                    pre_atom_value = predictions.squeeze(-1)
                    mask = (pre_atom_value != 0)
                    valid_predictions = predictions.squeeze(-1)[mask]
                    valid_labels = atom_labels[mask]
                    total_num = mask.sum().float()
                    if total_num > 0:
                        loss = loss_func(predictions.squeeze(-1), atom_labels) / total_num
                    else:
                        loss = torch.tensor(0.0, device=device)

                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    pbar.update(1)
                    pbar.set_postfix({'loss': f'{loss.item():.6f}'})

            average_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch} finished with average loss: {average_loss:.6f}")

        with torch.no_grad():
            model.eval()
            total_roc_auc_atom = 0
            total_accuracy_atom = 0
            total_precision_atom = 0
            total_sensitivity_atom = 0
            total_count = 0

            for iter, batch_data in enumerate(val_loader):
                
                batched_graph, batched_graph_c, atom_feats_batch, protein_feats_batch, atom_labels_batch = batch_data
                batched_graph = batched_graph.to(device)
                batched_graph_c = batched_graph_c.to(device)
                atom_feats_batch = [feat.to(device) for feat in atom_feats_batch]
                protein_feats_batch = [feat.to(device) for feat in protein_feats_batch]
                atom_labels_batch = [al.to(device) for al in atom_labels_batch]

                predictions = model(
                    batched_graph, 
                    batched_graph_c, 
                    atom_feats_batch, 
                    protein_feats_batch
                )

                probabilities = torch.sigmoid(predictions).squeeze(-1).cpu().numpy()
                atom_labels = torch.cat(atom_labels_batch, dim=0).squeeze(-1)  # [total_num_atoms]
                predictions = predictions.squeeze(-1)  # [total_num_atoms]
                predictions_class = (predictions > 0.5).float()
                all_true_labels.extend(atom_labels.cpu().numpy())
                all_predictions.extend(probabilities)

                if len(atom_labels) > 0 and len(torch.unique(atom_labels)) > 1:
                    roc_auc_atom = roc_auc_score(atom_labels.cpu().numpy(), predictions.cpu().numpy())
                    accuracy_atom = accuracy_score(atom_labels.cpu().numpy(), predictions_class.cpu().numpy())
                    precision_atom = precision_score(atom_labels.cpu().numpy(), predictions_class.cpu().numpy(), zero_division=0)
                    sensitivity_atom = recall_score(atom_labels.cpu().numpy(), predictions_class.cpu().numpy(), zero_division=0)

                    total_roc_auc_atom += roc_auc_atom
                    total_accuracy_atom += accuracy_atom
                    total_precision_atom += precision_atom
                    total_sensitivity_atom += sensitivity_atom
                    total_count += 1
                else:
                    print("Skipped batch due to lack of positive samples or only one class present in atom_labels.")

            if total_count > 0:
                avg_roc_auc_atom = total_roc_auc_atom / total_count
                avg_accuracy_atom = total_accuracy_atom / total_count
                avg_precision_atom = total_precision_atom / total_count
                avg_sensitivity_atom = total_sensitivity_atom / total_count
                print(f"Validation Results - Epoch: {epoch}, Sensitivity: {avg_sensitivity_atom:.4f}, "
                    f"ROC AUC: {avg_roc_auc_atom:.4f}, Accuracy: {avg_accuracy_atom:.4f}, "
                    f"Precision: {avg_precision_atom:.4f}")
                
                if avg_sensitivity_atom > best_sensitivity_atom:
                        best_sensitivity_atom = avg_sensitivity_atom
                        best_epoch = epoch
                        lr_str = str(learning_rate).replace('.', '_')
                        model_filename = f'model_save/som/best_model_smi_split_fold{val_fold}_lr{lr_str}_epoch{epoch}.pth'
                        # model_filename = f'best_model_lr{lr_str}_epoch{epoch}.pth'
                        torch.save(model.state_dict(), model_filename)
                        print(f"New best model saved with sensitivity: {best_sensitivity_atom:.4f} at epoch {epoch}")
            else:
                print("No validation batches with positive samples.")

            np.savez(f'data/concat/pred/prediction_sing_sub_results_fold_{val_fold}.npz', true_labels=np.array(all_true_labels), predictions=np.array(all_predictions))

            all_true_labels = []
            all_predictions = []

            result_roc_auc_atom += avg_roc_auc_atom
            result_accuracy_atom += avg_accuracy_atom
            result_precision_atom += avg_precision_atom
            result_sensitivity_atom += avg_sensitivity_atom

            print(f'Fold {val_fold} results:')
            print(f' - ROC-AUC: {avg_roc_auc_atom:.4f}')
            print(f' - Accuracy: {avg_accuracy_atom:.4f}')
            print(f' - Precision: {avg_precision_atom:.4f}')
            print(f' - Sensitivity: {avg_sensitivity_atom:.4f}')

    final_avg_roc_auc_atom = result_roc_auc_atom / num_folds
    final_avg_accuracy_atom = result_accuracy_atom / num_folds
    final_avg_precision_atom = result_precision_atom / num_folds
    final_avg_sensitivity_atom = result_sensitivity_atom / num_folds

    print('Final averaged results over all folds:')
    print(f' - ROC-AUC: {final_avg_roc_auc_atom:.4f}')
    print(f' - Accuracy: {final_avg_accuracy_atom:.4f}')
    print(f' - Precision: {final_avg_precision_atom:.4f}')
    print(f' - Sensitivity: {final_avg_sensitivity_atom:.4f}')

    return final_avg_roc_auc_atom, final_avg_accuracy_atom, final_avg_precision_atom, final_avg_sensitivity_atom

path = None  
is_atom_level_loss = True  
learning_rate = 2e-5 
num_epochs = 105

final_avg_roc_auc_atom, final_avg_accuracy_atom, final_avg_precision_atom, final_avg_sensitivity_atom = train_5_fold(
    df=df,
    path=path,
    is_atom_level_loss=is_atom_level_loss,
    learning_rate=learning_rate,
    num_epochs=num_epochs
)
print('Final averaged results over all folds:')
print(f' - ROC-AUC: {final_avg_roc_auc_atom:.4f}')
print(f' - Accuracy: {final_avg_accuracy_atom:.4f}')
print(f' - Precision: {final_avg_precision_atom:.4f}')
print(f' - Sensitivity: {final_avg_sensitivity_atom:.4f}')
