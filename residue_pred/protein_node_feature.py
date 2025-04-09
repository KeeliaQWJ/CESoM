import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
import sys
import torch
import time
from tqdm.auto import tqdm
from collections import defaultdict
from functools import partial
import py3Dmol
from IPython.display import IFrame, SVG, display, HTML
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdChemReactions
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import IPythonConsole
from rdkit import Chem
from pandarallel import pandarallel
from webapp.utils import (
    EasIFAInferenceAPI,
    ECSiteBinInferenceAPI,
    ECSiteSeqBinInferenceAPI,
    EnzymeActiveSiteESMGearNetAPI,
    UniProtParserMysql,
    get_structure_html_and_active_data,
    cmd,
)
from data_loaders.rxn_dataloader import process_reaction
from data_loaders.enzyme_rxn_dataloader import get_rxn_smiles
from common.utils import calculate_scores_vbin_test
import pandas as pd
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Process protein node features.')
parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV file.')
parser.add_argument('--output_pkl', type=str, required=True, help='Path to output pkl file.')
args = parser.parse_args()

csv_file_path = args.input_csv
df = pd.read_csv(csv_file_path)

if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    for i in range(num_devices):
        print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No CUDA devices available.")

# rxn_model_path="../checkpoints/reaction_attn_net/model-ReactionMGMTurnNet_train_in_uspto_at_2023-04-05-23-46-25"
# model_checkpoint_path = "../checkpoints/global_step_57000"

easifa_seq_predictor = ECSiteSeqBinInferenceAPI(
    device="cuda:0",
    model_checkpoint_path="../checkpoints/enzyme_site_no_gearnet_prediction_model/train_in_uniprot_ecreact_cluster_split_merge_dataset_limit_100_at_2024-05-20-05-13-33/global_step_24000",
    rxn_model_path="../checkpoints/reaction_attn_net/model-ReactionMGMTurnNet_train_in_uspto_at_2023-04-05-23-46-25"
)

enzyme_predictor = EnzymeActiveSiteESMGearNetAPI(
            device="cuda:0",
            model_checkpoint_path=model_checkpoint_path,
            max_enzyme_aa_length=600,
            pred_tolist=True,
        )

df = pd.read_csv(csv_file_path)

df = df[df["sequence"].notnull() & df["sequence"].str.strip().astype(bool)].reset_index(drop=True)
df['sequence'] = df['sequence'].astype(str)

def truncate_sequence(seq, max_length=600):
    return seq[:max_length] if len(seq) > max_length else seq

df['sequence'] = df['sequence'].apply(truncate_sequence)
structure_dir = "/home/wenjia/project/EasIFA/som/data/cyp/structure"

df['pdb_file_path'] = df['pdb_files'].apply(lambda x: os.path.join(structure_dir, x))
missing_pdb_files = df[~df['pdb_file_path'].apply(os.path.exists)]
if not missing_pdb_files.empty:
    print("missing PDB:")
    print(missing_pdb_files[['uniprot_id', 'pdb_file_path']])

    df = df[df['pdb_file_path'].apply(os.path.exists)].reset_index(drop=True)
else:
    print("all PDB done")

predictions = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    rxn = row["Reaction_SMILES"] 
    aa_sequence = row["sequence"]
    pdb_file_name = row['pdb_files'] 
    pdb_file_path = row['pdb_file_path']

    
    try:
        # result = easifa_seq_predictor.inference(rxn=rxn, aa_sequence=aa_sequence)
        result = enzyme_predictor.inference(
                enzyme_sequence=aa_sequence,
                enzyme_structure_path=pdb_file_path
            )
        if result is None:
            print(f"推理返回了 None, 索引: {idx}")
            predictions.append(None)
        else:
            pred = result
            predictions.append(pred)
            # protein_node_features_list.append(protein_node_feature_ones.cpu().numpy())
    except Exception as e:
        print(f" {idx} : {e}")
        predictions.append(None)

df["prediction"] = predictions

df.reset_index(drop=True).to_pickle(args.output_pkl)
