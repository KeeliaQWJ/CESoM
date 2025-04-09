#!/usr/bin/env python3
import pandas as pd
import torch
import esm
import numpy as np
from tqdm import tqdm
import numpy as np

def ensure_list(x):
    if isinstance(x, str):
        import ast
        try:
            return ast.literal_eval(x)
        except (SyntaxError, ValueError):
            print(f"无法解析的字符串: {x}")
            return []
    elif isinstance(x, list):
        return x
    else:
        return []

def generate_esm_features(sequences, model, batch_converter, device):
    """
    Generate ESM features for a list of protein sequences.
    Returns a list of tensors with shape [sequence_length, feature_dim].
    """
    data = [("protein", seq) for seq in sequences]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]
    esm_features = []
    for i, (_, seq) in enumerate(data):
        seq_len = len(seq)
        # Skip the special token at position 0
        seq_feat = token_representations[i, 1:seq_len+1].cpu()
        esm_features.append(seq_feat)
    return esm_features

def main():
    # an example DataFrame with sample sequences and binding_mask info
    # data = {
    #     "sequence": [
    #         "MKAILVVLLYTFATANAD", 
    #         "GTEAQTRLLSLALVAA"
    #     ],
    #     "binding_mask": [
    #         "[0,0,0,...,1,1,0,0,0]",  
    #     ]
    # }
    # df = pd.DataFrame(data)
    df = pd.read_csv('/home/wenjiaqian/project/EasIFA/som/data/AQA_with_mask.csv')
    # Process binding_mask using the ensure_list function
    df["binding_mask"] = df["binding_mask"].apply(ensure_list)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.to(device)
    model.eval()
    
    batch_size = 2  # Adjust batch size as needed
    active_site_features_list = []
    
    total_batches = (len(df) + batch_size - 1) // batch_size
    for start_idx in tqdm(range(0, len(df), batch_size), total=total_batches, desc="Processing batches"):
        end_idx = min(start_idx + batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        sequences = batch_df["sequence"].tolist()
        predictions = batch_df["binding_mask"].tolist()
        
        esm_features = generate_esm_features(sequences, model, batch_converter, device)
        for i in range(len(sequences)):
            seq_len = len(sequences[i])
            pred = predictions[i]
            feat = esm_features[i]
            if len(pred) != seq_len:
                print(f"Invalid binding_mask at index {start_idx + i}")
                active_site_features_list.append(None)
                continue
            
            pred = np.array(pred)
            active_indices = np.where(pred == 1)[0]
            active_site_feat = feat[active_indices]
            # print(f"Sequence {start_idx + i}: active_site_feat shape: {active_site_feat.shape}")
            active_site_features_list.append(active_site_feat)
    
    df["active_site_features"] = active_site_features_list
    
    # Example output: print the shape of active site features for each sequence
    for idx, row in df.iterrows():
        print(f"Sequence index {idx}:")
        print(f"Sequence: {row['sequence']}")
        print(f"Binding mask (parsed): {row['binding_mask']}")
        if row["active_site_features"] is not None:
            print(f"Active site features shape: {row['active_site_features'].shape}")
        else:
            print("Active site features: None")

if __name__ == "__main__":
    main()