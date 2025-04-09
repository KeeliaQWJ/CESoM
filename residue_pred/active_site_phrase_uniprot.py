#!/usr/bin/env python3
import numpy as np
import pandas as pd
import torch
import esm

def create_binding_mask(active_site_str, sequence):
    """
    Create a binding mask for a protein sequence based on an active_site string.
    The binding mask is a numpy array of the same length as the sequence,
    where positions marked as active sites are set to 1 and others to 0.
    
    Example active_site_str: "23-31/46,100/204/241,298-303"
    Note: Positions are 1-indexed.
    """
    mask = np.zeros(len(sequence), dtype=int)
    # Split the string by comma to separate different items
    items = active_site_str.split(',')
    positions = []
    for item in items:
        item = item.strip()
        if not item:
            continue
        # Split each item by '/' to handle cases like "23-31/46"
        sub_items = item.split('/')
        for sub_item in sub_items:
            sub_item = sub_item.strip()
            if '-' in sub_item:
                try:
                    start_str, end_str = sub_item.split('-')
                    start = int(start_str) - 1  # Convert to 0-indexed
                    end = int(end_str)          # End is inclusive in input; slice end is exclusive
                    positions.append((start, end))
                except Exception as e:
                    print(f"Failed to parse range '{sub_item}': {e}")
            else:
                try:
                    pos = int(sub_item) - 1
                    positions.append((pos, pos + 1))
                except Exception as e:
                    print(f"Failed to parse position '{sub_item}': {e}")
    for start, end in positions:
        if start < 0:
            start = 0
        if end > len(sequence):
            end = len(sequence)
        mask[start:end] = 1
    return mask

def generate_esm_features2(sequence, active_site_str, model, batch_converter, device):
    """
    Generate ESM features for the active sites specified by active_site_str.
    
    Parameters:
        sequence (str): Protein sequence.
        active_site_str (str): Active site string.
        model: Preloaded ESM model.
        batch_converter: ESM model's batch converter.
        device: Torch device (e.g., "cuda" or "cpu").
    
    Returns:
        torch.Tensor: Features for the active site positions.
    """
    binding_mask = create_binding_mask(active_site_str, sequence)
    
    data = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]
    esm_features = token_representations[0, 1:len(sequence)+1].cpu()
    
    binding_mask = np.array(binding_mask)
    if binding_mask.shape[0] != len(sequence):
        print("Binding mask length does not match sequence length.")
        return None
    
    active_indices = np.where(binding_mask == 1)[0]
    active_site_features = esm_features[active_indices]
    return active_site_features

def main():
    df = pd.read_csv('/home/wenjiaqian/project/EasIFA/som/data/AQA_with_mask.csv')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.to(device)
    model.eval()
    
    def process_row(row):
        return generate_esm_features2(row["sequence"], row["active_site"], model, batch_converter, device)
    
    df["active_site_features"] = df.apply(process_row, axis=1)
    print(df["active_site_features"][:2])
    # df.to_pickle('data/test/df_with_active_site_features.pkl')
    print("Processing complete.")

    # # Example protein sequence and active_site string
    # sequence = "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPDWQNYTPGPGIRYPLKF"
    # active_site_str = "10-15,20/25,30"
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    # batch_converter = alphabet.get_batch_converter()
    # model.to(device)
    # model.eval()
    
    # active_features = generate_esm_features2(sequence, active_site_str, model, batch_converter, device)
    # if active_features is None:
    #     print("ESM feature generation failed.")
    # else:
    #     print("Active site features shape:", active_features.shape)

if __name__ == "__main__":
    main()