#!/usr/bin/env python3
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG, display
import numpy as np
import torch

def hsv_to_rgb(h, s, v):
    """
    Convert HSV values to RGB.
    """
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = h60 % 6
    hi = int(h60f) % 6
    f = h60f - hi
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    if hi == 0:
        r, g, b = v, t, p
    elif hi == 1:
        r, g, b = q, v, p
    elif hi == 2:
        r, g, b = p, v, t
    elif hi == 3:
        r, g, b = p, q, v
    elif hi == 4:
        r, g, b = t, p, v
    elif hi == 5:
        r, g, b = v, p, q
    return r, g, b

def main():

    df_test = pd.read_pickle('data/test/chomt_with_predictions.pkl')
    
    molecule_index = 1  # Change this index as needed
    
    # Extract SMILES and atom predictions
    all_smiles = df_test['smiles'].apply(lambda x: x.split('.')[0])
    all_predictions = df_test['atom_predictions']
    smiles = all_smiles[molecule_index]
    predictions_for_molecule = np.array(all_predictions[molecule_index])
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # Convert predictions to a tensor and get the min and max values for normalization
    y = torch.tensor(predictions_for_molecule)
    minY = y.min().item()
    maxY = y.max().item()
    
    # Define dictionaries for highlights and highlight radii
    highlights = {}
    highlight_radii = {}
    
    # Iterate over each atom and set highlights and annotations
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        label = y[idx].item()
        atom.SetProp('atomNote', f"{label:.2f}")
        normalized_value = (label - minY) / (maxY - minY) if maxY > minY else 0.5
        rgb_color = hsv_to_rgb(200, normalized_value, 0.8)  # Blue hue
        highlights[idx] = rgb_color
        highlight_radii[idx] = 0.4
    
    drawer = rdMolDraw2D.MolDraw2DSVG(450, 300)
    options = drawer.drawOptions()
    options.atomHighlightsAreCircles = True
    options.useSVG = True
    
    # Draw the molecule with highlights and annotations
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=list(highlights.keys()),
        highlightAtomColors=highlights,
        highlightAtomRadii=highlight_radii
    )
    drawer.FinishDrawing()
    
    svg = drawer.GetDrawingText().replace('svg:', '')
    display(SVG(svg))

if __name__ == "__main__":
    main()