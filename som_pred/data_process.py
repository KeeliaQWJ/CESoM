import pandas as pd
from rdkit import Chem

def tag_reaction_atoms(reaction_smiles_list):
    tagged_smiles_list = []
    for reaction_smiles in reaction_smiles_list:
        reaction_smiles = reaction_smiles.replace("[H+].", "").replace(".[H+]", "") 
        substrate_smiles, product_smiles = reaction_smiles.split(">>")
        substrate_mol = Chem.MolFromSmiles(substrate_smiles)
        product_mol = Chem.MolFromSmiles(product_smiles)
        
        substrate_atoms = {atom.GetAtomMapNum(): atom for atom in substrate_mol.GetAtoms() if atom.GetAtomMapNum()}
        product_atoms = {atom.GetAtomMapNum(): atom for atom in product_mol.GetAtoms() if atom.GetAtomMapNum()}
        
        added_atom_map_nums = set(product_atoms.keys()) - set(substrate_atoms.keys())
        removed_atom_map_nums = set(substrate_atoms.keys()) - set(product_atoms.keys())

        for atom_map_num, substrate_atom in substrate_atoms.items():
            substrate_neighbors_map_nums = {neighbor.GetAtomMapNum() for neighbor in substrate_atom.GetNeighbors() if neighbor.GetAtomMapNum()}
            
            if atom_map_num in product_atoms:
                product_atom = product_atoms[atom_map_num]
                product_neighbors_map_nums = {neighbor.GetAtomMapNum() for neighbor in product_atom.GetNeighbors() if neighbor.GetAtomMapNum()}
                
                if substrate_neighbors_map_nums != product_neighbors_map_nums:
                    substrate_atom.SetProp('reactionTag', '1')
            else:
                substrate_atom.SetProp('reactionTag', '1')  
                for neighbor in substrate_atom.GetNeighbors():
                    if neighbor.GetAtomMapNum():  
                        neighbor.SetProp('reactionTag', '1')

        for atom_num in added_atom_map_nums:
            if atom_num in product_atoms:  
                atom = product_atoms[atom_num]
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetAtomMapNum():
                        neighbor.SetProp('reactionTag', '1')
                        if neighbor.GetAtomMapNum() in substrate_atoms:
                            substrate_atoms[neighbor.GetAtomMapNum()].SetProp('reactionTag', '1')
        
        for atom_num in added_atom_map_nums.union(removed_atom_map_nums):
            if atom_num in substrate_atoms:
                atom = substrate_atoms[atom_num]
                for neighbor in atom.GetNeighbors():
                    neighbor.SetProp('reactionTag', '1')
        
        if not added_atom_map_nums and not removed_atom_map_nums:
            product_atom_map_num_to_idx = {}
            for atom in product_mol.GetAtoms():
                map_num = atom.GetAtomMapNum()
                if map_num: 
                    product_atom_map_num_to_idx[map_num] = atom.GetIdx()
            for bond in substrate_mol.GetBonds():
                start_atom_map_num = bond.GetBeginAtom().GetAtomMapNum()
                end_atom_map_num = bond.GetEndAtom().GetAtomMapNum()

                start_atom_idx_in_product = product_atom_map_num_to_idx.get(start_atom_map_num, None)
                end_atom_idx_in_product = product_atom_map_num_to_idx.get(end_atom_map_num, None)

                if start_atom_idx_in_product is not None and end_atom_idx_in_product is not None:
                    product_bond = product_mol.GetBondBetweenAtoms(start_atom_idx_in_product, end_atom_idx_in_product)
                    if product_bond and bond.GetBondType() != product_bond.GetBondType():
                        bond.GetBeginAtom().SetProp('reactionTag', '1')
                        bond.GetEndAtom().SetProp('reactionTag', '1')
                    elif not product_bond:
                        bond.GetBeginAtom().SetProp('reactionTag', '1')  
                        bond.GetEndAtom().SetProp('reactionTag', '1')    
        
        for atom in substrate_mol.GetAtoms():
            atom.SetAtomMapNum(0)
        
        for atom in substrate_mol.GetAtoms():
            if atom.HasProp('reactionTag'):
                if atom.GetProp('reactionTag') == '1':
                    atom.SetAtomMapNum(1)
        
        tagged_smiles = Chem.MolToSmiles(substrate_mol, True)
        tagged_smiles_list.append(tagged_smiles)
        
    return tagged_smiles_list

if __name__ == "__main__":
    file_path = 'enzymemap/data/enzymemap_brenda2023.csv'  
    data = pd.read_csv(file_path)
    processed_data = data['mapped']
    processed_rxn = tag_reaction_atoms(processed_data)

    data['mapped_rxn_new'] = processed_rxn
    print(data['mapped_rxn_new'].head())
