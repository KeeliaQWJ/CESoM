import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

def get_ring_info_dict(mol):
    ring_info = {}
    ri = mol.GetRingInfo()
    for atom in mol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if not map_num:
            continue
        min_size = None
        for ring in ri.AtomRings():
            if atom.GetIdx() in ring:
                ring_size = len(ring)
                if min_size is None or ring_size < min_size:
                    min_size = ring_size
        ring_info[map_num] = min_size
    return ring_info

def tag_reaction_atoms(reaction_smiles_list):
    tagged_smiles_list = []
    
    for reaction_smiles in reaction_smiles_list:
        reaction_smiles = reaction_smiles.replace("[H+].", "").replace(".[H+]", "")
        try:
            substrate_smiles, product_smiles = reaction_smiles.split(">>")
        except:
            tagged_smiles_list.append("")
            continue
        substrate_mol = Chem.MolFromSmiles(substrate_smiles)
        product_mol = Chem.MolFromSmiles(product_smiles)
        if not substrate_mol or not product_mol:
            tagged_smiles_list.append("")
            continue

        substrate_atoms = {atom.GetAtomMapNum(): atom for atom in substrate_mol.GetAtoms() if atom.GetAtomMapNum()}
        product_atoms = {atom.GetAtomMapNum(): atom for atom in product_mol.GetAtoms() if atom.GetAtomMapNum()}
        sub_ring_info = get_ring_info_dict(substrate_mol)
        prod_ring_info = get_ring_info_dict(product_mol)
        
        for map_num in substrate_atoms:
            if map_num in prod_ring_info:
                sub_size = sub_ring_info.get(map_num)
                prod_size = prod_ring_info.get(map_num)
                if sub_size != prod_size:
                    substrate_atoms[map_num].SetProp('reactionTag', '1')
        substrate_ri = substrate_mol.GetRingInfo()
        product_atom_map_num_to_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in product_mol.GetAtoms() if atom.GetAtomMapNum()}
        
        for bond in substrate_mol.GetBonds():
            start_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            start_map = start_atom.GetAtomMapNum()
            end_map = end_atom.GetAtomMapNum()
            if not start_map or not end_map:
                continue
            in_ring = substrate_ri.AreAtomsInSameRing(start_atom.GetIdx(), end_atom.GetIdx())
            start_idx = product_atom_map_num_to_idx.get(start_map)
            end_idx = product_atom_map_num_to_idx.get(end_map)
            if start_idx is None or end_idx is None:
                if in_ring:
                    start_atom.SetProp('reactionTag', '1')
                    end_atom.SetProp('reactionTag', '1')
                continue
            product_bond = product_mol.GetBondBetweenAtoms(start_idx, end_idx)
            if product_bond:
                if bond.GetBondType() != product_bond.GetBondType() and in_ring:
                    start_atom.SetProp('reactionTag', '1')
                    end_atom.SetProp('reactionTag', '1')
            else:
                if in_ring:
                    start_atom.SetProp('reactionTag', '1')
                    end_atom.SetProp('reactionTag', '1')
                    
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
            if atom.HasProp('reactionTag'):
                atom.SetAtomMapNum(1)
        tagged_smiles = Chem.MolToSmiles(substrate_mol, canonical=True)
        tagged_smiles_list.append(tagged_smiles)
    return tagged_smiles_list

if __name__ == "__main__":
    file_path = 'enzymemap/data/enzymemap_brenda2023.csv'
    data = pd.read_csv(file_path)
    processed_data = data['mapped']
    processed_rxn = tag_reaction_atoms(processed_data)
    data['mapped_rxn_new'] = processed_rxn
    print(data['mapped_rxn_new'].head())
