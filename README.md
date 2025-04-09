# Environment Setup for the Project

This project relies on a conda environment specified in the `cesom.yml` file.

## Installation Steps

Create and activate the conda environment:

    ```bash
    conda env create -f cesom.yml
    conda activate cesom
    ```
---

## Using EasIFA for Enzymatic Active Site Prediction

1. Clone the EasIFA repository:

    ```bash
    git clone git@github.com:wangxr0526/EasIFA.git
    ```

2. Predict protein node features:

    - Ensure your data folder includes both the protein sequence and the corresponding `.pdb` file for the structure.
    - Run the following script to compute residue-level predictions:

      ```bash
      python residue_pred/Protein_node_feature.py
      ```

3. Generate residue embeddings:

    ```bash
    python residue_pred/active_site_phrase_easifa.py
    ```

    This script uses the predictions from the previous step to create residue embeddings.

4. Alternative Uniprot-based residue embeddings:

    - If you have enzyme proteins from Uniprot (by ID) and have corresponding active site information (e.g., `331/411`), you can directly run:

      ```bash
      python residue_pred/active_site_phrase_uniprot.py
      ```

    - This will generate residue embeddings without requiring a local `.pdb` file.

---

## Model and Training

All model-related code and training scripts are located in the `som_pred` directory.

1. Download the processed dataset and pretrained models and place them inside the `data` and `model_save` folders, respectively.
2. Run the following script to perform testing:

    ```bash
    python som_pred/SOM_triatt_test.py
    ```

3. Visualize results:

    ```bash
    python som_pred/molecule_pred_visualization.py
    ```

These scripts allow you to evaluate and visualize the performance of the trained model on your data.
