# Immunotherapy response prediction

Project which focuses on using GNN models with and without attention in order to predict immunotherapy response of patients with melanoma.

### Running the notebooks
Notebooks 01, 02, 03 must be run in order to obtain output files (ex. slide features, thresholded edges, patch data etc.). 
Notebook 06 experiments with patch level features and different ways of normalizing them.
Notebook 07 runs experiments with different types of GNNs (GCN, GAT, SAGE) on the same train-test split.

### Running PiNet
Before running PiNet, please ensure that the *DEEPMLO* folder is located in the same parent directory as *pinet* folder.
During the first run of the script, the raw data from DEEPMLO dataset will be converted and stored in *processed* folder.
To ensure the train-test split is the same, Notebook 07 should be run before PiNet. It will generate train_test_indices.txt file that is an output to PiNet dataloaders.
Alternatively, uncomment the part of the code that handles train set shuffling.
