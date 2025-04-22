# Graph-NIDS: Detecting Network Intrusions via HGT-based Edge Classification on Traffic Graphs

This repository contains the code and resources for a Network Intrusion Detection System (NIDS) built using graph neural networks. Specifically, it leverages the Heterogeneous Graph Transformer (HGT) model on the UNSW-NB15 dataset to classify network traffic flows by treating them as edges in a graph.

---

## Overview

The project aims to detect network intrusions by modeling network traffic data as a heterogeneous graph. An HGT model is trained on this graph structure to perform edge classification, identifying edges (representing network flows between IP addresses) as either 'Normal' or 'Intrusion'.

---

## Workflow

![workflow_graph](https://github.com/user-attachments/assets/028025b6-4425-4cdb-a169-7572090f0235)

---

## Features

* **Heterogeneous Graph Construction:** Builds graphs with multiple node types (IP addresses, ports, protocols) and edge types representing network flows and relationships.
* **HGT Model:** Utilizes the Heterogeneous Graph Transformer architecture for learning on the complex graph structure.
* **Edge Classification:** Classifies network flows (represented as edges between IP nodes) as normal or intrusion.
* **Dataset Processing:** Includes code for sampling, stratifying, and preprocessing the UNSW-NB15 dataset.

---

## Dataset

* **Source:** The UNSW-NB15 Dataset. Specifically, the `UNSW-NB15_1.csv` file containing ~700,000 network flow records was used as the base.
* **Sampling:** 240,000 rows were sampled from the base dataset using stratification based on the 'Label' column (intrusion vs. normal) to maintain class distribution.
* **Splitting:** The sampled data was further split into:
    * Training Set: 160,000 rows
    * Validation Set: 40,000 rows
    * Test Set: 40,000 rows
* **Dataset Creation Code:** The process for creating these datasets is detailed in the `Graph_Dataset_Code.ipynb` notebook.
* **Provided Datasets:** The generated `train_set.csv`, `val_set.csv`, and `test_set.csv` files are included in the `Datasets.zip` archive in this repository.

---

## Methodology

1.  **Preprocessing:** The raw flow data undergoes preprocessing including handling special values, encoding categorical features (proto, service, state), and scaling numerical features using StandardScaler (fitted on the training set).
2.  **Graph Construction:** Heterogeneous graphs are built for the train, validation, and test sets using the preprocessed data. Node types include `ip`, `port`, and `proto`. Edge types represent relationships like `('ip', 'flows_to', 'ip')`, `('ip', 'uses_port', 'port')`, and `('port', 'uses_proto', 'proto')`, along with their inverses for message passing. The preprocessed flow features are assigned as attributes to the `('ip', 'flows_to', 'ip')` edges.
3.  **HGT Model Training:** An HGT model is defined and trained for 10 epochs on the training graph to classify the `('ip', 'flows_to', 'ip')` edges. HGT was selected for its ability to effectively capture relationships across different node and edge types within the heterogeneous network traffic graph. The model's performance is monitored on the validation graph during training.
4.  **Evaluation:** After training, the model is evaluated on the test graph. An optimal probability threshold (0.4724) was determined using the validation set to maximize the F1-score for the intrusion class before final testing.
5.  **Implementation Details:** The entire process of graph building, model definition, training, and evaluation is implemented in the `HGT_Training_Code.ipynb` notebook.

---

## Graph Structure Visualization

![graph_pic](https://github.com/user-attachments/assets/cde4d2ac-da74-4f13-b6c0-f7ae781fb2f0)

---

## Results

The model achieved the following performance after 10 epochs of training and threshold adjustment:

**Final Training Epoch (Epoch 10):**
* Train Loss: 0.6016 | Val Loss: 0.4200
* Train Accuracy: 0.7437 | Val Accuracy: 0.9880
* Train F1 (Intrusion): 0.1361 | Val F1 (Intrusion): 0.8308
* Train Recall (Intrusion): 0.6361 | Val Recall (Intrusion): 0.9299

**Test Metrics (Optimal Threshold: 0.4724):**
* Test Accuracy: 0.9883
* Test F1 (Intrusion): 0.8405
* Test Recall (Intrusion): 0.9740
* Test Precision (Intrusion): 0.7392

---

## Performance Metric Plot

![Final metrics](https://github.com/user-attachments/assets/147d69a1-f198-4d60-9fd8-0f9ad0fced41)

---

## Repository Structure
```
Graph_NIDS/
├── Datasets.zip              # Zip archive containing train_set.csv, val_set.csv, test_set.csv
├── HGT_Training_Code.ipynb   # Jupyter notebook for graph building, HGT model training, and evaluation
├── Graph_Dataset_Code.ipynb  # Jupyter notebook showing the dataset sampling and splitting process
├── NUSW-NB15_features.csv    # Description of features in the UNSW-NB15 dataset
└── UNSW-NB15_LIST_EVENTS.csv # List of event types in the UNSW-NB15 dataset
```

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/bharateesha2004/Graph_NIDS.git](https://github.com/bharateesha2004/Graph_NIDS.git)
    cd Graph_NIDS
    ```
2.  **(Optional but recommended) Create a virtual environment:**
    ```bash
    python -m venv graph_env
    source graph_env/bin/activate  # On Windows use `graph_env\Scripts\activate`
    ```
3.  **Install required libraries:**
    ```bash
    pip install pandas numpy torch scikit-learn torch-geometric tqdm matplotlib networkx
    ```
    *Note: Ensure you install the correct PyTorch version compatible with your system (CPU/GPU) before installing torch-geometric. Refer to the official PyTorch and PyG installation guides.*

---

## Usage

1.  **Prepare Data:** Unzip the `Datasets.zip` file. This should create a directory (e.g., `Datasets/`) containing `train_set.csv`, `val_set.csv`, and `test_set.csv`.
2.  **Review Dataset Creation (Optional):** Open `Graph_Dataset_Code.ipynb` to understand how the train/val/test CSV files were generated from the original `UNSW-NB15_1.csv`.
3.  **Run Training and Evaluation:**
    * Open the `HGT_Training_Code.ipynb` notebook.
    * **Important:** Modify the data paths (e.g., `DATA_PATH`, `TRAIN_FILE`, `VALID_FILE`, `TEST_FILE` variables near the beginning of the notebook) to point to the location where you unzipped the datasets, especially if not running in an environment like Kaggle.
    * Ensure your environment (CPU/GPU) is set up correctly. The notebook includes device detection code. *(Note: Training may be significantly faster on a GPU).*
    * Run the cells sequentially to perform data loading, preprocessing, graph construction, model training, and evaluation.

---

## Acknowledgements

This project utilizes the UNSW-NB15 dataset. We acknowledge the creators and providers of this dataset.

* **Dataset Homepage:** [https://research.unsw.edu.au/projects/unsw-nb15-dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
* **Citation Recommendation:** Please refer to the dataset homepage for the recommended papers to cite if you use this dataset in your work.

---

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs, feature requests, or improvements.

---

## License

MIT License
