# README

## Project Title
**Can AI solve the Common Cold**

## Author
**Samantha Levy**

## Overview
This project implements and evaluates a novel variant of the GPT model, called **pHGate GPT**, designed to identify biologically meaningful signals related to endosomal acidification. Inspired by the pH-triggered conformational switch in the influenza hemagglutinin (HA) protein, the model integrates a learnable gating mechanism into the self-attention layers of a standard GPT architecture. This gating mechanism modulates attention flow based on the semantic content of the input sequence, mimicking how biological systems activate processes conditionally.

A baseline **Standard GPT** model is also included for comparison. The project evaluates both models using macro-level (Eval1) and micro-level (Eval2) analysis frameworks.

****Note Before Running main.py***
Before executing main.py, please ensure that the following preprocessing steps are completed:

*Datasets.py* – This script downloads all necessary biological data from external databases (e.g., NCBI, KEGG, UniProt). Run this first to populate the raw data directory.

*BioTokenizer.py* – This script processes the downloaded data into a machine-learning-ready format. It tokenizes biological sequences using a sliding window and saves them as labeled JSON files for training.



## Repository Structure
```
├── Datasets.py                 # Handles dataset preparation and data downloading tasks
├── Biotokenizer.py             # Implements a custom tokenizer for handling biological sequence data
├── Dataset_Construct.py        # Data loading and tokenization (BioTokenizerV2)
├── StandardGPT.py              # Standard GPT model implementation
├── pHGateGPT.py                # pHGate GPT model implementation
├── StandardGPT_Training.py     # Training script for Standard GPT
├── pHGate_Training.py          # Training script for pHGate GPT
├── Eval1_Macro.py              # Macro-level evaluation (loss, perplexity, attention focus)
├── Eval2_Micro.py              # Micro-level evaluation (gating behavior, enzyme detection)


├── loss_comparison_macro.png   # Plot comparing training loss (Eval1)
├── time_comparison_macro.png   # Plot comparing total training time (Eval1)
├── gating_heatmap.png          # Heatmap of gating activations
├── gating_vs_pH_scatter.png    # Scatter plot of gating strength vs pH threshold
├── training_metrics.json       # Training log for pHGate GPT
├── losses_phgate.json          # Loss history for pHGate GPT
├── README.md                   # This file
```

## Code Usage

### 1. Data Preparation
Run Datasets.py to get the biological data need to run the Biotokenizer.py
Run Biotokenizer.py then Dataset_Construct.py
Therefore ensuring you have  biological data tokenized using the `BioTokenizerV2` tool embedded within `Dataset_Construct.py`. 
This will produce a `tokenized_data.json` file containing labeled windows with `input_ids` and optional labels.

### 2. Training

#### a. Standard GPT

- Logs loss and perplexity
- Saves model checkpoint (`standard_gpt_checkpoint.pt`)

#### b. pHGate GPT

- Logs loss, perplexity, gate activation, and enzyme focus
- Saves checkpoint (`phgate_gpt_checkpoint.pt`)
- Saves logs to `training_metrics.json` and `losses_phgate.json`

#### c. Train StandardGPT 

- Training loop with loss and perplexity logging.
- Checkpoint saving functionality (`standard_gpt_checkpoint.pt`)
- Adjustable hyperparameters such as `num_epochs`, `learning_rate`, and `batch_size`

#### d. Train pHGate GPT

- Enhanced logging of gate activation values and enzyme focus metrics
- Checkpoint saving (`phgate_gpt_checkpoint.pt`) and metric storage in `training_metrics.json` and `losses_phgate.json`
- Configurable hyperparameters such as `vocab_size`, `num_layers`, and `dropout`

### 3. Evaluation

#### Macro-Level Evaluation (Eval1)

- Outputs average loss and perplexity
- Computes average attention focus
- Generates plots:
  - `loss_comparison_macro.png`
  - `time_comparison_macro.png`

#### Micro-Level Evaluation (Eval2)
- Analyzes each batch individually
- Records per-batch perplexity
- Computes gate activations
- Flags novel enzyme detections based on gating threshold
- Computes proxy `enzyme_focus` score

#### Optional: pHGate_plots 
- Produces heatmaps and scatterplots for pH gating diagnostics


## Output Files
- **`standard_gpt_checkpoint.pt`**: Checkpoint of Standard GPT model.
- **`phgate_gpt_checkpoint.pt`**: Checkpoint of pHGate GPT model.
- **`loss_comparison_macro.png`**: Training loss curve comparing both models.
- **`time_comparison_macro.png`**: Total training time bar chart.
- **`training_metrics.json`**: Detailed logging of pHGate GPT training.
- **`losses_phgate.json`**: Epoch-level loss history.

## Dependencies
- Python 3.8+
- torch
- numpy
- matplotlib
- tqdm
- transformers
- requests
- beautifulsoup4
- seaborn


## Key Takeaways
- pHGate GPT shows superior convergence and generalization.
- The gating mechanism improves model sensitivity to biologically relevant motifs.
- Eval2 demonstrates that gating is essential for identifying "novel enzymes."
