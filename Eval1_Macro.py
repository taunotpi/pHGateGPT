"""
Created on April 09, 2025

@author: samlevy

Evaluation 1 a Macro point of view
"""
import os
import json
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from Dataset_Construct import TokenizedBioDataset
from torch.nn.utils.rnn import pad_sequence

# Custom collate function with dynamic macro-level padding.
def macro_collate_fn(batch):
    """
    Custom collate function for dynamically padding input_ids and labels.
    Designed for macro-level evaluation where the focus is on
    overall performance and comparison between models.
    """
    input_ids = [item["input_ids"] for item in batch]
    if "labels" in batch[0]:
        labels = [item["labels"] for item in batch]
    else:
        labels = [item["input_ids"] for item in batch]

    # Use padding token value of 0
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)
    return {"input_ids": padded_input_ids, "labels": padded_labels}


# Macro-level evaluation function.
# Focused on calculating overall loss, perplexity, and other key metrics
# for comparing models' high-level performance.

def evaluate_language_model(model, dataloader, device, criterion, model_type):
    """
    Evaluate the language model at the macro level.
    This function calculates overall loss, perplexity, and key evaluation metrics
    for comparing the performance of  standard GPT vs. pHGate GPT.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    results = {"perplexity": [], "attention_focus": [], "novel_identification": []}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating (Macro-Level)"):
            tokens = batch["input_ids"].to(device)
            labels = batch["labels"]
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]

            # Forward pass for model evaluation
            if model_type == "phgate":
                output = model(inputs, return_attention=True, return_gating=True)
                logits = output["logits"]
                results["novel_identification"].append(compute_novel_enzyme_detection(output["gating_activations"]))
            else:
                output = model(inputs, return_attention=True)
                logits = output["logits"]

            # Calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            total_loss += loss.item() * (tokens.size(0) * (tokens.size(1) - 1))
            total_tokens += tokens.size(0) * (tokens.size(1) - 1)

            # Append metrics for macro analysis
            results["perplexity"].append(math.exp(loss.item()))
            results["attention_focus"].append(compute_enzyme_focus(output["attentions"], labels))

    # Calculate overall metrics
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity, results


# Helper functions that compute key metrics used in macro-level evaluation.
def compute_enzyme_focus(attentions, labels):
    """
    Compute average attention to enzyme-related tokens.
    Operating at the macro level, computing a single summary score
    for all enzyme-related tokens across the dataset.
    """
    enzyme_mask = (labels == 0)  # Mask for enzyme-related tokens
    focus_values = []

    for attn in attentions:
        # Average attention across all heads
        avg_attn = attn.mean(dim=1)  

        batch_size, seq_len, _ = avg_attn.shape
        enzyme_focus_per_batch = []

        for i in range(batch_size):
            enzyme_positions = enzyme_mask[i].nonzero(as_tuple=True)[0]  # Positions of enzyme tokens
            if len(enzyme_positions) == 0:
                continue  # Skip if no enzyme tokens in the sequence
            focus_score = avg_attn[i, :, enzyme_positions].mean().item()  # Attention score for enzyme tokens
            enzyme_focus_per_batch.append(focus_score)

        if enzyme_focus_per_batch:
            focus_values.append(np.mean(enzyme_focus_per_batch))

    return float(np.mean(focus_values)) if focus_values else 0.0


def compute_novel_enzyme_detection(gating_activations):
    """
    Detect novel enzyme-related tokens based on gating activations.
    This function summarizes the gating behavior of the model to identify
    whether novel enzyme tokens are being detected at the macro level.
    """
    threshold = 0.5  # Activation threshold
    avg_gate = np.mean([gate.mean().item() for gate in gating_activations])
    return avg_gate > threshold


# Main execution: macro-level evaluation and model comparison.

def compare_models():
    """
    Compare the performance of Standard GPT and pH-Gated GPT models
    using macro-level evaluation metrics like perplexity, attention focus,
    and novel enzyme detection.
    """
    dataset_path = "tokenized_data.json"
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the tokenized dataset
    dataset = TokenizedBioDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=macro_collate_fn)

    # Load models
    from StandardGPT import GPTModelStandard
    from pHGateGPT import PHGateGPT as GPTModelPHGate

    # Initialize models
    standard_model = GPTModelStandard(vocab_size=50257, embed_size=768, num_heads=12, num_layers=12, dropout=0.1,
                                      max_seq_len=1024)
    phgate_model = GPTModelPHGate(vocab_size=50257, embed_size=768, num_heads=12, num_layers=12, dropout=0.1,
                                  max_seq_len=1024, forward_expansion=4)

    standard_model.to(device)
    phgate_model.to(device)

    # Load trained model checkpoints
    standard_checkpoint = torch.load("standard_gpt_checkpoint.pt")
    phgate_checkpoint = torch.load("phgate_gpt_checkpoint.pt")
    standard_model.load_state_dict(standard_checkpoint['model_state_dict'])
    phgate_model.load_state_dict(phgate_checkpoint['model_state_dict'])

    # Define loss criterion (ignoring padding token)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Evaluate both models
    print("Evaluating Standard GPT (Macro-Level):")
    standard_loss, standard_perplexity, standard_results = evaluate_language_model(standard_model, dataloader, device,
                                                                                   criterion, model_type="standard")

    print("\nEvaluating pH-Gated GPT (Macro-Level):")
    phgate_loss, phgate_perplexity, phgate_results = evaluate_language_model(phgate_model, dataloader, device,
                                                                             criterion, model_type="phgate")

    # Print evaluation results
    print(f"\nStandard GPT - Loss: {standard_loss:.4f}, Perplexity: {standard_perplexity:.4f}")
    print(f"pH-Gated GPT - Loss: {phgate_loss:.4f}, Perplexity: {phgate_perplexity:.4f}")
    print(f"Standard GPT - Attention Focus: {standard_results['attention_focus']}")
    print(f"pH-Gated GPT - Attention Focus: {phgate_results['attention_focus']}")
    print(f"pH-Gated GPT - Novel Enzyme Detection: {phgate_results['novel_identification']}")


if __name__ == "__main__":
    compare_models()
