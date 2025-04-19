"""
Created on April 11, 2025

@author: samlevy

Evaluation 2 a Micro point of view
"""

import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from Dataset_Construct import TokenizedBioDataset
from torch.nn.utils.rnn import pad_sequence

# Custom collate function with dynamic micro-level padding.
def my_collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    if "labels" in batch[0]:
        labels = [item["labels"] for item in batch]
    else:
        labels = [item["input_ids"] for item in batch]

    # Pad sequences; assume padding token is 0.
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)
    return {"input_ids": padded_input_ids, "labels": padded_labels}

# Helper functions for micro-level metrics.
def compute_enzyme_focus(attentions, labels):
    """
    Compute average attention to enzyme-related tokens
    Assumes attentions: List of attention tensors [layer1, layer2, ...] of shape (batch, heads, seq_len, seq_len)
    and labels: tensor of shape (batch, seq_len)
    """
    enzyme_mask = (labels == 0)  
    micro_focus_values = []

    for attn in attentions:
        # attn shape: (batch, heads, seq_len, seq_len)
        avg_attn = attn.mean(dim=1)  # shape: (batch, seq_len, seq_len)
        batch_size, seq_len, _ = avg_attn.shape
        enzyme_focus_per_batch = []

        for i in range(batch_size):
            enzyme_positions = enzyme_mask[i].nonzero(as_tuple=True)[0]
            if len(enzyme_positions) == 0:
                continue
            # Attention paid to enzyme tokens (along the last dim = target positions)
            focus_score = avg_attn[i, :, enzyme_positions].mean().item()
            enzyme_focus_per_batch.append(focus_score)

        if enzyme_focus_per_batch:
            micro_focus_values.append(np.mean(enzyme_focus_per_batch))

    return float(np.mean(micro_focus_values)) if micro_focus_values else 0.0

def compute_novel_enzyme_detection(gating_activations):
    # Check if average gate activation across layers exceeds threshold.
    threshold = 0.5
    avg_gate = np.mean([gate.mean().item() for gate in gating_activations])
    return avg_gate > threshold

# Enhanced Evaluation Function for Language Modeling.
def evaluate_language_model(model, dataloader, device, criterion, model_type):
    model.eval()
    micro_total_loss = 0.0
    micro_total_tokens = 0
    micro_metrics = {"perplexity": [], "attention_focus": [], "novel_identification": [], "gating_summary": []}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="micro Evaluating"):
            tokens = batch["input_ids"].to(device)
            labels = batch["labels"]
            # For next-token prediction, input tokens are all but the last and targets are all but the first.
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]

            if model_type == "phgate":
                output = model(inputs, return_attention=True, return_gating=True)
                logits = output["logits"]
                gating_activations = output["gating_activations"]
                micro_novel_detection = compute_novel_enzyme_detection(gating_activations)
                micro_metrics["novel_identification"].append(micro_novel_detection)
                gating_layer_means = [gate.mean().item() for gate in gating_activations]
                micro_metrics["gating_summary"].append(gating_layer_means)
            else:
                output = model(inputs, return_attention=True)
                logits = output["logits"]

            loss = criterion(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            micro_total_loss += loss.item() * (tokens.size(0) * (tokens.size(1) - 1))
            micro_total_tokens += tokens.size(0) * (tokens.size(1) - 1)
            micro_metrics["perplexity"].append(math.exp(loss.item()))
            micro_metrics["attention_focus"].append(compute_enzyme_focus(output["attentions"], labels))

    micro_avg_loss = micro_total_loss / micro_total_tokens
    micro_perplexity = math.exp(micro_avg_loss)
    print("Evaluation complete!")
    print("Final Average Loss: {:.4f}, Perplexity: {:.4f}".format(micro_avg_loss, micro_perplexity))
    print("Gating Activation (per batch, if applicable):", micro_metrics.get("gating_summary", "N/A"))

    return micro_avg_loss, micro_perplexity, micro_metrics

# Training function with micro-level tracking.
def train_model(model, dataloader, optimizer, criterion, device, num_epochs, model_type):
    model.train()
    micro_epoch_losses = []
    start_time = time.time()

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_tokens = 0

        for batch in tqdm(dataloader, desc=f"micro Training Epoch {epoch + 1}/{num_epochs}"):
            tokens = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]

            optimizer.zero_grad()
            if model_type == "phgate":
                output = model(inputs, return_attention=True, return_gating=True)
                logits = output["logits"]
            else:
                output = model(inputs, return_attention=True)
                logits = output["logits"]

            loss = criterion(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * (tokens.size(0) * (tokens.size(1) - 1))
            total_tokens += tokens.size(0) * (tokens.size(1) - 1)

        micro_avg_loss = total_loss / total_tokens
        micro_epoch_losses.append(micro_avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {micro_avg_loss:.4f}")

    total_training_time = time.time() - start_time
    return micro_epoch_losses, total_training_time

# Main execution: setting up dataset and DataLoader.
def main():
    dataset_path = "tokenized_data.json"
    batch_size = 4
    num_epochs = 10
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TokenizedBioDataset(dataset_path)
    micro_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=my_collate_fn, drop_last=True)

    from StandardGPT import GPTModelStandard
    from pHGateGPT import PHGateGPT as GPTModelPHGate

    standard_model = GPTModelStandard(vocab_size=50257, embed_size=768, num_heads=12, num_layers=12, dropout=0.1,
                                      max_seq_len=1024)
    phgate_model = GPTModelPHGate(vocab_size=50257, embed_size=768, num_heads=12, num_layers=12, dropout=0.1,
                                  max_seq_len=1024, forward_expansion=4)

    standard_model.to(device)
    phgate_model.to(device)

    # Define optimizer.
    optimizer_standard = torch.optim.Adam(standard_model.parameters(), lr=learning_rate)
    optimizer_phgate = torch.optim.Adam(phgate_model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # micro Train and evaluate Standard GPT model
    print("micro Training Standard GPT:")
    standard_micro_losses, standard_micro_time = train_model(standard_model, micro_dataloader, optimizer_standard, criterion, device, num_epochs, model_type="standard")
    print(f"Standard GPT micro Training Time: {standard_micro_time:.2f} seconds")

    print("micro Evaluation for Standard GPT:")
    standard_micro_loss, standard_micro_ppl, standard_micro_metrics = evaluate_language_model(standard_model, micro_dataloader, device,
                                                                                   criterion, model_type="standard")
    print(f"Standard GPT - Final micro Average Loss: {standard_micro_loss:.4f}, micro Perplexity: {standard_micro_ppl:.4f}")
    print("Standard GPT - micro Attention Focus:",["attention_focus"])

    # micro Training and evaluation for pH-Gated GPT
    print("micro training pH-Gated GPT:")
    phgate_micro_losses, phgate_micro_time = train_model(phgate_model, micro_dataloader, optimizer_phgate, criterion, device, num_epochs, model_type="phgate")
    print(f"pH-Gated GPT micro Training Time: {phgate_micro_time:.2f} seconds")

    print("micro Evaluation for pH-Gated GPT:")
    phgate_micro_loss, phgate_micro_ppl, phgate_micro_metrics = evaluate_language_model(phgate_model, micro_dataloader, device,
                                                                             criterion, model_type="phgate")
    print(f"pH-Gated GPT - Final micro Average Loss: {phgate_micro_loss:.4f}, micro Perplexity: {phgate_micro_ppl:.4f}")
    print("pH-Gated GPT - micro Attention Focus:", phgate_micro_metrics["attention_focus"])
    print("pH-Gated GPT - micro Novel Enzyme Detection:", phgate_micro_metrics["novel_identification"])
    print("pH-Gated GPT - micro Gating Activation Summary (per batch):", phgate_micro_metrics["gating_summary"])

    # Plot training loss per epoch for both models
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, standard_micro_losses, label='Standard GPT')
    plt.plot(epochs, phgate_micro_losses, label='pH-Gated GPT')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()
    plt.show()

    # Track total training time vs. final performance
    plt.figure(figsize=(10, 6))
    plt.bar(['Standard GPT', 'pH-Gated GPT'], [standard_micro_time, phgate_micro_time], color=['blue', 'green'])
    plt.ylabel('Total Training Time (seconds)')
    plt.title('Total Training Time vs. Final Performance')
    plt.show()

if __name__ == "__main__":
    main()
