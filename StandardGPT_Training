"""
Created on April 02, 2025

@author: samlevy

Standard GPT Training Script (Baseline)

"""


import os
import json
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from StandardGPT import GPTModelStandard
from Dataset_Construct import TokenizedBioDataset, collate_fn


# Checkpoint Utility
def save_checkpoint(model, optimizer, epoch, filename="standard_gpt_checkpoint.pt"):
    """
    model.eval()
    Saves model and optimizer state for future recovery.

    Args:
        model (nn.Module): The model instance to save
        optimizer (Optimizer): Optimizer with its internal state
        epoch (int): Current training epoch
        filename (str): Path to save the checkpoint
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


# Hyperparameters and Setup
vocab_size = 50257  
embed_dim = 768  
num_heads = 12  
num_layers = 12  
dropout = 0.1
max_seq_len = 1024  
batch_size = 4
num_epochs = 3
learning_rate = 3e-4
data_path = "tokenized_data.json"  

# Load tokenized biological dataset
dataset = TokenizedBioDataset(data_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Initialize model and move to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTModelStandard(vocab_size, embed_dim, num_heads, num_layers, dropout, max_seq_len)
model.to(device)
model.train()

# Loss and optimization
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

total_steps_in_epoch = len(dataloader)

# Training Loop
for epoch in range(num_epochs):
    start_time_epoch = time.time()
    total_loss = 0.0

    for step, batch in enumerate(dataloader, start=1):
        # Move batch to the correct device (CPU/GPU)
        batch = {key: val.to(device) for key, val in batch.items()}
        optimizer.zero_grad()

        # Prepare input and target for next-token prediction
        inputs = batch['input_ids']
        targets = inputs[:, 1:]  # Predict next token
        inputs = inputs[:, :-1]  # Input excludes last token

        # Forward pass
        outputs = model(inputs)
        logits = outputs["logits"]
        loss = criterion(logits.view(-1, vocab_size), targets.reshape(-1))

        # Backpropagation
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Logging every 10 steps
        if step % 10 == 0 or step == total_steps_in_epoch:
            elapsed = time.time() - start_time_epoch
            progress = step / total_steps_in_epoch
            total_estimated = elapsed / progress if progress > 0 else 0
            remaining = total_estimated - elapsed
            print(f"Epoch {epoch + 1}, Step {step}/{total_steps_in_epoch} - Loss: {loss.item():.4f}, "
                  f"Elapsed: {elapsed:.2f}s, Remaining: {remaining:.2f}s")

    # Epoch summary
    avg_loss = total_loss / total_steps_in_epoch
    perplexity = math.exp(avg_loss)
    total_time_epoch = time.time() - start_time_epoch

    print(f"Epoch {epoch + 1} complete! Avg Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}, "
          f"Total Time: {total_time_epoch:.2f}s")

    # Save checkpoint after each epoch
    save_checkpoint(model, optimizer, epoch, filename="standard_gpt_checkpoint.pt")



