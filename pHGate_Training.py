"""
Created on March 30, 2025

@author: samlevy

pH-Gated GPT Training
"""

import os
import json
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Dataset_Construct import TokenizedBioDataset, collate_fn



# pH Gate Module and pH-Gated Blocks
class pHGate(nn.Module):
    """
    pHGate computes a learnable gating vector from an average of token embeddings.

    The gating mechanism uses a fully-connected layer and a sigmoid activation to produce a value 
    between 0 and 1, similar to the pH-dependent activation observed in biological systems.
    """

    def __init__(self, embed_size):
        super(pHGate, self).__init__()
        # Learnable weight matrix and bias for gating.
        self.fc = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        """
        Forward pass for the pHGate.

        Args:
            x (Tensor): Averaged token embeddings of shape (batch, 1, embed_size)

        Returns:
            Tensor: Gating vector (batch, 1, embed_size) with values scaled via sigmoid.
        """
        gate = torch.sigmoid(self.fc(x))
        return gate


class SelfAttentionPHGate(nn.Module):
    """
    Self-Attention module with integrated pH gating.

    This module performs multi-head attention with standard operations: projection, scaled dot-product,
    and reshaping. The implementation splits the embedding dimension across multiple heads and computes
    attention weights by performing dot product between queries and keys.
    """

    def __init__(self, embed_size, heads):
        super(SelfAttentionPHGate, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size, "Embedding size must be divisible by heads"

        # Linear projections for values, keys, and queries.
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        """
        Compute multi-head attention.

        Args:
            values (Tensor): Value tensor of shape (batch, seq_len, embed_size)
            keys (Tensor): Key tensor of shape (batch, seq_len, embed_size)
            query (Tensor): Query tensor of shape (batch, seq_len, embed_size)
        Returns:
            out (Tensor): Attention output (batch, seq_len, embed_size)
            attention (Tensor): Attention weights for each head.
        """
        N, seq_len, _ = query.shape
        # Reshape and permute to obtain (batch, heads, seq_len, head_dim)
        values = values.view(N, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        keys = keys.view(N, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        queries = query.view(N, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)

        # Compute raw attention scores using Einstein summation over query and key.
        energy = torch.einsum("nhqd,nhkd->nhqk", queries, keys)
        if mask is not None:
            # Positions with mask==0 are set to large negative values to nullify their contribution.
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy / math.sqrt(self.head_dim), dim=-1)

        # Compute weighted sum of values.
        out = torch.einsum("nhqk,nhkd->nhqd", attention, values)
        # Restore original dimensions: (batch, seq_len, embed_size)
        out = out.permute(0, 2, 1, 3).contiguous().view(N, seq_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out, attention


class TransformerBlockPHGate(nn.Module):
    """
    Transformer block with integrated pH gating.

    This block has a self-attention mechanism (with pH gate), a feed-forward network (with GELU activation),
    and residual connections with layer normalization. The pH gate scales the attention output similarly to a 
    biological pH-triggered activation.
    """

    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlockPHGate, self).__init__()
        self.attention = SelfAttentionPHGate(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        # Feed-forward network with expansion and GELU activation.
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
        self.ph_gate = pHGate(embed_size)

    def forward(self, value, key, query, mask, return_attention=False, return_metrics=False):
        """
        Forward pass for the transformer block.

        Args:
            value, key, query (Tensor): Inputs with shape (batch, seq_len, embed_size)
            mask (Tensor): Optional attention mask.
            return_attention (bool): Flag to return attention weights.
            return_metrics (bool): Flag to return additional metric (gate values).

        Returns:
            Depending on flags, returns the output tensor and optionally the gating values and/or attention weights.
        """
        # Compute standard self-attention.
        attn_output, attn_weights = self.attention(value, key, query, mask)
        # Compute an average representation of the query tokens (biologically analogous to an overall pH measurement).
        x_avg = query.mean(dim=1, keepdim=True)  # (batch, 1, embed_size)
        gate = self.ph_gate(x_avg)  # pH gate activation.
        # Dynamically scale the attention output.
        gated_attn_output = gate * attn_output

        # Residual connection with dropout and layer normalization.
        x = self.dropout(self.norm1(gated_attn_output + query))
        forward_out = self.feed_forward(x)
        out = self.dropout(self.norm2(forward_out + x))

        # Return additional metrics if requested.
        if return_metrics:
            if return_attention:
                return out, gate, attn_weights
            return out, gate
        if return_attention:
            return out, attn_weights
        return out


# pH-Gated GPT Model
class GPTModelPHGate(nn.Module):
    """
    pH-Gated GPT Model integrating a pH gate into the transformer blocks.

    The model has tpken and positional embeddings, followed by a stack of transformer blocks with pH gating.
    The final layer normalizes the output before projecting to vocabulary space for prediction.
    """

    def __init__(self, vocab_size, embed_size, num_heads, num_layers,
                 dropout=0.1, max_seq_len=1024, forward_expansion=4):
        super(GPTModelPHGate, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_size))
        self.layers = nn.ModuleList([
            TransformerBlockPHGate(embed_size, num_heads, dropout, forward_expansion)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, vocab_size, bias=False)

    def forward(self, x, attn_mask=None, return_attention=False, return_metrics=False):
        """
        Forward pass of the GPT model.

        Args:
            x (Tensor): Input tensor of token IDs with shape (batch, seq_len).
            attn_mask (Tensor): Causal mask for autoregressive decoding.
            return_attention (bool): Flag to return attention weights.
            return_metrics (bool): Flag to return gating activations.

        Returns:
            Dict: Contains logits and optionally lists of attention weights and gating activations from each layer.
        """
        b, t = x.size()
        token_embeddings = self.embed(x)  # (b, t, embed_size)
        pos_embeddings = self.pos_embed[:, :t, :]  # (1, t, embed_size)
        x = token_embeddings + pos_embeddings  

        gating_activations_list = []
        attentions_list = []

        for layer in self.layers:
            if return_metrics:
                if return_attention:
                    x, gate, attn_weights = layer(x, x, x, attn_mask, return_attention=True, return_metrics=True)
                    gating_activations_list.append(gate)
                    attentions_list.append(attn_weights)
                else:
                    x, gate = layer(x, x, x, attn_mask, return_metrics=True)
                    gating_activations_list.append(gate)
            else:
                if return_attention:
                    x, attn_weights = layer(x, x, x, attn_mask, return_attention=True)
                    attentions_list.append(attn_weights)
                else:
                    x = layer(x, x, x, attn_mask)
        x = self.ln_f(x)
        logits = self.head(x)  

        if return_metrics:
            return {
                "logits": logits,
                "gating_activations": gating_activations_list,
                "attentions": attentions_list
            }
        if return_attention:
            return {"logits": logits, "attentions": attentions_list}
        return {"logits": logits}



# Practice Enzyme Recognition Function
def compute_enzyme_focus(attentions, labels):
    """
     Function to compute enzyme recognition rate.

    This function averages the attention weights across all layers to emulate detecting key signals,
    analogous to identifying focus regions in pH-regulating processes.

    Args:
        attentions (list): List of attention weights from each layer.
        labels (Tensor): Target labels (unused in this [ractice] implementation).

    Returns:
        float: Average focus rate computed from attention weights.
    """
    focus_rates = [attn.mean().item() for attn in attentions]
    return sum(focus_rates) / len(focus_rates) if focus_rates else 0

# Utility Functions for Training and Evaluation

def save_checkpoint(model, optimizer, epoch, filename="phgate_gpt_checkpoint.pt"):
    """
    Save model and optimizer checkpoint to disk.

    Args:
        model (nn.Module): The model to save.
        optimizer (Optimizer): The optimizer whose state is to be saved.
        epoch (int): Current epoch number.
        filename (str): Path to save the checkpoint.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def log_metrics(log_file, epoch, step, loss, elapsed, remaining):
    """
    Log training metrics to a file.

    Args:
        log_file (str):Path of the log file.
         epoch (int): Current epoch.
        step (int): Current step in the epoch.
        loss (float): Loss at the given step.
        elapsed (float): Time elapsed in the epoch.
        remaining (float): Estimated remaiing time for the epoch.
    """
    with open(log_file, "a") as f:
        f.write(
            f"Epoch {epoch}, Step {step}: Loss = {loss:.4f}, Elapsed = {elapsed:.2f}s, Remaining = {remaining:.2f}s\n"
        )


def save_attention_visualization(attn_weights, layer=6, head=3, filename="phgate_attention_weights.png"):
    """
    attention weights visualization
 !!!!! Samantha Fix it!!!!

    """
    # Select the specific attention weights for the specified layer and head.
    attn = attn_weights[layer][0][head].detach().cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.imshow(attn, cmap='viridis')
    plt.colorbar()
    plt.title(f"Attention Weights (Layer {layer + 1}, Head {head + 1})")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.savefig(filename)
    plt.close()
    print(f"Attention visualization saved to {filename}")


# Main Training and Evaluation Function
def main():
    """
    Main function for training and evaluating the pH-Gated GPT model.

    Sets up the dataset, dataloader, model, optimizer, and training loop. During each epoch,
    it logs metrics, saves checkpoints, and evaluates the model via text generation and
    attention weight visualization.
    """
    # Hyperparameters Configuration
    vocab_size = 50257
    embed_size = 768
    num_heads = 12
    num_layers = 12
    dropout = 0.1
    max_seq_len = 1024
    forward_expansion = 4
    batch_size = 4
    num_epochs = 3
    learning_rate = 3e-4

    # File paths and dataset preparation
    data_path = "tokenized_data.json"  # Tokenized dataset file.
    log_file = "phgate_training_log.txt"
    checkpoint_file = "phgate_gpt_checkpoint.pt"

    dataset = TokenizedBioDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Model and Optimizer Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModelPHGate(vocab_size, embed_size, num_heads, num_layers, dropout, max_seq_len, forward_expansion)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    if os.path.exists(log_file):
        os.remove(log_file)

    total_steps = len(dataloader)
    #  Store training metrics across epochs.
    train_metrics = {"perplexity": [], "gating_activation": [], "enzyme_recognition": []}

    for epoch in range(1, num_epochs + 1):
        start_time_epoch = time.time()
        total_loss = 0.0
        # Lists per-batch metrics for the current epoch.
        epoch_perplexities = []
        epoch_gate_activations = []
        epoch_enzyme_recognition = []

        for step, batch in enumerate(dataloader, start=1):
            batch = {key: val.to(device) for key, val in batch.items()}
            optimizer.zero_grad()

            # Prepare inputs and targets for next-token prediction.
            inputs = batch['input_ids']
            targets = inputs[:, 1:]
            inputs = inputs[:, :-1]
            seq_len = inputs.size(1)
            # Creates causal mask to ensure autoregressive attention.
            mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0).to(device)

            # Forward pass with metric outputs (gating and attention).
            output = model(inputs, attn_mask=mask, return_attention=True, return_metrics=True)
            logits = output["logits"]
            loss = criterion(logits.view(-1, vocab_size), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Calculate perplexity as exponential
            perplexity = math.exp(loss.item())
            epoch_perplexities.append(perplexity)

            # Average the pH gate activations across transformer layers.
            gate_means = [gate.mean().item() for gate in output["gating_activations"]]
            avg_gate_activation = sum(gate_means) / len(gate_means) if gate_means else 0
            epoch_gate_activations.append(avg_gate_activation)

            # Compute enzyme recognition rate using a dummy function.
            enzyme_rate = compute_enzyme_focus(output["attentions"], targets)
            epoch_enzyme_recognition.append(enzyme_rate)

            elapsed = time.time() - start_time_epoch
            progress = step / total_steps
            total_estimated = elapsed / progress if progress > 0 else 0
            remaining = total_estimated - elapsed
            if step % 10 == 0 or step == total_steps:
                print(f"Epoch {epoch}, Step {step}/{total_steps} - Loss: {loss.item():.4f}, "
                      f"Perplexity: {perplexity:.4f}, Gate Activation: {avg_gate_activation:.4f}, "
                      f"Enzyme Recognition: {enzyme_rate:.4f}, Elapsed: {elapsed:.2f}s, Remaining: {remaining:.2f}s")
                log_metrics(log_file, epoch, step, loss.item(), elapsed, remaining)

        # Calculate epoch-level average
        avg_loss = total_loss / total_steps
        perplexity_epoch = sum(epoch_perplexities) / len(epoch_perplexities)
        avg_gate_activation_epoch = sum(epoch_gate_activations) / len(epoch_gate_activations)
        avg_enzyme_recognition_epoch = sum(epoch_enzyme_recognition) / len(epoch_enzyme_recognition)
        total_time_epoch = time.time() - start_time_epoch

        # Save loss history after every epoch
        if 'loss_history' not in train_metrics:
            train_metrics['loss_history'] = []

        train_metrics['loss_history'].append(avg_loss)  

        # Save the loss history to a JSON file
        with open("losses_phgate.json", "w") as f:
            json.dump(train_metrics['loss_history'], f)
        print("Saved loss history to losses_phgate.json")

        print(f"Epoch {epoch} complete! Avg Loss: {avg_loss:.4f}, Perplexity: {perplexity_epoch:.4f}, "
              f"Avg Gate Activation: {avg_gate_activation_epoch:.4f}, Enzyme Recognition: {avg_enzyme_recognition_epoch:.4f}, "
              f"Total Time: {total_time_epoch:.2f}s")
        log_metrics(log_file, epoch, total_steps, avg_loss, total_time_epoch, 0)
        save_checkpoint(model, optimizer, epoch, checkpoint_file)

        # Store metrics for current epoch.
        train_metrics["perplexity"].append(perplexity_epoch)
        train_metrics["gating_activation"].append(avg_gate_activation_epoch)
        train_metrics["enzyme_recognition"].append(avg_enzyme_recognition_epoch)
        with open("training_metrics.json", "w") as f:
            json.dump(train_metrics, f)

    # Evaluation: Attention Visualization
    model.eval()
    # Visualize attention weights for a sample batch. Samantha Fix it!!!!
    sample_batch = next(iter(dataloader))
    sample_batch = {key: val.to(device) for key, val in sample_batch.items()}
    seq_len = sample_batch['input_ids'].size(1) - 1
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0).to(device)
    outputs = model(sample_batch['input_ids'][:, :-1], attn_mask=mask, return_attention=True)
    attn_weights_list = outputs["attentions"]
    save_attention_visualization(attn_weights_list, layer=0, head=0, filename="phgate_attention_weights.png")


if __name__ == "__main__":
    main()
