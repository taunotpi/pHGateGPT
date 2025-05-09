"""
Created on March 25, 2025

@author: samlevy

pH-Gated GPT Model with Integrated Gating in the Self-Attention Layer

This module implements a GPT-style model enhanced with a learnable pH gate. Inspired by the pH-dependent activation
mechanism of the influenza virus’s hemagglutinin (HA) protein, the gating mechanism controls the flow of information in 
the self-attention layers. The pH gate dynamically modulates the attention outputs by scaling them with a sigmoid-derived 
gate, analogous to biological processes where enzyme activation is dependent on environmental pH.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from Dataset_Construct import TokenizedBioDataset, collate_fn
from torch.utils.data import DataLoader


# pH Gate Module
class pHGate(nn.Module):
    """
    pHGate computes a gating vector from the average token embeddings.

    The module applies a fully connected layer followed by a sigmoid activation to generate a gating value between 0 and 1.
    This value is used to dynamically scale the attention outputs, similar to how biological systems control enzyme activation.
    """

    def __init__(self, embed_size):
        super(pHGate, self).__init__()
        self.fc = nn.Linear(embed_size, embed_size)  # Wg and bg in GpH = σ(Wg ⋅ x + bg), representing learnable gating weights and bias


    def forward(self, x):
        """
        Forward pass for pHGate.

        Args:
            x (Tensor): Input tensor with shape (batch, 1, embed_size), representing averaged token embeddings.

        Returns:
            Tensor: Gating vector with values between 0 and 1, same shape as x.
        """
        # Compute the gating values using a linear transformation and sigmoid activation.
        gate = torch.sigmoid(self.fc(x))  # Apply sigmoid to Wg⋅x + bg, producing GpH values between 0 and 1
        return gate


# Self-Attention with pH Gating
class SelfAttentionPHGate(nn.Module):
    """
    Self-Attention module with integrated pH gating.

    Implements the multi-head attention mechanism as used in transformer models while computing scaled attention weights.
    The module reshapes, projects, and aggregates tokens from multiple heads, with a final linear projection to combine them.
    """

    def __init__(self, embed_size, heads):
        super(SelfAttentionPHGate, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        # Ensure that embedding size is exactly divisible by number of heads.
        assert self.head_dim * heads == embed_size, "Embedding size must be divisible by heads"

        # Weight matrices for values, keys, queries, and final projection.
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        """
        Compute the multi-head attention with optional mask.

        Args:
            values (Tensor): Tensor of values with shape (batch, seq_len, embed_size)
            keys (Tensor): Tensor of keys with shape (batch, seq_len, embed_size)
            query (Tensor): Tensor of queries with shape (batch, seq_len, embed_size)
            mask (Tensor): Attention mask (optional), where non-valid positions are marked with zeros.

        Returns:
            out (Tensor): Attention output with shape (batch, seq_len, embed_size)
            attention (Tensor): Attention weights for each head.
        """
        N, seq_len, _ = query.shape

        # Reshape and permute into (batch, heads, seq_len, head_dim)
        values = values.view(N, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        keys = keys.view(N, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        queries = query.view(N, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)

        # Compute dot products (scaled by the square root of head dimension for stability).
        energy = torch.einsum("nhqd,nhkd->nhqk", queries, keys)
        if mask is not None:
            # Apply the mask: positions with a zero in the mask are not attended to.
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy / math.sqrt(self.head_dim), dim=-1)

        # Compute weighted sum of values.
        out = torch.einsum("nhqk,nhkd->nhqd", attention, values)
        # Rearrange output back to original dimension: (batch, seq_len, embed_size)
        out = out.permute(0, 2, 1, 3).contiguous().view(N, seq_len, self.heads * self.head_dim)
        # Final linear projection.
        out = self.fc_out(out)
        return out, attention

# Transformer Block with pH Gate
class TransformerBlockPHGate(nn.Module):
    """
    Transformer block that integrates the pH gate into the self-attention mechanism.

    This block comprises a self-attention layer with pH gating, followed by residual connections, layer normalization,
    and a feed-forward network. It allows for conditional propagation of attention outputs based on the gating vector.
    """

    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlockPHGate, self).__init__()
        self.attention = SelfAttentionPHGate(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # Feed-forward network with GELU activation for smoother non-linear transformations.
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
        self.ph_gate = pHGate(embed_size)  # Instantiate the pH gate module.

    def forward(self, value, key, query, mask, return_attention=False, return_gating=False):
        """
        Forward pass for the transformer block.

        Args:
            value, key, query (Tensor): Inputs with shape (batch, seq_len, embed_size)
            mask (Tensor): Mask for attention mechanism.
            return_attention (bool): Whether to output attention weights.
            return_gating (bool): Whether to output gating activations.

        Returns:
            Dictionary containing:
                - output (Tensor): The final block output.
                - gate (Tensor, optional): pH gate activations.
                - attentions (Tensor, optional): Attention weights.
        """
        # Compute self-attention and obtain weights.
        attn_output, attn_weights = self.attention(value, key, query, mask)

        # Compute the average representation of the query tokens (biologically analogous to an average pH reading).
        x_avg = query.mean(dim=1, keepdim=True)  # (batch, 1, embed_size)
        # Generate the gating vector using the pH gate.
        gate = self.ph_gate(x_avg)  # (batch, 1, embed_size)
        # Apply the gate to modulate the attention output.
        gated_attn_output = gate * attn_output  # A_out = GpH * A; modulating attention by environmental context

        # Apply dropout and layer normalization as residual connections.
        x = self.dropout(self.norm1(gated_attn_output + query))
        # Feed-forward network for additional transformation.
        forward_out = self.feed_forward(x)
        out = self.dropout(self.norm2(forward_out + x))

        # Prepare output dictionary with optional gating and attention details.
        output_dict = {"output": out}
        if return_gating:
            output_dict["gate"] = gate
        if return_attention:
            output_dict["attentions"] = attn_weights
        return output_dict

# pH-Gated GPT Model

class PHGateGPT(nn.Module):
    """
    pH-Gated GPT Model modified to track gating activations.

    This is an autoregressive transformer model with an embedding layer, positional encoding, and multiple 
    stacked transformer blocks. The unique aspect is the integration of the pH gating mechanism that dynamically
    modulates attention outputs, inspired by biological analogues in pH-dependent protein activation.
    """

    def __init__(self, vocab_size, embed_size, num_heads, num_layers,
                 dropout=0.1, max_seq_len=1024, forward_expansion=4):
        super(PHGateGPT, self).__init__()
        # Token embedding layer.
        self.embed = nn.Embedding(vocab_size, embed_size)
        # Learnable positional embeddings.
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_size))

        # Stacked transformer layers with pH gate.
        self.layers = nn.ModuleList([
            TransformerBlockPHGate(embed_size, num_heads, dropout, forward_expansion)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_size)
        # Final projection layer to generate logits over the vocabulary.
        self.head = nn.Linear(embed_size, vocab_size, bias=False)

    def forward(self, x, attn_mask=None, return_attention=False, return_gating=False):
        """
        Forward pass for the pH-Gated GPT model.

        Args:
            x (Tensor): Input tensor of token IDs with shape (batch, seq_len).
            attn_mask (Tensor): Causal mask for autoregressive decoding.
            return_attention (bool): Whether to return attention weights.
            return_gating (bool): Whether to return gating activations.

        Returns:
            Dictionary containing:
                - logits (Tensor): The final logits for each token.
                - attentions (list, optional): List of attention weights from each layer.
                - gating_activations (list, optional): List of gating activations from each layer.
                - hidden_states (list, optional): Hidden states after each transformer block.
        """
        b, t = x.size()
        # Obtain token embeddings.
        token_embeddings = self.embed(x)  # (b, t, embed_size)
        # Extract corresponding positional embeddings.
        pos_embeddings = self.pos_embed[:, :t, :]
        # Combine token and positional embeddings.
        x = token_embeddings + pos_embeddings

        # Lists to save intermediate states.
        hidden_states = []
        gating_activations_list = []
        attentions_list = []

        # Pass through each transformer block.
        for layer in self.layers:
            # Determine the appropriate output based on flags.
            if return_gating:
                if return_attention:
                    layer_output = layer(x, x, x, attn_mask, return_attention=True, return_gating=True)
                    gate = layer_output["gate"]
                    attn_weights = layer_output["attentions"]
                    gating_activations_list.append(gate)
                    attentions_list.append(attn_weights)
                else:
                    layer_output = layer(x, x, x, attn_mask, return_gating=True)
                    gate = layer_output["gate"]
                    gating_activations_list.append(gate)
            else:
                if return_attention:
                    layer_output = layer(x, x, x, attn_mask, return_attention=True)
                    attn_weights = layer_output["attentions"]
                    attentions_list.append(attn_weights)
                else:
                    layer_output = layer(x, x, x, attn_mask)
            # Update x for the next layer and record hidden state.
            x = layer_output["output"]
            hidden_states.append(x)

        # Final normalization and logits computation.
        x = self.ln_f(x)
        logits = self.head(x)
        output = {"logits": logits}
        if return_attention:
            output["attentions"] = attentions_list
        if return_gating:
            output["gating_activations"] = gating_activations_list
            output["hidden_states"] = hidden_states
        return output

    def compute_gating(self, hidden_states):
        """
        Compute a summary of gating activations from the last layer's hidden state.

        Here, the gating activations are averaged over the tokens.

        Args:
            hidden_states (list): List containing the hidden states from each transformer block.

        Returns:
            numpy.ndarray: Numpy array representing the summary gating activations.
        """
        return torch.mean(hidden_states[-1], dim=1).detach().cpu().numpy()


# Example Usage
if __name__ == "__main__":
    # Hyperparameters Configuration
    embed_size = 128
    num_layers = 4
    heads = 8
    vocab_size = 1000
    max_length = 512
    dropout = 0.1
    forward_expansion = 4

    # Initialize the pH-Gated GPT model.
    model = PHGateGPT(vocab_size, embed_size, heads, num_layers, dropout, max_length, forward_expansion)

    # Create sample input: Batch size (B) and Sequence length (S).
    B = 32
    S = 48
    x = torch.randint(0, vocab_size, (B, S))

    # Create a causal mask for autoregressive prediction. This lower-triangular matrix ensures that 
    # each token only attends to previous tokens.
    mask = torch.tril(torch.ones(S, S)).unsqueeze(0).unsqueeze(0).to(x.device)

    # Perform a forward pass through the model with gating and attention tracking.
    outputs = model(x, attn_mask=mask, return_attention=True, return_gating=True)
    logits = outputs["logits"]
    gating_activations = outputs["gating_activations"]
    attentions = outputs["attentions"]

    # Display the dimensions of the computed logits.
    print("Logits shape:", logits.shape)

    # Compute and display a summary of gating activations from the final transformer block.
    gating_summary = model.compute_gating(outputs["hidden_states"])
    print("Gating summary shape:", gating_summary.shape)
