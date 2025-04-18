"""
Created on March 25, 2025

@author: samlevy

Standard GPT Model (Baseline)

This implementation is derived from the reference GPT model provided by the course instructor.
The core architecture remains the same: a stack of transformer blocks with multi-head self-attention.
Minor adjustments were made for logging, clarity, and alignment with how metrics are handled in
the pH-Gated GPT model for consistent comparison during experimentation.

Changes from the course's original implementation:
- Added detailed docstrings and inline biological/machine learning commentary.
- Integrated `return_attention` toggles to better analyze model behavior during training.
- Minor reshaping consistency and device handling in `__main__` for generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Self-Attention Mechanism
class SelfAttention(nn.Module):
    """
    Implements multi-head self-attention using scaled dot-product attention.

    Each token attends to every other token in the sequence, and attention scores
    are normalized using softmax. This allows the model to learn contextual relationships
    regardless of distance in the input sequence.
    """

    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        # Ensure embedding size is divisible by number of heads
        assert self.head_dim * heads == embed_size, "Embedding size must be divisible by heads"

        # Linear projections for Q, K, V
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # Final output projection
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        """
        Compute scaled dot-product attention.
        Args:
            values, keys, query: (batch, seq_len, embed_size)
            mask: (1, 1, seq_len, seq_len) – causal attention mask

        Returns:
            out: (batch, seq_len, embed_size) – attention-modulated output
            attention: (batch, heads, seq_len, seq_len) – attention weights
        """
        N, seq_len, _ = query.shape

        # Reshape input into multiple heads
        values = values.view(N, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        keys = keys.view(N, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        queries = query.view(N, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention energy computation: QK^T
        energy = torch.einsum("nhqd,nhkd->nhqk", queries, keys)  # (batch, heads, seq_len, seq_len)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize attention scores
        attention = torch.softmax(energy / math.sqrt(self.head_dim), dim=-1)

        # Apply attention to values
        out = torch.einsum("nhqk,nhkd->nhqd", attention, values)  # (batch, heads, seq_len, head_dim)
        out = out.permute(0, 2, 1, 3).contiguous().view(N, seq_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out, attention


# Transformer Block (No pH Gating)
class TransformerBlock(nn.Module):
    """
    A single transformer block consisting of:
    - Multi-head attention
    - Feed-forward network
    - LayerNorm and residual connections
    """

    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask, return_attention=False):
        attn_output, attn_weights = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attn_output + query))  # Residual connection
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))  # Residual connection

        if return_attention:
            return {"output": out, "attentions": attn_weights}
        return {"output": out}

# GPT Model (Baseline)
class GPTModelStandard(nn.Module):
    """
    GPT-style model built from a stack of transformer blocks.

    This model serves as the baseline for comparison against
    the pHGate GPT. Using positional embeddings
    and a stack of transformer layers to learn autoregressive token predictions.
    """

    def __init__(self, vocab_size, embed_size, num_heads, num_layers, dropout=0.1, max_seq_len=1024,
                 forward_expansion=4):
        super(GPTModelStandard, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_size))

        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, dropout, forward_expansion)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, vocab_size, bias=False)

    def forward(self, x, attn_mask=None, return_attention=False):
        """
        Forward pass for the standard GPT model.

        Args:
            x (Tensor): Input token indices of shape (batch, seq_len)
            attn_mask (Tensor): Optional mask for causal attention
            return_attention (bool): Return attention weights for visualization/debugging

        Returns:
            Dict containing logits and optionally attention weights.
        """
        b, t = x.size()
        token_embeddings = self.embed(x)  # (b, t, embed_size)
        pos_embeddings = self.pos_embed[:, :t, :]  # (1, t, embed_size)
        x = token_embeddings + pos_embeddings  # Positional encoding

        attn_weights_list = []
        for layer in self.layers:
            if return_attention:
                layer_output = layer(x, x, x, attn_mask, return_attention=True)
                x = layer_output["output"]
                attn_weights_list.append(layer_output["attentions"])
            else:
                x = layer(x, x, x, attn_mask)["output"]

        x = self.ln_f(x)
        logits = self.head(x)  # Predict next token distribution
        output = {"logits": logits}
        if return_attention:
            output["attentions"] = attn_weights_list
        return output



# Example Usage 
if __name__ == "__main__":

    # Hyperparameters
    embed_size = 128
    num_layers = 4
    heads = 8
    vocab_size = 1000
    max_length = 512
    dropout = 0.1
    forward_expansion = 4

    # Instantiate and Test Model
    model = GPTModelStandard(
        vocab_size, embed_size, heads, num_layers, dropout, max_length, forward_expansion
    )

    # Sample input (Batch size = 32, Sequence length = 48)
    B = 32
    S = 48
    x = torch.randint(0, vocab_size, (B, S))

    # Causal mask for autoregressive decoding
    mask = torch.tril(torch.ones(S, S)).unsqueeze(0).unsqueeze(0).to(x.device)

    # Forward pass through baseline model
    outputs = model(x, mask, return_attention=True)
    logits = outputs["logits"]
    attn_weights = outputs["attentions"]

    print("Logits shape:", logits.shape)
