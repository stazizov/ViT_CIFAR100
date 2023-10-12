import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    """Self-Attention layer.

    This class implements a self-attention mechanism, which computes the attention
    weights for each token in the input sequence based on its relationship with
    other tokens within the same sequence. It is commonly used in various natural
    language processing tasks, such as machine translation and sentiment analysis.

    Args:
        dimensions (int): The input and output dimensions of per token features.

    Attributes:
        input_dim (int): The input dimension of per token features.
        query (nn.Linear): Linear transformation for computing query vectors.
        key (nn.Linear): Linear transformation for computing key vectors.
        value (nn.Linear): Linear transformation for computing value vectors.
        softmax (nn.Softmax): Softmax function to normalize attention scores.

    """

    def __init__(self, dimensions):
        super(SelfAttention, self).__init__()
        self.input_dim = dimensions
        self.query = nn.Linear(dimensions, dimensions)
        self.key = nn.Linear(dimensions, dimensions)
        self.value = nn.Linear(dimensions, dimensions)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        """Compute self-attention weights and apply attention to the input sequence.

        Args:
            x (torch.Tensor): The input sequence tensor of shape (batch_size, seq_len, dimensions).

        Returns:
            torch.Tensor: The output tensor after applying self-attention, with the same shape as input x.

        """
        keys = self.key(x)
        values = self.value(x)
        queries = self.query(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / self.input_dim ** 0.5
        attention = torch.bmm(self.softmax(scores), values)
        return attention


class MultiHeadAttention(nn.Module):
    """Multihead attention.

    This class implements a multihead attention mechanism, which consists of
    multiple parallel attention heads that independently attend to different parts
    of the input sequence. It is commonly used in transformer-based models for
    tasks such as machine translation and language understanding.

    Args:
        dimensions (int): The input and output dimensions of per token features.
        n_heads (int): Number of attention heads.
        qkv_bias (bool): If True, bias is included in the query, key, and value projections.
        attn_p (float): Dropout probability applied to the query, key, and value tensors.
        proj_p (float): Dropout probability applied to the output tensor.

    Attributes:
        qkv (nn.Linear): Linear projection for the query, key, and value.
        attn_dropout (nn.Dropout): Dropout layer for attention.
        proj (nn.Linear): Linear mapping that takes the concatenated output of all
            attention heads and maps them into a new space.
        proj_dropout (nn.Dropout): Dropout layer for the projection.

    """
    def __init__(self, dimensions:int, n_heads:int, qkv_bias=True, attn_p=0., proj_p=0.) -> None:
        super().__init__()
        
        self.n_heads = n_heads
        self.dimensions = dimensions
        self.head_dim = dimensions // n_heads

        self.qkv = nn.Linear(dimensions, dimensions*3, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_p)
        
        self.proj = nn.Linear(dimensions, dimensions)
        self.proj_dropout = nn.Dropout(proj_p)
    
    def forward(self, x):
        """Compute multihead attention on the input sequence.

        Args:
            x (torch.Tensor): The input sequence tensor of shape (batch_size, n_patches + 1, dimensions).

        Returns:
            torch.Tensor: The output tensor after applying multihead attention, with the same shape as input x.

        """
        n_samples, n_tokens, dimensions = x.shape

        if dimensions != self.dimensions:
            raise ValueError("Input dimensions mismatch.")
        
        # (n_samples, n_patches + 1 (class_head), 3 * dimensions)
        qkv = self.qkv(x)
        # (n_samples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.reshape(
            n_samples, 
            n_tokens, 
            3, 
            self.n_heads, 
            self.head_dim 
        )
        # (3, n_samples, n_heads, n_patches + 1, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k , v = qkv[0], qkv[1], qkv[2]
        # (n_samples, n_heads, head_dim, n_patches + 1)
        k_t = k.transpose(-2, -1)
        #  (n_samples, n_heads, n_patches + 1, n_patches + 1)
        dp = (q @ k_t)/(self.head_dim **   .5)
        # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # (n_samples, n_heads, n_patches + 1, head_dim)
        weighted_avg = attn @ v
        # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2)
        # (n_samples, n_patches + 1, dimensions)
        weighted_avg = weighted_avg.flatten(2)
        x = self.proj(weighted_avg)
        x = self.proj_dropout(x)
        return x