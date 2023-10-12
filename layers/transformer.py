import torch
import torch.nn as nn
from layers.mlp import MLP
from layers.attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """Transformer block.
    
    Parameters: 
        dimensions(int): embedding dimensions.
        n_heads(int): Number of attention heads.
        mlp_ratio(float): Determines the hidden dimension size of the `MLP` module with respect to `dimensions`.
        qkv_bias(bool): If True then we include bias to the query, key and value projections.
        proj_p(float): Dropout probability.
        attn_p(float): Dropout probability.
    Attributes:
        norm1, norm2 (nn.LayerNorm): Layer Normalization.
        attn(MultiHeadAttention): Attention layer.
        mlp(MLP): MLP module.
    """
    def __init__(
            self, 
            dimensions, 
            n_heads, 
            mlp_ratio=4.0, 
            qkv_bias=True, 
            proj_p=0., 
            attn_p=0.
            ):

        super().__init__()
        self.norm1 = nn.LayerNorm(dimensions, eps= 1e-6)
        self.norm2 = nn.LayerNorm(dimensions, eps=1e-6)
        self.attn = MultiHeadAttention(
            dimensions=dimensions, 
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=proj_p
        )
        self.mlp = MLP(
            dimensions, 
            int(dimensions * mlp_ratio), 
            dimensions
        )
    
    def forward(self, x):
        """Run forward pass.
        Parameters:
            x(torch.tensor): shape `(n_samples, n_patches + 1, dimensions)`
        Returns:
            torch.tensor of shape (n_samples, n_patches + 1, dimensions)
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x 
    


