import torch.nn as nn

class PatchEmbedding(nn.Module):
    """Split image into patches and then embed them.

    Parameters:
        image_size (int): size of square image.
        patch_size (int): size of patches to split the image into.
        in_channels(int): number of channels in input image.
        embedding_dimension(int): the embedding dimension.
    Attributes:
        n_patches(int): number of patches inside of our image.
        projection(nn.Conv2d): The layer that split image into patches and convert them into embeddings. 
    """
    def __init__(
            self, 
            image_size : int, 
            patch_size : int, 
            in_channels : int = 3, 
            embedding_dimension : int = 768
            ) -> None:
        super().__init__()
        assert image_size > patch_size, "image_size should be greater than patch_size"
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2

        # non-overlapping convolution == converting into patches + embedding
        self.projection = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=embedding_dimension,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        """Run forward pass.
        Parameters: 
            x(torch.tensor): tensor of shape `(B, C, S, S)`, where B-batch_size, C-number of channels, S-image_size
        Returns:
            torch.tensor of shape `(B, P, E)`, where B-batch_size, P-patch_size, E-embedding_size
        """
        # (batch_size, embedding_dim, n_patches**0.5, n_patches**0.5)
        x = self.projection(x)
        # (batch_size, embedding_dim, n_patches)
        x = x.flatten(2)
        # (batch_size, n_patches, embedding_dim)
        x = x.transpose(1, 2)
        return x