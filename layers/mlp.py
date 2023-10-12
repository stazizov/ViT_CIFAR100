import torch.nn as nn

class MLP(nn.Module):
    """Multilayer perceptron.

    Parameters:
        in_features(int): Number of input features.
        hidden_features(int): Number of nodes in hidden layer.
        out_features(int): Number of output features.
        p_drop(float): Dropout probability.
    
    Attribute:
        fc(nn.Linear): The first linear layer.
        act(nn.GELU): gaussian error linear unit.
        fc2(nn.Linear): The second linear layer.
        dropout(nn.Dropout): The dropout layer.
    """
    def __init__(self, in_features:int, hidden_features:int, out_features:int, p_drop:float=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        """Run forward pass.
        Parameters:
            x(torch.tensor): shape `(n_samples, n_patches + 1, in_features)`
        Returns:
            torch.tensor of shape (n_samples, n_patches + 1, out_features)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x