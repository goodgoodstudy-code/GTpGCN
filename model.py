import torch
import torch.nn as nn
from torch.nn import functional as F
from abc import abstractmethod
from torch.nn import TransformerEncoderLayer
from torch import Tensor
from typing import Optional
import numpy as np
import torch_geometric as tg
from torch_geometric.nn import GCNConv

# Define a GCN model
class GCN(torch.nn.Module):
    def __init__(self, input_dim, num_classes, dropout, hgc, lg):
        super(GCN, self).__init__()

        # Define the size of hidden layers
        hidden = [hgc for i in range(lg)]
        self.dropout = dropout
        self.relu = torch.nn.ReLU(inplace=True)
        self.lg = lg

        # Define a list of GCN layers using GCNConv
        self.gconv = nn.ModuleList()

        # Initialize convolution layers
        for i in range(lg):
            in_channels = input_dim if i == 0 else hidden[i-1]
            # Using GCNConv instead of ChebConv
            self.gconv.append(GCNConv(in_channels, hidden[i]))

        # Classification layers
        self.cls = nn.Sequential(
            torch.nn.Linear(hidden[lg-1], 256),
            torch.nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            torch.nn.Linear(256, num_classes)
        )

        self.model_init()

    def model_init(self):
        # Initialize parameters of the model
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x, edge_index):
        # First graph convolution layer
        x = self.relu(self.gconv[0](x, edge_index))

        # Subsequent graph convolution layers
        for i in range(1, self.lg):
            x = F.dropout(x, self.dropout, self.training)
            x = self.relu(self.gconv[i](x, edge_index))

        # Classification layer
        logit = self.cls(x)

        return logit

# Define a ChebNet variant (version 1)
class ChebGCNv1(torch.nn.Module):
    def __init__(self, input_dim, num_classes, dropout, hgc, lg, K=3):
        super(ChebGCNv1, self).__init__()
        hidden = [hgc for i in range(lg)]
        self.dropout = dropout
        bias = False
        self.relu = torch.nn.ReLU(inplace=True)
        self.lg = lg
        self.gconv = nn.ModuleList()

        # Define Chebyshev convolution layers
        for i in range(lg):
            in_channels = input_dim if i == 0 else hidden[i-1]
            self.gconv.append(tg.nn.ChebConv(in_channels, hidden[i], K=K, normalization='sym', bias=bias))

        # Classification layers
        self.cls = nn.Sequential(
            torch.nn.Linear(hidden[lg-1], 256),
            torch.nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            torch.nn.Linear(256, num_classes)
        )

        self.model_init()

    def model_init(self):
        # Initialize parameters of the model
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x, edge_index):
        x = self.relu(self.gconv[0](x, edge_index))
        for i in range(1, self.lg):
            x = F.dropout(x, self.dropout, self.training)
            x = self.relu(self.gconv[i](x, edge_index))
        logit = self.cls(x)

        return logit
    
class ChebGCNv2(torch.nn.Module):
    def __init__(self, input_dim, num_classes, dropout, hgc, lg, K=3):
        super(ChebGCNv2, self).__init__()
        hidden = [hgc for i in range(lg)]  # Define hidden layer sizes
        self.dropout = dropout
        bias = False
        self.relu = torch.nn.ReLU(inplace=True)
        self.lg = lg
        self.gconv = nn.ModuleList()

        # Define Chebyshev convolution layers
        for i in range(lg):
            in_channels = input_dim if i == 0 else hidden[i-1]
            self.gconv.append(tg.nn.ChebConv(in_channels, hidden[i], K=K, normalization='sym', bias=bias))

        # Classification layers
        self.cls = nn.Sequential(
            torch.nn.Linear(hidden[lg-1], 256),
            torch.nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            torch.nn.Linear(256, num_classes)
        )

        self.model_init()

    def model_init(self):
        # Initialize parameters of the model
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x, edge_index, edge_weight=None):
        # Apply the first Chebyshev convolution layer
        x = self.relu(self.gconv[0](x, edge_index, edge_weight))
        
        # Apply the subsequent Chebyshev convolution layers
        for i in range(1, self.lg):
            x = F.dropout(x, self.dropout, self.training)
            x = self.relu(self.gconv[i](x, edge_index, edge_weight))
        logit = self.cls(x)

        return logit

# ------ Base module definition ------ #
class BaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, time_series: torch.Tensor, node_feature: torch.Tensor) -> torch.Tensor:
        pass

# ------ Interpretable Transformer Encoder ------ #
# Extract attention weights generated by the model when processing data
# This is useful for understanding and interpreting the decision-making process
class InterpretableTransformerEncoder(TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        """
        d_model: The input and output dimension of the model
        nhead: Number of attention heads
        """
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first, norm_first, device, dtype)
        self.attention_weights: Optional[Tensor] = None

    # Override the self-attention function from the parent TransformerEncoderLayer
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x, weights = self.self_attn(x, x, x,
                                    attn_mask=attn_mask,
                                    key_padding_mask=key_padding_mask,
                                    need_weights=True)
        # Store the attention weights, which indicate the influence between each element in the sequence
        self.attention_weights = weights
        return self.dropout1(x)

    # Function to retrieve the attention weights
    def get_attention_weights(self) -> Optional[Tensor]:
        return self.attention_weights
    


class graph_Transformer(BaseModel):

    def __init__(self, args):
        super().__init__()
        ext_featuresize = args.ext_featuresize
        input_feature_size = args.node_sz  # Number of nodes, e.g., 200
        self.transformer1 = InterpretableTransformerEncoder(
            d_model=input_feature_size, nhead=4, dim_feedforward=1024, batch_first=True
        )
        self.transformer2 = InterpretableTransformerEncoder(
            d_model=input_feature_size, nhead=4, dim_feedforward=1024, batch_first=True
        )
        # Define an optional pooling layer if needed
        # self.SAGPool = SAGPool(input_feature_size, ratio=0.5)
        self.activations = None
        self.gradients = None

        # Dimension reduction layers
        self.reduce_dim1 = nn.Linear(input_feature_size, 100)  # (batch_size, 200, 100)
        self.reduce_dim2 = nn.Linear(100, 50)  # (batch_size, 200, 50)
        self.reduce_dim3 = nn.Linear(50, 20)  # (batch_size, 200, 20)
        self.leaky_relu = nn.LeakyReLU()

        # Fully connected layers for classification
        self.fc1 = nn.Linear(input_feature_size * 20, 256)  # Reduce to 256
        self.fc2 = nn.Linear(256, 64)  # Reduce to 64
        self.fc3 = nn.Linear(64, ext_featuresize)
        self.fc4 = nn.Linear(ext_featuresize, 2)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(ext_featuresize)
        self.dropout = nn.Dropout(0.3)

    def forward(self, node_feature: torch.Tensor):
        """
        Forward pass for the graph transformer model.

        Args:
            node_feature (torch.Tensor): Node features, input size [batch_size, 200, 200]

        Returns:
            output (torch.Tensor): Final classification output
            extract_feature (torch.Tensor): Extracted feature representations
        """
        bz, _, _ = node_feature.shape  # Get batch size
        assignments = []  # Store pooling results (if applicable)

        # Apply the first transformer encoder
        node_feature = self.transformer1(node_feature)  # node_feature: [batch_size, 200, 200]
        
        # Uncomment the following line if a second transformer encoder is required
        # node_feature = self.transformer2(node_feature)

        # Dimension reduction through linear layers
        node_feature = self.reduce_dim1(node_feature)
        node_feature = self.leaky_relu(node_feature)
        node_feature = self.reduce_dim2(node_feature)
        node_feature = self.leaky_relu(node_feature)
        node_feature = self.reduce_dim3(node_feature)
        node_feature = self.leaky_relu(node_feature)

        # Reshape node features for fully connected layers
        node_feature = node_feature.view(bz, -1)

        # Fully connected layers with batch normalization and activation
        node_feature = self.fc1(node_feature)  # Reduce to 256 dimensions
        node_feature = self.batch_norm1(node_feature)
        node_feature = self.leaky_relu(node_feature)
        node_feature = self.dropout(node_feature)
        node_feature = self.fc2(node_feature)  # Reduce to 64 dimensions
        node_feature = self.batch_norm2(node_feature)
        node_feature = self.leaky_relu(node_feature)

        # Additional fully connected layers
        node_feature = self.fc3(node_feature)
        node_feature = self.batch_norm3(node_feature)
        node_feature = self.leaky_relu(node_feature)

        # Extracted features for further analysis
        extract_feature = node_feature

        # Final output layer
        output = self.fc4(node_feature)

        return output, extract_feature  # Return final output and extracted features

    def loss(self, assignments):
        pass



