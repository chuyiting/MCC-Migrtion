'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file MCNetworkUtils.py

    \brief Helper functions to build neural networks.

    \copyright Copyright (c) 2018 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import math

############################################################################# Network Utils
class MLP2Hidden(nn.Module):
    def __init__(self, num_input_features, hidden1_units, hidden2_units, num_out_features, 
                 use_dropout=False, use_init_bn=True, keep_prob=0.8, bn_momentum=0.01, eps=0.001):
        super().__init__()
        
        self.use_init_bn = use_init_bn
        self.use_dropout = use_dropout

        # Initialize layers
        # self.ln_init = nn.LayerNorm(num_input_features)
        self.bn_init = nn.BatchNorm1d(num_input_features, momentum=bn_momentum, eps=eps) if use_init_bn else None
        self.fc1 = nn.Linear(num_input_features, hidden1_units)
        self.ln1 = nn.LayerNorm(hidden1_units)
        self.bn1 = nn.BatchNorm1d(hidden1_units, momentum=bn_momentum, eps=eps)
        self.fc2 = nn.Linear(hidden1_units, hidden2_units)
        self.ln2 = nn.LayerNorm(hidden2_units)
        self.bn2 = nn.BatchNorm1d(hidden2_units, momentum=bn_momentum, eps=eps)
        self.fc3 = nn.Linear(hidden2_units, num_out_features)

        # Dropout layer
        if use_dropout:
            self.dropout = nn.Dropout(p=1 - keep_prob) 
            self.dropout2 = nn.Dropout(p=1 - keep_prob) 

    def forward(self, features):
        if self.use_init_bn:
            features = self.bn_init(features)

        # Hidden layer 1
        x = self.fc1(features)
        x = self.bn1(x)
        x = F.relu(x)

        # Hidden layer 2
        if self.use_dropout:
            x = self.dropout(x) 
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Output layer
        if self.use_dropout:
            x = self.dropout2(x) 
        x = self.fc3(x)

        return x

class MLP1Hidden(nn.Module):
    def __init__(self, num_input_features, hidden_units, num_out_features, use_dropout=False, keep_prob=1.0):
        """
        Initialize the MLP with one hidden layer.

        Args:
            num_input_features (int): Number of input features.
            hidden_units (int): Number of units in the hidden layer.
            num_out_features (int): Number of output features.
            use_dropout (bool): Boolean that indicates if dropout should be used in the MLP.
        """
        super().__init__()
        self.use_dropout = use_dropout

        # Define layers
        self.hidden_layer = nn.Linear(num_input_features, hidden_units)
        self.batch_norm_hidden = nn.BatchNorm1d(hidden_units, momentum=0.01)
        self.output_layer = nn.Linear(hidden_units, num_out_features)

        # Dropout layer
        if use_dropout:
            self.dropout = nn.Dropout(p=(1.0 - keep_prob))  # keep_prob is defined during the forward pass

    def forward(self, features):
        """
        Forward pass through the MLP.

        Args:
            features (tensor): Input features (nxm tensor).
            keep_prob (float): Probability to keep an input in the MLP (for dropout).
            is_training (bool): Indicates if the MLP is executed in a training mode or not.

        Returns:
            tensor: Output features from the MLP.
        """
        # Hidden layer with batch normalization and ReLU activation
        hidden = self.hidden_layer(features)
        hidden = self.batch_norm_hidden(hidden)
        hidden = torch.relu(hidden)

        # Apply dropout if in training mode
        if self.use_dropout:
            hidden = self.dropout(hidden)

        # Output layer
        output = self.output_layer(hidden)
        return output


class Conv1x1(nn.Module):
    def __init__(self, num_inputs, num_out_features):
        """
        Initialize the Conv1x1 layer.

        Args:
            num_inputs (int): Number of input features.
            num_out_features (int): Number of output features.
        """
        super().__init__()
        
        # Define the linear layer
        self.linear_layer = nn.Linear(num_inputs, num_out_features)
        
        # Initialize weights using Xavier (Glorot) initialization
        nn.init.xavier_uniform_(self.linear_layer.weight)
        nn.init.zeros_(self.linear_layer.bias)

    def forward(self, inputs):
        """
        Forward pass through the Conv1x1 layer.

        Args:
            inputs (tensor): Input features (nxm tensor).

        Returns:
            tensor: Transformed output features.
        """
        # Perform the linear transformation
        reduced_output = self.linear_layer(inputs)
        return reduced_output


class BatchNormReLUDropout(nn.Module):
    def __init__(self, in_features, use_dropout=False, keep_prob=0.8, bn_momentum = 0.01, eps=0.001):
        """
        Initialize the BatchNormReLUDropout layer.

        Args:
            in_features (int): Number of input features.
            use_dropout (bool): Boolean that indicates if dropout should be used in the layer.
            keep_prob (float): Probability of keeping a unit during dropout.
        """
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(in_features, momentum=bn_momentum, eps=eps)
        # self.layer_norm = nn.LayerNorm(in_features)
        self.relu = nn.ReLU()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=1 - keep_prob) if use_dropout else None

    def forward(self, x):
        """
        Forward pass for the BatchNormReLUDropout layer.

        Args:
            x (torch.Tensor): Input features.
            is_training (bool): Indicates if the model is in training mode.

        Returns:
            torch.Tensor: Output features after BatchNorm, ReLU, and optional Dropout.
        """
        x = self.batch_norm(x)
        # x = self.layer_norm(x)
        x = self.relu(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x

