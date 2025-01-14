# model/model.py

import torch
import torch.nn as nn


def weights_init_xavier_uniform(m):
    """
    Apply Xavier (Glorot) Uniform initialization to linear layers.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class EnergyPredictor(nn.Module):
    def __init__(self, z_size, distance_size, forces_size, hidden_size, mode='positions'):
        """
        Parameters:
        ----------
        z_size : int
            Number of atomic numbers (max_atoms).
        distance_size : int
            Size of the flattened distance matrix input.
        forces_size : int
            Size of the flattened forces input (max_atoms * 3).
        hidden_size : int
            Size of the hidden layers.
        mode : str
            Operational mode: 'positions' or 'forces'.
        """
        super(EnergyPredictor, self).__init__()
        self.mode = mode.lower()
        if self.mode not in ['positions', 'forces']:
            raise ValueError("Invalid mode. Choose 'positions' or 'forces'.")

        # Z Encoder (Atomic Numbers)
        self.z_encoder = nn.Sequential(
            nn.Linear(z_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Input Encoder based on mode
        if self.mode == 'positions':
            # Distance Matrix Encoder
            self.input_encoder = nn.Sequential(
                nn.Linear(distance_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            )
        elif self.mode == 'forces':
            # Forces Encoder
            self.input_encoder = nn.Sequential(
                nn.Linear(forces_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            )

        # Combiner for Energy Prediction
        self.combiner_energy = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Output: Energy
        )

        # Initialize weights
        self.apply(weights_init_xavier_uniform)

    def forward(self, Z, input_data, mask):
        """
        Forward pass for energy prediction.

        Parameters:
        ----------
        Z : torch.Tensor
            Tensor of atomic numbers, shape (batch_size, z_size).
        input_data : torch.Tensor
            Tensor of distance matrices or flattened forces.
            - For 'positions' mode: shape (batch_size, distance_size).
            - For 'forces' mode: shape (batch_size, forces_size).
        mask : torch.Tensor
            Tensor to mask padded inputs, same shape as input_data.

        Returns:
        -------
        torch.Tensor
            Predicted energies, shape (batch_size, 1).
        """
        # Encode atomic numbers
        z_emb = self.z_encoder(Z)  # (batch_size, hidden_size)

        # Encode input data based on mode
        input_emb = self.input_encoder(input_data * mask)  # (batch_size, hidden_size)

        # Combine embeddings
        combined = torch.cat([z_emb, input_emb], dim=1)  # (batch_size, hidden_size * 2)

        # Predict energy
        energy = self.combiner_energy(combined)  # (batch_size, 1)
        return energy
