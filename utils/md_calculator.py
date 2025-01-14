# utils/md_calculator.py

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes


class NeuralNetworkPotential(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, model, device, max_atoms, chemical_symbols, **kwargs):
        """
        Initializes the NeuralNetworkPotential calculator.

        Args:
            model (nn.Module): Trained energy predictor model.
            device (torch.device): Device to perform computations on.
            max_atoms (int): Maximum number of atoms expected in the molecules.
            chemical_symbols (list): List of chemical symbols present in the dataset.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.model = model
        self.device = device
        self.max_atoms = max_atoms
        self.chemical_symbols = chemical_symbols
        self.model.eval()  # Set model to evaluation mode

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        if atoms is None:
            atoms = self.atoms

        atomic_numbers = atoms.get_atomic_numbers()
        positions = atoms.get_positions()
        num_atoms = len(atomic_numbers)

        # Padding
        Z_padded = np.pad(atomic_numbers, (0, self.max_atoms - num_atoms), 'constant')
        pos_padded = np.pad(positions, ((0, self.max_atoms - num_atoms), (0, 0)), 'constant')
        pos_flat = pos_padded.flatten().astype(np.float32)

        # Concatenate Z and positions
        inputs = np.concatenate([Z_padded, pos_flat]).astype(np.float32)
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device=self.device).unsqueeze(0)
        inputs_tensor.requires_grad = True

        with torch.no_grad():
            pred_energy = self.model(inputs_tensor).item()

        # Compute forces
        pred_energy_tensor = self.model(inputs_tensor)
        forces = -torch.autograd.grad(
            outputs=pred_energy_tensor,
            inputs=inputs_tensor,
            grad_outputs=torch.ones_like(pred_energy_tensor),
            retain_graph=False,
            create_graph=False
        )[0].cpu().numpy().reshape(-1, 3)[:num_atoms]

        self.results['energy'] = pred_energy
        self.results['forces'] = forces
