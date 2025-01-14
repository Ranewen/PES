# main.py

import os
import shlex
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import ase.data
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt

# Optional: Uncomment if using WandB
# import wandb

# ------------------------------------------------
# Import from your local modules (adjust paths)
# ------------------------------------------------
# Ensure that 'model.model' and 'utils.metrics' are accessible.
# You might need to adjust the Python path or project structure accordingly.

from model.model import EnergyPredictor  # Ensure model/model.py exists with EnergyPredictor
from utils.metrics import calculate_metrics  # Ensure utils/metrics.py exists with calculate_metrics


# ------------------------------------------------
# Data Parsing and Preprocessing Functions
# ------------------------------------------------

def parse_extxyz_with_forces(file_path):
    """
    Parses an .extxyz file and extracts atomic_numbers, positions, chemical_symbols, energies, and forces.
    Also sorts atoms by atomic number and position to ensure consistent ordering.

    Parameters:
    ----------
    file_path : str
        Path to the .extxyz file.

    Returns:
    -------
    data : list of dict
        Each dict contains:
            - 'atomic_numbers': list of atomic numbers
            - 'chemical_symbols': list of chemical symbols
            - 'positions': (N, 3) ndarray of atomic positions
            - 'forces': (N, 3) ndarray of atomic forces
            - 'energy': float, energy of the structure
    """
    data = []
    with open(file_path, "r") as f:
        lines = f.readlines()

    num_lines = len(lines)
    i = 0
    structure_idx = 0

    print(f"DEBUG: Total line count: {num_lines}")

    while i < num_lines:
        # 1) Read the number of atoms line
        num_atoms_line = lines[i].strip()
        if not num_atoms_line.isdigit():
            i += 1
            continue

        n_atoms = int(num_atoms_line)
        i += 1

        if i + 1 + n_atoms > num_lines:
            print(
                f"DEBUG: Structure {structure_idx}: not enough lines to parse "
                f"({1 + n_atoms} needed, but only {num_lines - i} remain). Skipping the rest."
            )
            break

        # 2) Comment line
        comment_line = lines[i].strip()
        i += 1

        properties = {}
        try:
            for token in shlex.split(comment_line):
                if '=' in token:
                    k, v = token.split('=', 1)
                    properties[k] = v
        except Exception as e:
            print(f"DEBUG: Failed to parse comment line at structure {structure_idx}: {e}")
            continue

        # Extract energy if present
        energy_str = properties.get('energy', '0.0')
        try:
            energy = float(energy_str)
        except ValueError:
            print(f"DEBUG: Could not parse energy '{energy_str}' at structure {structure_idx}. Setting energy=0.0")
            energy = 0.0

        # 3) Atom lines
        atom_lines = lines[i:i + n_atoms]
        i += n_atoms

        species = []
        positions = []
        forces = []

        for idx, line in enumerate(atom_lines):
            tokens = line.split()
            if len(tokens) < 7:
                print(f"DEBUG: Skipping malformed atomic line: {line.strip()}")
                continue

            try:
                symbol = tokens[0]
                px, py, pz = float(tokens[1]), float(tokens[2]), float(tokens[3])
                fx, fy, fz = float(tokens[-3]), float(tokens[-2]), float(tokens[-1])
            except (ValueError, IndexError) as e:
                print(f"DEBUG: Error parsing atom line: {line.strip()}, error={e}")
                continue

            species.append(symbol)
            positions.append([px, py, pz])
            forces.append([fx, fy, fz])

        if not species:
            print(f"DEBUG: Structure {structure_idx} has no valid atoms. Skipping.")
            structure_idx += 1
            continue

        positions = np.array(positions, dtype=np.float32)
        forces = np.array(forces, dtype=np.float32)
        atomic_numbers = [ase.data.atomic_numbers[sym] for sym in species]

        # Sort atoms by atomic number and then by position to ensure consistent ordering
        sorted_indices = np.lexsort((positions[:, 2], positions[:, 1], positions[:, 0], atomic_numbers))
        atomic_numbers_sorted = np.array(atomic_numbers)[sorted_indices]
        species_sorted = np.array(species)[sorted_indices]
        positions_sorted = positions[sorted_indices]
        forces_sorted = forces[sorted_indices]

        data.append({
            'atomic_numbers': atomic_numbers_sorted.tolist(),
            'chemical_symbols': species_sorted.tolist(),
            'positions': positions_sorted,
            'forces': forces_sorted,
            'energy': energy
        })

        structure_idx += 1

    if not data:
        raise ValueError("No valid structures found in the .extxyz file.")

    print(f"DEBUG: Total processed structures: {len(data)}")
    return data


def integrate_forces_for_energy(data, reference_energy=0.0):
    """
    Numerically integrates forces from multiple reference structures to approximate
    the energy of each structure in 'data' via a single-step trapezoid rule.

    Handles structures with varying numbers of atoms by using a separate reference structure for each unique atom count.

    Parameters:
    ----------
    data : list of dict
        Each dict should have:
            - 'positions': (N, 3) ndarray of atomic positions
            - 'forces': (N, 3) ndarray of atomic forces
            - 'energy': float (original or dummy)
            - 'atomic_numbers': list of atomic numbers
            - 'chemical_symbols': list of chemical symbols
    reference_energy : float
        The absolute energy to assign to each reference structure.

    Returns:
    -------
    data : list of dict
        The same list, but each structure will have a new key 'predicted_energy'
        based on the force integration.
    """
    from collections import defaultdict

    # Group structures by their atom count
    atom_count_to_indices = defaultdict(list)
    for idx, sample in enumerate(data):
        atom_count = len(sample['atomic_numbers'])
        atom_count_to_indices[atom_count].append(idx)

    # Iterate over each group and perform integration
    for atom_count, indices in atom_count_to_indices.items():
        if not indices:
            continue

        # Select the first structure in the group as the reference
        ref_idx = indices[0]
        ref = data[ref_idx]
        R0 = ref['positions']
        F0 = ref['forces']
        ref['predicted_energy'] = reference_energy

        print(f"\nIntegrating energies for atom count: {atom_count}")
        print(f"Reference Structure Index: {ref_idx}, Assigned Energy: {reference_energy}")

        # Integrate energy for each structure in the group (excluding the reference)
        for target_idx in indices[1:]:
            target = data[target_idx]
            R1 = target['positions']
            F1 = target['forces']

            # Compute average forces
            F_avg = 0.5 * (F0 + F1)

            # Compute displacement
            dR = R1 - R0

            # Compute delta energy using trapezoid rule
            delta_E = -np.sum(F_avg * dR)

            # Assign predicted energy
            E1 = reference_energy + delta_E
            target['predicted_energy'] = E1

            print(f"Structure {target_idx}: Integrated Energy = {E1:.6f} eV")

    return data


def compute_distance_matrix(positions, debug=False):
    """
    Compute NxN pairwise distance matrix from Nx3 positions.

    Parameters:
    ----------
    positions : ndarray
        Array of shape (N, 3) containing atomic positions.
    debug : bool
        If True, print debug information.

    Returns:
    -------
    distance_matrix : ndarray
        Array of shape (N, N) containing pairwise distances.
    """
    if debug:
        print("DEBUG: Compute Distance Matrix Started")
        print(f"DEBUG: Atom Positions Shape: {positions.shape}")
        print(f"DEBUG: Atom Positions:\n{positions}")

    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distance_matrix = np.linalg.norm(diff, axis=-1)

    if debug:
        symmetric = np.allclose(distance_matrix, distance_matrix.T, atol=1e-6)
        if not symmetric:
            print("DEBUG: Distance Matrix is not symmetric!")
    return distance_matrix


def preprocess_data(data, chemical_symbols, max_atoms=44, mode='positions'):
    """
    Preprocess data to compute/pad distance matrices or prepare forces, energies, etc.

    Parameters:
    ----------
    data : list of dicts
        Parsed data from .extxyz files.
    chemical_symbols : list of str
        Allowed chemical symbols (e.g., ['H', 'C', 'O', 'N']).
    max_atoms : int
        Maximum number of atoms to pad/truncate to (default: 44).
    mode : str
        'positions' or 'forces' to determine preprocessing steps.

    Returns:
    -------
    processed_data : list of dicts with tensors
    """
    import torch

    print(f"DEBUG: Preprocessing data for mode: {mode}")

    processed_data = []

    for sample_idx, sample in enumerate(data):
        try:
            atomic_numbers = sample['atomic_numbers']
            chemical_symbols_in_struct = sample['chemical_symbols']
            positions = sample['positions']
            energy = sample['energy']
            forces = sample['forces']

            # Filter by allowed chemical symbols
            filtered_indices = [
                idx for idx, symbol in enumerate(chemical_symbols_in_struct)
                if symbol in chemical_symbols
            ]
            if not filtered_indices:
                continue

            atomic_numbers = np.array(atomic_numbers, dtype=np.float32)[filtered_indices]
            positions = positions[filtered_indices]
            forces = forces[filtered_indices]

            num_atoms = len(atomic_numbers)

            if mode == 'positions':
                distance_matrix = compute_distance_matrix(positions, debug=False)
                # Pad distance matrix to (max_atoms, max_atoms)
                distance_padded = np.pad(
                    distance_matrix,
                    ((0, max_atoms - num_atoms), (0, max_atoms - num_atoms)),
                    'constant', constant_values=0
                )

                # Pad atomic numbers to (max_atoms,)
                Z_padded = np.pad(atomic_numbers, (0, max_atoms - num_atoms), 'constant', constant_values=0)

                Z_tensor = torch.tensor(Z_padded, dtype=torch.float32)
                distance_tensor = torch.tensor(distance_padded, dtype=torch.float32)
                energy_tensor = torch.tensor(energy, dtype=torch.float32)

                processed_data.append({
                    'Z': Z_tensor,
                    'distance_matrix': distance_tensor,
                    'energy': energy_tensor,
                    'num_atoms': num_atoms
                })

            elif mode == 'forces':
                # Flatten forces and pad to (max_atoms * 3,)
                forces_flat = forces.flatten()
                padding_length = (max_atoms - num_atoms) * 3
                forces_padded = np.pad(forces_flat, (0, padding_length), 'constant', constant_values=0)

                # Pad atomic numbers to (max_atoms,)
                Z_padded = np.pad(atomic_numbers, (0, max_atoms - num_atoms), 'constant', constant_values=0)

                Z_tensor = torch.tensor(Z_padded, dtype=torch.float32)
                forces_tensor = torch.tensor(forces_padded, dtype=torch.float32)
                energy_tensor = torch.tensor(energy, dtype=torch.float32)

                processed_data.append({
                    'Z': Z_tensor,
                    'forces': forces_tensor,
                    'energy': energy_tensor,
                    'num_atoms': num_atoms
                })

            else:
                raise ValueError("Invalid mode. Choose 'positions' or 'forces'.")

        except Exception as e:
            print(f"DEBUG: Error processing sample {sample_idx}: {e}")
            continue

    print(f"DEBUG: Total processed samples after preprocessing: {len(processed_data)}")
    return processed_data


# ------------------------------------------------
# Dataset Class
# ------------------------------------------------

class MoleculeDataset(Dataset):
    def __init__(self, data, chemical_symbols, mode='positions'):
        """
        Initialize with raw data + preprocess.

        Parameters:
        ----------
        data : list of dicts
            Parsed data from .extxyz files.
        chemical_symbols : list of str
            Allowed chemical symbols (e.g., ['H', 'C', 'O', 'N']).
        mode : str
            'positions' or 'forces' to determine preprocessing steps.
        """
        self.mode = mode
        self.processed_data = preprocess_data(data, chemical_symbols, mode=mode)

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]


# ------------------------------------------------
# Collate Functions
# ------------------------------------------------

def collate_fn_positions(batch):
    """
    Custom collate function for 'positions' mode.

    Parameters:
    ----------
    batch : list of dicts
        Each dict contains 'Z', 'distance_matrix', 'energy', 'num_atoms'.

    Returns:
    -------
    tuple:
        - batched_Z: Tensor (batch_size, max_atoms)
        - batched_distance: Tensor (batch_size, max_atoms * max_atoms)
        - batched_mask: Tensor (batch_size, max_atoms * max_atoms)
        - batched_energy: Tensor (batch_size, 1)
    """
    batched_Z = []
    batched_distance = []
    batched_mask = []
    batched_energy = []

    for sample in batch:
        Z = sample['Z']
        distance = sample['distance_matrix']
        num_atoms = sample['num_atoms']
        energy = sample['energy']

        batched_Z.append(Z)
        distance_flat = distance.view(-1)  # Flatten to (max_atoms * max_atoms,)
        batched_distance.append(distance_flat)

        # Create mask based on num_atoms
        mask = torch.zeros((distance.size(0), distance.size(1)), dtype=torch.float32)
        mask[:num_atoms, :num_atoms] = 1.0
        mask_flat = mask.view(-1)  # Flatten to (max_atoms * max_atoms,)
        batched_mask.append(mask_flat)

        batched_energy.append(energy)

    batched_Z = torch.stack(batched_Z)  # (batch_size, max_atoms)
    batched_distance = torch.stack(batched_distance)  # (batch_size, max_atoms * max_atoms)
    batched_mask = torch.stack(batched_mask)  # (batch_size, max_atoms * max_atoms)
    batched_energy = torch.stack(batched_energy).unsqueeze(1)  # (batch_size, 1)

    return batched_Z, batched_distance, batched_mask, batched_energy



def collate_fn_forces(batch):
    """
    Custom collate function for 'forces' mode.

    Parameters:
    ----------
    batch : list of dicts
        Each dict contains 'Z', 'forces', 'energy', 'num_atoms'.

    Returns:
    -------
    tuple:
        - batched_Z: Tensor (batch_size, max_atoms)
        - batched_forces: Tensor (batch_size, max_atoms * 3)
        - batched_mask: Tensor (batch_size, max_atoms * 3)
        - batched_energy: Tensor (batch_size, 1)
    """
    batched_Z = []
    batched_forces = []
    batched_mask = []
    batched_energy = []

    for sample in batch:
        Z = sample['Z']
        forces = sample['forces']
        num_atoms = sample['num_atoms']
        energy = sample['energy']

        batched_Z.append(Z)
        batched_forces.append(forces)

        # Create mask based on num_atoms
        mask = torch.zeros((forces.size(0),), dtype=torch.float32)
        mask[:num_atoms * 3] = 1.0  # Each atom has 3 force components
        batched_mask.append(mask)

        batched_energy.append(energy)

    batched_Z = torch.stack(batched_Z)  # (batch_size, max_atoms)
    batched_forces = torch.stack(batched_forces)  # (batch_size, max_atoms * 3)
    batched_mask = torch.stack(batched_mask)  # (batch_size, max_atoms * 3)
    batched_energy = torch.stack(batched_energy).unsqueeze(1)  # (batch_size, 1)

    return batched_Z, batched_forces, batched_mask, batched_energy


# ------------------------------------------------
# Dataset Splitting Function
# ------------------------------------------------

def split_dataset(dataset, n_train, n_val, n_test, seed=42):
    """
    Split a dataset into three subsets (train, val, test) of sizes n_train, n_val, n_test.

    Parameters:
    ----------
    dataset : Dataset
        The full dataset.
    n_train : int
        Number of training samples.
    n_val : int
        Number of validation samples.
    n_test : int
        Number of test samples.
    seed : int
        Random seed for reproducibility.

    Returns:
    -------
    tuple:
        - train_dataset: Subset
        - val_dataset: Subset
        - test_dataset: Subset
    """
    total_samples = len(dataset)
    assert n_train + n_val + n_test <= total_samples, \
        f"Sum of splits exceeds dataset size ({total_samples})."

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total_samples, generator=generator)

    train_indices = perm[:n_train]
    val_indices = perm[n_train:n_train + n_val]
    test_indices = perm[n_train + n_val:n_train + n_val + n_test]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    return train_dataset, val_dataset, test_dataset


# ------------------------------------------------
# Evaluation Function
# ------------------------------------------------

def evaluate_model(model, test_loader, device, criterion, mode='positions'):
    """
    Evaluate model on test_loader and return test loss + MSE/MAE/R^2 + energy metrics.

    Parameters:
    ----------
    model : torch.nn.Module
        Trained model.
    test_loader : DataLoader
        DataLoader for the test set.
    device : torch.device
        Device to run evaluation on.
    criterion : torch.nn.Module
        Loss function.
    mode : str
        'positions' or 'forces' to determine evaluation steps.

    Returns:
    -------
    test_loss : float
        Average test loss.
    mse : float
        Mean Squared Error.
    mae : float
        Mean Absolute Error.
    r2 : float
        R-squared metric.
    metrics : dict
        Additional metrics from `calculate_metrics`.
    """
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            if mode == 'positions':
                Z, distance_matrix, mask, energies = batch
                Z = Z.to(device)
                distance_matrix = distance_matrix.to(device)
                mask = mask.to(device)
                energies = energies.to(device)

                pred_energy = model(Z, distance_matrix, mask)
            elif mode == 'forces':
                Z, forces, mask, energies = batch
                Z = Z.to(device)
                forces = forces.to(device)
                mask = mask.to(device)
                energies = energies.to(device)

                pred_energy = model(Z, forces, mask)
            else:
                raise ValueError("Invalid mode. Choose 'positions' or 'forces'.")

            loss = criterion(pred_energy, energies)
            test_loss += loss.item() * Z.size(0)

            all_preds.append(pred_energy.cpu())
            all_targets.append(energies.cpu())

    test_loss /= len(test_loader.dataset)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = calculate_metrics(all_preds, all_targets)
    # Standard metrics
    mse = nn.MSELoss()(all_preds, all_targets).item()
    mae = nn.L1Loss()(all_preds, all_targets).item()
    var_targets = torch.var(all_targets, unbiased=False).item()
    r2 = 1 - mse / var_targets if var_targets != 0 else float('nan')

    return test_loss, mse, mae, r2, metrics


# ------------------------------------------------
# Configuration Loading Function
# ------------------------------------------------

def load_config(config_path):
    """
    Load YAML configuration file.

    Parameters:
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns:
    -------
    config : dict
        Configuration parameters.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


# ------------------------------------------------
# Training Function
# ------------------------------------------------

def train_model(config, mode='positions'):
    """
    Main training function that trains the EnergyPredictor on energies.

    Parameters:
    ----------
    config : dict
        Configuration parameters loaded from YAML.
    mode : str
        'positions' or 'forces' to determine training mode.
    """
    # 1) Reproducibility
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    np.random.seed(config['dataset_seed'])

    # 2) Parse dataset
    data = parse_extxyz_with_forces(config['dataset_file_name'])

    # 3) Integrate forces to predict energies if in 'forces' mode
    if mode == 'forces':
        data = integrate_forces_for_energy(
            data,
            reference_energy=config.get('reference_energy', 0.0)
        )

        # Optionally overwrite 'energy' with 'predicted_energy'
        overwrite_energy = config.get('overwrite_energy', False)
        if overwrite_energy:
            for d in data:
                if 'predicted_energy' in d:
                    d['energy'] = d['predicted_energy']

    # 4) Create Dataset
    dataset = MoleculeDataset(data, config['chemical_symbols'], mode=mode)
    total_samples = len(dataset)
    print(f"DEBUG: Total dataset size: {total_samples}")

    # 5) Determine split sizes
    n_train = min(config['n_train'], total_samples)
    n_val = min(config['n_val'], total_samples - n_train)
    n_test = total_samples - n_train - n_val
    print(f"DEBUG: Train size: {n_train}, Val size: {n_val}, Test size: {n_test}")

    # 6) Perform splits
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset, n_train, n_val, n_test, seed=config['dataset_seed']
    )

    # 7) DataLoaders
    if mode == 'positions':
        collate_function = collate_fn_positions
    elif mode == 'forces':
        collate_function = collate_fn_forces
    else:
        raise ValueError("Invalid mode. Choose 'positions' or 'forces'.")

    train_loader = DataLoader(train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=config['shuffle'],
                              drop_last=False,
                              collate_fn=collate_function)

    val_loader = DataLoader(val_dataset,
                            batch_size=config['validation_batch_size'],
                            shuffle=False,
                            drop_last=False,
                            collate_fn=collate_function)

    test_loader = DataLoader(test_dataset,
                             batch_size=config['validation_batch_size'],
                             shuffle=False,
                             drop_last=False,
                             collate_fn=collate_function)

    # 8) Instantiate model
    sample = dataset[0]
    if mode == 'positions':
        distance_size = sample['distance_matrix'].numel()
        forces_size = 0  # Not used in positions mode
    elif mode == 'forces':
        distance_size = 0  # Not used in forces mode
        forces_size = sample['forces'].numel()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = EnergyPredictor(
        z_size=sample['Z'].shape[0],  # (max_atoms,)
        distance_size=distance_size,
        forces_size=forces_size,
        hidden_size=config['num_features'],
        mode=mode
    ).to(device)

    # 9) Define loss & optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config['learning_rate']),
        amsgrad=config['optimizer_amsgrad'],
        eps=float(config['optimizer_eps']),
        weight_decay=float(config['optimizer_weight_decay'])
    )
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5,
        threshold=1e-4, min_lr=1e-6
    )

    # (Optional) EMA
    if config.get('use_ema', False):
        from torch.optim.swa_utils import AveragedModel
        ema_model = AveragedModel(model)
        ema_decay = config.get('ema_decay', 0.99)

    # (Optional) wandb
    if config.get('wandb', False):
        wandb.init(project=config['wandb_project'], name=config['run_name'])
        wandb.config.update(config)

    # Create output directory
    os.makedirs(config.get('root', 'results/molecule_distance_calc'), exist_ok=True)

    # -----------------------
    # Loss Tracking
    # -----------------------
    train_losses = []
    val_losses = []

    # 10) Training Loop
    for epoch in range(1, config['max_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['max_epochs']}")
        model.train()
        running_loss = 0.0

        # --- TRAIN ---
        for batch_idx, batch in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch}/{config['max_epochs']}")
        ):
            if mode == 'positions':
                Z, distance_matrix, mask, energies = batch
                Z = Z.to(device)
                distance_matrix = distance_matrix.to(device)
                mask = mask.to(device)
                energies = energies.to(device)

                optimizer.zero_grad()
                pred_energy = model(Z, distance_matrix, mask)
                loss = criterion(pred_energy, energies)

            elif mode == 'forces':
                Z, forces, mask, energies = batch
                Z = Z.to(device)
                forces = forces.to(device)
                mask = mask.to(device)
                energies = energies.to(device)

                optimizer.zero_grad()
                pred_energy = model(Z, forces, mask)
                loss = criterion(pred_energy, energies)

            else:
                raise ValueError("Invalid mode. Choose 'positions' or 'forces'.")

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf loss encountered at epoch {epoch}, batch {batch_idx}. Skipping...")
                continue

            loss.backward()
            optimizer.step()

            # (Optional) update EMA
            if config.get('use_ema', False):
                ema_model.update_parameters(model)

            running_loss += loss.item() * Z.size(0)

        epoch_train_loss = running_loss / n_train
        train_losses.append(epoch_train_loss)

        # --- VALIDATION ---
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if mode == 'positions':
                    Z, distance_matrix, mask, energies = batch
                    Z = Z.to(device)
                    distance_matrix = distance_matrix.to(device)
                    mask = mask.to(device)
                    energies = energies.to(device)

                    pred_energy = model(Z, distance_matrix, mask)
                elif mode == 'forces':
                    Z, forces, mask, energies = batch
                    Z = Z.to(device)
                    forces = forces.to(device)
                    mask = mask.to(device)
                    energies = energies.to(device)

                    pred_energy = model(Z, forces, mask)
                else:
                    raise ValueError("Invalid mode. Choose 'positions' or 'forces'.")

                loss = criterion(pred_energy, energies)
                val_running_loss += loss.item() * Z.size(0)

        epoch_val_loss = val_running_loss / n_val
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch}, Training Loss: {epoch_train_loss:.6f}, Validation Loss: {epoch_val_loss:.6f}")
        scheduler.step(epoch_val_loss)

        # (Optional) wandb logging
        if config.get('wandb', False):
            wandb.log({
                'epoch': epoch,
                'training_loss': epoch_train_loss,
                'validation_loss': epoch_val_loss
            })

        # 11) Save checkpoint
        if epoch % config['save_checkpoint_freq'] == 0:
            checkpoint_path = os.path.join(
                config['root'], f"{config['run_name']}_epoch{epoch}.pth"
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # 12) Save Final Model
    final_model_path = os.path.join(config['root'], f"{config['run_name']}_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at: {final_model_path}")

    # (Optional) save EMA model
    if config.get('use_ema', False):
        ema_model_path = os.path.join(config['root'], f"{config['run_name']}_ema_final.pth")
        torch.save(ema_model.module.state_dict(), ema_model_path)
        print(f"EMA model saved: {ema_model_path}")

    # -----------------------
    # PLOT THE TRAIN & VAL CURVES
    # -----------------------
    plt.figure(figsize=(7, 5))
    epochs = range(1, config['max_epochs'] + 1)
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title(f"Energy Prediction Training vs. Validation Loss ({mode.capitalize()})")
    plt.legend()
    plt.tight_layout()
    # Save the plot
    plt.savefig(os.path.join(config['root'], f"train_val_loss_{mode}.png"))
    plt.show()

    # -----------------------
    # TEST PHASE
    # -----------------------
    print("\n--- Test Phase ---")
    test_loss, test_mse, test_mae, test_r2, metrics = evaluate_model(model, test_loader, device, criterion, mode=mode)

    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    print(f"Test R²: {test_r2:.6f}")
    print(f"Energy MAE: {metrics.get('energy_mae', 'N/A')}")
    print(f"Energy RMSE: {metrics.get('energy_rmse', 'N/A')}")
    print(f"Energy R²: {metrics.get('energy_r2', 'N/A')}")

    test_results_path = os.path.join(config['root'], f"{config['run_name']}_test_results_{mode}.txt")
    with open(test_results_path, 'w') as f:
        f.write(f"Test Loss: {test_loss:.6f}\n")
        f.write(f"Test MSE: {test_mse:.6f}\n")
        f.write(f"Test MAE: {test_mae:.6f}\n")
        f.write(f"Test R²: {test_r2:.6f}\n")
        f.write(f"Energy MAE: {metrics.get('energy_mae', 'N/A')}\n")
        f.write(f"Energy RMSE: {metrics.get('energy_rmse', 'N/A')}\n")
        f.write(f"Energy R²: {metrics.get('energy_r2', 'N/A')}\n")
    print(f"Test results saved: {test_results_path}")

    # (Optional) wandb logging
    if config.get('wandb', False):
        wandb.log({
            'Test Loss': test_loss,
            'Test MSE': test_mse,
            'Test MAE': test_mae,
            'Test R²': test_r2,
            'Energy MAE': metrics.get('energy_mae', 'N/A'),
            'Energy RMSE': metrics.get('energy_rmse', 'N/A'),
            'Energy R²': metrics.get('energy_r2', 'N/A'),
        })
        wandb.finish()

    # -----------------------
    # PLOT PREDICTED vs. TRUE ON TEST SET
    # -----------------------
    print("\n--- Plot Predicted vs. True Energies on Test Set ---")
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            if mode == 'positions':
                Z, distance_matrix, mask, energies = batch
                Z = Z.to(device)
                distance_matrix = distance_matrix.to(device)
                mask = mask.to(device)
                energies = energies.to(device)

                pred_energy = model(Z, distance_matrix, mask)
            elif mode == 'forces':
                Z, forces, mask, energies = batch
                Z = Z.to(device)
                forces = forces.to(device)
                mask = mask.to(device)
                energies = energies.to(device)

                pred_energy = model(Z, forces, mask)
            else:
                raise ValueError("Invalid mode. Choose 'positions' or 'forces'.")

            all_preds.append(pred_energy.cpu())
            all_targets.append(energies.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy().flatten()
    all_targets = torch.cat(all_targets, dim=0).numpy().flatten()

    plt.figure(figsize=(6, 6))
    plt.scatter(all_targets, all_preds, alpha=0.5)
    # Diagonal line for reference
    mn = min(all_targets.min(), all_preds.min())
    mx = max(all_targets.max(), all_preds.max())
    plt.plot([mn, mx], [mn, mx], 'k--', lw=2)
    plt.xlabel("True Energy (eV)")
    plt.ylabel("Predicted Energy (eV)")
    plt.title(f"Test Set: Predicted vs. True Energies ({mode.capitalize()})")
    plt.tight_layout()
    # Save the plot
    plt.savefig(os.path.join(config['root'], f"test_pred_vs_true_{mode}.png"))
    plt.show()

