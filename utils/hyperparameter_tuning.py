# utils/hyperparameter_tuning.py

import optuna
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from train.trainer import parse_extxyz_with_forces, preprocess_data_with_forces, MoleculeDataset, CombinedLoss, \
    EnergyPredictor
from model.gnn_model import GNNModel  # If using GNN
import yaml
from utils.metrics import calculate_metrics
import numpy as np


def objective(trial, config):
    # Suggest hyperparameters
    hidden_size = trial.suggest_int('hidden_size', 64, 256)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    num_layers = trial.suggest_int('num_layers', 2, 5)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)

    # Parse dataset
    molecule_data = parse_extxyz_with_forces(config['dataset_file_name'])
    dataset = MoleculeDataset(molecule_data, config['chemical_symbols'], max_atoms=config.get('max_atoms', None))

    # Split dataset
    n_train = config['n_train']
    n_val = config['n_val']
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val],
                                              generator=torch.Generator().manual_seed(config['dataset_seed']))

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['validation_batch_size'], shuffle=False)

    # Initialize model
    input_size = dataset.processed_data[0]['inputs'].shape[0]
    model = EnergyPredictor(input_size=input_size, hidden_size=hidden_size, pool_size=1)

    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define loss and optimizer
    criterion = CombinedLoss(energy_weight=1.0, force_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop (simplified for hyperparameter tuning)
    for epoch in range(config['tuning_epochs']):
        model.train()
        running_loss = 0.0
        for inputs, energies, true_forces in train_loader:
            inputs = inputs.to(device)
            energies = energies.to(device).unsqueeze(1)
            true_forces = true_forces.to(device)

            # Ensure that positions require gradients
            inputs.requires_grad_(True)

            optimizer.zero_grad()
            pred_energy = model(inputs)

            # Compute forces as gradients of energy w.r.t inputs
            pred_forces = -torch.autograd.grad(
                outputs=pred_energy,
                inputs=inputs,
                grad_outputs=torch.ones_like(pred_energy),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]

            # Reshape forces to match true forces
            pred_forces = pred_forces.view(-1, true_forces.shape[1])

            # Compute loss
            loss = criterion(pred_energy, energies, pred_forces, true_forces)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / n_train

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, energies, true_forces in val_loader:
                inputs = inputs.to(device)
                energies = energies.to(device).unsqueeze(1)
                true_forces = true_forces.to(device)

                inputs.requires_grad_(True)
                pred_energy = model(inputs)
                pred_forces = -torch.autograd.grad(
                    outputs=pred_energy,
                    inputs=inputs,
                    grad_outputs=torch.ones_like(pred_energy),
                    create_graph=False,
                    retain_graph=False,
                    only_inputs=True
                )[0]
                pred_forces = pred_forces.view(-1, true_forces.shape[1])

                loss = criterion(pred_energy, energies, pred_forces, true_forces)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= n_val
        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss


def run_hyperparameter_tuning(config_path):
    config = yaml.safe_load(open(config_path, 'r'))
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, config), n_trials=50)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save the best hyperparameters
    with open('best_hyperparameters.yaml', 'w') as f:
        yaml.dump({'best_params': trial.params}, f)
