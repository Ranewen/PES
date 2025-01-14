# utils/metrics.py

import torch
import torch.nn as nn
import numpy as np


def calculate_metrics(pred_energy, true_energy):
    """
    Calculates various evaluation metrics for energy and force predictions.

    Args:
        pred_energy (torch.Tensor): Predicted energies.
        true_energy (torch.Tensor): True energies.
        pred_forces (torch.Tensor): Predicted forces.
        true_forces (torch.Tensor): True forces.

    Returns:
        dict: Dictionary containing MAE, RMSE, and R² for energy and forces.
    """
    metrics = {}

    # Energy metrics
    mae_energy = nn.L1Loss()(pred_energy, true_energy).item()
    mse_energy = nn.MSELoss()(pred_energy, true_energy).item()
    rmse_energy = np.sqrt(mse_energy)
    ss_tot = torch.sum((true_energy - torch.mean(true_energy)) ** 2).item()
    ss_res = torch.sum((true_energy - pred_energy) ** 2).item()
    r2_energy = 1 - ss_res / ss_tot if ss_tot != 0 else float('nan')

    metrics['energy_mae'] = mae_energy
    metrics['energy_rmse'] = rmse_energy
    metrics['energy_r2'] = r2_energy

    """ Force metrics
    mae_force = nn.L1Loss()(pred_forces, true_forces).item()
    mse_force = nn.MSELoss()(pred_forces, true_forces).item()
    rmse_force = np.sqrt(mse_force)
    ss_tot_force = torch.sum((true_forces - torch.mean(true_forces)) ** 2).item()
    ss_res_force = torch.sum((true_forces - pred_forces) ** 2).item()
    r2_force = 1 - ss_res_force / ss_tot_force if ss_tot_force != 0 else float('nan')

    metrics['force_mae'] = mae_force
    metrics['force_rmse'] = rmse_force
    metrics['force_r2'] = r2_force
"""
    return metrics


