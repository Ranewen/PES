# utils/data_augmentation.py

import numpy as np
import torch
import ase
from ase import Atoms
from ase.io import write


def random_rotation(positions):
    """
    Applies a random rotation to atomic positions.

    Args:
        positions (numpy.ndarray): Atomic positions of shape (N, 3).

    Returns:
        numpy.ndarray: Rotated positions.
    """
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, np.pi)
    z = np.random.uniform(0, 2 * np.pi)

    # Rotation matrix using Euler angles
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta), -np.sin(theta)],
                   [0, np.sin(theta), np.cos(theta)]])

    Ry = np.array([[np.cos(phi), 0, np.sin(phi)],
                   [0, 1, 0],
                   [-np.sin(phi), 0, np.cos(phi)]])

    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])

    R = Rz @ Ry @ Rx
    rotated_positions = positions @ R.T
    return rotated_positions


def add_noise(positions, noise_level=0.01):
    """
    Adds Gaussian noise to atomic positions.

    Args:
        positions (numpy.ndarray): Atomic positions of shape (N, 3).
        noise_level (float): Standard deviation of the Gaussian noise.

    Returns:
        numpy.ndarray: Noisy positions.
    """
    noise = np.random.normal(0, noise_level, positions.shape)
    noisy_positions = positions + noise
    return noisy_positions


def augment_data(data, augmentation_methods=['rotation', 'noise'], noise_level=0.01):
    """
    Augments the dataset with specified augmentation methods.

    Args:
        data (list of dict): Original dataset.
        augmentation_methods (list): List of augmentation methods to apply.
        noise_level (float): Noise level for the 'noise' augmentation.

    Returns:
        list of dict: Augmented dataset.
    """
    augmented_data = []
    for sample in data:
        augmented_sample = sample.copy()
        positions = sample['positions']

        if 'rotation' in augmentation_methods:
            rotated_positions = random_rotation(positions)
            augmented_sample['positions'] = rotated_positions
            augmented_data.append(augmented_sample.copy())

        if 'noise' in augmentation_methods:
            noisy_positions = add_noise(positions, noise_level=noise_level)
            augmented_sample['positions'] = noisy_positions
            augmented_data.append(augmented_sample.copy())

        # Original data
        augmented_data.append(sample)

    return augmented_data
