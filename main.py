import argparse
import os
from train.trainer import train_model, load_config


def main():
    parser = argparse.ArgumentParser(description='Energy Prediction from Positions or Forces')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the YAML configuration file.')
    parser.add_argument('--mode', type=str, choices=['positions', 'forces'], required=True,
                        help='Mode of operation: "positions" to predict energy from positions, '
                             '"forces" to predict energy from forces.')
    args = parser.parse_args()

    config = load_config(args.config)

    train_model(config, mode=args.mode)


if __name__ == '__main__':
    main()


if __name__ == "__main__":
    main()
