"""Command line interface for liver disease prediction."""
import argparse
import logging
import sys
from pathlib import Path

import yaml

from liver_disease_prediction.models.train import train_model
from liver_disease_prediction.utils.logger import setup_logging


def get_project_root() -> Path:
    """Get the project root directory."""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        project_root = get_project_root()
        config_file = project_root / config_path
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found at {config_file}")
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration: {e}")


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='Liver Disease Prediction')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # Create necessary directories
        project_root = get_project_root()
        for path in [config['data']['processed_data_path'],
                    config['output']['models_path'],
                    config['output']['plots_path'],
                    config['output']['metrics_path']]:
            (project_root / path.lstrip('./')).mkdir(parents=True, exist_ok=True)

        # Train model
        train_model(config)
        logger.info("Training completed successfully")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 