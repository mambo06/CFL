"""
Command-line argument parser and configuration loader.
Based on: https://github.com/AstraZeneca/SubTab
"""

import sys
from argparse import ArgumentParser

import torch as th

from utils.utils import get_runtime_and_model_config, print_config


class ArgParser(ArgumentParser):
    """
    Custom ArgumentParser that prints helpful message on error.
    """
    
    def error(self, message):
        """
        Override error method to show help message.
        
        Args:
            message (str): Error message to display.
        """
        sys.stderr.write(f'Error: {message}\n\n')
        self.print_help()
        sys.exit(2)


def get_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = ArgParser(
        description='Federated Learning with SubTab Framework',
        formatter_class=lambda prog: ArgumentParser.formatter_class(
            prog, max_help_position=40, width=100
        )
    )
    
    # Dataset configuration
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="covtype",
        help="Name of the dataset (must have corresponding config file)"
    )
    
    # Device configuration
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "-g", "--gpu",
        dest='gpu',
        action='store_true',
        help="Use CUDA GPU (if available)"
    )
    device_group.add_argument(
        "-m", "--mps",
        dest='mps',
        action='store_true',
        help="Use Apple Silicon GPU (if available)"
    )
    device_group.add_argument(
        "-ng", "--no_gpu",
        dest='use_cpu',
        action='store_true',
        help="Force CPU usage"
    )
    parser.set_defaults(gpu=True, mps=False, use_cpu=False)
    
    parser.add_argument(
        "-dn", "--device_number",
        type=str,
        default='0',
        help="GPU device number (e.g., '0' for cuda:0)"
    )
    
    # Experiment tracking
    parser.add_argument(
        "-ex", "--experiment",
        type=int,
        default=1,
        help="Experiment number for MLFlow tracking"
    )
    
    # Federated learning configuration
    fl_group = parser.add_argument_group('Federated Learning Options')
    
    fl_group.add_argument(
        "-lc", "--local",
        dest='local',
        action='store_true',
        help="Run in local mode (non-federated)"
    )
    fl_group.add_argument(
        "-e", "--epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    fl_group.add_argument(
        "-c", "--clients",
        type=int,
        default=4,
        help="Number of federated learning clients"
    )
    fl_group.add_argument(
        "-cd", "--client_drop",
        type=float,
        default=0.0,
        help="Client drop rate (0.0 to 1.0)"
    )
    fl_group.add_argument(
        "-dd", "--data_drop",
        type=float,
        default=0.0,
        help="Data drop rate (0.0 to 1.0)"
    )
    fl_group.add_argument(
        "-nc", "--noniid_clients",
        type=float,
        default=0.0,
        help="Non-IID client rate (0.0 to 1.0)"
    )
    fl_group.add_argument(
        "-ci", "--class_imbalance",
        type=float,
        default=0.0,
        help="Class imbalance rate (0.0 to 1.0)"
    )
    fl_group.add_argument(
        "-bg", "--base_global",
        dest='base_global',
        action='store_true',
        help="Use base global model"
    )
    
    return parser.parse_args()


def get_config(args):
    """
    Load configuration from YAML files and merge with command-line arguments.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    
    Returns:
        dict: Complete configuration dictionary.
    """
    # Load runtime and model config from YAML files
    config = get_runtime_and_model_config(args)
    
    # Determine device
    config["device"] = _get_device(args)
    
    # Add federated learning parameters
    config['local'] = args.local
    config['epochs'] = args.epochs
    config['fl_cluster'] = args.clients
    config['client_drop_rate'] = args.client_drop
    config['data_drop_rate'] = args.data_drop
    config['client_imbalance_rate'] = args.noniid_clients
    config['class_imbalance'] = args.class_imbalance
    config['baseGlobal'] = args.base_global
    
    return config


def _get_device(args):
    """
    Determine the device to use based on arguments and availability.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    
    Returns:
        torch.device: Device to use for computation.
    """
    # Force CPU if requested
    if args.use_cpu:
        print("🖥️  Using CPU (forced by user)")
        return th.device('cpu')
    
    # Try CUDA GPU
    if args.gpu and th.cuda.is_available():
        device = th.device(f'cuda:{args.device_number}')
        print(f"🚀 Using CUDA GPU: {device}")
        return device
    
    # Try Apple Silicon GPU
    if args.mps and th.backends.mps.is_built():
        device = th.device('mps')
        print("🍎 Using Apple Silicon GPU (MPS)")
        return device
    
    # Default to CPU
    print("🖥️  Using CPU (no GPU available)")
    return th.device('cpu')


def print_config_summary(config, args=None):
    """
    Print configuration summary for verification.
    
    Args:
        config (dict): Configuration dictionary.
        args (argparse.Namespace, optional): Command-line arguments.
    """
    print("\n" + "=" * 100)
    print(f"{'CONFIGURATION SUMMARY':^100}")
    print("=" * 100 + "\n")
    
    print("📋 Model Configuration:")
    print_config(config)
    
    if args is not None:
        print("\n" + "-" * 100)
        print("\n⚙️  Command-line Arguments:")
        print_config(vars(args))
    
    print("\n" + "=" * 100 + "\n")


def validate_config(config):
    """
    Validate configuration parameters.
    
    Args:
        config (dict): Configuration dictionary.
    
    Raises:
        ValueError: If configuration parameters are invalid.
    """
    # Validate rates are between 0 and 1
    rate_params = [
        'client_drop_rate', 
        'data_drop_rate', 
        'client_imbalance_rate', 
        'class_imbalance'
    ]
    
    for param in rate_params:
        if param in config:
            value = config[param]
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Parameter '{param}' must be between 0.0 and 1.0, "
                    f"got {value}"
                )
    
    # Validate positive integers
    if config.get('epochs', 1) < 1:
        raise ValueError("Number of epochs must be at least 1")
    
    if config.get('fl_cluster', 1) < 1:
        raise ValueError("Number of clients must be at least 1")
    
    print("✓ Configuration validated successfully")


def get_config_string(config):
    """
    Generate a string representation of key configuration parameters.
    Useful for experiment naming and logging.
    
    Args:
        config (dict): Configuration dictionary.
    
    Returns:
        str: Formatted configuration string.
    """
    mode = "local" if config.get('local', False) else "FL"
    
    config_str = (
        f"{config.get('dataset', 'unknown')}-"
        f"{mode}-"
        f"e{config.get('epochs', 0)}-"
        f"c{config.get('fl_cluster', 0)}-"
        f"cd{config.get('client_drop_rate', 0):.2f}-"
        f"dd{config.get('data_drop_rate', 0):.2f}-"
        f"nc{config.get('client_imbalance_rate', 0):.2f}-"
        f"ci{config.get('class_imbalance', 0):.2f}"
    )
    
    return config_str


# Example usage and testing
if __name__ == "__main__":
    # Parse arguments
    args = get_arguments()
    
    # Get configuration
    config = get_config(args)
    
    # Validate configuration
    try:
        validate_config(config)
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)
    
    # Print summary
    print_config_summary(config, args)
    
    # Print config string
    print(f"📝 Config string: {get_config_string(config)}")
