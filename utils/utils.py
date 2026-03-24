"""
Utility functions for configuration, directory management, and reproducibility.
Based on: https://github.com/AstraZeneca/SubTab
"""

import cProfile
import os
import pstats
import random as python_random
import sys

import numpy as np
import torch
import yaml
from sklearn import manifold
from texttable import Texttable


def set_seed(options):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        options (dict): Configuration dictionary containing 'seed' parameter.
    """
    seed_value = options["seed"]
    np.random.seed(seed_value)
    python_random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    # Set CUDA seeds if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_dir(dir_path):
    """
    Create a directory if it doesn't exist.
    
    Args:
        dir_path (str): Path to directory to create.
    """
    os.makedirs(dir_path, exist_ok=True)


def make_dir(directory_path, new_folder_name):
    """
    Create a subdirectory and return the full path.
    
    Args:
        directory_path (str): Parent directory path.
        new_folder_name (str): Name of new subdirectory.
    
    Returns:
        str: Full path to the created directory.
    """
    full_path = os.path.join(directory_path, new_folder_name)
    create_dir(full_path)
    return full_path


def set_dirs(config):
    """
    Set up directory structure for saving results.
    
    Directory structure:
        results/
        └── framework/
            ├── training/
            │   └── model_mode/
            │       ├── model/
            │       ├── plots/
            │       └── loss/
            └── evaluation/
                ├── clusters/
                └── reconstructions/
    
    Args:
        config (dict): Configuration dictionary with paths and framework info.
    """
    paths = config["paths"]
    results_dir = paths["results"]
    
    # Create main results directory
    results_dir = make_dir(results_dir, "")
    
    # Create framework-specific directory
    framework_dir = make_dir(results_dir, config["framework"])
    
    # Create training directories
    training_dir = make_dir(framework_dir, "training")
    model_mode_dir = make_dir(training_dir, config["model_mode"])
    make_dir(model_mode_dir, "model")
    make_dir(model_mode_dir, "plots")
    make_dir(model_mode_dir, "loss")
    
    # Create evaluation directories
    evaluation_dir = make_dir(framework_dir, "evaluation")
    make_dir(evaluation_dir, "clusters")
    make_dir(evaluation_dir, "reconstructions")


def load_yaml_config(config_path):
    """
    Load YAML configuration file.
    
    Args:
        config_path (str): Path to YAML config file.
    
    Returns:
        dict: Configuration dictionary.
    
    Raises:
        SystemExit: If config file cannot be read.
    """
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        sys.exit(f"❌ Error: Config file not found: {config_path}")
    except yaml.YAMLError as e:
        sys.exit(f"❌ Error: Invalid YAML in {config_path}: {e}")
    except Exception as e:
        sys.exit(f"❌ Error reading config file {config_path}: {e}")


def get_runtime_and_model_config(args):
    """
    Load runtime and dataset-specific configuration.
    
    Args:
        args (argparse.Namespace): Command-line arguments containing dataset name.
    
    Returns:
        dict: Merged configuration dictionary.
    """
    # Load runtime configuration
    config = load_yaml_config("./config/runtime.yaml")
    
    # Add dataset information
    config["model_config"] = args.dataset
    config["dataset"] = args.dataset
    
    # Merge with model-specific configuration
    config = update_config_with_model(config)
    
    return config


def update_config_with_model(config):
    """
    Update configuration with model-specific settings.
    
    Args:
        config (dict): Base configuration dictionary.
    
    Returns:
        dict: Updated configuration with model settings.
    """
    # Load default model configuration
    model_config = load_yaml_config("./config/default.yaml")
    
    # Merge configurations (model_config overrides base config)
    config.update(model_config)
    
    return config


def get_runtime_and_model_config_with_dataset_name(dataset):
    """
    Load configuration using dataset name directly.
    
    Args:
        dataset (str): Name of the dataset.
    
    Returns:
        dict: Configuration dictionary.
    """
    # Load runtime configuration
    config = load_yaml_config("./config/runtime.yaml")
    
    # Add dataset information
    config["model_config"] = dataset
    config["dataset"] = dataset
    
    # Merge with model-specific configuration
    config = update_config_with_model(config)
    
    return config


def update_config_with_model_dims(data_loader, config):
    """
    Update configuration with input feature dimensions from data.
    
    Args:
        data_loader (Loader): Data loader instance.
        config (dict): Configuration dictionary.
    
    Returns:
        dict: Updated configuration with feature dimensions.
    """
    # Get first batch
    x, y = next(iter(data_loader.trainFL_loader))
    
    # Get feature dimensions
    n_features = x.shape[-1]
    
    # Insert feature dimension as first layer dimension
    config["dims"].insert(0, n_features)
    
    print(f"✓ Model input dimension set to {n_features}")
    
    return config


def run_with_profiler(main_fn, config):
    """
    Run function with profiler to analyze performance.
    
    Args:
        main_fn (callable): Main function to profile.
        config (dict): Configuration dictionary to pass to main_fn.
    """
    print("🔍 Starting profiler...")
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run the main function
    main_fn(config)
    
    profiler.disable()
    
    # Print statistics sorted by number of calls
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    
    print("\n" + "=" * 80)
    print("PROFILER RESULTS")
    print("=" * 80)
    stats.print_stats(20)  # Print top 20 functions


def tsne(latent, n_components=2, random_state=0):
    """
    Reduce dimensionality of embeddings using t-SNE.
    
    Args:
        latent (np.ndarray): High-dimensional embeddings.
        n_components (int): Number of dimensions to reduce to.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        np.ndarray: Low-dimensional embeddings.
    """
    tsne_model = manifold.TSNE(
        n_components=n_components,
        init='pca',
        random_state=random_state,
        verbose=0
    )
    
    return tsne_model.fit_transform(latent)


def print_config(config):
    """
    Print configuration in a formatted table.
    
    Args:
        config (dict or argparse.Namespace): Configuration to print.
    """
    # Convert Namespace to dict if needed
    if not isinstance(config, dict):
        config = vars(config)
    
    # Sort keys alphabetically
    sorted_keys = sorted(config.keys())
    
    # Initialize table with header
    table = Texttable()
    table.set_cols_align(["l", "l"])
    table.set_cols_valign(["m", "m"])
    table.set_cols_width([30, 50])
    
    # Add header and rows
    rows = [["Parameter", "Value"]]
    for key in sorted_keys:
        # Format key: replace underscores and capitalize
        formatted_key = key.replace("_", " ").capitalize()
        value = config[key]
        
        # Format value for better readability
        if isinstance(value, (list, tuple)) and len(str(value)) > 50:
            value = f"{type(value).__name__} (length: {len(value)})"
        elif isinstance(value, dict) and len(str(value)) > 50:
            value = f"Dict (keys: {len(value)})"
        
        rows.append([formatted_key, value])
    
    table.add_rows(rows)
    print(table.draw())


def save_config(config, save_path):
    """
    Save configuration to YAML file.
    
    Args:
        config (dict): Configuration dictionary.
        save_path (str): Path where to save the config file.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert non-serializable objects to strings
    serializable_config = {}
    for key, value in config.items():
        if isinstance(value, torch.device):
            serializable_config[key] = str(value)
        else:
            serializable_config[key] = value
    
    # Save to YAML
    with open(save_path, 'w') as file:
        yaml.dump(serializable_config, file, default_flow_style=False)
    
    print(f"✓ Configuration saved to {save_path}")


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to config file.
    
    Returns:
        dict: Loaded configuration.
    """
    config = load_yaml_config(config_path)
    
    # Convert device string back to torch.device if present
    if 'device' in config:
        config['device'] = torch.device(config['device'])
    
    return config


def get_device_info():
    """
    Get information about available compute devices.
    
    Returns:
        dict: Dictionary with device availability information.
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'mps_available': torch.backends.mps.is_built(),
        'cpu_count': os.cpu_count(),
    }
    
    if info['cuda_available']:
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_version'] = torch.version.cuda
    
    return info


def print_device_info():
    """Print information about available compute devices."""
    info = get_device_info()
    
    print("\n" + "=" * 60)
    print("DEVICE INFORMATION")
    print("=" * 60)
    print(f"CUDA Available: {info['cuda_available']}")
    
    if info['cuda_available']:
        print(f"  Device Count: {info['cuda_device_count']}")
        print(f"  Device Name: {info['cuda_device_name']}")
        print(f"  CUDA Version: {info['cuda_version']}")
    
    print(f"MPS Available: {info['mps_available']}")
    print(f"CPU Count: {info['cpu_count']}")
    print("=" * 60 + "\n")


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model (torch.nn.Module): PyTorch model.
    
    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num):
    """
    Format large numbers with K, M, B suffixes.
    
    Args:
        num (int or float): Number to format.
    
    Returns:
        str: Formatted number string.
    """
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(num)


# Example usage and testing
if __name__ == "__main__":
    # Print device information
    print_device_info()
    
    # Example configuration
    example_config = {
        'dataset': 'covtype',
        'epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001,
        'device': torch.device('cpu'),
        'seed': 42,
    }
    
    # Print configuration
    print("\nExample Configuration:")
    print_config(example_config)
    
    # Test seed setting
    set_seed(example_config)
    print("\n✓ Random seeds set for reproducibility")
