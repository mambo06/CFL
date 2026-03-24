"""
Evaluation script for Federated Learning with SubTab.
Based on: https://github.com/AstraZeneca/SubTab
"""

import copy
import sys

import mlflow
import numpy as np
import torch as th
from tqdm import tqdm

from src.model import CFL
from utils.arguments import get_arguments, get_config
from utils.eval_utils import (
    model_eval, 
    plot_clusters, 
    append_tensors_to_lists, 
    concatenate_lists, 
    aggregate
)
from utils.load_data_new import Loader
from utils.utils import set_dirs, run_with_profiler, update_config_with_model_dims

th.manual_seed(1)


def evaluate_embeddings(data_loader, model, config, client, mode='test', n_data=None):
    """
    Evaluate model using learned embeddings.
    
    Args:
        data_loader (Loader): Data loader instance.
        model (CFL): CFL model instance.
        config (dict): Configuration dictionary.
        client (int): Client ID.
        mode (str): 'train' or 'test' mode.
        n_data (int, optional): Number of data samples to use.
    
    Returns:
        tuple or None: Evaluation results (precision, recall, f1, accuracy) or None.
    """
    print("\n" + "=" * 60)
    print(f"Extracting embeddings for {mode} set...")
    print(f"Dataset: {config['dataset']}")
    print("=" * 60)
    
    # Get encoder and set to evaluation mode
    encoder = model.encoder.to(config["device"])
    encoder.eval()
    
    # Select appropriate data loader
    if n_data is not None:
        loader = data_loader.trainFL_loader if mode == 'train' else data_loader.test_loader
    else:
        loader = data_loader.trainFL_loader
    
    # Extract embeddings
    z, labels = _extract_embeddings(loader, model, encoder, config)
    
    # Split into train/test based on training_data_ratio
    z_train, z_test, y_train, y_test = _split_data(
        z, labels, config['training_data_ratio']
    )
    
    # Evaluate on test set
    if mode == 'test':
        print("\n" + "*" * 60)
        print("Evaluating with Logistic Regression on embeddings")
        print("*" * 60)
        
        suffix = f"-Dataset-{n_data}" if n_data is not None else ""
        prefix = f"Client-{client}{suffix}-contrastive-"
        
        description = "Sweeping C parameter (smaller = stronger regularization)"
        
        return model_eval(
            config, z_train, y_train, prefix,
            z_test=z_test, y_test=y_test,
            description=description, nData=n_data
        )
    else:
        return z, labels


def evaluate_original_data(data_loader, config, client, mode='test', n_data=None):
    """
    Evaluate using original data without embeddings.
    
    Args:
        data_loader (Loader): Data loader instance.
        config (dict): Configuration dictionary.
        client (int): Client ID.
        mode (str): 'train' or 'test' mode.
        n_data (int, optional): Number of data samples to use.
    
    Returns:
        tuple or None: Evaluation results (precision, recall, f1, accuracy) or None.
    """
    print("\n" + "=" * 60)
    print(f"Evaluating original data for {mode} set...")
    print("=" * 60)
    
    # Select appropriate data loader
    if n_data is not None:
        loader = data_loader.trainNS_loader if mode == 'train' else data_loader.testNS_loader
    else:
        loader = data_loader.trainNS_loader
    
    # Extract original features and labels
    z, labels = _extract_original_features(loader)
    
    # Split into train/test
    z_train, z_test, y_train, y_test = _split_data(
        z, labels, config['training_data_ratio']
    )
    
    print(f"Train shape: {z_train.shape}, Test shape: {z_test.shape}")
    
    # Evaluate on test set
    if mode == 'test':
        print("\n" + "*" * 60)
        print("Evaluating with Logistic Regression on original data")
        print("*" * 60)
        
        suffix = f"-Dataset-{n_data}" if n_data is not None else ""
        if config.get('baseGlobal', False):
            suffix += '-baseGlobal'
        
        prefix = f"Client-{client}{suffix}-original-"
        description = "Sweeping C parameter (smaller = stronger regularization)"
        
        return model_eval(
            config, z_train, y_train, prefix,
            z_test=z_test, y_test=y_test,
            description=description, nData=n_data
        )
    else:
        return z, labels


def _extract_embeddings(loader, model, encoder, config):
    """
    Extract embeddings from data using the encoder.
    
    Args:
        loader (DataLoader): PyTorch data loader.
        model (CFL): CFL model instance.
        encoder (nn.Module): Encoder network.
        config (dict): Configuration dictionary.
    
    Returns:
        tuple: (embeddings, labels) as numpy arrays.
    """
    z_list, label_list = [], []
    
    # Progress bar
    pbar = tqdm(loader, total=len(loader), desc="Extracting embeddings")
    
    for x, label in pbar:
        # Generate subsets
        x_tilde_list = model.subset_generator(x, mode='test')
        
        # Extract latent representations for each subset
        latent_list = []
        for xi in x_tilde_list:
            x_batch = model._tensor(xi)
            _, latent, _ = encoder(x_batch)
            latent_list.append(latent)
        
        # Aggregate latent representations
        latent = aggregate(latent_list, config)
        
        # Append to lists
        z_list, label_list = append_tensors_to_lists(
            [z_list, label_list],
            [latent, label.int()]
        )
    
    # Concatenate all batches
    z = concatenate_lists([z_list])
    labels = concatenate_lists([label_list])
    
    return z, labels


def _extract_original_features(loader):
    """
    Extract original features without model transformation.
    
    Args:
        loader (DataLoader): PyTorch data loader.
    
    Returns:
        tuple: (features, labels) as numpy arrays.
    """
    z_list, label_list = [], []
    
    # Progress bar
    pbar = tqdm(loader, total=len(loader), desc="Extracting features")
    
    for x, label in pbar:
        z_list, label_list = append_tensors_to_lists(
            [z_list, label_list],
            [x, label.int()]
        )
    
    # Concatenate all batches
    z = concatenate_lists([z_list])
    labels = concatenate_lists([label_list])
    
    return z, labels


def _split_data(z, labels, train_ratio):
    """
    Split data into train and test sets.
    
    Args:
        z (np.ndarray): Feature matrix.
        labels (np.ndarray): Labels.
        train_ratio (float): Ratio of training data (0.0 to 1.0).
    
    Returns:
        tuple: (z_train, z_test, y_train, y_test)
    """
    n_samples = z.shape[0]
    split_idx = int(n_samples * train_ratio)
    
    z_train = z[:split_idx]
    z_test = z[split_idx:]
    y_train = labels[:split_idx]
    y_test = labels[split_idx:]
    
    return z_train, z_test, y_train, y_test


def evaluate(data_loader, config, client, n_data=None):
    """
    Main evaluation function.
    
    Args:
        data_loader (Loader): Data loader instance.
        config (dict): Configuration dictionary.
        client (int): Client ID.
        n_data (int or list, optional): Number of data samples or list of sizes.
    
    Returns:
        list: [original_results, embedding_results]
    """
    # Initialize results
    results_list = [None, None]
    
    # Instantiate model
    model = CFL(config)
    
    # Load model weights if not using base global
    if not config.get('baseGlobal', False):
        model.load_models(client)
    
    with th.no_grad():
        # Evaluate original dataset
        if config.get('original_dataset', False):
            print("\n" + "🔍 " + "=" * 58)
            print("EVALUATING ORIGINAL DATASET")
            print("=" * 60)
            
            if n_data is not None and isinstance(n_data, list):
                for n in n_data:
                    result = evaluate_original_data(
                        data_loader, config, client, mode="test", n_data=n
                    )
            else:
                result = evaluate_original_data(
                    data_loader, config, client, mode="test", n_data=n_data
                )
            
            results_list[0] = copy.deepcopy(result)
            
            print(f"\n✓ Results saved to ./results/{config['framework']}/evaluation/")
        
        # Exit if using base global model
        if config.get('baseGlobal', False):
            return results_list
        
        # Evaluate embeddings
        print("\n" + "🔍 " + "=" * 58)
        print("EVALUATING LEARNED EMBEDDINGS")
        print("=" * 60)
        
        if n_data is not None and isinstance(n_data, list):
            for n in n_data:
                result = evaluate_embeddings(
                    data_loader, model, config, client, mode="test", n_data=n
                )
        else:
            result = evaluate_embeddings(
                data_loader, model, config, client, mode="test", n_data=n_data
            )
        
        results_list[1] = copy.deepcopy(result)
        
        print(f"\n✓ Results saved to ./results/{config['framework']}/evaluation/")
    
    return results_list


def main(config, client=1, n_data=None):
    """
    Main evaluation function.
    
    Args:
        config (dict): Configuration dictionary.
        client (int): Client ID.
        n_data (int or list, optional): Number of data samples or list of sizes.
    
    Returns:
        list: Evaluation results [original_results, embedding_results].
    """
    # Set up directories
    set_dirs(config)
    
    # Load data
    print(f"\n📊 Loading data for client {client}...")
    data_loader = Loader(
        config,
        dataset_name=config["dataset"],
        drop_last=False,
        client=client
    )
    
    # Update config with model dimensions
    config = update_config_with_model_dims(data_loader, config)
    
    # Run evaluation
    results = evaluate(data_loader, config, client, n_data)
    
    return results


if __name__ == "__main__":
    # Parse arguments
    args = get_arguments()
    config = get_config(args)
    
    # Configure for evaluation
    config["framework"] = config["dataset"]
    config["validate"] = False
    config["add_noise"] = False
    
    # Run with MLFlow tracking if enabled
    if config.get("mlflow", False):
        experiment_name = f"SubTab_Evaluation_{args.experiment}"
        mlflow.set_experiment(experiment_name=experiment_name)
        
        with mlflow.start_run():
            if config.get("profile", False):
                run_with_profiler(main, config)
            else:
                main(config)
    else:
        if config.get("profile", False):
            run_with_profiler(main, config)
        else:
            main(config)
