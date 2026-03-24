"""
Federated Learning Evaluation Script
Based on: https://github.com/AstraZeneca/SubTab
"""

import json
from pathlib import Path
import numpy as np

import eval
from utils.arguments import get_arguments, get_config


def load_dataset_info(dataset_name):
    """
    Load dataset information from JSON file.
    
    Args:
        dataset_name (str): Name of the dataset
        
    Returns:
        dict: Dataset information including task_type, cat_policy, and norm
    """
    info_path = Path(f'data/{dataset_name}/info.json')
    
    if not info_path.exists():
        raise FileNotFoundError(f"Dataset info file not found: {info_path}")
    
    return json.loads(info_path.read_text())


def configure_for_evaluation(config):
    """
    Configure settings for evaluation mode.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        dict: Updated configuration
    """
    # Load dataset information
    dataset_info = load_dataset_info(config["dataset"])
    
    # Update config with dataset info
    config["framework"] = config["dataset"]
    config['task_type'] = dataset_info['task_type']
    config['cat_policy'] = dataset_info['cat_policy']
    config['norm'] = dataset_info['norm']
    config['learning_rate_reducer'] = config['learning_rate']
    
    # Evaluation mode settings
    config["add_noise"] = False
    config["validate"] = False
    config["original_dataset"] = True
    config['modeFL'] = True
    config['sampling'] = True
    
    return config


def aggregate_results(all_results):
    """
    Aggregate results from all clients.
    
    Args:
        all_results (list): List of tuples (original_results, embed_results) for each client
        
    Returns:
        tuple: (original_metrics, embed_metrics) where each is a dict with mean values
    """
    ori_prec_list, ori_rec_list, ori_f1_list = [], [], []
    embed_prec_list, embed_rec_list, embed_f1_list = [], [], []
    
    for ori_results, embed_results in all_results:
        # Collect original results
        if ori_results is not None:
            ori_prec, ori_rec, ori_f1, _ = ori_results
            ori_prec_list.append(ori_prec)
            ori_rec_list.append(ori_rec)
            ori_f1_list.append(ori_f1)
        
        # Collect embedding results
        if embed_results is not None:
            embed_prec, embed_rec, embed_f1, _ = embed_results
            embed_prec_list.append(embed_prec)
            embed_rec_list.append(embed_rec)
            embed_f1_list.append(embed_f1)
    
    # Calculate means for original results
    ori_metrics = None
    if ori_prec_list and ori_rec_list and ori_f1_list:
        ori_metrics = {
            'precision': np.mean(ori_prec_list),
            'recall': np.mean(ori_rec_list),
            'f1': np.mean(ori_f1_list),
            'std_precision': np.std(ori_prec_list),
            'std_recall': np.std(ori_rec_list),
            'std_f1': np.std(ori_f1_list),
        }
    
    # Calculate means for embedding results
    embed_metrics = None
    if embed_prec_list and embed_rec_list and embed_f1_list:
        embed_metrics = {
            'precision': np.mean(embed_prec_list),
            'recall': np.mean(embed_rec_list),
            'f1': np.mean(embed_f1_list),
            'std_precision': np.std(embed_prec_list),
            'std_recall': np.std(embed_rec_list),
            'std_f1': np.std(embed_f1_list),
        }
    
    return ori_metrics, embed_metrics


def print_results(ori_metrics, embed_metrics):
    """
    Print aggregated results in a formatted manner.
    
    Args:
        ori_metrics (dict): Original data metrics
        embed_metrics (dict): Embedding metrics
    """
    print("\n" + "=" * 60)
    print("FEDERATED LEARNING EVALUATION RESULTS")
    print("=" * 60)
    
    if embed_metrics:
        print("\n📊 Embedding Results (Mean ± Std):")
        print(f"  Precision: {embed_metrics['precision']:.4f} ± {embed_metrics['std_precision']:.4f}")
        print(f"  Recall:    {embed_metrics['recall']:.4f} ± {embed_metrics['std_recall']:.4f}")
        print(f"  F1 Score:  {embed_metrics['f1']:.4f} ± {embed_metrics['std_f1']:.4f}")
    else:
        print("\n⚠️  No valid embedding results to calculate metrics.")
    
    if ori_metrics:
        print("\n📊 Original Data Results (Mean ± Std):")
        print(f"  Precision: {ori_metrics['precision']:.4f} ± {ori_metrics['std_precision']:.4f}")
        print(f"  Recall:    {ori_metrics['recall']:.4f} ± {ori_metrics['std_recall']:.4f}")
        print(f"  F1 Score:  {ori_metrics['f1']:.4f} ± {ori_metrics['std_f1']:.4f}")
    else:
        print("\n⚠️  No valid original results to calculate metrics.")
    
    print("\n" + "=" * 60)


def main(config):
    """
    Main evaluation function for federated learning.
    
    Args:
        config (dict): Configuration dictionary
    """
    # Configure for evaluation mode
    config = configure_for_evaluation(config)
    
    # Evaluate each client
    print(f"\n🚀 Starting evaluation for {config['fl_cluster']} clients...")
    all_results = []
    
    for client_id in range(config["fl_cluster"]):
        print(f"\n📍 Evaluating Client {client_id + 1}/{config['fl_cluster']}...")
        results = eval.main(config, client_id)
        all_results.append(results)
        print(f"✓ Client {client_id + 1} evaluation complete")
    
    # Aggregate results across all clients
    ori_metrics, embed_metrics = aggregate_results(all_results)
    
    # Print final results
    print_results(ori_metrics, embed_metrics)
    
    return ori_metrics, embed_metrics


if __name__ == "__main__":
    # Parse command line arguments
    args = get_arguments()
    
    # Load configuration
    config = get_config(args)
    
    # Run evaluation
    main(config)
