"""
Evaluation utilities for model assessment and visualization.
Based on: https://github.com/AstraZeneca/SubTab
"""

import csv
import functools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch as th
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC

from utils.colors import get_color_list
from utils.utils import tsne


def model_eval(config, z_train, y_train, suffix, z_test=None, y_test=None, 
               description="Logistic Regression", nData=None):
    """
    Evaluate model using classification metrics.
    
    Args:
        config (dict): Configuration dictionary.
        z_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        suffix (str): Filename suffix.
        z_test (np.ndarray, optional): Test features.
        y_test (np.ndarray, optional): Test labels.
        description (str): Description of evaluation method.
        nData (int, optional): Number of data samples.
    
    Returns:
        tuple: (precision, recall, f1_score, support) for test set.
    """
    print("\n" + ">" * 10 + " " + description)
    
    # Generate filename
    filename = _generate_filename(config, suffix)
    
    # Define regularization parameters
    if nData is None:
        c_values = [0.0001, 1, 10]
    else:
        c_values = [0.001]
    
    # Override with single value for consistency
    c_values = [0.01]
    
    results_list = []
    
    # Evaluate for each regularization parameter
    for c in c_values:
        print(f"\n{'*' * 10} C={c} {'*' * 10}")
        
        # Initialize classifier
        # clf = LogisticRegression(max_iter=1200, solver='lbfgs', C=c)
        clf = RandomForestClassifier(random_state=42, n_estimators=100)
        # clf = SVC(C=c, kernel='rbf', random_state=42)
        
        # Train classifier
        clf.fit(z_train, y_train)
        
        # Make predictions
        y_hat_train = clf.predict(z_train)
        y_hat_test = clf.predict(z_test)
        
        # Calculate metrics
        train_metrics = precision_recall_fscore_support(
            y_train, y_hat_train, average='weighted'
        )
        test_metrics = precision_recall_fscore_support(
            y_test, y_hat_test, average='weighted'
        )
        
        # Print results
        print(f"Training  - Precision: {train_metrics[0]:.4f}, "
              f"Recall: {train_metrics[1]:.4f}, "
              f"F1: {train_metrics[2]:.4f}")
        print(f"Test      - Precision: {test_metrics[0]:.4f}, "
              f"Recall: {test_metrics[1]:.4f}, "
              f"F1: {test_metrics[2]:.4f}")
        
        # Record results
        results_list.append({
            "model": f"RandomForest_C{c}",
            "train_precision": train_metrics[0],
            "train_recall": train_metrics[1],
            "train_f1": train_metrics[2],
            "test_precision": test_metrics[0],
            "test_recall": test_metrics[1],
            "test_f1": test_metrics[2],
        })
    
    # Save results
    _save_results(config, filename, results_list)
    
    return test_metrics


def _generate_filename(config, suffix):
    """
    Generate filename for saving results.
    
    Args:
        config (dict): Configuration dictionary.
        suffix (str): Filename suffix.
    
    Returns:
        str: Generated filename.
    """
    mode = "local" if config.get("local", False) else "FL"
    
    filename = (
        f"{suffix}"
        f"{config['epochs']}e-"
        f"{config['fl_cluster']}c-"
        f"{config['client_drop_rate']}cd-"
        f"{config['data_drop_rate']}dd-"
        f"{config['client_imbalance_rate']}nc-"
        f"{config['class_imbalance']}ci-"
        f"{config['dataset']}-"
        f"{mode}"
    )
    
    return filename


def _save_results(config, filename, results_list):
    """
    Save evaluation results to CSV file.
    
    Args:
        config (dict): Configuration dictionary.
        filename (str): Filename for saving.
        results_list (list): List of result dictionaries.
    """
    # Create directory if it doesn't exist
    results_dir = f"./results/{config['dataset']}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Full file path
    file_path = os.path.join(results_dir, f"{filename}.csv")
    
    # Save to CSV
    keys = results_list[0].keys()
    with open(file_path, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results_list)
    
    print(f"\n{'=' * 10}")
    print(f"✓ Results saved to: {file_path}")


def plot_clusters(config, z, clabels, suffix, plot_suffix="_inLatentSpace"):
    """
    Wrapper function to visualize clusters.
    
    Args:
        config (dict): Configuration dictionary.
        z (np.ndarray): Embeddings for plotting.
        clabels (np.ndarray): Class labels.
        suffix (str): Filename suffix.
        plot_suffix (str): Additional suffix for plot name.
    """
    # Get unique labels
    n_clusters = len(np.unique(clabels))
    
    # Create legend labels
    cluster_legends = [str(i) for i in range(n_clusters)]
    
    # Visualize clusters
    visualise_clusters(
        config, z, clabels, suffix,
        plt_name=f"classes{plot_suffix}",
        legend_title="Classes",
        legend_labels=cluster_legends
    )


def visualise_clusters(config, embeddings, labels, suffix, plt_name="test",
                       alpha=1.0, legend_title=None, legend_labels=None, ncol=1):
    """
    Plot clusters using PCA and t-SNE embeddings.
    
    Args:
        config (dict): Configuration dictionary.
        embeddings (np.ndarray): High-dimensional embeddings.
        labels (np.ndarray): Class labels.
        suffix (str): Filename suffix.
        plt_name (str): Plot filename.
        alpha (float): Transparency of scatter points.
        legend_title (str): Title for legend.
        legend_labels (list): Labels for legend.
        ncol (int): Number of columns in legend.
    """
    # Get color palette
    color_list, _ = get_color_list()
    palette = {str(i): color_list[i] for i in range(len(color_list))}
    
    # Prepare labels
    y = labels.reshape(-1)
    y = list(map(str, y.tolist()))
    
    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), facecolor='w')
    fig.subplots_adjust(hspace=0.1, wspace=0.3)
    
    # PCA visualization
    pca = PCA(n_components=2, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    
    axs[0].set_title('PCA Projection', fontsize=14, fontweight='bold')
    sns.scatterplot(
        x=embeddings_pca[:, 0], y=embeddings_pca[:, 1],
        ax=axs[0], palette=palette, hue=y, s=20, alpha=alpha, legend=False
    )
    axs[0].set_xlabel('PC1', fontsize=12)
    axs[0].set_ylabel('PC2', fontsize=12)
    
    # t-SNE visualization
    embeddings_tsne = tsne(embeddings)
    
    axs[1].set_title('t-SNE Projection', fontsize=14, fontweight='bold')
    sns_plt = sns.scatterplot(
        x=embeddings_tsne[:, 0], y=embeddings_tsne[:, 1],
        ax=axs[1], palette=palette, hue=y, s=20, alpha=alpha
    )
    axs[1].set_xlabel('t-SNE 1', fontsize=12)
    axs[1].set_ylabel('t-SNE 2', fontsize=12)
    
    # Configure legend
    _configure_legend(sns_plt, fig, ncol, legend_labels, legend_title)
    
    # Remove individual legends
    if axs[1].get_legend():
        axs[1].get_legend().remove()
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    
    # Save figure
    _save_plot(config, suffix, plt_name)
    
    plt.close()


def _configure_legend(sns_plt, fig, ncol, labels, title=None):
    """
    Configure plot legend.
    
    Args:
        sns_plt: Seaborn plot object.
        fig: Matplotlib figure object.
        ncol (int): Number of columns in legend.
        labels (list): Legend labels.
        title (str, optional): Legend title.
    """
    # Get handles and labels
    handles, legend_txts = sns_plt.get_legend_handles_labels()
    
    # Sort by numeric value
    legend_txts = [int(d) for d in legend_txts]
    legend_txts, handles = zip(*sorted(zip(legend_txts, handles)))
    
    # Create legend
    title = title or "Cluster"
    fig.legend(
        handles, labels, loc="center right",
        borderaxespad=0.1, title=title, ncol=ncol,
        frameon=True, fontsize=10
    )
    
    # Clean up axes
    sns_plt.set(xticklabels=[], yticklabels=[], xlabel=None, ylabel=None)
    sns_plt.tick_params(top=False, bottom=False, left=False, right=False)


def _save_plot(config, suffix, plt_name):
    """
    Save plot to file.
    
    Args:
        config (dict): Configuration dictionary.
        suffix (str): Filename suffix.
        plt_name (str): Plot name.
    """
    # Create directory
    root_path = os.path.dirname(os.path.dirname(__file__))
    plot_dir = os.path.join(
        root_path, "results", config["framework"],
        "evaluation", "clusters"
    )
    os.makedirs(plot_dir, exist_ok=True)
    
    # Save plot
    fig_path = os.path.join(plot_dir, f"{suffix}{plt_name}.png")
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    
    print(f"✓ Plot saved to: {fig_path}")


def save_np2csv(np_list, save_as="test.csv"):
    """
    Save numpy arrays to CSV file.
    
    Args:
        np_list (list): List containing [features, labels].
        save_as (str): Output filename.
    """
    # Extract features and labels
    X, y = np_list
    y = np.array(y, dtype=np.int8)
    
    # Create column names
    columns = ["label"] + [str(i) for i in range(X.shape[1])]
    
    # Concatenate data
    data = np.concatenate((y.reshape(-1, 1), X), axis=1)
    
    # Create DataFrame
    df = pd.DataFrame(data=data, columns=columns)
    
    # Save to CSV
    df.to_csv(save_as, index=False)
    
    print(f"✓ DataFrame saved to: {save_as}")
    print(f"  Shape: {df.shape}")
    print(f"  Preview:\n{df.head()}")


def append_tensors_to_lists(list_of_lists, list_of_tensors):
    """
    Append tensors to lists after converting to numpy arrays.
    
    Args:
        list_of_lists (list): List of lists to append to.
        list_of_tensors (list): List of PyTorch tensors.
    
    Returns:
        list: Updated list of lists.
    """
    for i, tensor in enumerate(list_of_tensors):
        list_of_lists[i].append(tensor.cpu().numpy())
    
    return list_of_lists


def concatenate_lists(list_of_lists):
    """
    Concatenate lists of numpy arrays.
    
    Args:
        list_of_lists (list): List of lists containing numpy arrays.
    
    Returns:
        np.ndarray or list: Concatenated arrays.
    """
    concatenated = [np.concatenate(lst) for lst in list_of_lists]
    
    return concatenated[0] if len(concatenated) == 1 else concatenated


def aggregate(latent_list, config):
    """
    Aggregate latent representations from multiple subsets.
    
    Args:
        latent_list (list): List of latent representations (torch tensors).
        config (dict): Configuration dictionary.
    
    Returns:
        torch.Tensor: Aggregated representation.
    
    Raises:
        ValueError: If aggregation method is not recognized.
    """
    aggregation_method = config.get("aggregation", "mean")
    
    if aggregation_method == "mean":
        return sum(latent_list) / len(latent_list)
    
    elif aggregation_method == "sum":
        return sum(latent_list)
    
    elif aggregation_method == "concat":
        return th.cat(latent_list, dim=-1)
    
    elif aggregation_method == "max":
        return functools.reduce(th.max, latent_list)
    
    elif aggregation_method == "min":
        return functools.reduce(th.min, latent_list)
    
    else:
        raise ValueError(
            f"Unknown aggregation method: {aggregation_method}. "
            f"Valid options: mean, sum, concat, max, min"
        )


def compute_metrics(y_true, y_pred, average='weighted'):
    """
    Compute classification metrics.
    
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        average (str): Averaging method for multi-class metrics.
    
    Returns:
        dict: Dictionary containing precision, recall, f1, and support.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=average
    )
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support
    }


def print_metrics(metrics, dataset_name="Test"):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics (dict): Dictionary of metrics.
        dataset_name (str): Name of dataset (e.g., "Train", "Test").
    """
    print(f"\n{dataset_name} Metrics:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    if 'support' in metrics:
        print(f"  Support:   {metrics['support']}")


# Example usage
if __name__ == "__main__":
    # Example: Test aggregation methods
    latent1 = th.randn(32, 128)
    latent2 = th.randn(32, 128)
    latent_list = [latent1, latent2]
    
    config = {"aggregation": "mean"}
    
    aggregated = aggregate(latent_list, config)
    print(f"Aggregated shape: {aggregated.shape}")
    
    # Example: Test metrics computation
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 1, 0, 1, 2])
    
    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics)
