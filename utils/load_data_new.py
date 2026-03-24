"""
Data loader for federated learning with tabular and image datasets.
Based on: https://github.com/AstraZeneca/SubTab
"""

import os
import numpy as np
import torch
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.datasets import fetch_covtype
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets as dts
import torchvision.transforms as transforms


class Loader:
    """Main data loader for federated learning."""

    def __init__(self, config, dataset_name, client, drop_last=True, shuffle=True, kwargs=None):
        """
        Initialize Pytorch data loader for federated learning.

        Args:
            config (dict): Configuration dictionary with options and arguments
            dataset_name (str): Name of the dataset to load
            client (int): Client ID for federated learning
            drop_last (bool): Whether to drop last incomplete batch
            shuffle (bool): Whether to shuffle data
            kwargs (dict): Additional parameters for DataLoader
        """
        if kwargs is None:
            kwargs = {}
            
        self.client = client
        self.config = config
        batch_size = config["batch_size"]
        
        # Set data paths
        paths = config["paths"]
        file_path = os.path.join(paths["data"], dataset_name)
        
        torch.manual_seed(5)

        # Get datasets with and without shuffling (NS = No Shuffle for Pearson correlation)
        train_fl_dataset, validation_fl_dataset, test_dataset = self.get_dataset(
            dataset_name, file_path, ns=False
        )
        train_ns_dataset, validation_ns_dataset, _ = self.get_dataset(
            dataset_name, file_path, ns=True
        )

        # Create data loaders
        self.trainFL_loader = DataLoader(
            train_fl_dataset, batch_size=batch_size, shuffle=shuffle, 
            drop_last=drop_last, **kwargs
        )
        self.validationFl_loader = DataLoader(
            validation_fl_dataset, batch_size=batch_size, shuffle=shuffle, 
            drop_last=drop_last, **kwargs
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=shuffle, 
            drop_last=drop_last, **kwargs
        )
        self.trainNS_loader = DataLoader(
            train_ns_dataset, batch_size=batch_size, shuffle=False, 
            drop_last=drop_last, **kwargs
        )
        self.validationNS_loader = DataLoader(
            validation_ns_dataset, batch_size=batch_size, shuffle=False, 
            drop_last=drop_last, **kwargs
        )

    def get_dataset(self, dataset_name, file_path, ns=False):
        """
        Returns training, validation, and test datasets.
        
        Args:
            dataset_name (str): Name of the dataset
            file_path (str): Path to dataset files
            ns (bool): No shuffle flag for Pearson correlation
            
        Returns:
            tuple: (train_dataset, validation_dataset, test_dataset)
        """
        # Map dataset names to loader classes
        loader_map = {'default_loader': TabularDataset}
        
        # Get dataset class (default to TabularDataset)
        dataset_class = loader_map.get(dataset_name, loader_map['default_loader'])
        
        # Create datasets
        train_fl_dataset = dataset_class(
            self.config, datadir=file_path, dataset_name=dataset_name, 
            mode='train_fl', client=self.client, ns=ns
        )
        validation_fl_dataset = dataset_class(
            self.config, datadir=file_path, dataset_name=dataset_name, 
            mode='validation', client=self.client, ns=ns
        )
        test_dataset = dataset_class(
            self.config, datadir=file_path, dataset_name=dataset_name, 
            mode='test', client=self.client, ns=ns
        )
        
        return train_fl_dataset, validation_fl_dataset, test_dataset


class ToTensorNormalize:
    """Convert ndarrays to Tensors."""
    
    def __call__(self, sample):
        """Assumes min-max scaling is done during preprocessing."""
        return torch.from_numpy(sample).float()


class TabularDataset(Dataset):
    """Dataset class for tabular data format."""
    
    def __init__(self, config, datadir, dataset_name, mode='train', 
                 client=0, transform=None, ns=False):
        """
        Initialize tabular dataset.

        Args:
            config (dict): Configuration dictionary
            datadir (str): Path to data directory
            dataset_name (str): Name of the dataset
            mode (str): 'train_fl', 'validation', or 'test'
            client (int): Client ID for federated learning
            transform (callable): Transformation function for data
            ns (bool): No shuffle flag (skip Pearson reordering if True)
        """
        self.client = client
        self.config = config
        self.mode = mode
        self.paths = config["paths"]
        self.dataset_name = dataset_name
        self.data_path = os.path.join(self.paths["data"], dataset_name)
        self.ns = ns  # Skip Pearson correlation reordering if True
        self.transform = transform or ToTensorNormalize()
        
        # Load data
        self.data, self.labels = self._load_data()

    def __len__(self):
        """Returns number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Returns a single sample and its label."""
        sample = self.data[idx]
        label = int(self.labels[idx])
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label

    def _load_data(self):
        """
        Loads dataset and returns features and labels.
        
        Returns:
            tuple: (data, labels) as numpy arrays
        """
        # Load appropriate dataset
        dataset_loaders = {
            "mnist": self._load_mnist,
            "blog": self._load_blog,
            "income": self._load_income,
            "cifar10": self._load_cifar,
            "syn": self._load_syn,
            "covtype": self._load_covtype,
        }
        
        loader_func = dataset_loaders.get(self.dataset_name.lower())
        if loader_func is None:
            raise ValueError(
                f"Dataset '{self.dataset_name}' not found. "
                f"Available datasets: {list(dataset_loaders.keys())}"
            )
        
        x_train, y_train, x_test, y_test = loader_func()

        # Validate training data ratio
        training_data_ratio = self.config["training_data_ratio"]
        if self.config["validate"] and training_data_ratio >= 1.0:
            raise ValueError(
                "training_data_ratio must be < 1.0 when validation is enabled."
            )

        # Feature partitioning for federated learning
        x_train, x_test = self._partition_features(x_train, x_test)

        # Split training and validation data
        x_train, y_train, x_val, y_val = self._split_train_val(
            x_train, y_train, training_data_ratio
        )

        # Apply federated learning modifications (data dropping, class imbalance)
        if self.config.get('modeFL', False):
            x_train, x_val = self._apply_fl_modifications(
                x_train, y_train, x_val, y_val
            )

        # Update number of classes
        self._update_num_classes(y_val)

        # Validate data preprocessing
        self._validate_data(x_val)

        # Select data based on mode
        data, labels = self._select_mode_data(
            x_train, y_train, x_val, y_val, x_test, y_test
        )

        # Apply Pearson correlation reordering if needed
        if not self.ns:
            pearson_order = self._calculate_pearson_order(data)
            data = data[:, pearson_order]

        return data, labels

    def _partition_features(self, x_train, x_test):
        """Partition features across clients for federated learning."""
        np.random.seed(0)  # Consistent permutation across clients
        
        # Trim features to be divisible by number of clients
        n_features = x_train.shape[1]
        n_clients = self.config['fl_cluster']
        n_features_per_client = n_features // n_clients
        total_features = n_features_per_client * n_clients
        
        x_train = x_train[:, :total_features]
        x_test = x_test[:, :total_features]
        
        # Shuffle and partition features
        feat_shuffle = np.random.permutation(total_features)
        min_idx = n_features_per_client * self.client
        max_idx = n_features_per_client * (self.client + 1)
        
        x_train = x_train[:, feat_shuffle][:, min_idx:max_idx]
        x_test = x_test[:, feat_shuffle][:, min_idx:max_idx]
        
        return x_train, x_test

    def _split_train_val(self, x_train, y_train, training_ratio):
        """Split training data into train and validation sets."""
        np.random.seed(np.random.randint(10))
        idx = np.arange(x_train.shape[0])
        
        # Calculate split index (aligned to batch size)
        batch_size = self.config['batch_size']
        n_train = int(len(idx) * training_ratio)
        n_train = (n_train // batch_size) * batch_size + batch_size
        
        tr_idx = idx[:n_train]
        val_idx = idx[n_train:]
        
        x_val = x_train[val_idx]
        y_val = y_train[val_idx]
        x_train = x_train[tr_idx]
        y_train = y_train[tr_idx]
        
        return x_train, y_train, x_val, y_val

    def _apply_fl_modifications(self, x_train, y_train, x_val, y_val):
        """Apply data dropping and class imbalance for federated learning."""
        n_clients = self.config['fl_cluster']
        drop_rate = self.config['client_drop_rate']
        imbalance_rate = self.config['client_imbalance_rate']
        
        # Client data dropping
        if self.client < int(n_clients * drop_rate):
            data_drop = self.config['data_drop_rate']
            n_drop_train = int(x_train.shape[0] * data_drop)
            n_drop_val = int(x_val.shape[0] * data_drop)
            
            x_train[:n_drop_train] = 0
            x_val[:n_drop_val] = 0
        
        # Class imbalance
        elif (int(n_clients * drop_rate) <= self.client < 
              int(n_clients * (drop_rate + imbalance_rate))):
            
            np.random.seed(0)
            n_classes = self.config['n_classes']
            class_imbalance = self.config['class_imbalance']
            
            # Select classes to reduce
            n_reduced_classes = n_classes - int(class_imbalance * n_classes)
            reduced_classes = np.random.choice(
                n_classes, n_reduced_classes, replace=False
            )
            
            # Apply to training data
            train_idx = np.arange(x_train.shape[0])
            affected_idx = train_idx[np.isin(y_train, reduced_classes)]
            n_zero = int(len(affected_idx) * class_imbalance)
            zero_idx = np.random.choice(affected_idx, size=n_zero, replace=False)
            x_train[zero_idx] = 0
            
            # Apply to validation data
            val_idx = np.arange(x_val.shape[0])
            affected_idx = val_idx[np.isin(y_val, reduced_classes)]
            n_zero = int(len(affected_idx) * class_imbalance)
            zero_idx = np.random.choice(affected_idx, size=n_zero, replace=False)
            # x_val[zero_idx] = 0  # Uncomment if needed
        
        return x_train, x_val

    def _update_num_classes(self, y_val):
        """Update number of classes in config if needed."""
        n_classes = len(np.unique(y_val))
        if self.config["n_classes"] != n_classes:
            print(f"{'>' * 10} Number of classes changed from "
                  f"{self.config['n_classes']} to {n_classes} {'<' * 10}")
            self.config["n_classes"] = n_classes

    def _validate_data(self, x_val):
        """Check if data preprocessing is correct."""
        max_val = np.max(np.abs(x_val))
        if max_val > 20:
            raise ValueError(
                f"Data preprocessing may be incorrect. "
                f"Max absolute value in features is {max_val}. "
                f"Expected values should be normalized (typically < 20)."
            )

    def _select_mode_data(self, x_train, y_train, x_val, y_val, x_test, y_test):
        """Select appropriate data based on mode."""
        mode_map = {
            "train_fl": (x_train, y_train),
            "validation": (x_val, y_val),
            "test": (x_test, y_test),
        }
        
        if self.mode not in mode_map:
            raise ValueError(
                f"Invalid mode '{self.mode}'. "
                f"Use one of: {list(mode_map.keys())}"
            )
        
        return mode_map[self.mode]

    def _calculate_pearson_order(self, data):
        """Calculate Pearson correlation ordering for features."""
        correlation_matrix = np.corrcoef(data.T)
        return np.argsort(correlation_matrix[0])

    # Dataset loading methods
    def _load_mnist(self):
        """Load MNIST dataset."""
        mnist_path = "./data/mnist"
        
        with open(f"{mnist_path}/train.npy", 'rb') as f:
            x_train = np.load(f)
            y_train = np.load(f)
        
        with open(f"{mnist_path}/test.npy", 'rb') as f:
            x_test = np.load(f)
            y_test = np.load(f)
        
        # Flatten and normalize
        x_train = x_train.reshape(-1, 28 * 28) / 255.0
        x_test = x_test.reshape(-1, 28 * 28) / 255.0
        
        return x_train, y_train, x_test, y_test

    def _load_blog(self):
        """Load Blog dataset."""
        x_train = np.load("./data/blog/xtrain.npy")
        y_train = np.load("./data/blog/ytrain.npy")
        x_test = np.load("./data/blog/xtest.npy")
        y_test = np.load("./data/blog/ytest.npy")
        
        return x_train, y_train, x_test, y_test

    def _load_income(self):
        """Load Income dataset."""
        x_train = np.load("./data/income/train_feat_std.npy")
        y_train = np.load("./data/income/train_label.npy")
        x_test = np.load("./data/income/test_feat_std.npy")
        y_test = np.load("./data/income/test_label.npy")
        
        return x_train, y_train, x_test, y_test

    def _load_cifar(self):
        """Load CIFAR-10 dataset and convert to grayscale."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), 
                (0.247, 0.243, 0.261)
            )
        ])
        
        train_dataset = dts.CIFAR10(
            root="./data/", train=True, download=False, transform=transform
        )
        test_dataset = dts.CIFAR10(
            root="./data/", train=False, download=False, transform=transform
        )
        
        x_train, y_train = zip(*train_dataset)
        x_test, y_test = zip(*test_dataset)
        
        # Convert to grayscale and flatten
        x_train = np.array([self._rgb_to_grey(img) for img in x_train])
        x_test = np.array([self._rgb_to_grey(img) for img in x_test])
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        
        return x_train, y_train, x_test, y_test

    def _load_syn(self):
        """Load synthetic dataset."""
        with h5py.File('./data/syn/syn.hdf5', 'r') as f:
            np.random.seed(25)
            n_samples = f['labels'].shape[0]
            idx = np.arange(n_samples)
            np.random.shuffle(idx)
            
            split_idx = int(n_samples * 0.9)
            
            x_train = f['features'][:].T[idx][:split_idx]
            y_train = f['labels'][:][idx][:split_idx]
            x_test = f['features'][:].T[idx][split_idx:]
            y_test = f['labels'][:][idx][split_idx:]
        
        return x_train, y_train, x_test, y_test

    def _load_covtype(self):
        """Load Forest Covertype dataset."""
        cov_type = fetch_covtype()
        X = normalize(cov_type.data, norm="l1")
        y = cov_type.target
        
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=1
        )
        
        return x_train, y_train, x_test, y_test

    @staticmethod
    def _rgb_to_grey(rgb_tensor):
        """Convert RGB image tensor to grayscale and flatten."""
        r, g, b = rgb_tensor[0], rgb_tensor[1], rgb_tensor[2]
        grey = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return grey.flatten().numpy()
