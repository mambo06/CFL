"""
Contrastive Federated Learning (CFL) Model
Trains an Autoencoder with a Projection network using SubTab framework.
Based on: https://github.com/AstraZeneca/SubTab
"""

import gc
import itertools
import os

import numpy as np
import pandas as pd
import torch as th

from utils.loss_functionsV1 import JointLoss
from utils.model_plot import save_loss_plot
from utils.model_utils import AEWrapper
from utils.utils import set_seed, set_dirs

th.autograd.set_detect_anomaly(True)


class CFL:
    """
    Contrastive Federated Learning Model.
    Trains an Autoencoder with projection network using SubTab framework.
    """

    def __init__(self, options):
        """
        Initialize CFL model.

        Args:
            options (dict): Configuration dictionary containing model parameters.
        """
        self.options = options
        self.device = options["device"]
        self.model_dict = {}
        self.summary = {}
        
        # Set random seed for reproducibility
        set_seed(self.options)
        
        # Set up paths and directories
        self._set_paths()
        set_dirs(self.options)
        
        # Check if we need combinations of projections
        self.is_combination = (
            self.options["contrastive_loss"] or 
            self.options["distance_loss"]
        )
        
        # Initialize network
        print("Building CFL Network...")
        self.set_autoencoder()
        self._set_scheduler()
        
        # Initialize loss tracking
        self.loss = {
            "tloss_b": [],  # Total loss per batch
            "tloss_e": [],  # Total loss per epoch
            "vloss_e": [],  # Validation loss per epoch
            "closs_b": [],  # Contrastive loss per batch
            "rloss_b": [],  # Reconstruction loss per batch
            "zloss_b": [],  # Z-space loss per batch
            "tloss_o": []   # Original total loss
        }

    def set_autoencoder(self):
        """Set up the autoencoder model, optimizer, and loss function."""
        # Instantiate autoencoder
        self.encoder = AEWrapper(self.options)
        self.model_dict["encoder"] = self.encoder
        
        # Move model to device (GPU/CPU)
        for model in self.model_dict.values():
            model.to(self.device)
        
        # Set up joint loss function
        self.joint_loss = JointLoss(self.options)
        
        # Set up optimizer
        parameters = [model.parameters() for model in self.model_dict.values()]
        self.optimizer_ae = self._adam(parameters, lr=self.options["learning_rate"])
        
        # Update summary
        self.summary["recon_loss"] = []

    def fit(self, x):
        """
        Fit model to a batch of data.

        Args:
            x (torch.Tensor): Input batch data.

        Returns:
            tuple: (total_loss, contrastive_loss, reconstruction_loss, z_loss)
        """
        self.set_mode(mode="training")
        
        # Concatenate original data with itself for reconstruction comparison
        x_orig = self.process_batch(x, x)
        
        # Generate subsets with added noise
        x_tilde_list = self.subset_generator(x, mode="train")
        
        # Get combinations of subsets if needed
        if self.is_combination:
            x_tilde_list = self.get_combinations_of_subsets(x_tilde_list)
        
        # Calculate losses
        tloss, closs, rloss, zloss = self.calculate_loss(x_tilde_list, x_orig)
        
        return tloss, closs, rloss, zloss

    def calculate_loss(self, x_tilde_list, x_orig):
        """
        Calculate losses for all subsets.

        Args:
            x_tilde_list (list): List of corrupted subsets.
            x_orig (torch.Tensor): Original uncorrupted data.

        Returns:
            tuple: Average losses (total, contrastive, reconstruction, z-space)
        """
        total_loss, contrastive_loss, recon_loss, zrecon_loss = [], [], [], []

        # Process each subset
        for xi in x_tilde_list:
            # Prepare input
            x_input = xi if self.is_combination else self.process_batch(xi, xi)
            
            # Forward pass
            z, latent, x_recon = self.encoder(x_input)
            
            # Use subset as target if reconstruct_subset is enabled
            if self.options["reconstruction"] and self.options["reconstruct_subset"]:
                x_target = x_input
            else:
                x_target = x_orig
            
            # Compute losses
            tloss, closs, rloss, zloss = self.joint_loss(z, x_recon, x_target)
            
            # Accumulate losses
            total_loss.append(tloss)
            contrastive_loss.append(closs)
            recon_loss.append(rloss)
            zrecon_loss.append(zloss)

        # Compute average losses
        n = len(total_loss)
        avg_total_loss = sum(total_loss) / n
        avg_contrastive_loss = sum(contrastive_loss) / n
        avg_recon_loss = sum(recon_loss) / n
        avg_zrecon_loss = sum(zrecon_loss) / n

        return avg_total_loss, avg_contrastive_loss, avg_recon_loss, avg_zrecon_loss

    def validate(self, x):
        """
        Compute validation loss.

        Args:
            x (torch.Tensor): Validation batch data.

        Returns:
            float: Average validation loss.
        """
        with th.no_grad():
            self.set_mode(mode="evaluation")
            
            # Generate subsets
            x_tilde_list = self.subset_generator(x, mode="test")
            
            # Get combinations if needed
            if self.is_combination:
                x_tilde_list = self.get_combinations_of_subsets(x_tilde_list)
            
            # Prepare original data
            x_orig = self.process_batch(x, x)
            
            # Calculate validation losses
            val_losses = []
            for xi in x_tilde_list:
                x_input = xi if self.is_combination else self.process_batch(xi, xi)
                z, latent, x_recon = self.encoder(x_input)
                val_loss, _, _, _ = self.joint_loss(z, x_recon, x_orig)
                val_losses.append(val_loss)
            
            # Return average validation loss
            return sum(val_losses) / len(val_losses)

    def subset_generator(self, x, mode="test", skip=None):
        """
        Generate subsets with added noise.

        Args:
            x (torch.Tensor): Input data to be divided into subsets.
            mode (str): 'train' or 'test' mode.
            skip (list): List of subset indices to skip (unused currently).

        Returns:
            list: List of corrupted subsets.
        """
        if skip is None:
            skip = [-1]
        
        n_subsets = self.options["n_subsets"]
        
        # Create copies of data for each subset
        subset_list = [x.clone() for _ in range(n_subsets)]
        
        # Add noise to each subset
        x_tilde_list = []
        use_high_masking = False
        
        for subset in subset_list:
            use_high_masking = not use_high_masking
            x_bar = subset
            
            # Add noise if enabled
            if self.options["add_noise"]:
                x_bar_noisy = self.generate_noisy_xbar(x_bar)
                
                # Generate binary mask with varying masking ratio
                p_m = self.options["masking_ratio"]
                if not use_high_masking:
                    p_m = 1 - p_m
                
                mask = np.random.binomial(1, p_m, x_bar.shape)
                
                # Apply mask
                x_bar = x_bar * (1 - mask) + x_bar_noisy * mask
            
            x_tilde_list.append(x_bar)
        
        return x_tilde_list

    def generate_noisy_xbar(self, x):
        """
        Generate noisy version of input data.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            np.ndarray: Corrupted version of input.
        """
        n_samples, n_features = x.shape
        noise_type = self.options["noise_type"]
        noise_level = self.options["noise_level"]
        
        x_bar = np.zeros([n_samples, n_features])
        
        # Swap noise: randomly shuffle data column-wise
        if noise_type == "swap_noise":
            for i in range(n_features):
                idx = np.random.permutation(n_samples)
                x_bar[:, i] = x[idx, i].cpu().numpy()
        
        # Gaussian noise
        elif noise_type == "gaussian_noise":
            x_np = x.cpu().numpy()
            x_bar = x_np + np.random.normal(
                float(th.mean(x)), noise_level, x.shape
            )
        
        # Zero-out noise
        else:
            x_bar = x_bar
        
        return x_bar

    def get_combinations_of_subsets(self, x_tilde_list):
        """
        Generate combinations of subsets.

        Args:
            x_tilde_list (list): List of subsets [x1, x2, x3, ...].

        Returns:
            list: List of concatenated subset pairs [(x1,x2), (x1,x3), ...].
        """
        # Get all pairwise combinations
        subset_combinations = list(itertools.combinations(x_tilde_list, 2))
        
        # Concatenate each pair
        concatenated_subsets = []
        for xi, xj in subset_combinations:
            x_batch = self.process_batch(xi, xj)
            concatenated_subsets.append(x_batch)
        
        return concatenated_subsets

    def process_batch(self, xi, xj):
        """
        Concatenate two inputs and convert to tensor on device.

        Args:
            xi (torch.Tensor): First input.
            xj (torch.Tensor): Second input.

        Returns:
            torch.Tensor: Concatenated batch on device.
        """
        x_batch = th.cat((xi, xj), dim=0)
        return self._tensor(x_batch)

    def save_train_params(self, client):
        """
        Save training parameters and loss plots.

        Args:
            client (int): Client ID.
        """
        prefix = self._generate_filename_prefix(client)
        
        # Save loss plot
        save_loss_plot(self.loss, self._plots_path, prefix)
        
        # Save loss data as CSV
        loss_df = pd.DataFrame(dict([
            (k, pd.Series(v)) for k, v in self.loss.items()
        ]))
        loss_df.to_csv(f"{self._loss_path}/{prefix}-losses.csv")

    def save_weights(self, client):
        """
        Save model weights.

        Args:
            client (int): Client ID.
        """
        prefix = self._generate_filename_prefix(client)
        
        for model_name, model in self.model_dict.items():
            save_path = f"{self._model_path}/{model_name}_{prefix}.pt"
            th.save(model, save_path)
        
        print("✓ Model weights saved successfully.")

    def load_models(self, client):
        """
        Load saved model weights.

        Args:
            client (int): Client ID.
        """
        prefix = self._generate_filename_prefix(client)
        
        for model_name in self.model_dict:
            load_path = f"{self._model_path}/{model_name}_{prefix}.pt"
            model = th.load(
                load_path, 
                map_location=self.device, 
                weights_only=False
            )
            setattr(self, model_name, model.eval())
            print(f"✓ {model_name} loaded successfully.")
        
        print("✓ All models loaded successfully.")

    def set_mode(self, mode="training"):
        """
        Set model mode (training or evaluation).

        Args:
            mode (str): 'training' or 'evaluation'.
        """
        for model in self.model_dict.values():
            if mode == "training":
                model.train()
            else:
                model.eval()

    def _generate_filename_prefix(self, client):
        """
        Generate filename prefix for saving/loading.

        Args:
            client (int): Client ID.

        Returns:
            str: Filename prefix.
        """
        config = self.options
        mode = "local" if config["local"] else "FL"
        
        prefix = (
            f"Client-{client}-"
            f"{config['epochs']}e-"
            f"{config['fl_cluster']}c-"
            f"{config['client_drop_rate']}cd-"
            f"{config['data_drop_rate']}dd-"
            f"{config['client_imbalance_rate']}nc-"
            f"{config['class_imbalance']}ci-"
            f"{config['dataset']}-"
            f"{mode}"
        )
        
        return prefix

    def _set_paths(self):
        """Set up directory paths for saving results."""
        results_path = os.path.join(
            self.options["paths"]["results"], 
            self.options["framework"]
        )
        model_mode = self.options["model_mode"]
        
        self._results_path = results_path
        self._model_path = os.path.join(
            results_path, "training", model_mode, "model"
        )
        self._plots_path = os.path.join(
            results_path, "training", model_mode, "plots"
        )
        self._loss_path = os.path.join(
            results_path, "training", model_mode, "loss"
        )

    def _set_scheduler(self):
        """Set up learning rate scheduler."""
        self.scheduler = th.optim.lr_scheduler.StepLR(
            self.optimizer_ae, step_size=1, gamma=0.99
        )

    def _adam(self, params, lr=1e-4):
        """
        Set up AdamW optimizer.

        Args:
            params (list): List of model parameters.
            lr (float): Learning rate.

        Returns:
            torch.optim.AdamW: Optimizer instance.
        """
        return th.optim.AdamW(
            itertools.chain(*params), 
            lr=lr, 
            betas=(0.9, 0.999), 
            eps=1e-07
        )

    def _tensor(self, data):
        """
        Convert numpy array to torch tensor on device.

        Args:
            data (np.ndarray or torch.Tensor): Input data.

        Returns:
            torch.Tensor: Data as tensor on device.
        """
        if type(data).__module__ == np.__name__:
            data = th.from_numpy(np.float32(data))
        
        return data.to(self.device).float()

    def print_model_summary(self):
        """Display model architecture."""
        print("=" * 80)
        print(f"{'CFL MODEL ARCHITECTURE':^80}")
        print("=" * 80)
        print(f"\nMode: {self.options['model_mode'].upper()}\n")
        print(self.encoder)
        print("=" * 80)
