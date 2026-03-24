import copy
import yaml
from tqdm import tqdm
from pathlib import Path
import json

import numpy as np
import torch

import eval_training as eval
from src.model import CFL
from utils.arguments import get_arguments, get_config
from utils.load_data_new import Loader
from utils.utils import update_config_with_model_dims


class Server:
    """Federated Learning Server for model aggregation."""
    
    def __init__(self, model):
        self.global_model = model
        self.global_dict = self.global_model.encoder.state_dict()
        
    def aggregate_models(self, client_models):
        """FedAvg aggregation of client models."""
        for k in self.global_dict.keys():
            self.global_dict[k] = torch.stack([
                client.get_model_params()[k].float() 
                for client in client_models 
                if client.tloss is not None
            ]).mean(0)
        
    def distribute_model(self):
        """Return global model parameters."""
        return self.global_dict


class Client:
    """Federated Learning Client."""
    
    def __init__(self, model, dataloader, client_number):
        self.model = copy.deepcopy(model)
        self.dataloader = copy.deepcopy(dataloader)
        self.client_number = copy.deepcopy(client_number)
        self.tloss = None
        
    def train(self, shuffle_seed=None):
        """Train client model on one batch."""
        x, y = next(iter(self.dataloader))

        # Shuffle data if seed provided
        if shuffle_seed is not None:
            np.random.seed(shuffle_seed)
            idx = np.random.permutation(x.shape[0])
            x = x[idx]
            y = y[idx]

        self.model.optimizer_ae.zero_grad()
        tloss, closs, rloss, zloss = self.model.fit(x)

        # Log losses
        self.model.loss["tloss_o"].append(tloss.item())
        self.model.loss["tloss_b"].append(tloss.item())
        self.model.loss["closs_b"].append(closs.item())
        self.model.loss["rloss_b"].append(rloss.item())
        self.model.loss["zloss_b"].append(zloss.item())

        tloss.backward()
        self.tloss = tloss
        return tloss
    
    def get_model_params(self):
        """Return client model parameters."""
        return copy.deepcopy(self.model.encoder.state_dict())

    def step(self):
        """Perform optimizer step."""
        self.model.optimizer_ae.step()

    def set_model(self, params):
        """Update client model with new parameters."""
        self.model.encoder.load_state_dict(params)


def run(config, save_weights=True):
    """Main federated learning training loop."""
    
    # Initialize data loader and model
    ds_loader = Loader(config, dataset_name=config["dataset"], client=0)
    config = update_config_with_model_dims(ds_loader, config)
    global_model = CFL(config)
    
    # Initialize server
    server = Server(global_model)
    
    # Initialize clients
    clients = []
    for clt in range(config["fl_cluster"]):
        loader = Loader(config, dataset_name=config["dataset"], client=clt).trainFL_loader
        client = Client(global_model, loader, clt)
        clients.append(client)

    total_batches = len(loader)
    
    # Training loop
    for epoch in range(config["epochs"]):
        epoch_loss = 0
        
        for batch_idx in tqdm(range(total_batches), desc=f"Epoch {epoch+1}/{config['epochs']}"):
            # Train each client
            for client in clients:
                loss = client.train(shuffle_seed=epoch)
                epoch_loss += loss.item()
                client.step()
            
            # Aggregate models
            server.aggregate_models(clients)

            # Distribute updated model to clients
            for client in clients:
                client.set_model(server.distribute_model())
                
        # Log epoch loss
        avg_epoch_loss = epoch_loss / (config['fl_cluster'] * total_batches)
        print(f'Epoch {epoch+1} Loss: {avg_epoch_loss:.6f}')

    # Save client models and configs
    for n, client in enumerate(clients):
        model = client.model
        model.saveTrainParams(n)
        
        if save_weights:
            model.save_weights(n)

        # Generate filename prefix
        prefix = (
            f"Client-{n}-{config['epochs']}e-{config['fl_cluster']}c-"
            f"{config['client_drop_rate']}cd-{config['data_drop_rate']}dd-"
            f"{config['client_imbalance_rate']}nc-{config['class_imbalance']}ci-"
            f"{config['dataset']}-"
        )
        prefix += "local" if config["local"] else "FL"

        # Save config
        config_path = f"{model._results_path}/config_{prefix}.yml"
        with open(config_path, 'w') as config_file:
            yaml.dump(config, config_file, default_flow_style=False)


def main(config):
    """Setup configuration and run training."""
    
    # Load dataset info
    dataset_info_path = Path(f'data/{config["dataset"]}/info.json')
    dataset_info = json.loads(dataset_info_path.read_text())
    
    # Update config with dataset info
    config["framework"] = config["dataset"]
    config['task_type'] = dataset_info['task_type']
    config['cat_policy'] = dataset_info['cat_policy']
    config['norm'] = dataset_info['norm']
    config['learning_rate_reducer'] = config['learning_rate']
    config['modeFL'] = True
    config['sampling'] = True
    
    # Run training
    cfg = copy.deepcopy(config)
    run(config, save_weights=True)
    
    # Run evaluation
    eval.main(cfg)


if __name__ == "__main__":
    # Get arguments and config
    args = get_arguments()
    config = get_config(args)
    
    # Run main
    main(config)
