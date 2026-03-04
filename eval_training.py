# https://github.com/AstraZeneca/SubTab


import copy
import eval
from utils.arguments import get_arguments, get_config, print_config_summary

import json
from pathlib import Path
import numpy as np



def main(config):
    
    config["framework"] = config["dataset"]
    config['task_type'] = json.loads(Path('data/'+config["dataset"]+'/info.json').read_text())['task_type']
    config['cat_policy'] = json.loads(Path('data/'+config["dataset"]+'/info.json').read_text())['cat_policy']
    config['norm'] = json.loads(Path('data/'+config["dataset"]+'/info.json').read_text())['norm']
    config['learning_rate_reducer'] = config['learning_rate']
    # Get a copy of autoencoder dimensions
    # Disable adding noise since we are in evaluation mode
    config["add_noise"] = False
    # Turn off valiation
    config["validate"] = False
    config["original_dataset"] = False
    # Get all of available training set for evaluation (i.e. no need for validation set)
    # config["training_data_ratio"] = 0.1

    all_results = []
    for client in range(config["fl_cluster"]):

        results = eval.main(config, client)
        all_results.append(results)


    oriPrecL, oriRecL, oriFL = [], [], []
    embedPrecL, embedRecL, embedFL = [], [], []

    for item in all_results:
        print(item)
        ori, embed = item
        if ori is not None:
            oriPrec, oriRec, oriF, _ = ori
            oriPrecL.append(oriPrec)
            oriRecL.append(oriRec)
            oriFL.append(oriF)
        
        if embed is not None:
            embedPrec, embedRec, embedF, _ = embed
            embedPrecL.append(embedPrec)
            embedRecL.append(embedRec)
            embedFL.append(embedF)

    # Calculate mean for embed results
    if embedPrecL and embedRecL and embedFL:
        embed_mean = (
            f'Mean of embed results.\n \
            Precision : {np.mean(np.array(embedPrecL))} \n \
            Recall : {np.mean(np.array(embedRecL))} \n \
            F1 : {np.mean(np.array(embedFL))} '
        )
    else:
        embed_mean = "No valid embed results to calculate the mean."

    # Calculate mean for original results
    if oriPrecL and oriRecL and oriFL:
        ori_mean = (
            f'Mean of original results. \n \
            Precision : {np.mean(np.array(oriPrecL))} \n \
            Recall : {np.mean(np.array(oriRecL))} \n \
            F1 : {np.mean(np.array(oriFL))} '
        )
    else:
        ori_mean = "No valid original results to calculate the mean."

    print(embed_mean)
    print(ori_mean)


if __name__ == "__main__":
    # Get parser / command line arguments
    args = get_arguments()
    # Get configuration file
    config = get_config(args)
    # Overwrite the parent folder name for saving results
    # config["framework"] = config["dataset"]
    # Get a copy of autoencoder dimensions
    # dims = copy.deepcopy(config["dims"])
    config['sampling']=True

    # print_config_summary(config, args)
    main(config)
    

