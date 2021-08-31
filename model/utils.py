import torch
import torch.nn as nn
from utils import log_info
import torch.nn.functional as F


def load_weights(network, load_weights):
    log_info(f'LOADING WEIGHTS {load_weights}')
    if torch.cuda.is_available():
        weights = torch.load(load_weights)
    else:
        weights = torch.load(load_weights, map_location='cpu')
    if 'model' in weights.keys():
        weights = weights['model']  # for BTS, weights.keys()==dict_keys(['global_step', 'model', 'optimizer'])
    # ---- automatically check DataParallel ---
    if list(weights.keys())[0].find('module') != -1:
        network = nn.DataParallel(network)
    # --------------------
    weights = {k: v for k, v in weights.items() if "post_process_layer" not in k}
    network.load_state_dict(weights)