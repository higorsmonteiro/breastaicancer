import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union, List
from sklearn.metrics import roc_auc_score, average_precision_score

#from hapcancer.model.model import build_model_singleyear
#
#def load_pretrained_singleyear(config):
#    pretrained_checkpoint_name = config['misc']['pretrained']['model_name']
#    file_name = config['misc']['pretrained']['file_name']
#    device  = config['misc']['device']
#    checkpoint_path = Path(config['misc']['checkpoint_path'])
#    
#    model = build_model_singleyear(config, device).to(device)
#
#    pretrained_path = checkpoint_path.joinpath(pretrained_checkpoint_name)
#    loaded_state = torch.load(pretrained_path.joinpath(file_name), map_location=device)
#    model.load_state_dict(loaded_state['model_state_dict'])
#    return model


def set_outer_lr(
    optimizer: torch.optim.Optimizer, 
    step: int, 
    warmup_steps: int, 
    target_lr: float, 
    verbose: Optional[bool] = False):
    if step < warmup_steps:
        lr = target_lr * (step + 1) / warmup_steps
    else:
        lr = target_lr
    for g in optimizer.param_groups:
        g['lr'] = lr
    if verbose:
        print(f"lr: {lr}")

def calculate_metrics(
    all_labels: Union[List, np.ndarray], 
    all_scores: Union[List, np.ndarray]
):
    '''
        ...
    '''
    if len(all_labels) > 0:
        y_true  = torch.cat(all_labels).numpy()
        y_score = torch.cat(all_scores).numpy()
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_score)
            ap  = average_precision_score(y_true, y_score)
        else:
            auc, ap = np.nan, np.nan
    else:
        auc, ap = np.nan, np.nan
    return auc, ap