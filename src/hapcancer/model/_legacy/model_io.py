import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

from typing import List, Optional, Union
from hapcancer.model.dataload.load_input import load_input

def set_trainable(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def configure_encoder_freeze(
    encoder: nn.Module,
    train_last_n_layers: int = 0,          # 0 = fully frozen; 1 = last layer; 2 = last two; ...
    train_layernorms: bool = True,         # keep LayerNorms trainable (cheap + stabilizing)
    train_input_proj: bool = False,        # if your input projection needs adapting
    train_time_encoding: bool = False,     # if your TimeEncoding has learnable params
    train_attention_pooling: bool = True   # attention_pooling is small; often worth training
):
    """
    Freezes the encoder and selectively unfreezes requested parts.
    Works with your MammogramTransformerEncoder wrapper.
    """
    # 1) Freeze everything in the encoder
    set_trainable(encoder, False)

    # 2) Optionally keep small, helpful parts trainable
    if train_input_proj and hasattr(encoder, "input_proj"):
        set_trainable(encoder.input_proj, True)

    if train_time_encoding and hasattr(encoder, "time_encoding"):
        set_trainable(encoder.time_encoding, True)

    if train_attention_pooling and hasattr(encoder, "attention_pooling"):
        set_trainable(encoder.attention_pooling, True)

    # 3) Unfreeze last N transformer blocks (if requested)
    # PyTorch's TransformerEncoder stores layers in encoder.transformer.layers (ModuleList)
    layers = getattr(encoder.transformer, "layers", None)
    if layers is None:
        raise AttributeError("Could not find encoder.transformer.layers (unexpected PyTorch version/structure).")

    if train_last_n_layers > 0:
        for block in layers[-train_last_n_layers:]:
            set_trainable(block, True)

    # 4) (Optional) always let LayerNorms learn—tiny parameter count, often beneficial
    if train_layernorms:
        for m in encoder.modules():
            if isinstance(m, nn.LayerNorm):
                set_trainable(m, True)
    
    


