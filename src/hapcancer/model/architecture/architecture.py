'''
    Define the architecture's modules for the final model.
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, List

from sklearn.preprocessing import StandardScaler

# =======================================================================
# ============================= TRANSFORMER =============================
# =======================================================================

class TimeEncoding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, timestamps):
        # timestamps: (B, T, 1), por ex. dias antes do atual
        x = torch.log1p(torch.clamp(timestamps, min=0.0))
        return self.mlp(x)
    
class AttentionPooling(nn.Module):
    '''
        ...
    '''
    def __init__(self, embed_dim):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, encoded_seq, attention_mask):
        """
            ...
        """
        B, T, D = encoded_seq.shape
        scores = self.scorer(encoded_seq).squeeze(-1)  # (B, T)

        # mask para -inf
        mask = ~attention_mask.bool()
        scores = scores.masked_fill(mask, float("-inf"))

        # caso B,T todo mascarado -> defina pesos=0
        attn = torch.softmax(scores, dim=1)
        attn = torch.where(torch.isfinite(attn), attn, torch.zeros_like(attn))

        pooled = torch.bmm(attn.unsqueeze(1), encoded_seq).squeeze(1)  # (B, D)
        return pooled
    
def _causal_mask(T, device):
    # máscara triangular superior True = bloquear
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

class MammogramTransformerEncoder(nn.Module):
    def __init__(self, input_dim=1024, embed_dim=64, num_heads=4, num_layers=3, dropout=0.1,
                 use_causal=False):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.time_encoding = TimeEncoding(embed_dim)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention_pooling = AttentionPooling(embed_dim)
        self.use_causal = use_causal

    def forward(self, mammogram_seq, timestamps, attention_mask):
        """
            mammogram_seq: (B, T, input_dim)
            timestamps:    (B, T, 1)  dias antes do atual
            attention_mask:(B, T)     1=real, 0=padding
            return: (B, D)
        """
        B, T, _ = mammogram_seq.shape
        x = self.input_proj(mammogram_seq) + self.time_encoding(timestamps)
        x = self.input_dropout(x)

        # key padding mask: True = posição a ignorar
        key_padding_mask = ~attention_mask.bool()  # (B, T)

        # -- do not use yet (I do not understand completely this)
        src_mask = _causal_mask(T, x.device) if self.use_causal else None

        # Transformer
        x = self.transformer(x, mask=src_mask, src_key_padding_mask=key_padding_mask)

        # Pooling atento
        pooled = self.attention_pooling(x, attention_mask)
        return pooled
    
# ========================================================================
# ============================ PREDICTION MLP ============================
# ========================================================================

class DynamicMLP(nn.Module):
    '''
    
    '''
    def __init__(
        self, 
        input_dim: int, 
        hidden_layers: List[int], 
        dropout: Optional[float] = 0.0, 
        activation: Optional[str] = 'relu', 
        use_batchnorm: Optional[bool] = False, 
        final_layer: Optional[bool]= True, 
        sigmoid: Optional[bool] = True
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        for h in hidden_layers:
            layers.append(nn.Linear(current_dim, h))

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))

            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'mish':
                layers.append(nn.Mish())
            else:
                raise ValueError(f"Unsupported activation: {activation}")

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            current_dim = h

        if final_layer: # -- if this MLP is a shared module, then it should be false
            layers.append(nn.Linear(current_dim, 1)) # -- output layer
            if sigmoid:
                layers.append(nn.Sigmoid()) # -- binary classification
        self.mlp = nn.Sequential(*layers)
        self.output_dim = current_dim

    def forward(self, x):
        z = self.mlp(x)
        return z
    

# ======================================================================================
# ============================ PREDICTION MLP - SINGLE YEAR ============================
# ======================================================================================

class CancerRiskSingleYearWithMLP(nn.Module):
    '''
        Architecture for cancer prediction with a transformer encoder for the sequence
        of past mammograms.

        So far, this architecture seems to be very expensive for training.

        Args:
        -----
            encoder:
            embed_dim:
            extra_feature_dim:
            mlp_config:
            sigmoid:
            device:

        Forward Args:
        -------------
            mammogram_seq:
            time_diffs:
            attention_mask:
            extra_features:
    '''
    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int,
        extra_feature_dim: int,
        mlp_config: dict,
        sigmoid: bool,
        device: Optional[str] = 'cpu'
    ):
        super().__init__()
        self.encoder = encoder.to(device)

        input_dim = embed_dim + extra_feature_dim
        self.mlp = DynamicMLP(
            input_dim=input_dim,
            hidden_layers=mlp_config['hidden_layers'],
            dropout=mlp_config.get('dropout', 0.0),
            activation=mlp_config.get('activation', 'relu'),
            use_batchnorm=mlp_config.get('use_batchnorm', False),
            sigmoid=sigmoid
        )

    def forward(self, mammogram_seq, time_diffs, attention_mask, extra_features):
        encoded_rep = self.encoder(mammogram_seq, time_diffs, attention_mask)
        if torch.isnan(encoded_rep).any():
            print("NaNs found in encoder output")
        combined_rep = torch.cat([encoded_rep, extra_features], dim=-1)
        risk_score = self.mlp(combined_rep)
        return risk_score

class CancerRiskSingleYearWithMLPNoTransformer(nn.Module):
    '''
        Architecture for cancer prediction without a transformer encoder.

        The model is a regular MLP network where the inputs are a vector
        representing the mammogram sequence plus the vector of extra features
        (risk factors + demographic).

        Args:
        -----
            embed_dim:
            extra_feature_dim:
            mlp_config:
            sigmoid:
            device:

        Forward Args:
        -------------
            mammogram_seq_vec:
            extra_features:
    '''
    def __init__(
        self,
        embed_dim: int,
        extra_feature_dim: int,
        mlp_config: dict,
        sigmoid: bool,
        device: Optional[str] = 'cpu'
    ):
        super().__init__()
        self.input_dim = embed_dim + extra_feature_dim + 1 # +1 for the variable delta_t in the mean mammogram vector
        self.mlp = DynamicMLP(
            input_dim=self.input_dim,
            hidden_layers=mlp_config['hidden_layers'],
            dropout=mlp_config.get('dropout', 0.0),
            activation=mlp_config.get('activation', 'relu'),
            use_batchnorm=mlp_config.get('use_batchnorm', False),
            sigmoid=sigmoid
        )
        self.scaler = StandardScaler()

    def forward(
        self,
        mammogram_seq_vec, 
        extra_features
    ):
        #extra_features = torch.tensor(self.scaler.fit_transform(extra_features), dtype=torch.float32)
        combined_rep = torch.cat([mammogram_seq_vec, extra_features], dim=-1)
        risk_score = self.mlp(combined_rep) # (B,1)
        return risk_score


