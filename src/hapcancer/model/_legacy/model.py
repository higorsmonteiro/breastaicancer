'''
    Define the model's architecture.
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, List

from hapcancer.model.dataload.datasets import CancerDatasetSingleYearTFIDF
from hapcancer.model.architecture import CancerRiskSingleYearWithMLPNoTransformer

class TimeEncoding(nn.Module):
    """
        Exponential decay time encoding: time-dependent contributions should fade smoothly as they become older.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Linear(1, embed_dim)
        self.decay_factor = nn.Parameter(torch.ones(embed_dim))  # learnable decay factor

    def forward(self, timestamps):
        return self.linear(timestamps) * torch.exp(-self.decay_factor * timestamps)
    
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
    def __init__(self, embed_dim):
        """
            Self-Attention pooling module to learn importance weights for each time step.
            - embed_dim: Size of the Transformer embeddings.
        """
        super().__init__()
        self.attention_weights = nn.Linear(embed_dim, 1)  # Linear layer to compute scores

    def forward(self, encoded_seq, attention_mask):
        """
            Computes a weighted sum of encoded mammograms based on learned attention scores.
        
            Inputs:
            - encoded_seq: (seq_len, batch_size, embed_dim) → Output from Transformer
            - attention_mask: (batch_size, seq_len) → 1 for real tokens, 0 for padding

            Returns:
            - pooled_rep: (batch_size, embed_dim) → Weighted sum of mammogram embeddings
        """
        # -- compute raw attention scores for each time step
        attention_scores = self.attention_weights(encoded_seq).squeeze(-1)  # (seq_len, batch_size)

        # -- apply mask: set scores to very low for padding tokens
        attention_scores = attention_scores.masked_fill(attention_mask.T == 0, -1e9)

        # -- compute softmax over time axis (seq_len)
        attention_weights = torch.softmax(attention_scores, dim=0)  # (seq_len, batch_size)

        # -- compute weighted sum of encoded mammograms
        pooled_rep = (attention_weights.unsqueeze(-1) * encoded_seq).sum(dim=0)  # (batch_size, embed_dim)
        return pooled_rep
    
class AttentionPooling(nn.Module):
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
        encoded_seq: (B, T, D)
        attention_mask: (B, T) 1=real, 0=padding (bool ou byte)
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


class MammogramTransformerEncoder(nn.Module):
    def __init__(self, input_dim=128, embed_dim=64, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.time_encoding = TimeEncoding(embed_dim) 

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention_pooling = AttentionPooling(embed_dim)

    def forward(self, mammogram_seq, timestamps, attention_mask):
        """
            Encodes the mammogram sequence using a Transformer.

            Inputs:
            - mammogram_seq: (batch_size, sequencia_len, input_dim)
            - timestamps: (batch_size, sequencia_len, 1) → Days before current mammogram
            - attention_mask: (batch_size, sequencia_len) (1 = real, 0 = padding)

            Returns:
            - encoded_rep: (batch_size, embed_dim)  # Final sequence representation
        """
        mammogram_emb = self.input_proj(mammogram_seq)  # (batch_size, sequencia_len, embed_dim)
        time_emb = self.time_encoding(timestamps)  # (batch_size, sequencia_len, embed_dim)
        input_emb = mammogram_emb + time_emb  # sum both inputs
        assert not torch.isnan(input_emb).any(), "NaNs before transformer!"

        input_emb = input_emb.permute(1, 0, 2)  # (sequencia_len, batch_size, embed_dim)

        #print(input_emb.device, attention_mask.device, time_emb.device)
        # -- set the attention mask
        src_key_padding_mask = attention_mask == 0  # (batch_size, sequencia_len)
        mask_sum = attention_mask.sum(dim=1)
        if (mask_sum == 0).any():
            print(attention_mask)
            print(mask_sum)
            print("Warning: some sequences are fully masked!")

        # -- apply transformer
        encoded_seq = self.transformer(input_emb, src_key_padding_mask=src_key_padding_mask)
        if torch.isnan(encoded_seq).any():
            print(encoded_seq)
        assert not torch.isnan(encoded_seq).any(), "NaNs after transformer!" 
        pooled_rep = self.attention_pooling(encoded_seq, attention_mask) 
        return pooled_rep 
    
def _causal_mask(T, device):
    # máscara triangular superior True = bloquear
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

class MammogramTransformerEncoder(nn.Module):
    def __init__(self, input_dim=1024, embed_dim=64, num_heads=4, num_layers=3, dropout=0.1,
                 dim_feedforward=None, use_causal=False):
        super().__init__()
        #if dim_feedforward is None:
        #    dim_feedforward = 4 * embed_dim

        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.time_encoding = TimeEncoding(embed_dim)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            #dim_feedforward=dim_feedforward,
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


class DynamicMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout=0.0, activation='relu', use_batchnorm=False, final_layer=True, sigmoid=True):
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

        if final_layer: # -- if this MLP will serve as a shared module, then it should be false
            layers.append(nn.Linear(current_dim, 1)) # -- output layer
            if sigmoid:
                layers.append(nn.Sigmoid()) # -- binary classification
        self.mlp = nn.Sequential(*layers)
        self.output_dim = current_dim

    def forward(self, x):
        z = self.mlp(x)
        return z

class CancerRiskMultiYearWithMLP(nn.Module):
    def __init__(self, encoder, embed_dim, extra_feature_dim, mlp_config, sigmoid_output=True, num_years=5, device='cpu'):
        """
            Predicts cumulative breast cancer risk for 1 to `num_years` years using a shared encoder and MLP,
            and additive hazards for monotonicity.

            Args:
                encoder: the Transformer-based sequence encoder
                embed_dim: output dimension of the encoder
                extra_feature_dim: dimension of structured input features
                mlp_config: dict with hidden_layers, dropout, activation, use_batchnorm, sigmoid (ignored)
                num_years: how many years ahead to predict (default 5)
        """
        super().__init__()
        self.encoder = encoder.to(device)
        self.num_years = num_years
        self.sigmoid_output = sigmoid_output

        input_dim = embed_dim + extra_feature_dim

        # -- shared MLP after encoder + features
        self.mlp = DynamicMLP(
            input_dim=input_dim,
            hidden_layers=mlp_config['hidden_layers'],
            dropout=mlp_config.get('dropout', 0.0),
            activation=mlp_config.get('activation', 'relu'),
            use_batchnorm=mlp_config.get('use_batchnorm', False),
            final_layer=False, # -- no single neuron output layer
            sigmoid=False  # no sigmoid here — we apply it after hazard sum
        )

        # Base risk + one hazard head per year
        last_hidden = mlp_config['hidden_layers'][-1] if mlp_config['hidden_layers'] else input_dim
        self.base = nn.Linear(last_hidden, 1)
        self.hazards = nn.ModuleList([
            nn.Sequential(nn.Linear(last_hidden, 1), nn.ReLU())
            for _ in range(num_years)
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, mammogram_seq, timestamps, attention_mask, extra_features):
        """
            Returns risk scores for each year from 1 to `num_years`.

            Output:
                Tensor of shape (batch_size, num_years)
        """
        encoded_rep = self.encoder(mammogram_seq, timestamps, attention_mask)
        combined = torch.cat([encoded_rep, extra_features], dim=-1)

        mlp_out = self.mlp(combined)
        base_logit = self.base(mlp_out)

        accum = base_logit
        outputs = []
        for h in self.hazards:
            accum = accum + h(mlp_out)
            out = self.sigmoid(accum) if self.sigmoid_output else accum
            outputs.append(out)

        return torch.cat(outputs, dim=1)  # shape: (batch_size, num_years)

class CancerRiskMultiYearWithMLP(nn.Module):
    def __init__(self, encoder, embed_dim, extra_feature_dim, mlp_config,
                 num_years=5, device='cpu'):
        super().__init__()
        self.encoder = encoder.to(device)
        self.num_years = num_years

        input_dim = embed_dim + extra_feature_dim
        self.mlp = DynamicMLP(
            input_dim=input_dim,
            hidden_layers=mlp_config['hidden_layers'],
            dropout=mlp_config.get('dropout', 0.0),
            activation=mlp_config.get('activation', 'relu'),
            use_batchnorm=mlp_config.get('use_batchnorm', False),
            final_layer=False,
            sigmoid=False
        )

        last_hidden = mlp_config['hidden_layers'][-1] if mlp_config['hidden_layers'] else input_dim
        # one linear head per interval -> logits of the interval hazard
        self.hazards = nn.ModuleList([nn.Linear(last_hidden, 1) for _ in range(num_years)])
        # stable init: tiny hazards at start
        for h in self.hazards:
            nn.init.xavier_uniform_(h.weight)
            nn.init.constant_(h.bias, -4.0)  # ~1.8% initial hazard

    def forward(self, mammogram_seq, timestamps, attention_mask, extra_features):
        h = self.encoder(mammogram_seq, timestamps, attention_mask)
        z = torch.cat([h, extra_features], dim=-1)
        z = self.mlp(z)

        # interval hazard logits & probs
        logits_h = torch.cat([head(z) for head in self.hazards], dim=1)  # (B,T)
        hazards = torch.sigmoid(logits_h)                                # (B,T), in (0,1)

        # survival product S_k = Π_{t=1..k} (1 - h_t), cumulative risk F_k = 1 - S_k
        survival = torch.cumprod(1.0 - hazards, dim=1)                   # (B,T)
        risk_within_k = 1.0 - survival                                   # (B,T)

        return {"logits_h": logits_h, "hazards": hazards, "risk": risk_within_k}

# -- version used in the first results
class CancerRiskXYears_v2(nn.Module):
    def __init__(self, encoder, embed_dim, extra_feature_dim, mlp_config, device='cpu'):
        '''
        Args:
            encoder: transformer encoder
            embed_dim: output dim from encoder
            extra_feature_dim: size of structured features
            mlp_config: dictionary with:
                - hidden_layers: list of hidden layer sizes (e.g., [128, 64])
                - dropout: float
                - activation: string ('relu', 'gelu', etc)
                - use_batchnorm: bool
        '''
        super().__init__()
        self.encoder = encoder.to(device)

        input_dim = embed_dim + extra_feature_dim
        self.mlp = DynamicMLP(
            input_dim=input_dim,
            hidden_layers=mlp_config['hidden_layers'],
            dropout=mlp_config.get('dropout', 0.0),
            activation=mlp_config.get('activation', 'relu'),
            use_batchnorm=mlp_config.get('use_batchnorm', False),
            sigmoid=mlp_config.get('sigmoid', True)
        )

    def forward(self, mammogram_seq, timestamps, attention_mask, extra_features):
        encoded_rep = self.encoder(mammogram_seq, timestamps, attention_mask)
        if torch.isnan(encoded_rep).any():
            print("NaNs found in encoder output")
        combined_rep = torch.cat([encoded_rep, extra_features], dim=-1)
        risk_score = self.mlp(combined_rep)
        return risk_score

class CancerRiskSingleYearWithMLP(nn.Module):
    def __init__(
        self,
        encoder,
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


# -- instantiate the model
def build_model(config, device):
    '''
        Load the configuration file to instantiate the model.

        Args:
        -----
            config:

        Returns:
        --------
            model:
    '''
    embed_dim = config['model']['embed_dim']
    mammogram_input_dim = config['model']['mammogram_input_dim']
    extra_features_dim = config['model']['extra_features_dim']
    transformer_num_heads = config['model']['transformer_num_heads']
    transformer_num_layers = config['model']['transformer_num_layers']
    transformer_dropout = config['model']['transformer_dropout']
    mlp_config = config['model']['mlp_config']

    encoder = MammogramTransformerEncoder(input_dim=mammogram_input_dim, embed_dim=embed_dim, num_heads=transformer_num_heads, num_layers=transformer_num_layers, dropout=transformer_dropout)
    #model = CancerRiskOneYear(encoder=encoder, embed_dim=embed_dim, extra_feature_dim=extra_features_dim, layer_dict=layer_dict, device=device)
    model = CancerRiskXYears_v2(encoder=encoder, embed_dim=embed_dim, extra_feature_dim=extra_features_dim, mlp_config=mlp_config, device=device)      
    return model

def build_model_singleyear(config, device):
    '''
        ...
    '''
    embed_dim = config['model']['embed_dim']
    mammogram_input_dim = config['model']['mammogram_input_dim']
    extra_features_dim = config['model']['extra_features_dim']
    transformer_num_heads = config['model']['transformer_num_heads']
    transformer_num_layers = config['model']['transformer_num_layers']
    transformer_dropout = config['model']['transformer_dropout']
    mlp_config = config['model']['mlp_config']
    loss_function = config['training']['loss_function']

    risk_to_sigmoid_loss = {
        'compositional_auc_loss': True,
        'focal_loss': True,
        'average_precision_loss': False,
        'aucm_loss': False
    }
    sigmoid_output = risk_to_sigmoid_loss[loss_function]
    
    encoder = MammogramTransformerEncoder(
        input_dim=mammogram_input_dim, 
        embed_dim=embed_dim, 
        num_heads=transformer_num_heads, 
        num_layers=transformer_num_layers, 
        dropout=transformer_dropout
    )
    model = CancerRiskSingleYearWithMLP(
        encoder=encoder,
        embed_dim=embed_dim,
        extra_feature_dim=extra_features_dim,
        mlp_config=mlp_config,
        sigmoid=sigmoid_output,
        device=device
    )
    return model

def build_model_multiyear(config, device):
    '''
        If we decide to use multiple year network.
    '''
    
    embed_dim = config['model']['embed_dim']
    mammogram_input_dim = config['model']['mammogram_input_dim']
    extra_features_dim = config['model']['extra_features_dim']
    transformer_num_heads = config['model']['transformer_num_heads']
    transformer_num_layers = config['model']['transformer_num_layers']
    transformer_dropout = config['model']['transformer_dropout']
    mlp_config = config['model']['mlp_config']
    loss_function = config['training']['loss_function']

    risk_to_sigmoid_loss = {
        'compositional_auc_loss': True,
        'focal_loss': True,
        'average_precision_loss': False,
        'aucm_loss': False
    }
    sigmoid_output = risk_to_sigmoid_loss[loss_function]

    encoder = MammogramTransformerEncoder(
        input_dim=mammogram_input_dim, 
        embed_dim=embed_dim, 
        num_heads=transformer_num_heads, 
        num_layers=transformer_num_layers, 
        dropout=transformer_dropout
    )
    model = CancerRiskMultiYearWithMLP(
        encoder=encoder,
        embed_dim=embed_dim,
        extra_feature_dim=extra_features_dim,
        mlp_config=mlp_config,
        #sigmoid_output=sigmoid_output,
        num_years=5,
        device=device
    )
    return model

def build_model_singleyear_without_transformer(config, device):
    '''
    
    '''
    embed_dim = config['model']['embed_dim']
    mammogram_input_dim = config['model']['mammogram_input_dim']
    extra_features_dim = config['model']['extra_features_dim']
    transformer_num_heads = config['model']['transformer_num_heads']
    transformer_num_layers = config['model']['transformer_num_layers']
    transformer_dropout = config['model']['transformer_dropout']
    mlp_config = config['model']['mlp_config']
    loss_function = config['training']['loss_function']

    risk_to_sigmoid_loss = {
        'cross_entropy': False,
        'compositional_auc_loss': True,
        'focal_loss': True,
        'average_precision_loss': False,
        'aucm_loss': False
    }
    sigmoid_output = risk_to_sigmoid_loss[loss_function]
    model = CancerRiskSingleYearWithMLPNoTransformer(
        embed_dim=mammogram_input_dim,
        extra_feature_dim=extra_features_dim,
        mlp_config=mlp_config,
        sigmoid=sigmoid_output,
        device=device
    )
    return model

