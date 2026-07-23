'''
    Define the model's architecture.
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, List
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

from hapcancer.model.dataload.datasets import CancerDatasetSingleYearTFIDF
from hapcancer.model.architecture.architecture import CancerRiskSingleYearWithMLPNoTransformer


class CancerPredictionBaseModel:
    '''
        Base class to define the prediction model of the task. The
        idea is to be flexible enough so that it can use at least three
        different type of models: MLP, LightGBM and Logistic Regression.
    '''
    def __init__(self, config):
        self.config = config
        self.model_class = self.config['model']['model_class']

        self.mammogram_input_dim = self.config['model']['mammogram_input_dim']
        self.extra_features_dim = self.config['model']['extra_features_dim']
        self.device = self.config['misc']['device']
        self.mlp_config = None
        self.light_gbm_config = None
        self.logreg_config = None

        self.loss_function = None

        # -- depending on the type of loss function used, it should return either logits and probs.
        # -- If the model should return probabilities, then 'sigmoid_for_loss_fn' is True.
        self.sigmoid_for_loss_fn = {
            'cross_entropy': False,
            'compositional_auc_loss': True,
            'focal_loss': True,
            'average_precision_loss': False,
            'aucm_loss': False
        }

        self._model = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, v):
        raise Exception("model cannot be assigned an external value.")
    
    def fit(self, X, y):
        if hasattr(self.model, "fit"):
            self.model.fit(X, y)
        else:
            raise NotImplementedError("fit not implemented for this model type")
    
    def build_model(self):
        pass

class CancerPredictionModelMLP(CancerPredictionBaseModel):
    '''
        Build TF-IDF+MLP model.
    '''
    def build_model(self):
        mlp_config = self.config['model']['mlp_config']
        loss_function = self.config['training']['loss_function']
        sigmoid_output = self.sigmoid_for_loss_fn[loss_function]
        self._model = CancerRiskSingleYearWithMLPNoTransformer(
            embed_dim=self.mammogram_input_dim,
            extra_feature_dim=self.extra_features_dim,
            mlp_config=mlp_config,
            sigmoid=sigmoid_output,
            device=self.device
        )

class CancerPredictionModelLightGBM(CancerPredictionBaseModel):
    '''
        Build TF-IDF+LightGBM model.
    '''
    def build_model(self):
        light_gbm_config = self.config['model']['light_gbm_config']
        self._model = lgb.LGBMClassifier(**light_gbm_config)

class CancerPredictionModelLogReg(CancerPredictionBaseModel):
    '''
        Build TF-IDF+Logistic Regression model.
    '''
    def build_model(self):
        self._model = LogisticRegression(
            **self.config['model'].get('logreg_config', {}),
            max_iter=1000
        )



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

def build_model_singleyear_with_tfidf(config, device):
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

