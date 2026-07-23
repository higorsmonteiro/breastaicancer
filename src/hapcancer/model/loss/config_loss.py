import torch.optim as optim
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from hapcancer.model.loss.custom_loss import FocalLoss
#from .custom_loss import FocalLoss
from libauc.losses import AUCMLoss, AveragePrecisionLoss, CompositionalAUCLoss
from libauc.optimizers import PESG, SOAP, PDSCA

def get_optimizer_with_weight_decay(model, weight_decay):
    '''
        Select which parameters should be weight decayed.
        It seems that some specific set of parameters from the transformer's 
        architecture should be avoided, as performing weight decay on them 
        will break the training by generating NaN values during the optimization.

        Args:
        -----
            model:
                ...
            weight_decay:
                ... 

        Returns:
        --------
            optimizer_grouped_parameters:
                ...
    '''
    decay = []
    no_decay = []

    # -- weight decay on these parameters raise errors.
    params_to_exclude = [
        'bias', 'layerNorm.weight', 'layerNorm.bias', 'embedding', 'decay_factor'
    ]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # skip frozen weights
        if any(nd in name.lower() for nd in params_to_exclude):
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer_grouped_parameters = [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ]
    return optimizer_grouped_parameters

def get_flattened_params_with_decay(grouped_params):
    '''
        For optimizers like PESG and PDSCA, only grouping the params using our
        function 'get_optimizer_with_weight_decay' will not work, because the way
        those functions receive the params are a little bit different.
        Therefore, we perform some modifications to guarantee we optimize all parameters
        while avoiding performing regularization in some of the transformer's parameters
        (weight_decay=0).

        A good test is to check whether this same approach also work for Adam and SOAP, so
        that we write uniform code for each case.

        Args:
        -----
            grouped_params:
                dictionary. Output of our custom function 'get_optimizer_with_weight_decay'.
    '''
    flat_params = []
    for group in grouped_params:
        for param in group['params']:
            # Attach the specific weight decay setting to each parameter
            param._custom_weight_decay = group['weight_decay']
            flat_params.append(param)
    return flat_params

def config_loss(config, model, **kwargs):
    '''
        Configure Compositional AUC loss with PDSCA for our survival-product setup.
        We will feed flattened logits and labels, masked, with an interval-wise index.
    '''
    lr           = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    loss_name    = config['training']['loss_function']   # expect 'compositional_auc_loss'
    device       = config['misc']['device']

    # group parameters with/without weight decay
    optimizer_grouped_parameters = get_optimizer_with_weight_decay(model, weight_decay)

    if loss_name == 'compositional_auc_loss':
        # v2 is usually the stable pick; margin=1.0 is a sensible default according to the docs
        criterion = CompositionalAUCLoss(margin=1.0, version='v2', device=device)
        # PDSCA needs a flat parameter list with decay flags (my helper)
        flat_params = get_flattened_params_with_decay(optimizer_grouped_parameters)
        optimizer = PDSCA(flat_params, loss_fn=criterion, lr=lr, lr0=0.01, weight_decay=weight_decay, device=device)
    else:
        # maybe keep the other options from the previous code.
        criterion = None
        optimizer = optim.Adam(optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay, epoch_decay=0)
    return criterion, optimizer

def config_loss_many(config, model, device, **kwargs):
    '''
        Given the configuration file, set up the loss function and the optimizer
        to be used during model training.
    '''
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    loss_function = config['training']['loss_function']

    criterion, optimizer = None, None
    if loss_function=='focal_loss':
        focal_gamma = config['training']['focal_gamma']
        focal_alpha = config['training']['focal_alpha']
        criterion = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        #optimizer_grouped_parameters = get_optimizer_with_weight_decay(model, weight_decay)
        #optimizer = optim.Adam(optimizer_grouped_parameters, lr=learning_rate, weight_decay=weight_decay)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif loss_function=='average_precision_loss': # it did work
        criterion = AveragePrecisionLoss(data_len=kwargs['data_len'], margin=1.0, device=device)
        #optimizer_grouped_parameters = get_optimizer_with_weight_decay(model, weight_decay)
        #optimizer = SOAP(optimizer_grouped_parameters, lr=learning_rate, weight_decay=weight_decay)
        optimizer = SOAP(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif loss_function=='aucm_loss': # it didn't work
        criterion = AUCMLoss(margin=1.0, imratio=kwargs['imratio'], device=device)
        #optimizer_grouped_parameters = get_optimizer_with_weight_decay(model, weight_decay)
        #flat_params = get_flattened_params_with_decay(optimizer_grouped_parameters)
        #optimizer = PESG(flat_params, loss_fn=criterion, lr=learning_rate, weight_decay=weight_decay, device=device)
        optimizer = PESG(model.parameters(), loss_fn=criterion, lr=learning_rate, weight_decay=weight_decay, device=device)
    elif loss_function=='compositional_auc_loss':
        criterion = CompositionalAUCLoss(margin=1.0, imratio=kwargs['imratio'], device=device)
        #criterion = CompositionalAUCLoss(margin=1.0, version='v2', device=device)
        #flat_params = get_flattened_params_with_decay(optimizer_grouped_parameters)
        #optimizer = PDSCA(flat_params, loss_fn=criterion, lr=learning_rate, lr0=0.01, weight_decay=weight_decay, device=device)
        optimizer = PDSCA(model.parameters(), loss_fn=criterion, lr=learning_rate, weight_decay=weight_decay, device=device)
    elif loss_function=="cross_entropy":
        criterion = BCEWithLogitsLoss()
        if kwargs['bce_optimizer'].lower()=="adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif kwargs['bce_optimizer'].lower()=="sgd":
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        #optimizer_grouped_parameters = get_optimizer_with_weight_decay(model, weight_decay)
        #optimizer = optim.Adam(optimizer_grouped_parameters, lr=learning_rate, weight_decay=weight_decay)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        pass
    return criterion, optimizer

def make_masked_loss(loss_fn):
    """
        Wraps a loss function of the form (preds, labels, index)
        to apply it over valid (non-censored) entries only.

        Returns:
        - masked_loss(preds, labels, mask, index)
    """
    def masked_loss(preds, labels, mask, index):
        valid = mask.bool()
        preds_flat = preds[valid]
        labels_flat = labels[valid]
        index_flat = index[valid]
        if valid.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=preds.device)
        return loss_fn(preds_flat, labels_flat, index=index_flat)
    return masked_loss