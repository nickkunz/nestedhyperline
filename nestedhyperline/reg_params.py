## load libraries
import numpy as np
from hyperopt import hp

## regularization hyper-params
def reg_params():
    
    """ utilized for bayesian hyper-parameter optimization, returns
    ridge and lasso regression parameter ranges (search space) """
    
    reg_params = {
        
        ## lambda shrinkage
        'alpha': hp.loguniform(
            label = 'alpha',
            low = np.log(0.001),
            high = np.log(100)
        )
    }
    
    return reg_params

## elastic-net hyper-params
def net_params():
    
    """ utilized for bayesian hyper-parameter optimization, returns
    elastic-net regression parameter ranges (search space) """
    
    net_params = {
        
        ## lambda shrinkage
        'alpha': hp.loguniform(
            label = 'alpha',
            low = np.log(0.001),
            high = np.log(100)
        ),
        
        # l2 and l1 mixture
        'l1_ratio': hp.uniform(
            label = 'l1_ratio',
            low = 0.00,
            high = 1.00
        )
    }
    
    return net_params
