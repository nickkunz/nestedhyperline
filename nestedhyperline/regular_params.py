## load libraries
import numpy as np
from hyperopt import hp

## ridge and lasso hyper-params
def reg_params():

    """ Utilized for Bayesian Optimization. Returns Ridge and 
    LASSO regression parameter ranges (search space). """

    reg_params = {
        
        ## lambda shrinkage
        'alpha': hp.loguniform(
            label = 'alpha',
            low = np.log(0.00001),
            high = np.log(10000)
        )
    }

    return reg_params

## elastic-net hyper-params
def net_params():

    """ Utilized for Bayesian Optimization. Returns Elastic-Net 
    regression parameter ranges (search space) """

    net_params = {

        ## lambda shrinkage
        'alpha': hp.loguniform(
            label = 'alpha',
            low = np.log(0.00001),
            high = np.log(10000)
        ),

        # l2 and l1 mixture
        'l1_ratio': hp.uniform(
            label = 'l1_ratio',
            low = 0.00,
            high = 1.00
        )
    }

    return net_params
