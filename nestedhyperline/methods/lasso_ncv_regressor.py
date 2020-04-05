## load libraries
import random as rd
from sklearn.linear_model import Lasso
from nestedhyperline.reg_params import reg_params
from nestedhyperline.argument_quality import ArgumentQuality
from nestedhyperline.ncv_optimizer import ncv_optimizer

## lasso regression
def lasso_ncv_regressor(
    
    data,          ## pandas dataframe, clean (no nan's)
    y,             ## string, header of y reponse variable
    loss = "root_mean_squared_error", ## string, objective function to minimize
    k_outer = 5,   ## pos int, k number of outer folds (1 < k < n)
    k_inner = 5,   ## pos int, k number of inner folds (1 < k < n)
    n_evals = 25,  ## pos int, number of evals for bayesian optimization
    seed = rd.randint(0, 9999),  ## pos int, fix for reproduction
    verbose = True               ## bool, display output
    ):
    
    ## conduct input quality checks
    ArgumentQuality(
        data = data,
        y = y,
        loss = loss,
        k_outer = k_outer,
        k_inner = k_inner,
        n_evals = n_evals,
        seed = seed,
        verbose = verbose
    )
    
    ## initiate modeling method
    method = Lasso
    params = reg_params()
    
    ## nested cross-valid bayesian hyper-param optimization
    ncv_results = ncv_regressor(
        
        ## main func args
        data = data,
        y = y,
        loss = loss,
        k_outer = k_outer,
        k_inner = k_inner,
        n_evals = n_evals,
        seed = seed,
        verbose = verbose,
        
        ## pred func args
        method = method,
        params = params
    )
    
    ## regression results object
    return ncv_results
