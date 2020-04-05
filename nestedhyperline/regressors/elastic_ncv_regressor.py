## load libraries
import random as rd
from sklearn.linear_model import ElasticNet
from nestedhyperline.reg_params import net_params
from nestedhyperline.argument_quality import ArgumentQuality
from nestedhyperline.ncv_optimizer import ncv_optimizer

## elastic-net regression
def elastic_ncv_regressor(
    
    data,          ## pandas dataframe, clean (no nan's)
    y,             ## string, header of y reponse variable
    loss = "root_mean_squared_error", ## string, objective function to minimize
    k_outer = 5,   ## pos int, k number of outer folds (1 < k < n)
    k_inner = 5,   ## pos int, k number of inner folds (1 < k < n)
    n_evals = 25,  ## pos int, number of evals for bayesian optimization
    seed = rd.randint(0, 9999),  ## pos int, fix for reproduction
    verbose = True               ## bool, display output
    ):
    
    """
    conducts elastic-net l2 and l1 regularization for linear regression 
    prediction problems
    
    designed for rapid prototyping, quickly obtains prediction results by 
    compromising implementation details and flexibility
    
    applicable only to linear regression problems, unifies three important 
    supervised learning techniques for structured data:
    
    1) nested k-fold cross validation (minimize bias)
    2) bayesian optimization (efficient hyper-parameter tuning)
    3) linear regularization (reduce model complexity)

    bayesian hyper-parameter optimization is conducted utilizing tree prezen
    estimation, linear regularization is conducted utilizing l2 and l1
    
    returns custom regression object:
    - root mean squared error (or other specified regression metric)
    - list of root mean squared errors on outer-folds
    
    data: 
    - pandas dataframe (n > 2)
    - clean (no nan's)
    
    y: 
    - string
    - header of y reponse variable
    
    loss:
    - string
    - objective function to minimize
    - default "root_mean_squared_error"
    - supports any error metric found in sklearn
    
    k_outer:
    - pos int
    - k number of outer folds (1 < k < n)
    
    k_inner:
    - pos int
    - k number of inner folds (1 < k < n)
    
    n_evals:
    - pos int
    - number of evals for bayesian optimization
    - default 25
    
    seed:
    - pos int
    - fix to reproduce results
    
    verbose:
    - bool
    - display function output
    """
    
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
    method = ElasticNet
    params = net_params()
    
    ## nested cross-valid bayesian hyper-param optimization
    ncv_results = ncv_optimizer(
        
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
