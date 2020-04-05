## load libraries
from sklearn.linear_model import Ridge, Lasso, ElasticNet

## boosting method (xgboost, lightgbm, catboost)
def reg_select(method, params, seed):
    
    ## ridge and lasso
    if method in [Ridge, Lasso]:
        
        ## hyper-param specification
        method_params = method(
            
            ## learned params
            alpha = params["alpha"],
            
            ## specified params
            random_state = seed
        )
    
    ## lightgbm
    if method == ElasticNet:
        ## hyper-param specification
        method_params = method(
            
            ## learned params
            alpha = params["alpha"],
            l1_ratio = params["l1_ratio"],
            
            ## specified params
            random_state = seed
        )
    
    ## returns boosting method and params
    return method_params
