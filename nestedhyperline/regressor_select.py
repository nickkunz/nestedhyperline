## load libraries
from sklearn.linear_model import Ridge, Lasso, ElasticNet

## boosting method (xgboost, lightgbm, catboost)
def reg_select(method, params, random_state):

    ## ridge and lasso
    if method in [Ridge, Lasso]:

        ## hyper-param specification
        method_params = method(

            ## learned params
            alpha = params["alpha"],

            ## max number of iterations
            max_iter = 100000,

            ## specified params
            random_state = random_state

        )

    ## lightgbm
    if method == ElasticNet:
        ## hyper-param specification
        method_params = method(

            ## learned params
            alpha = params["alpha"],
            l1_ratio = params["l1_ratio"],

            ## max number of iterations
            max_iter = 100000,

            ## specified params
            random_state = random_state
        )

    ## returns boosting method and params
    return method_params
