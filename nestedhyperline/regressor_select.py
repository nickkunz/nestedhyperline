## load libraries
from sklearn.linear_model import Ridge, Lasso, ElasticNet

## regularization method
def reg_select(method, params, random_state):

    """ Selects the specified linear regularization method and determines
    the maximum number of Bayesian Optimization iterations. Choose between 
    Ridge, LASSO, and Elastic-Net. """

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

    ## elastic-net
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

    return method_params
