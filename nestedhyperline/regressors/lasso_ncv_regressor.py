## load libraries
import random as rd
from sklearn.linear_model import Lasso
from nestedhyperline.regular_params import reg_params
from nestedhyperline.argument_quality import ArgumentQuality
from nestedhyperline.ncv_optimizer import ncv_optimizer

## lasso regression
def lasso_ncv_regressor(

    data,
    y,
    loss = "rmse",
    k_outer = 5,
    k_inner = 5,
    n_evals = 1000,
    random_state = rd.randint(0, 9999),
    standardize = True,
    verbose = True
    ):

    """
    Conducts LASSO L1 Regularization for Linear Regression prediction problems.

    Designed for rapid prototyping. Quickly obtains prediction results by
    compromising implementation details and flexibility.

    Applicable only to linear regression problems. Unifies three important
    supervised learning techniques for structured data:

    1) Nested K-Fold Cross Validation (minimize bias)
    2) Bayesian Optimization (efficient hyper-parameter tuning)
    3) Linear Regularization (reduce model complexity)

    Bayesian hyper-parameter optimization is conducted utilizing Tree Prezen
    Estimation. Linear Regularization is conducted utilizing L1 shrinkage.

    Arguments:

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

    random_state:
    - pos int
    - fix to reproduce results

    standardize:
    - bool
    - standardizes data

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
        random_state = random_state,
        standardize = standardize,
        verbose = verbose
    )

    ## initiate model
    method = Lasso
    params = reg_params()

    ## nested cross-valid bayesian optimization
    ncv_results = ncv_optimizer(

        ## main func args
        data = data,
        y = y,
        loss = loss,
        k_outer = k_outer,
        k_inner = k_inner,
        n_evals = n_evals,
        random_state = random_state,
        standardize = standardize,
        verbose = verbose,

        ## pred func args
        method = method,
        params = params
    )

    ## regression results
    return ncv_results
