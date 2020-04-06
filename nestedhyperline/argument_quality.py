## load libraries
import pandas as pd

## create input quality checks for regressor and classifier
class ArgumentQuality():
    def __init__(self, data, y, loss, k_outer, k_inner, n_evals, seed, verbose):
        
        """
        Conducts input checks on arguments found in regressor functions:
        ridge_ncv_regressor(), lasso_ncv_regressor(), and elastic_ncv_regressor().
        Ensure proper usage and execution. This class is intended to raise errors
        if invalid inputs are passed through the regressor function arguments.
        """
        
        self.data = data
        self.y = y
        self.loss = loss
        self.k_outer = k_outer
        self.k_inner = k_inner
        self.n_evals = n_evals
        self.seed = seed
        self.verbose = verbose
        
        ## quality check for dataframe
        if isinstance(self.data, pd.DataFrame) is False:
             raise ValueError("must pass pandas dataframe into 'data' argument")
        
        ## quality check for missing values in dataframe
        if self.data.isnull().values.any():
             raise ValueError("dataframe cannot contain missing values")
        
        ## quality check for y
        if isinstance(self.y, str) is False:
             raise ValueError("'y' must be a string")
        
        if self.y in self.data.columns.values is False:
             raise ValueError("'y' must be an header name (string) found in the dataframe")
        
        ## quality check for loss 
        if isinstance(self.loss, str) is False:
             raise ValueError("'loss' must be a string")
        
        if self.loss in [
            "explained_variance",
            "max_error",
            "mean_absolute_error",
            "mean_squared_error",
            "root_mean_squared_error",
            "mean_squared_log_error",
            "median_absolute_error",
            "r2",
            "mean_poisson_deviance",
            "mean_gamma_deviance"
            ] is False:
                raise ValueError("'loss' must be an accepted sklearn scoring param")
        
        ## quality check for k-fold outer argument
        if self.k_outer > len(self.data):
             raise ValueError("'k_outer' is greater than number of observations (rows)")
        
        if self.k_outer < 2:
             raise ValueError("'k_outer' must be a positive integer greater than 1")
        
        ## quality check for k-fold inner argument
        if self.k_inner > len(self.data):
             raise ValueError("'k_inner' is greater than number of observations (rows)")
        
        if self.k_inner < 2:
             raise ValueError("'k_inner' must be a positive integer greater than 1")
        
        ## quality check for number of evaluations
        if self.n_evals < 1:
             raise ValueError("'n_evals' must be a positive integer")
        
        ## quality check for random seed
        if self.seed < 1:
             raise ValueError("'seed 'must be a positive integer")
        
        ## quality check for verbose
        if isinstance(self.verbose, bool) is False:
             raise ValueError("'verbose' must be boolean")
