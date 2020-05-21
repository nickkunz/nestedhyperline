## load libraries
import pandas as pd

## argument input quality checks for regressor
class ArgumentQuality():
     def __init__(self, data, y, loss, k_outer, k_inner, n_evals,
          random_state, standardize, verbose):

          """ Conducts input checks on arguments found in regressor
          functions: ridge_ncv_regressor(), lasso_ncv_regressor(), 
          and elastic_ncv_regressor().
          
          Ensures proper usage and execution. This class is intended
          to raise errors if invalid inputs are passed through the
          regressor function arguments. """

          self.data = data
          self.y = y
          self.loss = loss
          self.k_outer = k_outer
          self.k_inner = k_inner
          self.n_evals = n_evals
          self.random_state = random_state
          self.standardize = standardize
          self.verbose = verbose

          ## check for dataframe
          if isinstance(self.data, pd.DataFrame) is False:
               raise ValueError("must pass pandas dataframe into 'data' argument")

          if len(self.data) < 3:
               raise ValueError("dataframe must contain more than 3 observations")

          ## check for missing values in dataframe
          if self.data.isnull().values.any():
               raise ValueError("dataframe cannot contain missing values")

          ## check for y
          if isinstance(self.y, str) is False:
               raise ValueError("'y' must be a string")

          if self.y in self.data.columns.values is False:
               raise ValueError("'y' must be an header name (string) found in the dataframe")

          ## check for loss
          if isinstance(self.loss, str) is False:
               raise ValueError("'loss' must be a string")

          if self.loss in [
               "explained variance", "ev",
               "max error", "me",
               "mean absolute error", "mae",
               "mean squared error", "mse",
               "root mean squared error", "rmse",
               "mean squared log error", "msle",
               "median absolute error", "mdae",
               "r2",
               "mean poisson deviance", "mpd",
               "mean gamma deviance", "mgd"
               ] is False:
                    raise ValueError("'loss' must be an accepted sklearn scoring param")

          ## check for outer k-fold
          if self.k_outer > 20:
               raise ValueError("'k_outer' must be a positive integer between 2 and 20")

          if self.k_outer < 2:
               raise ValueError("'k_outer' must be a positive integer between 2 and 20")

          ## check for inner k-fold
          if self.k_inner > 20:
               raise ValueError("'k_inner' must be a positive integer between 2 and 20")

          if self.k_inner < 2:
               raise ValueError("'k_inner' must be a positive integer between 2 and 20")

          ## check for number of evaluations
          if self.n_evals < 1:
               raise ValueError("'n_evals' must be a positive integer")

          ## check for random state
          if self.random_state < 1:
               raise ValueError("'random_state' must be a positive integer")

          ## check for standardization
          if isinstance(self.standardize, bool) is False:
               raise ValueError("'standardize' must be boolean")

          ## check for verbose
          if isinstance(self.verbose, bool) is False:
               raise ValueError("'verbose' must be boolean")
