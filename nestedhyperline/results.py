## load libraries
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.ticker as tkr
import matplotlib.cm as cm
import seaborn as sns
import numpy as np

## store regression results
class RegressResults():
    def __init__(self, y, model, params, 
                 coef_list, trials_list, error_list,
                 standardize, k_outer, n_evals):

        """
        Calculates and stores the average of all Outer K-Fold Cross-Validation 
        specified errors (Mean Squared Error, Mean Absolute Error, etc.), as 
        well as all the attributes from the parent Results object. Returns to 
        main function ncv_optimizer().
        """

        self.y = y
        self.model = model
        self.params = params
        self.coef_list = coef_list
        self.trials_list = trials_list
        self.error_list = error_list
        self.standardize = standardize
        self.k_outer = k_outer
        self.n_evals = n_evals

    ## average errors across outer k-folds
    def error_mean(self):
        error_mean = round(np.average(
            self.error_list

            ## round results
            ), ndigits = 6
        )

        return error_mean
