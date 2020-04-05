## load libraries
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

## store regression results
class RegressResults():
    def __init__(self, trials, model, params, error_list):
        
        """
        calculates and stores the average of all outer k-fold cross-validation 
        specified errors (mean squared error, mean absolute error, etc.), as 
        well as all the attributes from the parent Results object, returned to 
        main function ncv_optimizer()
        """
        
        self.trials = trials
        self.model = model
        self.params = params
        self.error_list = error_list
    
    ## average rmse results across outer k-folds
    def error_mean(self):
        error_mean = round(np.average(
            self.error_list
            
            ## round results
            ), ndigits = 6
        )
        
        return error_mean
