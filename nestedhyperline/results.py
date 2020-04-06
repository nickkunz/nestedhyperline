## load libraries
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

## store regression results
class RegressResults():
    def __init__(self, trials, model, params, error_list):
        
        """
        Calculates and stores the average of all Outer K-Fold Cross-Validation 
        specified errors (Mean Squared Error, Mean Absolute Error, etc.), as 
        well as all the attributes from the parent Results object. Returns to 
        main function ncv_optimizer().
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
