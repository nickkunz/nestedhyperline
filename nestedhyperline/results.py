## load libraries
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

## store regression results
class RegressResults():
    def __init__(self, model, params, trials_list, y_pred_list, error_list):
        
        """
        Calculates and stores the average of all Outer K-Fold Cross-Validation 
        specified errors (Mean Squared Error, Mean Absolute Error, etc.), as 
        well as all the attributes from the parent Results object. Returns to 
        main function ncv_optimizer().
        """
        
        self.model = model
        self.params = params
        self.trials_list = trials_list
        self.y_pred_list = y_pred_list
        self.error_list = error_list
    
    ## average rmse results across outer k-folds
    def error_mean(self):
        error_mean = round(np.average(
            self.error_list
            
            ## round results
            ), ndigits = 6
        )
        
        return error_mean
