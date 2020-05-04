## load libraries
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.ticker as tkr
import matplotlib.cm as cm
import seaborn as sns
import numpy as np

## internal
from nestedhyperline.plot_prep import plot_prep

## store regression results
class RegressResults():
    def __init__(self, y, cols, model, params, 
                 trials_list, error_list, coef_list,
                 standardize, k_outer, n_evals):

        """
        Calculates and stores the average of all Outer K-Fold Cross-Validation 
        specified errors (Mean Squared Error, Mean Absolute Error, etc.), as 
        well as all the attributes from the parent Results object. Returns to 
        main function ncv_optimizer().
        """

        self.y = y
        self.cols = cols
        self.model = model
        self.params = params
        self.trials_list = trials_list
        self.error_list = error_list
        self.coef_list = coef_list
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
    
    ## plot convergence of lambda across outer k-folds
    def plot_lambda(self,

        ## pre-process args
        trials_list = trials_list,
        params = pararms,
        k_outer = k_outer,
        n_evals = n_evals,
        standardize = standardize,

        ## settings args
        lw_dot = 1.30,
        lw_sld = 1.1,
        plt_hgt = 9,
        plt_wdt = 20,
        ax_fnt_sze = 11,
        sty = 'whitegrid'
        ):

        ## plot pre-processor
        plt_data = plot_prep(
            trials_list = trials_list,
            params = pararms,
            k_outer = k_outer,
            n_evals = n_evals,
            standardize = standardize
        )

        ## style
        sns.set_style(
            style = sty
        )

        ## size
        fig = plt.figure(
            figsize = (plt_wdt, plt_hgt)
        )

        ## title
        ax = fig.add_subplot(
            title = "Nested K-Fold Cross-Validation: Bayesian Optimized Convergence of Log(λ)"
        )

        ## y axis
        ax.set_ylabel(
            ylabel = 'Log(λ)', 
            fontsize = ax_fnt_sze
        )

        ax.set_ylim(
            bottom = plt_data['lamb_min'] - (plt_data['lamb_min'] * 0.33), 
            top = plt_data['lamb_max'] + (plt_data['lamb_max'] * 0.33)
        )

        ax.set_yscale(
            value = 'log'
        )

        ## x axis
        ax.set_xlabel(
            xlabel = 'Number of Evaluations', 
            fontsize = ax_fnt_sze
        )

        ax.set_xlim(
            left = -(n_evals * 0.010), 
            right = n_evals + (n_evals * 0.010)
        )

        ## bayesian optimization
        for i in range(k_outer):
            ax.scatter(
                x = range(n_evals),
                y = plt_data['lamb_list'][i],
                color = plt_data['colors'][i], ## dot colors
                alpha = 0.75,  ## dot opacity
                s = 12  ## dot size
            )

        ## lowest average test error line
        ax.axhline(
            y = plt_data['lamb_mean_test'],
            color = 'black',
            linestyle = 'dotted',
            linewidth = lw_dot
        )

        ## lowest average train-valid error line
        ax.axhline(
            y = plt_data['lamb_mean_min'],
            color = 'black',
            linewidth = lw_sld
        )

        ## legend
        ax.legend(
            handles = plt_data['leg_lines'], 
            labels = plt_data['leg_labels'],
            loc = 'lower right',
            fontsize = 'small'
        )

        ## display
        plt.show()
