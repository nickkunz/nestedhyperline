## load libraries
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.ticker as tkr
import matplotlib.cm as cm
import seaborn as sns

## internal
from nestedhyperline.plot_prep import plot_prep

## store regression results
class RegressResults():
    def __init__(self, y, cols, model, params, 
                 trials_list, error_list, coef_list,
                 standardize, k_outer, n_evals):

        """ Calculates the average of all Outer K-Fold Cross-Validation 
        specified loss function errors (Mean Squared Error, Mean Absolute 
        Error, etc.), as well as all the associated plots. Returns to main 
        function ncv_optimizer(). """

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

    ## calculate mean error
    def error_mean(self, ndigits = 6):

        ## average errors across outer k-folds
        error_mean = round(np.average(
            self.error_list

            ## round results
            ), ndigits = ndigits
        )

        ## results 
        return error_mean
    
    ## plot mean error
    def plot_error_mean(self,

        ## viz settings
        lw_dot = 1.30,
        lw_sld = 1.1,
        dt_opa = 
        plt_hgt = 9,
        plt_wdt = 20,
        ax_fnt_sze = 11,
        sty = 'whitegrid'
        ):

        ## plot pre-processor
        plt_data = plot_prep(
            trials_list = self.trials_list,
            params = self.params,
            k_outer = self.k_outer,
            n_evals = self.n_evals,
            standardize = self.standardize
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
            title = 'Nested K-Fold Cross-Validation: Test Set Errors as a Function of Log(λ)'
        )

        ## y axis
        ax.set_ylabel(
            ylabel = 'Test Set Error',
            fontsize = ax_fnt_sze
        )

        ## x axis
        ax.set_xlabel(
            xlabel = 'Log(λ)',
            fontsize = ax_fnt_sze
        )

        ax.set_xlim(
            left = plt_data['lamb_min'] - (plt_data['lamb_min'] * 0.10), 
            right = plt_data['lamb_max'] + (plt_data['lamb_max'] * 0.10)
        )

        ax.set_xscale(
            value = 'log'
        )

        ## cross-valid test errors
        for i in range(self.k_outer):
            ax.scatter(
                x = plt_data['lamb_list'][i],
                y = plt_data['loss_list'][i],
                color = plt_data['colors'][i], ## dot colors
                alpha = 0.50,  ## dot opacity
                s = 10  ## dot size
            )

        ## lowest average test set error line
        ax.axvline(
            x = plt_data['lamb_mean_min'],
            color = 'black', 
            linestyle = 'dotted',
            linewidth = lw_dot
        )

        ## lowest average train-valid error line
        ax.axvline(
            x = plt_data['lamb_mean_min'],
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

    ## plot convergence of lambda
    def plot_lambda(self,

        ## viz settings
        lw_dot = 1.30,
        lw_sld = 1.1,
        plt_hgt = 9,
        plt_wdt = 20,
        ax_fnt_sze = 11,
        sty = 'whitegrid'
        ):

        ## plot pre-processor
        plt_data = plot_prep(
            trials_list = self.trials_list,
            params = self.params,
            k_outer = self.k_outer,
            n_evals = self.n_evals,
            standardize = self.standardize
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
            left = -(self.n_evals * 0.010), 
            right = self.n_evals + (self.n_evals * 0.010)
        )

        ## bayesian optimization
        for i in range(self.k_outer):
            ax.scatter(
                x = range(self.n_evals),
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

    ## plot convergence of alpha
    def plot_alpha(self,

        ## viz settings
        lw_dot = 1.30,
        lw_sld = 1.1,
        plt_hgt = 9,
        plt_wdt = 20,
        ax_fnt_sze = 11,
        sty = 'whitegrid'
        ):

        ## plot pre-processor
        plt_data = plot_prep(
            trials_list = self.trials_list,
            params = self.params,
            k_outer = self.k_outer,
            n_evals = self.n_evals,
            standardize = self.standardize
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
            title = "Nested K-Fold Cross-Validation: Bayesian Optimized Convergence of α"
        )

        ## y axis
        ax.set_ylabel(
            ylabel = 'α', 
            fontsize = ax_fnt_sze
        )

        ax.set_ylim(
            bottom = plt_data['alpha_min'] - 0.02, 
            top = plt_data['alpha_max'] + 0.02,
        )

        ## x axis
        ax.set_xlabel(
            xlabel = 'Number of Evaluations', 
            fontsize = ax_fnt_sze
        )

        ax.set_xlim(
            left = -(self.n_evals * 0.010), 
            right = self.n_evals + (self.n_evals * 0.010)
        )

        ## bayesian optimization
        for i in range(self.k_outer):
            ax.scatter(
                x = range(self.n_evals),
                y = plt_data['alpha_list'][i],
                color = plt_data['colors'][i],  ## dot colors
                alpha = 0.75,  ## dot opacity
                s = 12  ## dot size
            )

        ## lowest average test error line
        ax.axhline(
            y = plt_data['alpha_mean_test'],
            color = 'black',
            linestyle = 'dotted',
            linewidth = lw_dot
        )

        ## lowest average train-valid error line
        ax.axhline(
            y = plt_data['alpha_mean_min'],
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

    ## plot regularization
    def plot_regular(self,

        ## viz settings
        lw_dot = 1.30,
        lw_sld = 1.1,
        plt_hgt = 9,
        plt_wdt = 20,
        ax_fnt_sze = 11,
        sty = 'whitegrid'
        ):

        ## plot pre-processor
        plt_data = plot_prep(
            trials_list = self.trials_list,
            params = self.params,
            k_outer = self.k_outer,
            n_evals = self.n_evals,
            standardize = self.standardize
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
        if self.standardize is True:
            ax = fig.add_subplot(
                title = 'Nested K-Fold Cross-Validation: Standardized Coefficient Regularization as a Function of Log(λ)'
            )
        else:
            ax = fig.add_subplot(
                title = 'Nested K-Fold Cross-Validation: Coefficient Regularization as a Function of Log(λ)'
            )

        ## y axis
        if self.standardize is True:
            ax.set_ylabel(
                ylabel = 'Coefficients',
                fontsize = ax_fnt_sze
            )
        else:
            ax.set_ylabel(
                ylabel = 'Standardized Coefficients',
                fontsize = ax_fnt_sze
            )

        ## x axis
        ax.set_xlabel(
            xlabel = 'Log(λ)',
            fontsize = ax_fnt_sze
        )

        ax.set_xlim(
            left = plt_data['lamb_min'] - (plt_data['lamb_min'] * 0.10),
            right = plt_data['lamb_max'] + (plt_data['lamb_max'] * 0.10)
        )

        ax.set_xscale(
            value = 'log'
        )

        ## plot coefficient regularization
        lamb_list_sort = [None] * self.k_outer
        coef_list_sort = [None] * self.k_outer

        for i in range(self.k_outer):

            ## sort lambda values
            lamb_list_sort[i], coef_list_sort[i] = zip(*sorted(zip(
                plt_data['lamb_list'][i], plt_data['coef_list'][i]
                )
              )
            )

            ## coefficients lines
            ax.plot(
                lamb_list_sort[i],
                coef_list_sort[i],
                color = plt_data['colors'][i],  ## line colors
                alpha = 0.40  ## line opacity
            )

        ## plot lowest average test set error line
        ax.axvline(
            x = plt_data['lamb_mean_min'],
            color = 'black', 
            linestyle = 'dotted',
            linewidth = lw_dot
        )

        ## plot lowest average train-valid error line
        ax.axvline(
            x = plt_data['lamb_mean_test'],
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

    ## plot coefficients
    def plot_coef(self,

        ## viz settings
        lw_dot = 1.30,
        lw_sld = 1.1,
        plt_hgt = 9,
        plt_wdt = 20,
        ax_fnt_sze = 11,
        sty = 'whitegrid'
        ):

        ## plot pre-processor
        plt_data = plot_prep(
            trials_list = self.trials_list,
            params = self.params,
            k_outer = self.k_outer,
            n_evals = self.n_evals,
            standardize = self.standardize
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
        if self.standardize is True:
            ax = fig.add_subplot(
                title = 'Nested K-Fold Cross-Validation: Standardized Coefficients of Bayesian Optimized Log(λ)'
            )
        else:
            ax = fig.add_subplot(
                title = 'Nested K-Fold Cross-Validation: Coefficients of Bayesian Optimized Log(λ)'
            )

        ## y axis
        if self.standardize is True:
            ax.set_ylabel(
                ylabel = 'Standardized Coefficients',
                fontsize = ax_fnt_sze
            )
        else:
            ax.set_ylabel(
                ylabel = 'Coefficients',
                fontsize = ax_fnt_sze
            )

        bar_num = np.arange(
            len(self.cols) - 1
        )

        bar_width = (
            (plt_wdt / self.k_outer) * (0.04 + (self.k_outer * 0.0006))
        )

        ## x axis
        x_labels = self.cols.drop(labels = self.y)

        ax.set_xlabel(
            xlabel = 'Explanatory Features',
            fontsize = ax_fnt_sze
        )

        ax.set_xlim(
            left = 0, 
            right = bar_num[-1] + 1
        )

        ## x ticks
        ax.set_xticks(
            ticks = np.arange(len(self.cols))
        )

        ax.set_xticklabels('')

        ax.set_xticks(
            ticks = bar_num + 0.5,
            minor = True
        )

        ax.set_xticklabels(
            labels = x_labels,
            rotation = 90,
            minor = True
        )

        for t in ax.xaxis.get_ticklabels():
            t.set_horizontalalignment('center')

        ## plot coefficients
        for i in range(self.k_outer):
            ax.bar(
                x = (bar_num + bar_width) + (bar_width * i),
                height = list(self.coef_list[i]),
                width = bar_width,
                facecolor = plt_data['colors'][i], 
                edgecolor = plt_data['colors'][i],
                linewidth = 0,
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