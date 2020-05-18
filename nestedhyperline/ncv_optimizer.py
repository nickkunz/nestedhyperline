## load libraries
import numpy as np
import pandas as pd
import warnings as wn

## mested k-fold cross-validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

## bayesian optimization and modeling
from hyperopt import fmin, tpe, Trials, STATUS_OK
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

## internal
from nestedhyperline.results import RegressResults
from nestedhyperline.regressor_select import reg_select

## nested cross-validation and bayesian optimization
def ncv_optimizer(

    ## main func args
    data, y, loss, k_outer, k_inner, n_evals, 
    random_state, standardize, verbose,

    ## pred func args
    method, params
    ):

    """ The main underlying function. Quickly obtains prediction results 
    by compromising implementation details and flexibility.

    Applicable only to linear regression problems. Unifies three important 
    supervised learning techniques for structured data:

    1) Nested K-Fold Cross Validation (minimize bias)
    2) Bayesian Optimization (efficient hyper-parameter tuning)
    3) Linear Regularization (reduce model complexity)

    Bayesian hyper-parameter optimization is conducted utilizing Tree 
    Prezen Estimation. Linear Regularization is conducted utilizing 
    specified method.

    Returns custom regression object containing:
    - Root Mean Squared Error (RMSE) or other specified regression metric
    - List of RMSE on Outer K-Folds """

    ## suppress warning messages
    wn.filterwarnings(
        action = 'ignore',
        category = DeprecationWarning
    )

    wn.filterwarnings(
        action = 'ignore',
        category = FutureWarning
    )

    ## loss function
    if loss == "explained variance" or loss == "ev":
        loss_func = metrics.explained_variance_score

    if loss == "max error" or loss == "me":
        loss_func = metrics.max_error

    if loss == "mean absolute error" or loss == "mae":
        loss_func = metrics.mean_absolute_error

    if loss == "mean squared error" or loss == "mse":
        loss_func = metrics.mean_squared_error

    if loss == "root mean squared error" or loss == "rmse":
        loss_func = metrics.mean_squared_error

    if loss == "median absolute error" or loss == "mdae":
        loss_func = metrics.median_absolute_error

    if loss == "r2":
        loss_func = metrics.r2_score

    if loss == "mean poisson deviance" or loss == "mpd":
        loss_func = metrics.mean_poisson_deviance

    if loss == "mean gamma deviance" or loss == "mgd":
        loss_func = metrics.mean_gamma_deviance

    ## reset data index
    data.reset_index(
        inplace = True,
        drop = True
    )

    ## store feature names
    column_names = data.columns
    
    ## standardize explanatory features x
    if standardize == True:
        data = StandardScaler().fit_transform(data)
        data = pd.DataFrame(
            data = data,
            columns = column_names
        )

    ## test set prediction stores
    trials_list = []
    error_list = []
    coef_list = []
    model_list = []
    params_list = []

    ## outer k-fold specification
    k_folds_outer = KFold(
        n_splits = k_outer,
        shuffle = False
    )

    ## split data into train-valid and test sets
    for train_valid_index, test_index in k_folds_outer.split(data):

        ## explanatory features x
        x_train_valid, x_test = data.drop(y, axis = 1).iloc[
            train_valid_index], data.drop(y, axis = 1).iloc[
                test_index]

        ## response variable y
        y_train_valid, y_test = data[y].iloc[
            train_valid_index], data[y].iloc[
                test_index]

        ## objective function
        def obj_func(params):

            """ objective function to minimize utilizing
            bayesian hyper-parameter optimization """

            ## inner k-folds
            k_folds_inner = KFold(
                n_splits = k_inner,
                shuffle = False
            )

            ## split data into train and valid sets
            for train_index, valid_index in k_folds_inner.split(x_train_valid):

                ## explanatory features x
                x_train, x_valid = x_train_valid.iloc[
                    train_index], x_train_valid.iloc[
                        valid_index]

                ## response variable y
                y_train, y_valid = y_train_valid.iloc[
                    train_index], y_train_valid.iloc[
                        valid_index]

                ## method and params
                model = reg_select(
                    method = method,
                    params = params,
                    random_state = random_state
                )

                ## training set
                model = model.fit(
                    X = x_train,
                    y = y_train
                )

                ## store coefficients
                coef = model.coef_

                ## predict on valid set
                y_pred = model.predict(x_valid)

                ## calculate loss
                if loss == "root mean squared error" or loss == "rmse":

                    ## squared root loss
                    error = loss_func(
                        y_true = y_valid,
                        y_pred = y_pred,
                        squared = False,  ## false returns rmse
                    )
                else:
                    ## squared loss
                    error = loss_func(
                        y_true = y_valid,
                        y_pred = y_pred
                    )

            ## average loss
            error_mean = np.average(error)

            ## return average cross-valid loss
            return {
                'loss': error_mean, 
                'coef': coef,
                'status': STATUS_OK
            }

        ## record results
        trials = Trials()

        ## conduct bayesian optimization
        params_opt = fmin(
            fn = obj_func,
            space = params,
            algo = tpe.suggest,  ## tree parzen estimation
            max_evals = n_evals,
            trials = trials,
            show_progressbar = verbose
        )

        ## model with optimal hyper-params
        model_opt = reg_select(
            method = method,
            params = params_opt,
            random_state = random_state
        )

        ## train on entire train-valid set
        model_opt = model_opt.fit(
            X = x_train_valid,
            y = y_train_valid
        )

        ## store coefficients
        coef = model_opt.coef_

        ## predict on test set
        y_pred = model_opt.predict(x_test)

        ## calculate loss
        if loss == "root mean squared error" or loss == "rmse":

            ## squared root loss
            error_list.append(
                loss_func(
                    y_true = y_test,
                    y_pred = y_pred,
                    squared = False  ## false returns rmse
                )
            )
        else:
            ## squared loss
            error_list.append(
                loss_func(
                    y_true = y_test,
                    y_pred = y_pred
                )
            )

        ## outer cross-valid results
        model_list.append(model_opt)
        params_list.append(params_opt)
        trials_list.append(trials)
        coef_list.append(coef)

    ## custom object
    return RegressResults(
        y = y,
        cols = column_names,
        model = model_list,
        params = params_list,
        trials_list = trials_list,
        error_list = error_list,
        coef_list = coef_list,
        standardize = standardize,
        k_outer = k_outer,
        n_evals = n_evals
    )
