## load libraries
import numpy as np
import warnings as wn

## mested k-fold cross-validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

## bayesian hyper-parameter optimization and modeling
from hyperopt import fmin, tpe, Trials, STATUS_OK

## performance evaluation
from sklearn import metrics

## internal
from nestedhyperline.results import RegressResults
from nestedhyperline.regressor_select import reg_select

## nested cross-validation and bayesian hyper-param optimization
def ncv_optimizer(

    ## main func args
    data, y, loss, k_outer, k_inner, n_evals, seed, verbose,

    ## pred func args
    method, params
    ):

    """
    The main underlying function designed for rapid prototyping. Quickly obtains 
    prediction results by compromising implementation details and flexibility.

    Applicable only to linear regression problems. Unifies three important 
    supervised learning techniques for structured data:

    1) Nested K-Fold Cross Validation (minimize bias)
    2) Bayesian Optimization (efficient hyper-parameter tuning)
    3) Linear Regularization (reduce model complexity)

    Bayesian hyper-parameter optimization is conducted utilizing Tree Prezen
    Estimation. Linear Regularization is conducted utilizing specified method.

    Returns custom regression object containing:
    - Root Mean Squared Error (RMSE) or other specified regression metric
    - List of RMSE on outer-folds
    """

    ## suppress warning messages
    wn.filterwarnings(
        action = 'ignore',
        category = DeprecationWarning
    )

    wn.filterwarnings(
        action = 'ignore',
        category = FutureWarning
    )

    ## set loss function
    if loss == "explained variance" or loss == "ev":
        error_to_score = loss
        error = metrics.explained_variance_score

    if loss == "max error" or loss == "me":
        error_to_score = loss
        error = metrics.max_error

    if loss == "mean absolute error" or loss == "mae":
        error_to_score = "mean_absolute_error"
        error = metrics.mean_absolute_error

    if loss == "mean squared error" or loss == "mse":
        error_to_score = "neg_mean_squared_error"
        error = metrics.mean_squared_error

    if loss == "root mean squared error" or loss == "rmse":
        error_to_score = "neg_root_mean_squared_error"
        error = metrics.mean_squared_error

    if loss == "median absolute error" or loss == "mdae":
        error_to_score = "neg_median_absolute_error"
        error = metrics.median_absolute_error

    if loss == "r2":
        error_to_score = loss
        error = metrics.r2_score

    if loss == "mean poisson deviance" or loss == "mpd":
        error_to_score = "neg_mean_poisson_deviance"
        error = metrics.mean_poisson_deviance

    if loss == "mean gamma deviance" or loss == "mgd":
        error_to_score = "neg_mean_gamma_deviance"
        error = metrics.mean_gamma_deviance

    ## reset data index
    data.reset_index(
        inplace = True,
        drop = True
    )

    ## test set prediction stores
    y_test_list = []
    y_pred_list = []
    trials_list = []
    error_list = []

    ## outer k-folds
    k_folds_outer = KFold(
        n_splits = k_outer,
        shuffle = False
    )

    ## split data into training-validation and test sets
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
        def obj_fun(params):

            """ objective function to minimize utilizing
            bayesian hyper-parameter optimization """

            ## inner k-folds
            k_folds_inner = KFold(
                n_splits = k_inner,
                shuffle = False
            )

            ## split data into training-and validation test sets
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
                    seed = seed
                )

                ## training  set
                model = model.fit(
                    X = x_train,
                    y = y_train
                )

                ## make prediction on validation set
                y_pred = model.predict(x_valid)

                ## store coefficients
                coef = model.coef_

                ## calculate loss
                if loss == "root_mean_squared_error":
                    
                    ## squared loss
                    loss = error(
                        y_true = y_valid,
                        y_pred = y_pred,
                        squared = True,
                    )

                else:
                    loss = error(
                        y_true = y_valid,
                        y_pred = y_pred
                    )

            ## average loss
            loss_mean = np.average(loss)

            ## return averaged cross-valid scores and status report
            return {
                'loss': loss_mean, 
                'coef': coef,
                'status': STATUS_OK
            }

        ## record results
        trials = Trials()

        ## conduct bayesian optimization, inner loop cross-valid
        params_opt = fmin(
            fn = obj_fun,
            space = params,
            algo = tpe.suggest,  ## tree parzen estimation
            max_evals = n_evals,
            trials = trials,
            show_progressbar = verbose
        )

        ## modeling method with optimal hyper-params
        model_opt = reg_select(
            method = method,
            params = params_opt,
            seed = seed
        )

        ## train on entire training-validation set
        model_opt = model_opt.fit(
            X = x_train_valid,
            y = y_train_valid
        )

        ## make prediction on test set
        y_pred = model_opt.predict(x_test)

        ## store outer cross-validation results
        y_test_list.append(y_test)
        y_pred_list.append(y_pred)
        trials_list.append(trials)

        ## calculate loss
        if loss == "root_mean_squared_error":
            
            ## squared loss
            error_list.append(
                error(
                    y_true = y_test,
                    y_pred = y_pred,
                    squared = True
                )
            )

        else:
            error_list.append(
                error(
                    y_true = y_test,
                    y_pred = y_pred
                )
            )

    ## custom regression object
    return RegressResults(
        model = model_opt,
        params = params_opt,
        trials_list = trials_list,
        y_pred_list = y_pred_list,
        error_list = error_list,
        coef_list = coef_list
    )
