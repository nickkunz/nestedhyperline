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
    if loss == "explained_variance" or loss == "ev":
        error_to_score = loss
        error = metrics.explained_variance_score
    
    if loss == "max_error" or loss == "me":
        error_to_score = loss
        error = metrics.max_error
    
    if loss == "mean_absolute_error" or loss == "mae":
        error_to_score = "mean_absolute_error"
        error = metrics.mean_absolute_error
    
    if loss == "mean_squared_error" or loss == "mse":
        error_to_score = "neg_mean_squared_error"
        error = metrics.mean_squared_error
    
    if loss == "root_mean_squared_error" or loss == "rmse":
        error_to_score = "neg_root_mean_squared_error"
        error = metrics.mean_squared_error
    
    if loss == "median_absolute_error" or loss == "mdae":
        error_to_score = "neg_median_absolute_error"
        error = metrics.median_absolute_error
    
    if loss == "r2":
        error_to_score = loss
        error = metrics.r2_score
    
    if loss == "mean_poisson_deviance" or loss == "mpd":
        error_to_score = "neg_mean_poisson_deviance"
        error = metrics.mean_poisson_deviance
    
    if loss == "mean_gamma_deviance" or loss == "mgd":
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
    x_test_list = []
    error_list = []
    
    ## outer loop k-folds
    k_folds_outer = KFold(
        n_splits = k_outer,
        shuffle = False
    )
    
    ## split data into training-validation and test sets by k-folds
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
            
            ## method, params, and objective
            model = reg_select(
                method = method,
                params = params,
                seed = seed
            )
            
            ## inner loop cross-valid
            cv_scores = cross_val_score(
                estimator = model,
                X = x_train_valid,
                y = y_train_valid,
                scoring = error_to_score,
                cv = KFold(
                    n_splits = k_inner,
                    random_state = seed,
                    shuffle = False
                ),
                n_jobs = -1  ## utilize all cores
            )
            
            ## average the minimized inner loop cross-valid scores
            cv_scores_mean = 1 - np.average(cv_scores)
            
            ## return averaged cross-valid scores and status report
            return {'loss': cv_scores_mean, 'status': STATUS_OK}
        
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
        x_test_list.append(x_test)
        
        ## calculate error
        if loss == "root_mean_squared_error":
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
    
    return RegressResults(
        trials = trials,
        model = model_opt,
        params = params_opt,
        error_list = error_list
    )
