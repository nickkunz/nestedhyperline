## load libraries
import numpy as np
import pandas as pd
import warnings as wn

## mested k-fold cross-validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold

## bayesian hyper-parameter optimization and modeling
from hyperopt import fmin, tpe, Trials, STATUS_OK

## performance evaluation
from sklearn import metrics

## internal
from nestedhyperline.argument_quality import ArgumentQuality
from nestedhyperline.method_select import reg_select

## nested cross-validation and bayesian hyper-param optimization
def ncv_optimizer(
    
    ## main func args
    data, y, loss, k_outer, k_inner, n_evals, seed, verbose,
    
    ## pred func args
    method, params
    ):
    
    """
    main underlying function, designed for rapid prototyping, quickly obtain
    prediction results by compromising implementation details and flexibility
    
    can be applied to regression, multi-class classification, and binary
    classification problems, unifies three important supervised learning
    techniques for structured data:
    
    1) nested k-fold cross validation (minimize bias)
    2) bayesian optimization (efficient hyper-parameter tuning)
    3) gradient boosting (flexible and extensive prediction)

    bayesian hyper-parameter optimization is conducted utilizing tree prezen
    estimation, gradient boosting is conducted utilizing user specified methods
    
    returns custom object depending on the type of prediction
    - regressor: root mean squared error (or other regression metric)
    - classifier: accuracy, prec-recall-f1-support, confusion matrix, roc auc
    - all cases: feature importance plot, hyperopt trials object
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
    if loss == "explained_variance":
        error_to_score = loss
        error = metrics.explained_variance_score
    
    if loss == "max_error":
        error_to_score = loss
        error = metrics.max_error
    
    if loss == "mean_absolute_error":
        error_to_score = "mean_absolute_error"
        error = metrics.mean_absolute_error
    
    if loss == "mean_squared_error":
        error_to_score = "neg_mean_squared_error"
        error = metrics.mean_squared_error
    
    if loss == "root_mean_squared_error":
        error_to_score = "neg_root_mean_squared_error"
        error = metrics.mean_squared_error
    
    if loss == "median_absolute_error":
        error_to_score = "neg_median_absolute_error"
        error = metrics.median_absolute_error
    
    if loss == "r2":
        error_to_score = loss
        error = metrics.r2_score
    
    if loss == "mean_poisson_deviance":
        error_to_score = "neg_mean_poisson_deviance"
        error = metrics.mean_poisson_deviance
    
    if loss == "mean_gamma_deviance":
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
