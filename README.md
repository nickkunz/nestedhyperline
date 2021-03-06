<div align="center">
  <img src="https://github.com/nickkunz/nestedhyperline/blob/master/media/logos/nestedhyperline_banner.png">
</div>

## Nested Cross-Validation for Bayesian Optimized Linear Regularization
[![PyPI version](https://badge.fury.io/py/nestedhyperline.svg)](https://badge.fury.io/py/nestedhyperline)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Build Status](https://travis-ci.com/nickkunz/nestedhyperline.svg?branch=master)](https://travis-ci.com/nickkunz/nestedhyperline)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/1a851e718e1441adb251c14458d20b3b)](https://www.codacy.com/manual/nickkunz/nestedhyperline?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=nickkunz/nestedhyperline&amp;utm_campaign=Badge_Grade)
![GitHub last commit](https://img.shields.io/github/last-commit/nickkunz/nestedhyperline)

## Description
A Python implementation that unifies Nested K-Fold Cross-Validation, Bayesian Hyperparameter Optimization, and Linear Regularization. Designed for rapid prototyping on small to mid-sized data sets (can be manipulated within memory). Quickly obtains high quality prediction results by abstracting away tedious hyperparameter tuning and implementation details in favor of usability and implementation speed. Bayesian Hyperparamter Optimization utilizes Tree Parzen Estimation (TPE) from the <a href="https://github.com/hyperopt/hyperopt">Hyperopt</a> package. Linear Regularization can be conducted one of three ways. Select between <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html">Ridge</a>, <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html">Lasso</a>, or <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html">Elastic-Net</a>. Useful where linear regression is applicable.

## Features
1. Consistent syntax across all Linear Regularization methods.
2. Supported Linear Regularization methods: <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html">Ridge</a>, <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html">Lasso</a>, <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html">Elastic-Net</a>.
3. Returns custom object that includes performance metrics and plots.
4. Developed for readability, maintainability, and future improvement.

## Requirements
1. Python 3
2. NumPy
3. Pandas
4. MatPlotLib
5. Seaborn
6. Scikit-Learn
7. Hyperopt

## Installation
```python
## install pypi release
pip install nestedhyperline

## install developer version
pip install git+https://github.com/nickkunz/nestedhyperline.git
```

## Usage
```python
## load libraries
from nestedhyperline import regressors
from sklearn import datasets
import pandas

## load data
housing_sklearn = datasets.load_boston()
housing = pandas.DataFrame(housing_sklearn.data, columns = housing_sklearn.feature_names)
housing['target'] = pandas.Series(housing_sklearn.target)

## conduct lasso regression
results = regressors.lasso_ncv_regressor(
    data = housing,
    y = 'target',
    loss = 'mse',
    k_inner = 3,
    k_outer = 3,
    n_evals = 300
)

## preview performance
results.error_mean()

## preview plots
results.plot_error_mean()
results.plot_lambda()
results.plot_regular()
results.plot_coef()
```

## Examples
https://github.com/nickkunz/nestedhyperline/blob/master/examples/nestedhyperline_example_ridge.ipynb

## License
© Nick Kunz, 2019. Licensed under the General Public License v3.0 (GPLv3).

## Contributions
NestedHyperLine is open for improvements and maintenance. Your help is valued to make the package better for everyone.

## References
Bergstra, J., Bardenet, R., Bengio, Y., Kegl, B. (2011). Algorithms for Hyper-Parameter Optimization. https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf.

Bergstra, J., Yamins, D., Cox, D. D. (2013). Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures. 
Proceedings of the 30th International Conference on International Conference on Machine Learning. 28:I115–I123. http://proceedings.mlr.press/v28/bergstra13.pdf.

Hoerl, Arthur E., Kennard, Robert W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. American Statistical Association and American Society for Quality Stable. 12(1):55-67. https://doi.org/10.1080/00401706.1970.10488634.

Tibshirani, R. (1996).  Regression Shrinkage and Selection Via the Lasso. Journal of the Royal Statistical Society: Series B (Methodological). 58(1):267-288. https://doi.org/10.1111/j.2517-6161.1996.tb02080.x.

Zou, H., Hastie, T. (2005). Regularization and Variable Selection via the Elastic Net. Journal of the Royal Statistical Society: Series B (Statistical Methodology). 67: 301-320. https://doi.org/10.1111/j.1467-9868.2005.00503.x.
