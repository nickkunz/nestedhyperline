from setuptools import setup

setup(
    name = 'nestedhyperline',
    version = '0.0.6',
    description = 'A wrapper for conducting Nested Cross-Validation for Bayesian Optimized Linear Regularization',
    long_description = open('README.md').read(),
    long_description_content_type = "text/markdown",
    author = 'Nick Kunz',
    author_email = 'nick.kunz@me.com',
    url = 'https://github.com/nickkunz/nestedhyperline',
    classifiers = [
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'
        ],
    keywords = [
        'nested cross-validation',
        'bayesian optimization',
        'linear regularization',
        'ridge',
        'lasso',
        'elastic net'
    ],
    packages = [
        'nestedhyperline',
        'nestedhyperline.regressors'
    ],
    include_package_data = True,
    install_requires = [
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'sklearn',
        'hyperopt'
    ],
    tests_require = ['nose'],
    test_suite = 'nose.collector'
)
