# PyTorch functions package

This is a small package to get functions for train and test your machine learning model, then make predictions with it.  
This package is intended for small project use. For bigger project, itβs better to use [PyTorch Lightening](https://github.com/PyTorchLightning/pytorch-lightning).

# Plan π

1. Follow instructions on [PyPI website](https://packaging.python.org/tutorials/packaging-projects/)
2. Write functions :
    - train
    - valid
    - test
    - predict
3. Write some unit testing codes.
4. Make it "wrappable" for scikit-learn usage.
5. Update and maintaint it.

# DONE β
- Functions are written.
- String stub documentation.

# TODO π―
- Score in train / valid / test : Make it general for all wanted scores (only accuracy at the moment).
- Remove mean at the return of validation and test functions (designed for accuracy).
- Check CPU treatment
