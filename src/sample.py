"""Sample module for a project

This file demonstrates the usual way of structuring your business logic. Usually, you
have a high-level docstring that explains the purpose of the module. We use the
`reStructuredText` (RST) format for our docstrings.

For classes and functions, we use the `numpydoc` styleguide to document their
parameters. You can delete this file or repurpose it to your own use-case.

.. _resStructuredText: https://sphinx-tutorial.readthedocs.io/cheatsheet/
.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/format.html
"""

from typing import List


class NaiveClassifier:
    """A classifier that only predicts a single class

    Demonstrates the usual class structure
    """

    def __init__(self, C: int = 1000, gamma: float = 0.01):
        """Initialize a class

        Parameters
        ----------
        C: int
            Regularization parameter
        gamma: float
            Defines how far the influence of a single training example reaches. Low
            values mean 'far', high values mean 'close'
        """
        self.is_fitted = False
        self.C = C
        self.gamma = gamma

    def fit(self, X, y):
        """Fit a model given a training set and its labels

        Parameters
        ----------
        X : array-like
            Training vectors of shape (n_samples, n_features)
        y : array-like
            Target values of shape (n_samples, )
        """
        self.is_fitted = True

    def predict(self, X) -> List[int]:
        """Perform classification for a batch of samples

        Parameters
        ----------
        X : array-like
            Test vectors to infer upon of shape (n_samples, n_features)

        Returns
        -------
        list of int
            Predicted labels for a particular model
        """
        return 1
