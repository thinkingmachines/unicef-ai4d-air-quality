"""Sample test for src.sample module"""

import pytest

from src.sample import NaiveClassifier


@pytest.fixture
def trained_classifier():
    X = [[1, 2], [3, 4], [5, 6]]
    y = [0, 1, 0]

    clf = NaiveClassifier()
    clf.fit(X, y)
    return clf


def test_classifier_should_be_fitted_after_training():
    # Sample data
    X = [[1, 2], [3, 4], [5, 6]]
    y = [0, 1, 0]

    clf = NaiveClassifier()
    clf.fit(X, y)
    assert clf.is_fitted


@pytest.mark.parametrize("test_sample", [[1, 2], [0, 4], [8, 9]])
def test_classifier_should_return_1_prediction(trained_classifier, test_sample):
    pred = trained_classifier.predict(test_sample)
    assert pred == 1
