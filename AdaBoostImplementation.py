'''
This program is an implementation of binary classification using AdaBoost.
'''

import numpy as np


# Single-node tree builder
class Stump():
    def __init__(self):
        self.polarity = 1
        self.featureIndex = None
        self.accuracy = None
        self.threshold = None

    def makePrediction(self, dataArray):
        numInstances = dataArray.shape[0]
        feature = dataArray[:, self.featureIndex]

        # Predictions are default set to one
        predictions = np.ones(numInstances)

        if self.polarity == 1:
            predictions[feature < self.threshold] = -1
        else:
            predictions[feature > self.threshold] = -1

        return predictions


class Adaboost():
    def __init__(self, n_clf=5):
        self.n_clf = n_clf

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights to 1/N
        w = np.full(n_samples, (1 / n_samples))

        self.clfs = []
        # Iterate through classifiers
        for _ in range(self.n_clf):
            clf = Stump()
            min_error = float('inf')

            # greedy search to find best threshold and feature
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    # predict with polarity 1
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1

                    # Error = sum of weights of misclassified samples
                    misclassified = w[y != predictions]
                    error = sum(misclassified)

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    # store the best configuration
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i
                        min_error = error

            # calculate alpha
            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))

            # calculate predictions and update weights
            predictions = clf.makePrediction(X)


            w *= np.exp(-clf.alpha * y * predictions)
            # Normalize to one
            w /= np.sum(w)

            # Save classifier
            self.clfs.append(clf)


        def predict(self, X):
            clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
            y_pred = np.sum(clf_preds, axis=0)
            y_pred = np.sign(y_pred)

            return y_pred
