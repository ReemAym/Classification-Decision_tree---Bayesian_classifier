
import numpy as np


class NaiveBayes:
    def __init__(self):
        self.class_probabilities = {}
        self.feature_probabilities = {}

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)

        # Calculate class probabilities
        for cls in self.classes:
            self.class_probabilities[cls] = np.sum(y == cls) / n_samples

        # Calculate feature probabilities
        for cls in self.classes:
            self.feature_probabilities[cls] = []
            for feature in range(n_features):
                if isinstance(X[0, feature], (int, float)):  # Continuous feature
                    feature_values = X[y == cls, feature]
                    mean = np.mean(feature_values)
                    std = np.std(feature_values)
                    self.feature_probabilities[cls].append((mean, std))
                else:  # Categorical feature
                    feature_values = X[y == cls, feature]
                    unique_values, counts = np.unique(feature_values, return_counts=True)
                    probabilities = (counts + 1) / (len(feature_values) + len(unique_values))  # Laplace smoothing
                    feature_prob_dict = dict(zip(unique_values, probabilities))
                    self.feature_probabilities[cls].append(feature_prob_dict)

    def _calculate_class_probability(self, x, cls):
        class_probability = self.class_probabilities[cls]
        for feature, value in enumerate(x):
            if isinstance(value, (int, float)):  # Continuous feature
                mean, std = self.feature_probabilities[cls][feature]
                # Use Gaussian probability density function for continuous features
                class_probability *= (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((value - mean) ** 2) / (2 * std ** 2))
            else:  # Categorical feature
                if value in self.feature_probabilities[cls][feature]:
                    class_probability *= self.feature_probabilities[cls][feature][value]
                else:
                    # Laplace smoothing for unseen values
                    class_probability *= 1 / (len(self.feature_probabilities[cls][feature]) + 1)
        return class_probability


    def predict(self, X):
        predictions = []
        for x in X:
            max_probability = float('-inf')
            predicted_class = None
            for cls in self.classes:
                class_probability = self._calculate_class_probability(x, cls)
                if class_probability > max_probability:
                    max_probability = class_probability
                    predicted_class = cls
            predictions.append(predicted_class)
        return predictions