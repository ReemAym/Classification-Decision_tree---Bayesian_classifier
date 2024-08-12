import numpy as np


# decision tree
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # Index of feature to split on
        self.threshold = threshold  # Threshold value for the split
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Predicted class if node is a leaf


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Stopping criteria
        if depth == self.max_depth or n_classes == 1 or n_samples < 2:
            return Node(value=np.bincount(y).argmax())

        # Find best split for continuous features
        best_gain = 0
        best_feature = None
        best_threshold = None
        for feature in range(n_features):
            if isinstance(X[0, feature], (int, float)):  # Check if the feature is continuous
                thresholds = np.unique(X[:, feature])
                for threshold in thresholds:
                    left_indices = np.where(X[:, feature] <= threshold)[0]
                    right_indices = np.where(X[:, feature] > threshold)[0]

                    if len(left_indices) == 0 or len(right_indices) == 0:
                        continue

                    gain = self._information_gain(y, y[left_indices], y[right_indices])
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = threshold
            else:  # Handle categorical features
                unique_values = np.unique(X[:, feature])
                for value in unique_values:
                    left_indices = np.where(X[:, feature] == value)[0]
                    right_indices = np.where(X[:, feature] != value)[0]

                    if len(left_indices) == 0 or len(right_indices) == 0:
                        continue

                    gain = self._information_gain(y, y[left_indices], y[right_indices])
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = None  # No threshold for categorical features

        if best_gain == 0:
            return Node(value=np.bincount(y).argmax())

        if best_threshold is not None:  # If a threshold is found, split using it
            left_indices = np.where(X[:, best_feature] <= best_threshold)[0]
            right_indices = np.where(X[:, best_feature] > best_threshold)[0]
        else:  # If no threshold (for categorical features), split based on value equality
            left_indices = np.where(X[:, best_feature] == unique_values[0])[0]
            right_indices = np.where(X[:, best_feature] != unique_values[0])[0]

        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))

    def _information_gain(self, y, left_y, right_y):
        parent_entropy = self._entropy(y)
        left_entropy = self._entropy(left_y)
        right_entropy = self._entropy(right_y)
        left_weight = len(left_y) / len(y)
        right_weight = len(right_y) / len(y)
        return parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)

    def predict(self, X):
        return np.array([self._predict_tree(x, self.root) for x in X])

    def _predict_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_tree(x, node.left)
        else:
            return self._predict_tree(x, node.right)

#naive



# # Read the CSV file
# file_path = "diabetes_prediction_dataset.csv"
# # data = pd.read_csv(file_path)
# percentage = float(input("Enter the percentage of data to read (0-100): "))
# # Read the specified percentage of rows randomly
# data = pd.read_csv(file_path).sample(frac=percentage / 100)
# # Display the first few rows of the dataset
# # print(data.head())
#
# # Split dataset into features and target variable
# X = data.drop(columns=['diabetes']).values
# y = data['diabetes'].values
#
# # Allow user to input the percentage of training and test data
# train_percentage = float(input("Enter the percentage of training data (0-100): "))
# test_percentage = float(input("Enter the percentage of test data (0-100): "))
# num_records = int(test_percentage / 100.0 * len(data))
# # Split the dataset into training and test sets
# X_train_DT, X_test_DT, y_train_DT, y_test_DT = train_test_split(X, y, train_size=train_percentage / 100,
#                                                     test_size=test_percentage / 100)
#
# # Instantiate and fit the decision tree model
# tree = DecisionTree(max_depth=5)
# tree.fit(X_train_DT, y_train_DT)
#
# # Make predictions on the test data
# y_pred_DT = tree.predict(X_test_DT)
#
# # Compute the accuracy of the model
# accuracy_DT = accuracy_score(y_test_DT, y_pred_DT)
#
# print("Accuracy of the decision tree model:", accuracy_DT)
#
# # Allow user to input a record of data to predict its class
# # record = []
# # for feature_name in data.columns[:-1]:  # Exclude the target column
# #     value = input(f"Enter the value of {feature_name}: ")
# #     if(feature_name == 'gender' or feature_name == 'smoking_history'):
# #          record.append(value)
# #     else:
# #         record.append(float(value))
# #
# # # Predict the class of the input record
# # predicted_class = tree.predict([record])[0]
# # print("Predicted class:", predicted_class)
#
# for i in range(num_records):
#     record_DT = y_pred_DT[i]  # Get the feature values of the record
#     # Predict the class of the input record
#     # predicted_class = tree.predict([record])[0]
#     print(f"Record {i + 1} - Predicted class: {record_DT}")
#
# #naive
#
# # Split the dataset into training and test sets
# X_train_N, X_test_N, y_train_N, y_test_N = train_test_split(X, y, train_size=train_percentage / 100,
#                                                     test_size=test_percentage / 100)
#
# # Instantiate and fit the Naive Bayes model
# naive_bayes = NaiveBayes()
# naive_bayes.fit(X_train_N, y_train_N)
#
# # Make predictions on the test data
# y_pred_N = naive_bayes.predict(X_test_N)
#
# # Compute the accuracy of the model
# accuracy_N = accuracy_score(y_test_N, y_pred_N)
# print("Accuracy of the Naive Bayes model:", accuracy_N)
#
# # Allow user to input a record of data to predict its class
# num_records = int(test_percentage / 100.0 * len(data))
# for i in range(num_records):
#     record_N = y_pred_N[i]  # Get the feature values of the record
#     print(f"Record {i + 1} - Predicted class: {record_N}")
#
# if accuracy_N > accuracy_DT:
#     print("Bayesian classifier is better")
# elif accuracy_N < accuracy_DT:
#     print("Decision tree classifier is better")
# else:
#     print("Two classifier have the same accuracy")





# Import the classes and functions from the previous code



