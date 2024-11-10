import numpy as np
class CustomDecisionTreeClassifier:
   def __init__(self, max_depth=None):
       self.max_depth = max_depth
       self.tree = None
   def gini_impurity(self, y):
       # Calculate Gini impurity for a list of labels
       class_counts = np.bincount(y)
       probabilities = class_counts / len(y)
       return 1 - np.sum(probabilities ** 2)
   def best_split(self, X, y):
       best_gini = float('inf')
       best_split = None
       best_left_y = None
       best_right_y = None
       best_feature = None
       best_value = None
       n_features = X.shape[1]
       for feature in range(n_features):
           # Iterate through all possible feature values
           unique_values = np.unique(X[:, feature])
           for value in unique_values:
               left_mask = X[:, feature] <= value
               right_mask = ~left_mask
               left_y = y[left_mask]
               right_y = y[right_mask]
               # Calculate Gini impurity for the split
               gini_left = self.gini_impurity(left_y)
               gini_right = self.gini_impurity(right_y)
               # Weighted Gini for the split
               gini_split = (len(left_y) / len(y)) * gini_left + (len(right_y) / len(y)) * gini_right
               # Update the best split if found a better one
               if gini_split < best_gini:
                   best_gini = gini_split
                   best_split = (feature, value)
                   best_left_y = left_y
                   best_right_y = right_y
       return best_split, best_left_y, best_right_y
   def build_tree(self, X, y, depth=0):
       # Stop if maximum depth is reached or if no more splits are possible
       if len(np.unique(y)) == 1 or (self.max_depth and depth >= self.max_depth):
           return np.argmax(np.bincount(y))  # Return the majority class
       # Find the best split
       split, left_y, right_y = self.best_split(X, y)
       if split is None:
           return np.argmax(np.bincount(y))  # Return the majority class if no valid split
       feature, value = split
       left_mask = X[:, feature] <= value
       right_mask = ~left_mask
       # Recursively build the left and right subtrees
       left_tree = self.build_tree(X[left_mask], left_y, depth + 1)
       right_tree = self.build_tree(X[right_mask], right_y, depth + 1)
       return {
           'feature': feature,
           'value': value,
           'left': left_tree,
           'right': right_tree
       }
   def fit(self, X, y):
       self.tree = self.build_tree(X, y)
   def predict_sample(self, sample, tree):
       # Recursive function to predict a single sample
       if isinstance(tree, dict):
           if sample[tree['feature']] <= tree['value']:
               return self.predict_sample(sample, tree['left'])
           else:
               return self.predict_sample(sample, tree['right'])
       else:
           return tree  # Leaf node returns the predicted class
   def predict(self, X):
       # Predict for multiple samples
       return np.array([self.predict_sample(sample, self.tree) for sample in X])
# Example usage:
X_train = np.array([[2, 3], [10, 15], [3, 2], [8, 8], [7, 9], [12, 10]])
y_train = np.array([0, 1, 0, 1, 1, 1])
# Initialize and train the classifier
clf = CustomDecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)
# Make a prediction on a new sample
X_test = np.array([[6, 7]])
predictions = clf.predict(X_test)
print("Predicted class:", predictions)