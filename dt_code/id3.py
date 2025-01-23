import arff
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
def entropy(p):
    if p > 0:
        return -p * np.log2(p)
    else:
        return 0

class DTClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self):
        """ Initialize class with chosen hyperparameters.
        Args:
            hidden_layer_widths (list(int)): A list of integers which defines the width of each hidden layer
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
        Example:
            DT  = DTClassifier()
        """
        self.graph = {():[]}

    def fit(self, X, y):
        """ Fit the data; Make the Desicion tree
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        X = X.astype(int)
        y = y.flatten().astype(int)
        self.num_features = X.shape[1]
        self.feature_class = X.max(axis=0) + 1
        self.num_class = int(y.max()) + 1
        self._split(X, y, [], (), ())
        self.root = (self.graph[()][0],)
        return self
    
    def _split(self, X, y, mask, parent, path):
        gain = self.calc_gain(X, y, mask)
        split_on = gain.argmin()
        node = path + (split_on,)
        self.graph[parent].append(split_on)
        self.graph[node] = []
        mask.append(split_on)
        F_split = X[:, split_on]
        for i in range(self.feature_class[split_on]):
            y_p = y[F_split==i]
            if len(y_p) == 0 or len(mask) > self.num_features:
                vals, counts = np.unique(y, return_counts=True)
                ind = counts.argmax()
                self.graph[node].append([vals[ind]])
            elif len(set(y_p)) == 1:
                y_val = set(y_p).pop()
                self.graph[node].append([y_val])
            else:
                new_path = path + ((split_on, i),)
                self._split(X[F_split==i], y[F_split==i], mask.copy(), node, new_path)
        
    def calc_gain(self, X, y, mask):
        n = X.shape[0]
        gain = []
        for i in range(self.num_features):
            if i in mask:
                gain.append(np.inf)
                continue
            F = X[:, i]
            P = np.vstack((F, y)).T
            info = 0
            for j in range(int(F.max()) + 1):
                d = sum(F==j)
                ratio = d / n
                if d == 0:
                    continue
                curr_entropy = sum(entropy(sum(P[F==j][:,-1]==k) / d) for k in range(self.num_class))
                info += ratio*curr_entropy
            gain.append(info)
        return np.array(gain)
        
    def predict(self, X):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        X = X.astype(int)
        y_hat = []
        for x in X:
            y_hat.append(self._find(x, self.root))
        return np.array(y_hat).reshape(-1, 1)

    def _find(self, x, node):
        z = x[node[-1]]
        v = self.graph[node][x[node[-1]]]
        if type(v) == list:
            return v[0]
        else:
            node = node[:-1] + ((node[-1], z),) + (v,)
            return self._find(x, node)
    
    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.
        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-li    def _shuffle_data(self, X, y):
        """
        y = y.astype(int)
        y_hat = self.predict(X)
        return (y_hat == y).sum() / len(y)
