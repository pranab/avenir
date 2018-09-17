#!/Users/pranab/Tools/anaconda/bin/python

# avenir-python: Machine Learning
# Author: Pranab Ghosh
# 
# Licensed under the Apache License, Version 2.0 (the "License"); you
# may not use this file except in compliance with the License. You may
# obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0 
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

# one class SVM for outlier detection

import sys
import numpy as np
from sklearn import svm

if not len(sys.argv[1]) == 5:
	print "wrong number of command line args"
	print "uasge: python  ocsvm.py training_data_size nu kernel gamma"
	exit(1)

train_size = int(sys.argv[1])
nu = float(sys.argv[2])
kernel = sys.argv[3]
gamma = float(sys.argv[4])

# Generate train data
X = 0.3 * np.random.randn(train_size, 2)
X_train = np.r_[X + 2, X - 2]

# Generate some regular novel observations
X = 0.3 * np.random.randn(20, 2)
#print X
X_test = np.r_[X + 2, X - 2]
#print "----------------"
#print X_test
# Generate some abnormal novel observations
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
print "X_outliers"
print X_outliers

# fit the model
# nu=0.1, kernel="rbf", gamma=0.1
clf = svm.OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
print "n_error_test"
print n_error_test
print "y_pred_outliers"
print y_pred_outliers
print "n_error_outliers"
print n_error_outliers