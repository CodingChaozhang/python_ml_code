# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 09:40:51 2018

@author: gear
github:  https://github.com/gear106
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz


def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    return data[:,:2], data[:,-1]


##################################---main-code---###############################

# step1 load datasets:
X, Y = create_data()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# step2 create DecisionTree:
#clf = DecisionTreeClassifier(criterion='gini')  # CART Tree
clf = DecisionTreeClassifier(criterion='entropy') # ID3
clf.fit(X, Y)

# step3 test Results:
score = clf.score(X_test, Y_test)
Y_pred = clf.predict(X_test)
print('test accuracy :%.2f%%' % (score*100))

# step4 export Tree structure:
tree = export_graphviz(clf, out_file='mytree.pdf')
with open('mytree.pdf') as f:
    dot_graph = f.read()
    
graphviz.Source(dot_graph)