import numpy as np
from sklearn import metrics

a = np.array([[0.8,0.2],[0.9,0.1],[0.2,0.7],[0.2,0.5],[0.2,0.8],[0.2,0.8]])
b = np.array([[1,0],[1,0],[0,1],[1,0],[0,1],[1,0]])

print(metrics.roc_auc_score(b,a))

a1 = [0.8,0.9,0.2,0.2,0.2,0.2]
b1 = [1,1,0,1,1,0]
print(metrics.roc_auc_score(b1,a1))

a2 = [0.2,0.1,0.7,0.5,0.8,0.8]
b2 = [0,0,1,0,1,0]
print(metrics.roc_auc_score(b2,a2))