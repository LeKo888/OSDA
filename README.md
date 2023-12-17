# osda_hm
Lazy FCA

## dataset
The Boston Housing Dataset  
Congressional Voting Record  
MNIST  

## lazy classification
(Take The Boston Housing Dataset as an example)

### 1.Data analysis

```python
import pandas as pd
import numpy as np

df = pd.read_csv('./boston.csv',
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'Degree'])
df['Degree'] = [x == '1' for x in df['Degree']]
df.sample(100)
```

```python
      CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD    TAX  \ ... \ Degree
0  0.00632  18.0   2.31     0  0.538  6.575  65.2  4.0900    1  296.0   ...    True
1  0.02731   0.0   7.07     0  0.469  6.421  78.9  4.9671    2  242.0   ...    False
2  0.02729   0.0   7.07     0  0.469  7.185  61.1  4.9671    2  242.0   ...    True
3  0.03237   0.0   2.18     0  0.458  6.998  45.8  6.0622    3  222.0   ...    True
4  0.06905   0.0   2.18     0  0.458  7.147  54.2  6.0622    3  222.0   ...    False
```

### 2.Unadjusted
```python
import fcalc
from sklearn.model_selection import train_test_split

X = df.iloc[:,;-1]
y = df['Degree']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pat_cls = fcalc.classifier.PatternBinaryClassifier(X_train.values, y_train.to_numpy())

pat_cls.predict(X_test.values)

from sklearn.metrics import accuracy_score
print("accuracy:",round(accuracy_score(y_test, pat_cls.predictions),4))
```
