# osda_hm
Lazy FCA

##dataset
The Boston Housing Dataset
Congressional Voting Record
MNIST

##lazy classification
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
0  0.00632  18.0   2.31     0  0.538  6.575  65.2  4.0900    1  296.0   ...
1  0.02731   0.0   7.07     0  0.469  6.421  78.9  4.9671    2  242.0   ...
2  0.02729   0.0   7.07     0  0.469  7.185  61.1  4.9671    2  242.0   ...
3  0.03237   0.0   2.18     0  0.458  6.998  45.8  6.0622    3  222.0   ...
4  0.06905   0.0   2.18     0  0.458  7.147  54.2  6.0622    3  222.0   ...
```

### 2.Unadjusted
