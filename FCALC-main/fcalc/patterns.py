import numpy as np

class IntervalPattern:
    def __init__(self, test, train) -> None:
        self.low = np.minimum(test, train)
        self.high = np.maximum(test, train)

class CategoricalPattern:
    def __init__(self, test, train) -> None:
        self.mask = list(map(lambda x, y: x == y, test, train))
        self.vals = test[self.mask]
        