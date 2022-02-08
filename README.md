```python
from symint import OptErrorIntervals
import numpy as np

arr = np.ones(10)
K = 5

intopt = OptErrorIntervals(arr, k=K, uncertainty=0.05)
lci, uci = intopt.locate()
parameters = intopt.evaluate()
```