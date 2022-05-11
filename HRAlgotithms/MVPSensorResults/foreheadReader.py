import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from datetime import datetime	
plt.close('all')
"""
Approach:
- load the data
- separate into windows based on min HR to ensure at least one peak and trough in there
- normalise window
- run moving average filter on window
- 
"""

#Data Loading
# data = pd.read_csv("20_4_Test1/test3.csv", names = ['time','red', 'IR', 'HR', 'SPo2'])
data = pd.read_csv("../../Datasets/Custom/09_11_FingertipHighPower50Hz.csv", names = ['red'])

plt.figure()
plt.plot(data['red'])
plt.show()