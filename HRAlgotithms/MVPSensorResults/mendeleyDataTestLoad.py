import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
plt.close('all')
dataMendeley = pd.read_csv("../../Datasets/ppg/trainData.csv")
print(dataMendeley.shape)
for i in range(1,10):
	plt.figure()
	plt.plot(dataMendeley[str(i)])
plt.show()