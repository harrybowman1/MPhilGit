import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from datetime import datetime	
import heartpy as hp
# plt.close('all')
"""
Approach:
- load the data
- separate into windows based on min HR to ensure at least one peak and trough in there
- normalise window
- run moving average filter on window
- 
"""

#Data Loading
path = "../../Datasets/Custom/09_12_Wrist_noMovement.csv"
dataType = "timeSingle"#"single""multi"ect
# data = pd.read_csv("20_4_Test1/test3.csv", names = ['time','red', 'IR', 'HR', 'SPo2'])
data = pd.read_csv(path, names = ['time', 'red', 'IR'])
dataOrig = data.copy()
if 0:
	data['timeS'] = data['time']/1000
	data['HR'] = [x if x > 0 else 0 for x in data['HR']]
elif 0:
	a = [datetime.strptime(i, "%Y-%m-%d %H:%M:%S.%f") for i in data['time']]
	data["timeS"] = [(((i-a[0]).seconds*1000000 + (i-a[0]).microseconds)/1e6) for i in a]
elif dataType == "timeSingle":
	dataFixed = data.reset_index()
	a = [datetime.strptime(i, "%Y-%m-%d %H:%M:%S.%f") for i in dataFixed['time']]
	dataFixed["timeS"] = [((i-a[0]).seconds*1000000 + (i-a[0]).microseconds)/1e6 for i in a]
	data = dataFixed

print(data.shape)
data = data[200:]

filtered = hp.filter_signal(data['red'], [0.7, 3.5], sample_rate=25, order=3, filtertype='bandpass')
wd, m = hp.process(filtered, sample_rate = 25, high_precision=True, clean_rr=True)
hp.plotter(wd, m, title = 'zoomed in section', figsize=(12,6))
# hp.plot_poincare(wd, m)
# plt.show()
fig, (ax0, ax1) = plt.subplots(2,1)
ax0.plot(data['red'])
ax1.plot(filtered)
plt.show()