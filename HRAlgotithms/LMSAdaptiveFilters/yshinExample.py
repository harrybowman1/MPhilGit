# LMS Adaptive Filter (Simple Two-Gene Network)
# Yong-Jun Shin (2019)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
plt.close('all')
N = 100                  # total number of data points
n = np.arange(0, N, 1)   # [0,..., N-1] (vector)          
x = np.empty(N)          # measured protein x concentration in uM (vector)
y = np.empty(N)          # measured protein y concentration in uM (vector)
x.fill(5)                # constant measured x protein concentration (= 5 uM)        
a1 = np.linspace(0.6, 0.3, num=N)  # a1 parameter decreasing from 0.6 to 0.3 (vector)
b1 = np.linspace(0.9, 0.2, num=N)  # b1 parameter decreasing from 0.9 to 0.2 (vector)
m = 0                    # mean of Gaussian noise
sd = 0.2                 # standard deviation of Gaussian noise 
x[0] = x[0] + np.random.normal(m, sd)   # initial measured x protein concentration (simulated)
y[0] = 0 + np.random.normal(m, sd)      # initial measured y protein concentration (simulated) 
yHat = np.empty(N)       # estimated protein y concentration in uM (vector)
a1Hat = np.empty(N)      # estimated a1 parameter (vector)                                                          
b1Hat = np.empty(N)      # estimated b1 parameter (vector)                                                        
e = np.empty(N)          # y protein estimation error in uM (vector) (= y - yHat)
a1Hat[1] = 0             # initial estimated a1 parameter                                                   
b1Hat[1] = 0             # initial estimated b1 parameter 
u = 0.001                # step size (mu)

for i in range (1, N-1):                             
    
    # simulated experiment
    x[i] = x[i] + np.random.normal(m, sd)                          # measured x protein concentration (simulated)
    y[i] = a1[i]*x[i-1] + b1[i]*y[i-1] + np.random.normal(m, sd)   # measured y protein concentration (simulated)
  
    # Least Mean Squares(LMS) adaptive filter                          
    yHat[i] = a1Hat[i]*x[i-1] + b1Hat[i]*y[i-1]      # estimated protein y concentration
    e[i] = y[i] - yHat[i]                            # y protein estimation error
    a1Hat[i+1] = a1Hat[i] + u*x[i-1]*e[i]            # estimated a1 parameter
    b1Hat[i+1] = b1Hat[i] + u*y[i-1]*e[i]            # estimated b1 parameter
    
x[N-1] = x[N-2] + np.random.normal(m, sd)                            # last measured x protein concentration (simulated)
y[N-1] = a1[N-1]*x[N-2] + b1[N-1]*y[N-2] + np.random.normal(m, sd)   # last measured y protein concentration (simulated)
yHat[N-1] = a1Hat[N-1] * x[N-2] + b1Hat[N-1] * y[N-2]                # last estimated protein y concentration
e[N-1] = y[N-1] - yHat[N-1]                                          # last y protein estimation error

plt.plot(n, x, 'y', label = 'x protein')                                         
plt.plot(n, y, 'g', label = 'y protein')
plt.plot(n, yHat, 'b', label = 'estimated y protein')
plt.plot(n, e, 'r', label = 'y protein estimation error')
plt.xlabel('time (i)')                                                         
plt.ylabel('protein concentration (uM)')
plt.legend(loc='upper right')
plt.title('LMS Adaptive Filter')
plt.axis([0, N, -3, 30])
plt.grid(True)
plt.show()
plt.figure()
plt2.plot(n, a1Hat, 'm', label = 'estimated a1')
plt2.plot(n, b1Hat, 'c', label = 'estimated b1')
plt2.xlabel('time (i)')                                                         
plt2.ylabel('parameter value')
plt2.legend(loc='upper right')
plt2.title('Estimated Parameter Values')
plt2.axis([0, N, 0, 2])
plt2.grid(True)
plt2.show()