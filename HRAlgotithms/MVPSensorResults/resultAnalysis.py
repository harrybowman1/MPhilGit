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
# path = "../../Datasets/Custom/Fingertip/53bpm.csv"
path = "../../Datasets/Custom/BackOfWrist/60down53bpm.csv"
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
#Should result data as a df with cols timeS and red

print(data.shape)

#Find Sample time
a = data['timeS']
sampleTime = np.array([a[i+1]-a[i] for i in range(len(a)-1)]).mean()

#Variables
plot = False
windowTime = 50
sampleTime = 0.02
sampleFreq = 1/sampleTime
# sampleFreq = 11
minHr = 40 #bpm
maxHr = 200 #bpm
minHrFreq = minHr/60
maxHrFreq = maxHr/60
minHrTime = 1/minHrFreq
maxHrTime = 1/maxHrFreq

smoothSize = 9
minHrSamples = int(math.floor(minHrTime/sampleTime))+1
maxHrSamples = int(math.floor(maxHrTime/sampleTime))+1
# maxHrSamples = 
#To ensure we get both peak and trough from hr, need to make sure window is minHR time * 2
windowSamples= minHrSamples*2
windowSamples = 50


def movingAverage(signal, nPoints):
	#Takes in the signal, and gets a moving average of nPoints
	#Takes in nPoints (odd) and replaces the middle value with the found value
	#Re-do times to be floor(nPoints/2) back
	#i.e. [1 2 8 1 2 2 10 1 1] with nPoints=5 to goes from len of 9 to 5 : 9-smoothsize-1 = 4
	samples = len(signal)
	smoothed = np.array([])
	for i in range(0, samples-nPoints):
		newVal = np.mean(signal[i:i+nPoints])
		smoothed = np.append(smoothed, newVal)
	return smoothed

def faultyDataDetector(signal, windowSamples):
	#If data too bumpy compared to prev readings (accumulate the values) then set to 0 and return flag for invalid
	#Returns info on the std, and validities, as well as the signal to use
	endNum = signal.shape[0] - signal.shape[0]%windowSamples #remove the end samples so as to make windowing work properly
	stdResult = np.array([])
	stdTracker = 1e6
	counter = 0
	validArray = np.array([])
	boolOutput = np.array([])
	#For all the windows
	for i in range(0,endNum,windowSamples):
		validWindow = 1
		counter+=1
		dataSub = signal[i:i+windowSamples] #get subset
		stdCurr = dataSub.std() #get std of the subset
		print(stdCurr, stdTracker)
		if counter > 3:
			# stdTracker = stdResult[-3:].std()*2 #Get the new threshold - 3 times the last 3 previous good std
			if stdCurr < stdTracker: #If its good then append
				print('stdCurr is less than stdTracker')
				stdResult = np.append(stdResult, stdCurr)
				validWindow = 1
				stdTracker = stdResult.std() *20 #Get the new threshold - 3 times the last 3 previous good std
			else:
				validWindow = 0
		else:
			stdResult = np.append(stdResult, stdCurr) #for the first 3, append all to the std result
			validWindow= 1
			print('first 3')
		
		# if validWindow:
		validArray = np.append(validArray,validWindow)
		if validWindow:
			boolOutput = np.append(boolOutput, [1 for i in range(windowSamples)])
		else:
			boolOutput = np.append(boolOutput, [0 for i in range(windowSamples)])

	fig, (ax1, ax2) = plt.subplots(1,2)
	ax1.plot([i*windowSamples for i in range(len(validArray))],validArray)
	ax2.plot(stdResult)
	plt.title('Fault Detector')
	plt.show()
	return boolOutput.astype(int), endNum, validArray, stdResult

def windowNormaliser(signal, windowSamples):
	#signal should be a numpy array of data points
	endNum = signal.shape[0] - signal.shape[0]%windowSamples #remove the end samples so as to make windowing work properly
	resultSignal = np.array([])
	#Normalise the data across windows of windowSamples across - determined by min heartbeat freq
	for i in range(0,endNum,windowSamples):
		dataSub = signal[i:i+windowSamples]
		centered = (dataSub - dataSub.mean())/dataSub.std()
		resultSignal = np.append(resultSignal, centered) #Len 598 - found by end num
		print('a')
	resultSignal = np.nan_to_num(resultSignal)
	return resultSignal, endNum

def peakFinder(signal, minDist):
	length = len(signal)
	meanHeight = np.mean(signal)
	stdHeight = np.std(signal)
	threshold = meanHeight+0.8*stdHeight
	peaks = np.array([])
	#Find the peaks 
	for i in range(1,length-5):
		#Middle of three points is the highest
		if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i+1] > signal[i+5] and signal[i-1] > signal[i-5]:
			#Above a threshold of mean+1.5*std 
			if signal[i]>threshold:
				peaks = np.append(peaks, i)
				print('peakFound')

	#Dist between points is at least minHr (40bpm)
	print("peaks before too close = ", peaks)
	print("Mindist = ", minDist)
	peaksTooClose = True
	peaksAll = peaks
	while peaksTooClose:
		toDelete = np.array([])
		peaksTooClose = False
		for i in range(len(peaks)-1):
			print("Start of loop")
			dist = peaks[i+1] - peaks[i]
			if dist < minDist:
				peaksTooClose = True
				if peaks[i+1]>peaks[i]:
					toDelete = np.append(toDelete, i+1)
				else:
					toDelete = np.append(toDelete, i)
				print("hitting continue")
				break
		print('after continue')
		peaks = np.delete(peaks, toDelete.astype(int))
	print("peaks after too close = ", peaks)
	return peaks.astype(int), threshold, peaksAll.astype(int)

def bpmCalc(peaksInd, sampleFreq):
	#for the peaks found, find the time between peaks
	p2pList = np.array([])
	hrList = np.array([])
	for i in range(len(peaksInd)-1):
		p2p = peaksInd[i+1]-peaksInd[i]
		p2pList = np.append(p2pList, p2p)
		print(p2p)
		timeInd = 1/sampleFreq
		Hr = 60/(p2p*timeInd) 
		print("HR = ",Hr)
		hrList = np.append(hrList, Hr)

	hrEst = hrList[2:-2].mean()
	return hrEst

#Do for red and ir
#for each window of data: center around 0, normalise, filter/smooth and plot at each step

# totalData = [np.array([]), np.array([])]
# smoothed = [np.array([]), np.array([])]
totalData = np.array([])
smoothed = np.array([])

#WINDOWED AND THEN SMOOTHED

#Do for finger data first	
validData,endNum,qw,wq = faultyDataDetector(data['red'].values, windowSamples)#[:400]#[100:500]
dataUsed = data[:endNum]
dataUsed['valid'] = validData 
dataUsed['red'] = dataUsed['red']*dataUsed['valid']
fingerData = dataUsed
totalData, endNum = windowNormaliser(fingerData['red'], windowSamples)
#Run the moving average filter across the entire sigal after it has been normaliseda
smoothed = movingAverage(totalData, smoothSize) #len = 598 - smoothsize
#Find Peaks - should be done using moving average filter for the windows - not the entire signal
peaksInd, threshold, peaksAll = peakFinder(smoothed,maxHrSamples)
#Calc av hr
avHr = bpmCalc(peaksInd, sampleFreq)

#Plot the resulting normalised then smoothedData
fig, (ax0, ax1, ax2, ax3) = plt.subplots(4,1, sharex = True)
smoothSizeHalf = math.ceil(smoothSize/2)
timeTotal = fingerData['timeS'][:endNum] 
timeSmoothed = timeTotal[math.floor(smoothSize/2):-math.ceil(smoothSize/2)].values
peaksTime = timeSmoothed[peaksInd]
ax0.plot(dataOrig['red'])
ax0.set_title('raw data')
ax1.plot(fingerData['red'], label = 'red')
ax1.set_title('raw data minus non-valid')
ax2.plot(totalData, label = "Windowed")
ax2.set_title('windowed data')
ax3.plot(smoothed)
ax3.set_title("Smoothed")

fig,ax3 = plt.subplots(1,1)
#Plot windowed and smoothed data
ax3.plot(timeSmoothed, smoothed, label = "Windowed and Smoothed")
#plot peaks found
ax3.scatter(peaksTime, smoothed[peaksInd], color = 'r')
# ax3.scatter(peaksAllTime, smoothed[peaksAll],color = 'g')
ax3.plot(timeSmoothed, [threshold for i in range(len(timeSmoothed))])
titleStr = 'Red data : HR est = ' + str(int(avHr))
ax3.set_title(titleStr)
ax3.legend()

plt.savefig(titleStr)

#Separate debugging plot
plt.figure()
plt.plot(timeTotal , totalData, label = "Windowed")
plt.plot(timeSmoothed, smoothed, label = "Windowed and Smoothed")
plt.scatter(peaksTime, smoothed[peaksInd], color = 'r')

plt.show()

#Plot the raw data
# plt.figure()
# plt.plot(fingerData['red'], label = 'red')
# # plt.plot(fingerData['IR'], label = 'IR')
# plt.legend()
# plt.show()


timeDiff = np.array([])
timeList = data['timeS'].values
for i in range(1,len(timeList)):
	timeDiff = np.append(timeDiff, timeList[i]-timeList[i-1])
# print(timeDiff.unique())fingerData


"""Archived
def plotSpecially(title, rangeLow, rangeHigh):
	fig,ax = plt.subplots()
	dataFingerSoft = data[rangeLow:rangeHigh]
	ax.plot( dataFingerSoft['timeS'], dataFingerSoft['red'])
	ax.set_ylabel("Red light reflected intensity")
	# ax2 = ax.twinx()
	# ax2.plot(dataFingerSoft['timeS'], dataFingerSoft['HR'])
	# ax2.set_ylabel("HR computed")
	# pltstr = "Finger Soft: " + str(dataFingerSoft['HR'].mean())
	plt.title(title)
	plt.savefig(title)


if plot:

	plotSpecially("Finger Soft", 100, 300)
	plotSpecially("Finger Hard", 310, 504)
	plotSpecially("Back of wrist", 538, 700)
	plotSpecially("INside wrist", 763, 930)
	plt.show()

"""
