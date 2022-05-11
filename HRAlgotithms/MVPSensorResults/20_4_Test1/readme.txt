Param for below:
	byte ledBrightness = 60; //Options: 0=Off to 255=50mA
	byte sampleAverage = 4; //Options: 1, 2, 4, 8, 16, 32
	byte ledMode = 2; //Options: 1 = Red only, 2 = Red + IR, 3 = Red + IR + Green WE DONT HAVE GREEN ON THIS CHIP
	byte sampleRate = 100; //Options: 50, 100, 200, 400, 800, 1000, 1600, 3200
	int pulseWidth = 411; //Options: 69, 118, 215, 411
	int adcRange = 4096; //Options: 2048, 4096, 8192, 16384

Test.csv - first one, no time
Test2.csv - not used, sensor/serial stopped working, but saved for posteritiy
Test3.csv - time included with red, ir, HR and SPo2. Good info

Param for below - same as above, but changed with sample rate to 400 - effective reading rate of 100hz
