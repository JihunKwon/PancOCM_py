'''
Created on Jan 11, 2019

@author: jihun
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import ocm_exp
import sys
import os
import ocm
#sys.path.append("C:\\Users\\jihun\\eclipse-workspace\\OCM\\OCM_Analysis\\alma_master\\alma_master\\alma")
plt.close("all")

ocm_list = [] #run1
ocm_list = [] #run2
ocm_list = [] #run3
ocm_list = [] #test1
#ocm_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_01_20180928\\run1.bin")
ocm_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_01_20181102\\runb1.bin")
#ocm_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_02_20181102\\run1.bin")
#ocm_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_02_20181220\\run1_ocm.bin")


fidx = 0
ocm_filename = ocm_list[fidx]
a = ocm_exp.ocm_exp(ocm_filename)

sec_per_trace = (a.ts2_us[0,1] - a.ts2_us[0,0]) #ts2_us (float array): NI timestamp

#in seconds
big_t = np.multiply(range(0,a.cnt),sec_per_trace) #cnt:num of full traces in the file
#big_t = 1*16621. max is 41.12

'''
#plot the data
fig, ax = plt.subplots()
plt.title('Full experiment')
plt.xlabel('number of full traces')
plt.ylabel('micro-seconds')
ax.imshow(a.ocm, aspect="auto")
fig.show()
print("Disp OCM")
'''

#plot the data
fig, ax = plt.subplots()
plt.title('Full experiment')
plt.xlabel('number of full traces')
plt.ylabel('micro-seconds')
ax.imshow(a.ocm[:,0:10000], aspect="auto")
fig.show()

#Plot along-time
data = a.ocm[1600,]

#low pass filter before sampling
f1 = np.ones([2000])
data_conv = np.convolve(data,f1,'same')
data_filt = np.divide(data_conv,f1.size)

fig, ax = plt.subplots()
ax.plot(range(0,a.cnt),data_filt)
fig.show()


#Plot along-time
data = a.ocm[1700,]
#low pass filter before sampling
f1 = np.ones([2000])
data_conv = np.convolve(data,f1,'same')
data_filt = np.divide(data_conv,f1.size)

fig, ax = plt.subplots()
ax.plot(range(0,a.cnt),data_filt)
fig.show()



#Plot along-time
data = a.ocm[1800,]
#low pass filter before sampling
f1 = np.ones([2000])
data_conv = np.convolve(data,f1,'same')
data_filt = np.divide(data_conv,f1.size)

fig, ax = plt.subplots()
ax.plot(range(0,a.cnt),data_filt)
fig.show()


#Plot along-time
data = a.ocm[1999,]
#low pass filter before sampling
f1 = np.ones([2000])
data_conv = np.convolve(data,f1,'same')
data_filt = np.divide(data_conv,f1.size)

fig, ax = plt.subplots()
ax.plot(range(0,a.cnt),data_filt)
fig.show()

np.size(data_filt)