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
#ocm_list.append("C:\\OCM_Data\\20181102_Panc_OCM\\Subject_20180928\\run1.bin")
#ocm_list.append("C:\\OCM_Data\\20181102_Panc_OCM\\Subject_01_20181102\\runb1.bin")
ocm_list.append("C:\\OCM_Data\\20181102_Panc_OCM\\Subject_02_20181102\\run1.bin")


fidx = 0
ocm_filename = ocm_list[fidx]
a = ocm_exp.ocm_exp(ocm_filename)

sec_per_trace = (a.ts2_us[0,1] - a.ts2_us[0,0]) #ts2_us (float array): NI timestamp

#in seconds
big_t = np.multiply(range(0,a.cnt),sec_per_trace) #cnt:num of full traces in the file
#big_t = 1*16621. max is 41.12

#plot the data
fig, ax = plt.subplots()
plt.title('Full experiment')
plt.xlabel('number of full traces')
plt.ylabel('micro-seconds')
ax.imshow(a.ocm, aspect="auto")
fig.show()
print("Disp OCM")

# cnt (int): number of traces in the file
print(a.cnt) #a.cnt = 166211 (x axis)

#Plot along-time
data = a.ocm[1500,]

#low pass filter before sampling
f1 = np.ones([1000])
data_conv = np.convolve(data,f1,'same')
data_filt = np.divide(data_conv,f1.size)

fig, ax = plt.subplots()
ax.plot(range(0,a.cnt),data_filt)
fig.show()

#First tp
fig, ax = plt.subplots()
ax.plot(big_t,data_filt)
plt.xlim(big_t[3000], big_t[28000])
fig.show()

#x axis: sec_per_trace
yf = scipy.fftpack.fft(data_filt)
xf = np.linspace(0.0, 1/2, a.cnt/2)
fig, ax = plt.subplots()
ax.plot(xf, 2/a.cnt * np.abs(yf[:a.cnt//2]))
plt.show()
plt.xlim(0.0, 0.003) #plt.xlim(-0.00001, 0.0005)
plt.ylim(0, 100)

#x axis: a.cnt
xf = np.linspace(0.0, 1/(2*sec_per_trace), len(big_t)/2)
fig, ax = plt.subplots()
ax.plot(xf, 2/len(big_t) * np.abs(yf[:len(big_t)//2]))
plt.show()
plt.xlim(0.0, 50) #plt.xlim(-0.00001, 0.0005)
plt.ylim(0, 100)
print("End of code")

'''
#Apply cross correlation before low pass filter
xcor = np.zeros([1500,a.cnt])
xcor1 = np.zeros([xcor.size])
xcor2 = np.zeros([xcor.size])

xcor1 = a.ocm[500:2000,5]
xcor1 = xcor1.astype(float)


#After calculating xcor1, shifting xcor2 in the horizontal axis.
#At every position of xcor2 (p), calculate correlation between xcor1 and xcor2
for p in range(0,a.cnt):
    xcor2 = a.ocm[500:2000,p]
    xcor2 = xcor2.astype(float)
    xcor[:,p] = np.correlate(xcor1,xcor2,'same')

'''


#try to estimate breathing trace from OCM data
#use the 5th trace just in case the first few have artifacts; could tweek this
xcor = np.ones([100,a.cnt]) #works for 1500, why?
xcor1 = np.ones([xcor.size])
xcor2 = np.ones([xcor.size])

xcor1 = a.ocm[1500:1600,5]
xcor1 = xcor1.astype(float)

#After calculating xcor1, shifting xcor2 in the horizontal axis.
#At every position of xcor2 (p), calculate correlation between xcor1 and xcor2
for p in range(0,a.cnt):
    xcor2 = a.ocm[1500:1600,p]
    xcor2 = xcor2.astype(float)
    xcor[:,p] = np.correlate(xcor1,xcor2,'same')
    

#find max of correlation
vo = np.ones(a.cnt) #vertex offset from reference
for p in range(0,a.cnt):
    vo[p] = np.argmax(xcor[:,p])-(xcor1.size/2)

#low pass filter offsets
f1 = np.ones([500])
vo_filt = np.convolve(vo,f1,'same')
vo_filt = np.divide(vo_filt,f1.size)
#normalize correlation result
vo_filt = np.divide(vo_filt,np.max(np.abs(vo_filt)))
vo_filt = np.subtract(vo_filt,np.average(vo_filt))
#plot the breathing trace and the correlation trace
fig, ax = plt.subplots()
ax.plot(big_t,vo_filt)
fig.show()

print("End of code")
