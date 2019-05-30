import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

plt.close('all')

out_list = []

#Jihun Local
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_01_20180928/run1.npy") #Before water
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_01_20180928/run2.npy") #After water
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_01_20180928/run3.npy") #10min After water
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_01_20181102/run1.npy")
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_01_20181102/run2.npy")
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_01_20181102/run3.npy")
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_02_20181102/run1.npy")
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_02_20181102/run2.npy")
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_02_20181102/run3.npy")
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_02_20181220/run1.npy")
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_02_20181220/run2.npy")
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_02_20181220/run3.npy")
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_03_20190228/run1.npy")
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_03_20190228/run2.npy")
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_03_20190228/run3.npy")
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_03_20190320/run1.npy")
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_03_20190320/run2.npy")
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_03_20190320/run3.npy")

#these are where the runs end in each OCM file
num_subject = 1;
#rep_list = [8196, 8196, 8196]# 3124 3401 3200
rep_list = [8196, 8196, 8196, 8192, 8192, 8192, 6932, 6932, 6932, 3690, 3690, 3690, 3401, 3401, 3401, 3690, 3690, 3690]

#these store data for each transducer, 5 breath holds, 15 runs
t0 = np.zeros([50,5,np.size(rep_list)])
t1 = np.zeros([50,5,np.size(rep_list)])
t2 = np.zeros([50,5,np.size(rep_list)])
t3 = np.zeros([50,5,np.size(rep_list)])

#stores mean squared difference
d0 = np.zeros([5,np.size(rep_list)])
d1 = np.zeros([5,np.size(rep_list)])
d2 = np.zeros([5,np.size(rep_list)])
d3 = np.zeros([5,np.size(rep_list)])

#for fidx in range(0,np.size(rep_list)):
fidx=0
in_filename = out_list[fidx]
ocm = np.load(in_filename)

#crop data
ocm = ocm[200:250,:] #Original code.
s, t = np.shape(ocm)

## First distribute ocm to ocm0, ocm1, ocm2
b = np.linspace(0,t-1,t)
b0 = np.mod(b,4)==0
ocm0 = ocm[:,b0]
b1 = np.mod(b,4)==1
ocm1 = ocm[:,b1]
b2 = np.mod(b,4)==2
ocm2 = ocm[:,b2]
b3 = np.mod(b,4)==3
ocm3 = ocm[:,b3]

#s=# of samples per trace
#t=# of total traces
s, t = np.shape(ocm0)

# filter the data
offset = np.ones([s,t])  # offset correction
hptr = np.ones([s,t])  # high pass filter
lptr = np.ones([s,t])  # low pass filter
lptra = np.ones([s,t])
lptr_new = np.ones([s,t])
lptr_norm = np.ones([s,t])  # Normalized
f1 = np.ones([5])
f2 = np.ones([10])
max_p = 0

depth = np.linspace(0, s - 1, s)
fig = plt.figure(figsize=(12,12))
X=10000

#for fidx in range(0, np.size(rep_list)):
for fidx in range(0,3):
    if fidx%3==0: #if before water
        for p in range(0, t):
            # high pass then low pass filter the data
            tr1 = ocm0[:,p]
            offset = signal.detrend(tr1)
            hptr[:,p] = np.convolve(offset,[1,-1],'same')
            tr2 = hptr[:,p]
            lptra[:,p] = np.convolve(tr2,f1,'same')
            tr3 = lptra[:,p]
            # square and envelope detect
            lptr[:,p] = np.convolve(np.sqrt(np.square(tr3)),f2,'same')
            ax0 = fig.add_subplot(311)
            a_bef = ax0.plot(depth,lptr[:,p])
            ax0.set_title('Before water, OCM0')

    elif fidx%3==1: #if After water
        for p in range(0, t):
            # high pass then low pass filter the data
            tr1 = ocm0[:,p]
            offset = signal.detrend(tr1)
            hptr[:,p] = np.convolve(offset,[1,-1],'same')
            tr2 = hptr[:,p]
            lptra[:,p] = np.convolve(tr2,f1,'same')
            tr3 = lptra[:,p]
            # square and envelope detect
            lptr[:,p] = np.convolve(np.sqrt(np.square(tr3)),f2,'same')
            ax0 = fig.add_subplot(311)
            a_aft = ax0.plot(depth,lptr[:,p])
            ax0.set_title('After water, OCM0')

    else: # if 10min after water
        for p in range(0, t):
            # high pass then low pass filter the data
            tr1 = ocm0[:,p]
            offset = signal.detrend(tr1)
            hptr[:,p] = np.convolve(offset,[1,-1],'same')
            tr2 = hptr[:,p]
            lptra[:,p] = np.convolve(tr2,f1,'same')
            tr3 = lptra[:,p]
            # square and envelope detect
            lptr[:,p] = np.convolve(np.sqrt(np.square(tr3)),f2,'same')
            ax0 = fig.add_subplot(313)
            a_10m = ax0.plot(depth,lptr[:,p])
            ax0.set_title('10min After water, OCM0')

        fig.tight_layout()
        plt.savefig('t012_{fidx}.png'.format(fidx=fidx))
