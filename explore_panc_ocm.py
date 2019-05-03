# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 22:24:37 2018

@author: Jeremy Bredfeldt
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

plt.close('all')

out_list = []

'''
Note: file name run1, run2 and run3 means: before, shortly after and 10 minutes after water, respectively.
      This run name is confusing because we also use only three OCM out of four in this study. 
'''

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
#rep_list = [8769, 8769, 8769, 8767, 8767, 8767, 7506, 7506, 7506]
num_subject = 1;
#rep_list = [8196, 8196, 8196, 8192, 8192, 8192, 6932, 6932, 6932, 3690, 3690, 3690, 3401, 3401, 3401, 3690, 3690, 3690]# 3124 3401 3200
rep_list = [8196, 8196, 8196]# 3124 3401 3200

#these store data for each transducer, 5 breath holds, 15 runs
t0 = np.zeros([500,5,np.size(rep_list)])
t1 = np.zeros([500,5,np.size(rep_list)])
t2 = np.zeros([500,5,np.size(rep_list)])
t3 = np.zeros([500,5,np.size(rep_list)])

#stores mean squared difference
d0 = np.zeros([5,np.size(rep_list)])
d1 = np.zeros([5,np.size(rep_list)])
d2 = np.zeros([5,np.size(rep_list)])
d3 = np.zeros([5,np.size(rep_list)])

for fidx in range(0,np.size(rep_list)):
    in_filename = out_list[fidx]
    ocm = np.load(in_filename)
    
    #crop data
    ocm = ocm[300:800,:] #Original code.

    #s=# of samples per trace
    #t=# of total traces
    s, t = np.shape(ocm)
    
    # ============================1: INITIAL CODES=====================================
    # filter the data
    offset = np.ones([s,t])  # offset correction
    hptr = np.ones([s,t])  # high pass filter
    lptr = np.ones([s,t])  # low pass filter
    lptra = np.ones([s,t])
    lptr_norm = np.ones([s,t])  # Normalized
    f1 = np.ones([5])
    f2 = np.ones([10])
    max_p = 0

    # My variables
    offset_my = np.ones([s,t])  # offset correction
    lptr_my = np.ones([s,t])  # low pass filter
    lptr_env_my = np.ones([s,t])  # low pass filter
    f1_my1 = np.ones([5])
    f2_my = np.ones([10]) # Envelop
    for p in range(0,t):

        # high pass then low pass filter the data
        tr1 = ocm[:,p]
        offset = signal.detrend(tr1)
        hptr[:,p] = np.convolve(offset,[1,-1],'same')
        tr2 = hptr[:,p]
        lptra[:,p] = np.convolve(tr2,f1,'same')
        tr3 = lptra[:,p]
        # square and envelope detect
        lptr[:,p] = np.convolve(np.sqrt(np.square(tr3)),f2,'same')
        # normalize
        max_temp = np.max(lptr[:,p])
        if max_p < max_temp:
            max_p = max_temp

        lptr_norm[:,p] = np.divide(lptr[:,p],np.max(lptr[:,p]))
        '''
        # ========================Visualize==============================================
        # This part shows how the signal changed after the filtering.
        depth = np.linspace(0, s - 1, s)
        fig = plt.figure(figsize=(12,8))

        ax0 = fig.add_subplot(311)
        a0 = ax0.plot(depth,ocm[:,p])
        a0off = ax0.plot(depth,offset[:])
        ax0.set_title('Raw and Offset')

        ax1 = fig.add_subplot(312)
        a1 = ax1.plot(depth,hptr[:,p])
        ax1.set_title('High pass')

        ax2 = fig.add_subplot(313)
        a2 = ax2.plot(depth,lptra[:,p])
        a3 = ax2.plot(depth,lptr[:,p])
        ax2.set_title('Low pass and Envelop detect')

        fig.tight_layout()
        plt.savefig('Filtered_wave_original.png')
        # =============================================================================
        '''

        '''
        # ============================2: MY TEST ANALYSIS=====================================
        # high pass then low pass filter the data
        tr1 = ocm[:,p]
        offset_my = signal.detrend(tr1)
        lptr_my[:, p] = np.convolve(offset_my, f1_my1, 'same')
        tr2 = lptr_my[:, p]
        # square and envelope detect
        lptr_env_my[:, p] = np.convolve(np.sqrt(np.square(tr2)), f2_my, 'same')
        # Some kind of normalization here
        lptr_norm[:,p] = np.divide(lptr_env_my[:,p],np.max(lptr_env_my[:,p]))

        
        # ========================Visualize==============================================
        # This part shows how the signal changed after the filtering.
        depth = np.linspace(0, s - 1, s)
        fig = plt.figure(figsize=(12,8))

        ax0 = fig.add_subplot(311)
        a0 = ax0.plot(depth,ocm[:,p])
        a0off = ax0.plot(depth,offset_my[:])
        ax0.set_title('Raw and Offset')

        ax1 = fig.add_subplot(312)
        a1 = ax1.plot(depth,lptr_my[:,p])
        a2 = ax1.plot(depth,lptr_env_my[:,p])
        ax1.set_title('Low pass 5 and Envelop detect')

        fig.tight_layout()
        plt.savefig('Filtered_wave_my.png')
        # =============================================================================
        '''

    ocm = lptr
    
    b = np.linspace(0,t-1,t)
    b0 = np.mod(b,4)==0
    ocm0 = ocm[:,b0]
    b1 = np.mod(b,4)==1
    ocm1 = ocm[:,b1]
    b2 = np.mod(b,4)==2
    ocm2 = ocm[:,b2]
    b3 = np.mod(b,4)==3
    ocm3 = ocm[:,b3]
    
    s, c0 = np.shape(ocm0)
    s, c1 = np.shape(ocm1)
    s, c2 = np.shape(ocm2)
    s, c3 = np.shape(ocm3)
    
# =============================================================================
    '''
 #plot the data
    fig, ax = plt.subplots()
    ax.imshow(np.abs(ocm0[100:600,:]), aspect="auto", vmin=0, vmax=1)
    fig, ax = plt.subplots()
    ax.imshow(np.abs(ocm1[100:600,:]), aspect="auto", vmin=0, vmax=1)
    fig, ax = plt.subplots()
    ax.imshow(np.abs(ocm2[100:600,:]), aspect="auto", vmin=0, vmax=1)
    fig, ax = plt.subplots()
    ax.imshow(np.abs(ocm3[100:600,:]), aspect="auto", vmin=0, vmax=1)
    ax.imshow(np.abs(ocm0[100:600,:]), aspect="auto", vmin=-4000, vmax=4000)
    plt.title('Full experiment')
    plt.xlabel('seconds')
    plt.ylabel('micro-seconds')
    fig.show()
    '''
# =============================================================================

    #compute mean of the breath hold, there are 5 breath holds per run, 3 runs per subject
    ocm0m = np.ones([s,5])
    ocm1m = np.ones([s,5])
    ocm2m = np.ones([s,5])
    ocm3m = np.ones([s,5])

    for i in range(0,5): #Distribute ocm signal from end to start
        ocm0m[:,i] = np.mean(np.abs(ocm0[:,ocm0.shape[1]-rep_list[fidx]*(i+1)-1:ocm0.shape[1]-rep_list[fidx]*i-1]),1)
        ocm1m[:,i] = np.mean(np.abs(ocm1[:,ocm1.shape[1]-rep_list[fidx]*(i+1)-1:ocm1.shape[1]-rep_list[fidx]*i-1]),1)
        ocm2m[:,i] = np.mean(np.abs(ocm2[:,ocm2.shape[1]-rep_list[fidx]*(i+1)-1:ocm2.shape[1]-rep_list[fidx]*i-1]),1)
        ocm3m[:,i] = np.mean(np.abs(ocm3[:,ocm3.shape[1]-rep_list[fidx]*(i+1)-1:ocm3.shape[1]-rep_list[fidx]*i-1]),1)

    #collect all the data so far
    t0[:,:,fidx] = ocm0m
    t1[:,:,fidx] = ocm1m
    t2[:,:,fidx] = ocm2m
    t3[:,:,fidx] = ocm3m

    sample_rate_MHz = 10
    us_per_sample = 1/sample_rate_MHz

    #in cm
    little_t = np.linspace(2.3,6.2,s)

'''
fig, (ax1, ax2, ax3) = plt.subplots(3,1)
ax1.plot(little_t, t2[:,1,6])
ax2.plot(little_t, t2[:,1,7])
ax3.plot(little_t, t2[:,1,8])
ax2.set_ylabel("OCM Amplitude (a.u.)")
ax3.set_xlabel("Depth (cm)")
'''

'''Method 2: rm0 is only from fidx=0,3,6,9, which means signal from first-run (before water) is used for subtraction
'''
#loop through subjects
for sub in range(0,num_subject):
    #mean for this run across all breath holds
    rm0 = np.mean(t0[:,:,sub*3],1)
    rm1 = np.mean(t1[:,:,sub*3],1)
    rm2 = np.mean(t2[:,:,sub*3],1)
    rm3 = np.mean(t3[:,:,sub*3],1)
    #loop through runs (before water, after water, 10min after water)
    for run in range(0,3):
        fidx = run + sub*3 #file number
        #loop through breath holds
        for bh in range(0,5):
            d0[bh,fidx] = np.mean(np.square(np.subtract(rm0,t0[:,bh,fidx])),0)
            d1[bh,fidx] = np.mean(np.square(np.subtract(rm1,t1[:,bh,fidx])),0)
            d2[bh,fidx] = np.mean(np.square(np.subtract(rm2,t2[:,bh,fidx])),0)
            d3[bh,fidx] = np.mean(np.square(np.subtract(rm3,t3[:,bh,fidx])),0)
  
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1)
ax1.boxplot(d0)
ax1.set_ylim(0,0.05)
ax2.boxplot(d1)
ax2.set_ylim(0,0.05)
ax3.boxplot(d2)
ax3.set_ylim(0,0.05)
ax4.boxplot(d3)
ax4.set_ylim(0,0.05)

sub1 = np.concatenate((d0[:,0:3],d1[:,0:3],d2[:,0:3]),0)
fix, ax = plt.subplots()
ax.boxplot(sub1)
ax.set_ylabel("Mean Squared Error")

out_txt = []
out_t = []

#Jihun Local
out_txt.append("/Users/Kwon/PycharmProjects/PancOCM/transducer0.txt")
out_txt.append("/Users/Kwon/PycharmProjects/PancOCM/transducer1.txt")
out_txt.append("/Users/Kwon/PycharmProjects/PancOCM/transducer2.txt")
out_txt.append("/Users/Kwon/PycharmProjects/PancOCM/transducer3.txt")

np.savetxt(out_txt[0],d0,fmt='%0.06f',delimiter=' ',newline='\n')
np.savetxt(out_txt[1],d1,fmt='%0.06f',delimiter=' ',newline='\n')
np.savetxt(out_txt[2],d2,fmt='%0.06f',delimiter=' ',newline='\n')
np.savetxt(out_txt[3],d3,fmt='%0.06f',delimiter=' ',newline='\n')
