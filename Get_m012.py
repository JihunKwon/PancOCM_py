'''
This code calculates the number of traces outisde of the envelope, which is between
median+m*SD and median-m*SD. To do this, first we need to get m that corresponds to the threshold we define (such as FPR = 1%)
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import csv
import statistics
import pickle

plt.close('all')
out_list = []

plt.rcParams['font.family'] ='sans-serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


#Jihun Local
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_01_20180928/run1.npy") #Before water
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_01_20180928/run2.npy") #After water
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_01_20181102/run1.npy")
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_01_20181102/run2.npy")
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_02_20181102/run1.npy")
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_02_20181102/run2.npy")
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_02_20181220/run1.npy")
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_02_20181220/run2.npy")
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_03_20190228/run1.npy")
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_03_20190228/run2.npy")
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_03_20190320/run1.npy")
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_03_20190320/run2.npy")
sr_list = ['s1r1', 's1r2', 's2r1', 's2r2', 's3r1', 's3r2']
rep_list = [8196, 8196, 8192, 8192, 6932, 6932, 3690, 3690, 3401, 3401, 3690, 3690]

'''
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_01_20180928/run1.npy") #Before water
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_01_20180928/run2.npy") #After water
sr_list = ['s1r1', 's1r1']
rep_list = [819, 819]
'''
num_train = 2
num_test = 10 - num_train
num_ocm = 3
num_bh = 5 # number of bh in each state


for fidx in range(0,np.size(rep_list)):
    Sub_run = sr_list[fidx]
    plt.rcParams["font.size"] = 11
    in_filename = out_list[fidx]
    ocm = np.load(in_filename)

    #crop data
    ocm = ocm[300:650, :]
    s, t = np.shape(ocm)

    # variables initialization
    median0 = np.zeros([s, num_bh])  # median
    median1 = np.zeros([s, num_bh])
    median2 = np.zeros([s, num_bh])
    if fidx % 2 == 0:
        median0_base = np.zeros([s])  # median of filtered signal
        median1_base = np.zeros([s])
        median2_base = np.zeros([s])
        sd0 = np.zeros([s])  # sd of (median - train)
        sd1 = np.zeros([s])
        sd2 = np.zeros([s])
        thr0 = np.zeros([s])  # threshold
        thr1 = np.zeros([s])
        thr2 = np.zeros([s])
    ocm0_filt = np.zeros([s, t])  # filtered signal (median based filtering)
    ocm1_filt = np.zeros([s, t])
    ocm2_filt = np.zeros([s, t])
    out0_test = np.zeros([num_bh])  # output test result
    out1_test = np.zeros([num_bh])
    out2_test = np.zeros([num_bh])

    # divide the data into each OCM and store absolute value
    b = np.linspace(0, t-1, t)
    b0 = np.mod(b,4) == 0
    ocm0 = ocm[:, b0]
    b1 = np.mod(b,4) == 1
    ocm1 = ocm[:, b1]
    b2 = np.mod(b,4) == 2
    ocm2 = ocm[:, b2]
    s, c0 = np.shape(ocm0)
    print('ocm0:', ocm0.shape)

    # first few traces are strange. Remove them
    ocm0 = ocm0[:, c0 - num_bh*rep_list[fidx]:]
    ocm1 = ocm1[:, c0 - num_bh*rep_list[fidx]:]
    ocm2 = ocm2[:, c0 - num_bh*rep_list[fidx]:]

    print('ocm0 new:', ocm0.shape)
    s, c0 = np.shape(ocm0)
    t_sub = int(c0 / num_bh)

    #### Median-based filtering ####
    for bh in range(0, num_bh):
        for depth in range(0, s):
            # get median for each bh
            median0[depth, bh] = statistics.median(ocm0[depth, bh*t_sub:(bh+1)*t_sub])
            median1[depth, bh] = statistics.median(ocm1[depth, bh*t_sub:(bh+1)*t_sub])
            median2[depth, bh] = statistics.median(ocm2[depth, bh*t_sub:(bh+1)*t_sub])
    # filtering all traces with median trace
    bh = -1
    for p in range(0, c0):
        if p % rep_list[fidx] == 0:
            bh = bh + 1
        for depth in range(0, s):
            # filter the signal (subtract median from each trace of corresponding bh)
            ocm0_filt[depth, p] = np.abs(ocm0[depth, p] - median0[depth, bh])
            ocm1_filt[depth, p] = np.abs(ocm1[depth, p] - median1[depth, bh])
            ocm2_filt[depth, p] = np.abs(ocm2[depth, p] - median2[depth, bh])

    #### Threshold generation ####
    # if state 1
    if fidx % 2 == 0:
        # Median-based filtering
        for depth in range(0, s):
            # Calculate median of baseline signal
            median0_base[depth] = statistics.median(ocm0_filt[depth, 0:t_sub*num_train])
            median1_base[depth] = statistics.median(ocm1_filt[depth, 0:t_sub*num_train])
            median2_base[depth] = statistics.median(ocm2_filt[depth, 0:t_sub*num_train])
            # Get SD of (Median - train)
            sd0[depth] = np.std(median0_base[depth] - ocm0_filt[depth, 0:t_sub*num_train])
            sd1[depth] = np.std(median1_base[depth] - ocm1_filt[depth, 0:t_sub*num_train])
            sd2[depth] = np.std(median2_base[depth] - ocm2_filt[depth, 0:t_sub*num_train])

        #### Get parameter m ####
        # m and OoE (out of envelop) distribution
        m_max = 100
        scale = 10  # number divides m
        count0 = np.zeros([m_max])
        count1 = np.zeros([m_max])
        count2 = np.zeros([m_max])
        flag0_m = 0
        flag1_m = 0
        flag2_m = 0
        for m in range(0, m_max):
            if m % 10 == 0:
                print('m:', m)
            thr0[:] = np.abs(median0_base[:]) + m / scale * sd0[:]
            thr1[:] = np.abs(median1_base[:]) + m / scale * sd1[:]
            thr2[:] = np.abs(median2_base[:]) + m / scale * sd2[:]
            # loop inside the training set
            for p in range(0, t_sub*num_train):
                flag0 = 0
                flag1 = 0
                flag2 = 0
                for depth in range(0, s):
                    # if not detected yet
                    if flag0 < 1:  # OCM0
                        # check every depth and count if it's larger than the threshold
                        if ocm0_filt[depth, p] > thr0[depth]:
                            count0[m] = count0[m] + 1
                            flag0 = 1
                    if flag1 < 1:  # OCM1
                        if ocm1_filt[depth, p] > thr1[depth]:
                            count1[m] = count1[m] + 1
                            flag1 = 1
                    if flag2 < 1:  # OCM2
                        if ocm2_filt[depth, p] > thr2[depth]:
                            count2[m] = count2[m] + 1
                            flag2 = 1

        fname = 'm012_' + str(Sub_run) + '_max' + str(m_max) + '_scale' + str(scale) + '_train' + str(num_train) + '.pkl'
        with open(fname, 'wb') as f:
            pickle.dump([count0, count1, count2], f)
