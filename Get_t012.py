import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pickle

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
#rep_list = [8769, 8769, 8769, 8767, 8767, 8767, 7506, 7506, 7506]
num_subject = 1;
#rep_list = [8196, 8196, 8196, 8192, 8192, 8192, 6932, 6932, 6932, 3690, 3690, 3690, 3401, 3401, 3401, 3690, 3690, 3690]# 3124 3401 3200
rep_list = [8196, 8196, 8196]# 3124 3401 3200

# these store data for each transducer, 5 breath holds, 15 runs
t0 = np.zeros([360, 5, np.size(rep_list)])
t1 = np.zeros([360, 5, np.size(rep_list)])
t2 = np.zeros([360, 5, np.size(rep_list)])

for fidx in range(0, np.size(rep_list)):
    # fidx = 16
    in_filename = out_list[fidx]
    ocm = np.load(in_filename)

    # crop data
    ocm = ocm[300:660, :]  # Original code.

    # s=# of samples per trace
    # t=# of total traces
    s, t = np.shape(ocm)

    # ============================1: INITIAL CODES=====================================
    # filter the data
    offset = np.ones([s, t])  # offset correction
    hptr = np.ones([s, t])  # high pass filter
    lptr = np.ones([s, t])  # low pass filter
    lptra = np.ones([s, t])
    lptr_norm = np.ones([s, t])  # Normalized
    f1 = np.ones([5])
    f2 = np.ones([10])
    max_p = 0

    # My variables
    offset_my = np.ones([s, t])  # offset correction
    lptr_my = np.ones([s, t])  # low pass filter
    lptr_env_my = np.ones([s, t])  # low pass filter
    f1_my1 = np.ones([5])
    f2_my = np.ones([10])  # Envelop
    for p in range(0, t):

        # high pass then low pass filter the data
        tr1 = ocm[:, p]
        offset = signal.detrend(tr1)
        hptr[:, p] = np.convolve(offset, [1, -1], 'same')
        tr2 = hptr[:, p]
        lptra[:, p] = np.convolve(tr2, f1, 'same')
        tr3 = lptra[:, p]
        # square and envelope detect
        lptr[:, p] = np.convolve(np.sqrt(np.square(tr3)), f2, 'same')
        # normalize
        max_temp = np.max(lptr[:, p])
        if max_p < max_temp:
            max_p = max_temp

        lptr_norm[:, p] = np.divide(lptr[:, p], np.max(lptr[:, p]))

    ocm = lptr_norm

    b = np.linspace(0, t - 1, t)
    b0 = np.mod(b, 4) == 0
    ocm0 = ocm[:, b0]
    b1 = np.mod(b, 4) == 1
    ocm1 = ocm[:, b1]
    b2 = np.mod(b, 4) == 2
    ocm2 = ocm[:, b2]

    s, c0 = np.shape(ocm0)
    s, c1 = np.shape(ocm1)
    s, c2 = np.shape(ocm2)

    # compute mean of the breath hold, there are 5 breath holds per run, 3 runs per subject
    ocm0m = np.ones([s, 5])
    ocm1m = np.ones([s, 5])
    ocm2m = np.ones([s, 5])

    for i in range(0, 5):  # Distribute ocm signal from end to start
        ocm0m[:, i] = np.mean(np.abs(ocm0[:, ocm0.shape[1] - rep_list[fidx] * (i + 1) - 1:ocm0.shape[1] - rep_list[fidx] * i - 1]), 1)
        ocm1m[:, i] = np.mean(np.abs(ocm1[:, ocm1.shape[1] - rep_list[fidx] * (i + 1) - 1:ocm1.shape[1] - rep_list[fidx] * i - 1]), 1)
        ocm2m[:, i] = np.mean(np.abs(ocm2[:, ocm2.shape[1] - rep_list[fidx] * (i + 1) - 1:ocm2.shape[1] - rep_list[fidx] * i - 1]), 1)

    # collect all the data so far
    t0[:, :, fidx] = ocm0m
    t1[:, :, fidx] = ocm1m
    t2[:, :, fidx] = ocm2m

with open('t012.pkl', 'wb') as f:
    pickle.dump([t0, t1, t2], f)
