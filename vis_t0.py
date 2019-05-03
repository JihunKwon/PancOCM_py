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
num_subject = 6;
rep_list = [8196, 8196, 8196, 8192, 8192, 8192, 6932, 6932, 6932, 3690, 3690, 3690, 3401, 3401, 3401, 3690, 3690, 3690]
#rep_list = [8196, 8196, 8196]

#these store data for each transducer, 5 breath holds, 15 runs
t0 = np.zeros([10,5,np.size(rep_list)])
t1 = np.zeros([10,5,np.size(rep_list)])
t2 = np.zeros([10,5,np.size(rep_list)])
t3 = np.zeros([10,5,np.size(rep_list)])

#stores mean squared difference
d0 = np.zeros([5,np.size(rep_list)])
d1 = np.zeros([5,np.size(rep_list)])
d2 = np.zeros([5,np.size(rep_list)])
d3 = np.zeros([5,np.size(rep_list)])

for fidx in range(0,np.size(rep_list)):
    in_filename = out_list[fidx]
    ocm = np.load(in_filename)

    #crop data
    ocm = ocm[100:110,:] #Original code.

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

    if (fidx % 3) == 2:
        #Visualize t0, t1, t2
        '''
        depth = np.linspace(0, s - 1, s)
        fig = plt.figure(figsize=(10,8))

        ax0 = fig.add_subplot(311)
        a0 = ax0.plot(depth,t0[:,0,fidx])
        a1 = ax0.plot(depth,t0[:,1,fidx])
        a2 = ax0.plot(depth,t0[:,2,fidx])
        a3 = ax0.plot(depth,t0[:,3,fidx])
        a4 = ax0.plot(depth,t0[:,4,fidx])
        ax0.set_title('OCM0')

        ax1 = fig.add_subplot(312)
        b0 = ax1.plot(depth,t1[:,0,fidx])
        b1 = ax1.plot(depth,t1[:,1,fidx])
        b2 = ax1.plot(depth,t1[:,2,fidx])
        b3 = ax1.plot(depth,t1[:,3,fidx])
        b4 = ax1.plot(depth,t1[:,4,fidx])
        ax1.set_title('OCM1')

        ax2 = fig.add_subplot(313)
        c0 = ax2.plot(depth,t2[:,0,fidx])
        c1 = ax2.plot(depth,t2[:,1,fidx])
        c2 = ax2.plot(depth,t2[:,2,fidx])
        c3 = ax2.plot(depth,t2[:,3,fidx])
        c4 = ax2.plot(depth,t2[:,4,fidx])
        ax2.set_title('OCM2')

        fig.tight_layout()
        plt.savefig('t012_{fidx}.png'.format(fidx=fidx))
        '''


        depth = np.linspace(0, s - 1, s)
        fig = plt.figure(figsize=(20,16))

        ax0 = fig.add_subplot(331)
        a0 = ax0.plot(depth,t0[:,0,fidx-2], label="line 1")
        a1 = ax0.plot(depth,t0[:,1,fidx-2], label="line 2")
        a2 = ax0.plot(depth,t0[:,2,fidx-2], label="line 3")
        a3 = ax0.plot(depth,t0[:,3,fidx-2], label="line 4")
        a4 = ax0.plot(depth,t0[:,4,fidx-2], label="line 5")
        ax0.set_title('OCM0, Before')

        ax1 = fig.add_subplot(332)
        b0 = ax1.plot(depth,t1[:,0,fidx-2], label="line 1")
        b1 = ax1.plot(depth,t1[:,1,fidx-2], label="line 2")
        b2 = ax1.plot(depth,t1[:,2,fidx-2], label="line 3")
        b3 = ax1.plot(depth,t1[:,3,fidx-2], label="line 4")
        b4 = ax1.plot(depth,t1[:,4,fidx-2], label="line 5")
        ax1.set_title('OCM1, Before')

        ax2 = fig.add_subplot(333)
        c0 = ax2.plot(depth,t2[:,0,fidx-2], label="line 1")
        c1 = ax2.plot(depth,t2[:,1,fidx-2], label="line 2")
        c2 = ax2.plot(depth,t2[:,2,fidx-2], label="line 3")
        c3 = ax2.plot(depth,t2[:,3,fidx-2], label="line 4")
        c4 = ax2.plot(depth,t2[:,4,fidx-2], label="line 5")
        ax2.set_title('OCM2, Before')

        #After
        ax0 = fig.add_subplot(334)
        a0 = ax0.plot(depth,t0[:,0,fidx-1], label="line 6")
        a1 = ax0.plot(depth,t0[:,1,fidx-1], label="line 7")
        a2 = ax0.plot(depth,t0[:,2,fidx-1], label="line 8")
        a3 = ax0.plot(depth,t0[:,3,fidx-1], label="line 9")
        a4 = ax0.plot(depth,t0[:,4,fidx-1], label="line 10")
        ax0.set_title('OCM0, After')

        ax1 = fig.add_subplot(335)
        b0 = ax1.plot(depth,t1[:,0,fidx-1], label="line 6")
        b1 = ax1.plot(depth,t1[:,1,fidx-1], label="line 7")
        b2 = ax1.plot(depth,t1[:,2,fidx-1], label="line 8")
        b3 = ax1.plot(depth,t1[:,3,fidx-1], label="line 9")
        b4 = ax1.plot(depth,t1[:,4,fidx-1], label="line 10")
        ax1.set_title('OCM1, After')

        ax2 = fig.add_subplot(336)
        c0 = ax2.plot(depth,t2[:,0,fidx-1], label="line 6")
        c1 = ax2.plot(depth,t2[:,1,fidx-1], label="line 7")
        c2 = ax2.plot(depth,t2[:,2,fidx-1], label="line 8")
        c3 = ax2.plot(depth,t2[:,3,fidx-1], label="line 9")
        c4 = ax2.plot(depth,t2[:,4,fidx-1], label="line 10")
        ax2.set_title('OCM2, After')

        #10 min
        ax0 = fig.add_subplot(337)
        a0 = ax0.plot(depth,t0[:,0,fidx], label="line 11")
        a1 = ax0.plot(depth,t0[:,1,fidx], label="line 12")
        a2 = ax0.plot(depth,t0[:,2,fidx], label="line 13")
        a3 = ax0.plot(depth,t0[:,3,fidx], label="line 14")
        a4 = ax0.plot(depth,t0[:,4,fidx], label="line 15")
        ax0.set_title('OCM0, 10min')

        ax1 = fig.add_subplot(338)
        b0 = ax1.plot(depth,t1[:,0,fidx], label="line 11")
        b1 = ax1.plot(depth,t1[:,1,fidx], label="line 12")
        b2 = ax1.plot(depth,t1[:,2,fidx], label="line 13")
        b3 = ax1.plot(depth,t1[:,3,fidx], label="line 14")
        b4 = ax1.plot(depth,t1[:,4,fidx], label="line 15")
        ax1.set_title('OCM1, 10min')

        ax2 = fig.add_subplot(339)
        c0 = ax2.plot(depth,t2[:,0,fidx], label="line 11")
        c1 = ax2.plot(depth,t2[:,1,fidx], label="line 12")
        c2 = ax2.plot(depth,t2[:,2,fidx], label="line 13")
        c3 = ax2.plot(depth,t2[:,3,fidx], label="line 14")
        c4 = ax2.plot(depth,t2[:,4,fidx], label="line 15")
        ax2.set_title('OCM2, 10min')

        fig.tight_layout()
        plt.savefig('t012_{fidx}.png'.format(fidx=fidx))
