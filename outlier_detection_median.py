import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import statistics
import matplotlib.animation as animation

plt.close('all')

out_list = []

#Jihun Local
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_01_20180928/run1.npy") #Before water
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_01_20180928/run2.npy") #After water
out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_01_20180928/run3.npy") #10min After water
#out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_01_20181102/run1.npy")
#out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_01_20181102/run2.npy")
#out_list.append("/Users/Kwon/OCM_Data/Panc_OCM/Subject_01_20181102/run3.npy")

#these are where the runs end in each OCM file
#rep_list = [8769, 8769, 8769, 8767, 8767, 8767, 7506, 7506, 7506]
num_subject = 1
num_bh = 15  # number of bh
batch = 3 # Devide each bh into three separate groups
num_ocm = 3 # # of OCM
rep_list = [8196, 8196, 8196]
#rep_list = [8192, 8192, 8192]


#these store data for each transducer, 5 breath holds, 15 runs
t0 = np.zeros([20,5,np.size(rep_list)])
t1 = np.zeros([20,5,np.size(rep_list)])
t2 = np.zeros([20,5,np.size(rep_list)])
t3 = np.zeros([20,5,np.size(rep_list)])

#stores mean squared difference
d0 = np.zeros([5,np.size(rep_list)])
d1 = np.zeros([5,np.size(rep_list)])
d2 = np.zeros([5,np.size(rep_list)])
d3 = np.zeros([5,np.size(rep_list)])

for fidx in range(0,np.size(rep_list)):
    in_filename = out_list[fidx]
    ocm = np.load(in_filename)

    #crop data
    ocm = ocm[300:650,:] #Original code.

    #s=# of samples per trace
    #t2=# of total traces
    s, t = np.shape(ocm)

    # filter the data
    hptr = np.ones([s,t])  # high pass filter
    lptr = np.ones([s,t])  # low pass filter
    mag = np.ones([s,t])   # magnitude
    mag_norm = np.ones([s,t])  # Normalized
    mag_norm_medi0 = np.ones([s, num_bh*batch])  # Normalized
    mag_norm_medi1 = np.ones([s, num_bh*batch])  # Normalized
    mag_norm_medi2 = np.ones([s, num_bh*batch])  # Normalized
    f1 = np.ones([5])
    max_p = 0
    ocm_mag = np.ones([s, t])  # store the magnitude of the filtered signal
    median_sub = np.ones([s, batch*num_bh])  # store the magnitude of the filtered signal

    #for sub in range(0, batch * num_bh):  # loop from 0 to 15 for each state
    for p in range(0,t):  # loop every t_sub
        # high pass then low pass filter the data
        tr1 = ocm[:,p]
        hptr[:,p] = np.convolve(tr1,[1,-1],'same')
        tr2 = hptr[:,p]
        lptr[:,p] = np.convolve(tr2,f1,'same')
        tr3 = lptr[:,p]
        # get magnitude
        mag[:,p] = np.abs(tr3)
        # normalize
        max_temp = np.max(mag[:,p])
        if max_p < max_temp:
            max_p = max_temp

        mag_norm[:,p] = np.divide(mag[:,p],np.max(mag[:,p]))

    print('mag_norm:', mag_norm.shape)  # mag_norm: (350, 166211)

    # Divide into each OCM
    b = np.linspace(0,t-1,t)
    b0 = np.mod(b,4)==0
    ocm0 = mag_norm[:,b0]
    b1 = np.mod(b,4)==1
    ocm1 = mag_norm[:,b1]
    b2 = np.mod(b,4)==2
    ocm2 = mag_norm[:,b2]

    s, c0 = np.shape(ocm0)
    s, c1 = np.shape(ocm1)
    s, c2 = np.shape(ocm2)
    print('ocm0:', ocm0.shape)
    t_sub = int(c0 / batch / num_bh)  # t_sub: # of traces in sub-region of the bh
    Thr0 = np.ones([t_sub])
    Thr1 = np.ones([t_sub])
    Thr2 = np.ones([t_sub])
    d = np.linspace(0, ocm0.shape[0] - 1, ocm0.shape[0])

    print('ocm0_value:', ocm0[10,:])

    # Divide each OCM into subregion (Each bh and each batch)
    for sub in range(0, batch*num_bh):
        for depth in range(0, s):
            mag_norm_medi0[depth, sub] = statistics.median(ocm0[depth, sub*t_sub:(sub+1)*t_sub])
            mag_norm_medi1[depth, sub] = statistics.median(ocm1[depth, sub*t_sub:(sub+1)*t_sub])
            mag_norm_medi2[depth, sub] = statistics.median(ocm2[depth, sub*t_sub:(sub+1)*t_sub])

    print('mag_norm_medi:', mag_norm_medi0[10, 0])
    print('mag_norm_medi.shape:', mag_norm_medi0.shape)

    ###### Get Threshold ######
    base_diff0 = np.ones([s, t_sub])
    base_diff1 = np.ones([s, t_sub])
    base_diff2 = np.ones([s, t_sub])
    base_sd0 = np.ones([s])
    base_sd1 = np.ones([s])
    base_sd2 = np.ones([s])

    # Calculate difference between median of baseline and each trace in baseline
    for depth in range(0, s):
        for p in range(0, t_sub):
            base_diff0[depth, p] = mag_norm_medi0[depth, 0] - ocm0[depth, p]
            base_diff1[depth, p] = mag_norm_medi1[depth, 0] - ocm1[depth, p]
            base_diff2[depth, p] = mag_norm_medi2[depth, 0] - ocm2[depth, p]

    # Calculate SD of difference
    base_sd0[:] = np.std(base_diff0[:, :], axis=1)
    base_sd1[:] = np.std(base_diff1[:, :], axis=1)
    base_sd2[:] = np.std(base_diff2[:, :], axis=1)

    # Get Threshold
    tolerance = 3
    target = 2

    # ===============OCM0===========================================================
    # Plot Baseline
    fig = plt.figure()
    ims = []
    plt.plot(d, mag_norm_medi0[:, 0], 'g', linewidth=2, linestyle='dashed', label="OCM0, baseline")
    plt.plot(d, mag_norm_medi0[:, 0] + tolerance * base_sd0[:], 'r', linewidth=2, linestyle='dashed', label="OCM0, baseline")
    plt.title("Median signal, Baseline")
    plt.xlabel("Depth")
    plt.ylabel("Magnitude (a.u.)")
    for p in range(0, 100):
        im = plt.plot(d, ocm0[:, p], 'b', linewidth=1, linestyle='solid')
        ims.append(im)
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    ani.save("OCM0_baseline.gif", writer="imagemagic")

    # Plot Test set
    fig = plt.figure()
    ims = []
    plt.plot(d, mag_norm_medi0[:, 0], 'g', linewidth=2, linestyle='dashed', label="OCM0, baseline")
    plt.plot(d, mag_norm_medi0[:, 0] + tolerance * base_sd0[:], 'r', linewidth=2, linestyle='dashed', label="OCM0, baseline")
    plt.title("Median signal, State 2")
    plt.xlabel("Depth")
    plt.ylabel("Magnitude (a.u.)")
    for p in range(t_sub*target, t_sub*target+100):
        im = plt.plot(d, ocm0[:, p], 'b', linewidth=1, linestyle='solid')
        ims.append(im)
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    ani.save("OCM0_test.gif", writer="imagemagic")
    # =============================================================================

    # ===============OCM1===========================================================
    # Plot Baseline
    fig = plt.figure()
    ims = []
    plt.plot(d, mag_norm_medi1[:, 0], 'g', linewidth=2, linestyle='dashed', label="OCM0, baseline")
    plt.plot(d, mag_norm_medi1[:, 0] + tolerance * base_sd1[:], 'r', linewidth=2, linestyle='dashed', label="OCM1, baseline")
    plt.title("Median signal, Baseline")
    plt.xlabel("Depth")
    plt.ylabel("Magnitude (a.u.)")
    for p in range(0, 100):
        im = plt.plot(d, ocm1[:, p], 'b', linewidth=1, linestyle='solid')
        ims.append(im)
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    ani.save("OCM1_baseline.gif", writer="imagemagic")

    # Plot Test set
    fig = plt.figure()
    ims = []
    plt.plot(d, mag_norm_medi1[:, 0], 'g', linewidth=2, linestyle='dashed', label="OCM0, baseline")
    plt.plot(d, mag_norm_medi1[:, 0] + tolerance * base_sd1[:], 'r', linewidth=2, linestyle='dashed', label="OCM1, baseline")
    plt.title("Median signal, State 2")
    plt.xlabel("Depth")
    plt.ylabel("Magnitude (a.u.)")
    for p in range(t_sub*target, t_sub*target+100):
        im = plt.plot(d, ocm1[:, p], 'b', linewidth=1, linestyle='solid')
        ims.append(im)
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    ani.save("OCM1_test.gif", writer="imagemagic")
    # =============================================================================


    # ===============OCM2===========================================================
    # Plot Baseline
    fig = plt.figure()
    ims = []
    plt.plot(d, mag_norm_medi2[:, 0], 'g', linewidth=2, linestyle='dashed', label="OCM0, baseline")
    plt.plot(d, mag_norm_medi2[:, 0] + tolerance * base_sd2[:], 'r', linewidth=2, linestyle='dashed', label="OCM2, baseline")
    plt.title("Median signal, Baseline")
    plt.xlabel("Depth")
    plt.ylabel("Magnitude (a.u.)")
    for p in range(0, 100):
        im = plt.plot(d, ocm2[:, p], 'b', linewidth=1, linestyle='solid')
        ims.append(im)
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    ani.save("OCM2_baseline.gif", writer="imagemagic")

    # Plot Test set
    fig = plt.figure()
    ims = []
    plt.plot(d, mag_norm_medi2[:, 0], 'g', linewidth=2, linestyle='dashed', label="OCM0, baseline")
    plt.plot(d, mag_norm_medi2[:, 0] + tolerance * base_sd2[:], 'r', linewidth=2, linestyle='dashed', label="OCM2, baseline")
    plt.title("Median signal, State 2")
    plt.xlabel("Depth")
    plt.ylabel("Magnitude (a.u.)")
    for p in range(t_sub*target, t_sub*target+100):
        im = plt.plot(d, ocm2[:, p], 'b', linewidth=1, linestyle='solid')
        ims.append(im)
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    ani.save("OCM2_test.gif", writer="imagemagic")
    # =============================================================================
