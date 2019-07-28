'''
This code check raw traces and remove if outlier is detected.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pickle
import seaborn as sns

plt.rcParams['font.family'] ='sans-serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

#sr_list = ['s1r1', 's1r2', 's2r1', 's2r2', 's3r1', 's3r2']
#sr_list = ['s1r1']
sr_list = ['s2r1']
for fidx in range(0,np.size(sr_list)):
    Sub_run = sr_list[fidx]
    #sr_name = 'ocm012_' + Sub_run + '.pkl'  # filtered
    sr_name = 'Raw_det_ocm012_' + Sub_run + '.pkl'  # raw

    with open(sr_name, 'rb') as f:
        ocm0_all, ocm1_all, ocm2_all = pickle.load(f)

    print(ocm0_all.shape)
    depth = np.linspace(0, ocm0_all.shape[0] - 1, ocm0_all.shape[0])
    num_max = ocm0_all.shape[1]  # Total num of traces

    # Split data to "before, train", "before, test", and "after, test".
    # first, get "before" components
    ocm_bef = np.zeros([ocm0_all.shape[0], ocm0_all.shape[1], 3])
    ocm_aft = np.zeros([ocm0_all.shape[0], ocm0_all.shape[1], 3])
    ocm_bef[:, :, 0] = ocm0_all[:, :, 0]
    ocm_bef[:, :, 1] = ocm1_all[:, :, 0]
    ocm_bef[:, :, 2] = ocm2_all[:, :, 0]
    ocm_aft[:, :, 0] = ocm0_all[:, :, 1]
    ocm_aft[:, :, 1] = ocm1_all[:, :, 1]
    ocm_aft[:, :, 2] = ocm2_all[:, :, 1]
    print('ocm_bef', ocm_bef.shape)
    print('ocm_aft', ocm_aft.shape)
    # Transpose
    ocm_bef = np.einsum('abc->bac', ocm_bef)
    ocm_aft = np.einsum('abc->bac', ocm_aft)
    print('ocm_bef', ocm_bef.shape)
    print('ocm_aft', ocm_aft.shape)
    # Shuffle "before"
    np.random.shuffle(ocm_bef)

    ratio = 0.2  # ratio of test set for before
    ocm_bef_train = ocm_bef[0:int(num_max*(1-ratio)), :, :]  # First 80% traces are for training
    ocm_bef_test = ocm_bef[int(num_max*(1-ratio)):num_max, :, :]  # Last 20% traces are for training
    print('ocm_bef_train:', ocm_bef_train.shape)
    print('ocm_bef_test:', ocm_bef_test.shape)

    # Calculate mean of "before test"
    ocm_bef_m = np.zeros([ocm_bef_train.shape[1], 3])
    ocm_bef_sd = np.zeros([ocm_bef_train.shape[1], 3])

    ocm_bef_m[:, 0] = np.mean(ocm_bef_train[:, :, 0], axis=0)
    ocm_bef_m[:, 1] = np.mean(ocm_bef_train[:, :, 1], axis=0)
    ocm_bef_m[:, 2] = np.mean(ocm_bef_train[:, :, 2], axis=0)
    ocm_bef_sd[:, 0] = np.std(ocm_bef_train[:, :, 0], axis=0)
    ocm_bef_sd[:, 1] = np.std(ocm_bef_train[:, :, 1], axis=0)
    ocm_bef_sd[:, 2] = np.std(ocm_bef_train[:, :, 2], axis=0)

    # Calculate mean of "after test"
    ocm_aft_m = np.zeros([ocm_aft.shape[1], 3])
    ocm_aft_sd = np.zeros([ocm_aft.shape[1], 3])

    ocm_aft_m[:, 0] = np.mean(ocm_aft[:, :, 0], axis=0)
    ocm_aft_m[:, 1] = np.mean(ocm_aft[:, :, 1], axis=0)
    ocm_aft_m[:, 2] = np.mean(ocm_aft[:, :, 2], axis=0)
    ocm_aft_sd[:, 0] = np.std(ocm_aft[:, :, 0], axis=0)
    ocm_aft_sd[:, 1] = np.std(ocm_aft[:, :, 1], axis=0)
    ocm_aft_sd[:, 2] = np.std(ocm_aft[:, :, 2], axis=0)

    # ========================Visualize==============================================

    fig = plt.figure(figsize=(6, 3))
    # This part shows raw signals
    ## Mean+3SD
    # OCM1
    ocm = 1
    ax1 = fig.add_subplot(111)
    a0 = ax1.plot(depth, ocm_bef_m[:, ocm], 'b', linewidth=2, label="Before")  # Before water
    plt.fill_between(range(ocm_bef_m.shape[0]), ocm_bef_m[:,ocm]-3*ocm_bef_sd[:,ocm], ocm_bef_m[:,ocm]+3*ocm_bef_sd[:,ocm], alpha=.3)
    a1 = ax1.plot(depth, ocm_aft_m[:, ocm], 'r', linewidth=2, label="After")  # After water
    plt.fill_between(range(ocm_aft_m.shape[0]), ocm_aft_m[:,ocm]-3*ocm_aft_sd[:,ocm], ocm_aft_m[:,ocm]+3*ocm_aft_sd[:,ocm], alpha=.3)
    ax1.set_title(r'OCM, $mean \pm 3SD$')
    ax1.set_ylim(-2,2)
    ax1.set_xlabel('Depth')
    ax1.set_ylabel('Intensity')
    plt.legend(loc='best')

    fig.tight_layout()
    fig.show()
    f_name = 'Mean+3SD.png'
    plt.savefig(f_name)
    # =============================================================================
