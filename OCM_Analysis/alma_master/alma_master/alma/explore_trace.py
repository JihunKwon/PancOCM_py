'''
Created on Mar 5, 2019

@author: jihun
This code is copied from explore_panc_ocm.py. Instead of using root mean square, try to come up with better indices
to characterize the OCM curve. What I tried is:

Envelope Detection: Use Hilbert transformation instead of root mean squared
'''

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
out_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_01_20180928\\run1.npy") #Before water
out_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_01_20180928\\run2.npy") #After water
out_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_01_20180928\\run3.npy") #10min After water
out_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_01_20181102\\run1.npy")
out_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_01_20181102\\run2.npy")
out_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_01_20181102\\run3.npy")
out_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_02_20181102\\run1.npy")
out_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_02_20181102\\run2.npy")
out_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_02_20181102\\run3.npy")
out_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_02_20181220\\run1.npy")
out_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_02_20181220\\run2.npy")
out_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_02_20181220\\run3.npy")
out_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_03_20190228\\run1.npy")
out_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_03_20190228\\run2.npy")
out_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_03_20190228\\run3.npy")


#these are where the runs end in each OCM file
#rep_list = [8769, 8769, 8769, 8767, 8767, 8767, 7506, 7506, 7506]
num_subject = 5;
rep_list = [8196, 8196, 8196, 8192, 8192, 8192, 6932, 6932, 6932, 3690, 3690, 3690, 3401, 3401, 3401]# 3124 3401 3200

#these store data for each transducer, 5 breath holds, 15 runs
'''
t0 = np.zeros([500,5,np.size(rep_list)])
t1 = np.zeros([500,5,np.size(rep_list)])
t2 = np.zeros([500,5,np.size(rep_list)])
t3 = np.zeros([500,5,np.size(rep_list)])
'''

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
    #fidx = 0
    in_filename = out_list[fidx]
    ocm = np.load(in_filename)
    
    
    #crop data
    ocm = ocm[300:800,:] #Original code. 
    #ocm = ocm[300+150:800,:] 
    #ocm = ocm[300:900,:] #600FOV
    #ocm = ocm[300+150:800+150,:]
    #ocm = ocm[200:900,:]
    
    
    #s=# of samples per trace
    #t=# of total traces
    s, t = np.shape(ocm)
    
    #filter the data
    hptr = np.ones([s,t]) #high pass filter
    lptr = np.ones([s,t]) #low pass filter
    lptr_0 = np.ones([s,t])
    lptr_1 = np.ones([s,t])
    lptr_2 = np.ones([s,t])
    lptr_3 = np.ones([s,t])
    bandtr = np.ones([s,t])
    lptr_hil = np.ones([s,t]) #low pass filter to hilbert transform
    lptr_hil_0 = np.ones([s,t]) #low pass filter to hilbert transform
    lptr_hil_1 = np.ones([s,t]) 
    lptr_hil_2 = np.ones([s,t]) 
    lptr_hil_3 = np.ones([s,t]) 
    lptr_hil_0_norm = np.ones([s,t]) #Normalized
    lptr_hil_1_norm = np.ones([s,t]) 
    lptr_hil_2_norm = np.ones([s,t]) 
    lptr_hil_3_norm = np.ones([s,t]) 
    lptr_raw = np.ones([s,t]) #low pass filter to hilbert transform
    f1 = np.ones([2])
    f2 = np.ones([10])
    
    b = np.linspace(0,t-1,t)
    b0 = np.mod(b,4)==0
    ocm0 = ocm[:,b0]
    b1 = np.mod(b,4)==1
    ocm1 = ocm[:,b1]
    b2 = np.mod(b,4)==2
    ocm2 = ocm[:,b2]
    b3 = np.mod(b,4)==3
    ocm3 = ocm[:,b3]
    
    tmax = (t+1)/4 - 1 #Sometimes OCM3 is slightly shorter than other OCMs. In that case, "-1" is necessarily
    for p in range(0,int(tmax)):
        #p=2000;
        #OCM0
        tr0 = ocm0[:,p] #raw
        #bandtr[:,p] = np.convolve(tr0,[1,-1],'same') #Band pass
        bandtr[:,p] = np.convolve(tr0,f1,'same')
        tr0_b = bandtr[:,p]
        envelope_0 = abs(signal.hilbert(tr0_b)) #Apply hilbert transform to detect envelope
        lptr_hil_0[:,p] = np.convolve(envelope_0,f2,'same')
        tr0_hil = lptr_hil_0[:,p]
        #normalize
        lptr_hil_0_norm[:,p] = np.divide(lptr_hil_0[:,p],np.max(lptr_hil_0[:,p]))
        
        #OCM1
        tr1 = ocm1[:,p] #raw
        #bandtr[:,p] = np.convolve(tr1,[1,-1],'same') #Band pass
        bandtr[:,p] = np.convolve(tr1,f1,'same')
        tr1_b = bandtr[:,p]
        envelope_1 = abs(signal.hilbert(tr1_b)) #Apply hilbert transform to detect envelope
        lptr_hil_1[:,p] = np.convolve(envelope_1,f2,'same')
        tr1_hil = lptr_hil_1[:,p]
        #normalize
        lptr_hil_1_norm[:,p] = np.divide(lptr_hil_1[:,p],np.max(lptr_hil_1[:,p]))
        
        #OCM2
        tr2 = ocm2[:,p] #raw
        #bandtr[:,p] = np.convolve(tr2,[1,-1],'same') #Band pass
        bandtr[:,p] = np.convolve(tr2,f1,'same')
        tr2_b = bandtr[:,p]
        envelope_2 = abs(signal.hilbert(tr2_b)) #Apply hilbert transform to detect envelope
        lptr_hil_2[:,p] = np.convolve(envelope_2,f2,'same')
        tr2_hil = lptr_hil_2[:,p]
        #normalize
        lptr_hil_2_norm[:,p] = np.divide(lptr_hil_2[:,p],np.max(lptr_hil_2[:,p]))
        
        #OCM3
        tr3 = ocm3[:,p] #raw
        #bandtr[:,p] = np.convolve(tr3,[1,-1],'same') #Band pass
        bandtr[:,p] = np.convolve(tr3,f1,'same')
        tr3_b = bandtr[:,p]
        envelope_3 = abs(signal.hilbert(tr3_b)) #Apply hilbert transform to detect envelope
        lptr_hil_3[:,p] = np.convolve(envelope_3,f2,'same')
        tr3_hil = lptr_hil_3[:,p]
        #normalize
        lptr_hil_3_norm[:,p] = np.divide(lptr_hil_3[:,p],np.max(lptr_hil_3[:,p]))
        
        '''
        #OCM0
        fig, ax = plt.subplots()
        ax.plot(tr0, label="raw")
        ax.plot(tr0_b, label="Band pass to raw")
        ax.plot(envelope_0, label="hilbert")
        ax.plot(tr0_hil, label="low pass to hilbert")
        plt.title("Low pass to raw and hilbert_OCM0")    
        plt.legend(loc='upper right')
        fig.savefig('low pass to raw and hilbert_OCM0.png', dpi=fig.dpi)
        #OCM1
        fig, ax = plt.subplots()
        ax.plot(tr1, label="raw")
        ax.plot(tr1_b, label="Band pass to raw")
        ax.plot(envelope_1, label="hilbert")
        ax.plot(tr1_hil, label="low pass to hilbert")
        plt.title("Low pass to raw and hilbert_OCM1")    
        plt.legend(loc='upper right')
        fig.savefig('low pass to raw and hilbert_OCM1.png', dpi=fig.dpi)
        #OCM2
        fig, ax = plt.subplots()
        ax.plot(tr2, label="raw")
        ax.plot(tr2_b, label="Band pass to raw")
        ax.plot(envelope_2, label="hilbert")
        ax.plot(tr2_hil, label="low pass to hilbert")
        plt.title("Low pass to raw and hilbert_OCM2")    
        plt.legend(loc='upper right')
        fig.savefig('low pass to raw and hilbert_OCM2.png', dpi=fig.dpi)        
        '''

    ocm0 = lptr_hil_0
    ocm1 = lptr_hil_1
    ocm2 = lptr_hil_2
    ocm3 = lptr_hil_3
    
   
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
    
    '''
    for i in range(0,5): #First run includes some extra ocm signal
        ocm0m[:,i] = np.mean(np.abs(ocm0[:,rep_list[fidx]*i:rep_list[fidx]*(i+1)]),1)
        ocm1m[:,i] = np.mean(np.abs(ocm1[:,rep_list[fidx]*i:rep_list[fidx]*(i+1)]),1)
        ocm2m[:,i] = np.mean(np.abs(ocm2[:,rep_list[fidx]*i:rep_list[fidx]*(i+1)]),1)
        ocm3m[:,i] = np.mean(np.abs(ocm3[:,rep_list[fidx]*i:rep_list[fidx]*(i+1)]),1)
    '''
    '''
    for i in range(0,5): #Distribute ocm signal from end to start
        ocm0m[:,i] = np.mean(np.abs(ocm0[:,ocm0.shape[1]-rep_list[fidx]*i-1:ocm0.shape[1]-rep_list[fidx]*(i+1)-1]),1)
        ocm1m[:,i] = np.mean(np.abs(ocm1[:,ocm1.shape[1]-rep_list[fidx]*i-1:ocm1.shape[1]-rep_list[fidx]*(i+1)-1]),1)
        ocm2m[:,i] = np.mean(np.abs(ocm2[:,ocm2.shape[1]-rep_list[fidx]*i-1:ocm2.shape[1]-rep_list[fidx]*(i+1)-1]),1)
        ocm3m[:,i] = np.mean(np.abs(ocm3[:,ocm3.shape[1]-rep_list[fidx]*i-1:ocm3.shape[1]-rep_list[fidx]*(i+1)-1]),1)
    '''
    
    for i in range(0,5): #Distribute ocm signal from end to start
        ocm0m[:,i] = np.mean(np.abs(ocm0[:,int((ocm0.shape[1]+1)/4)-rep_list[fidx]*(i+1)-1:int((ocm0.shape[1]+1)/4)-rep_list[fidx]*i-1]),1)
        ocm1m[:,i] = np.mean(np.abs(ocm1[:,int((ocm0.shape[1]+1)/4)-rep_list[fidx]*(i+1)-1:int((ocm1.shape[1]+1)/4)-rep_list[fidx]*i-1]),1)
        ocm2m[:,i] = np.mean(np.abs(ocm2[:,int((ocm0.shape[1]+1)/4)-rep_list[fidx]*(i+1)-1:int((ocm2.shape[1]+1)/4)-rep_list[fidx]*i-1]),1)
        ocm3m[:,i] = np.mean(np.abs(ocm3[:,int((ocm0.shape[1]+1)/4)-rep_list[fidx]*(i+1)-1:int((ocm3.shape[1]+1)/4)-rep_list[fidx]*i-1]),1)
    
    
    #collect all the data so far
    t0[:,:,fidx] = ocm0m
    t1[:,:,fidx] = ocm1m
    t2[:,:,fidx] = ocm2m
    t3[:,:,fidx] = ocm3m
    
    
    sample_rate_MHz = 10
    us_per_sample = 1/sample_rate_MHz

    #in cm
    little_t = np.linspace(2.3,6.2,s)
    
fig, (ax1, ax2, ax3) = plt.subplots(3,1)
ax1.plot(little_t, t2[:,1,6])
ax2.plot(little_t, t2[:,1,7])
ax3.plot(little_t, t2[:,1,8])
ax2.set_ylabel("OCM Amplitude (a.u.)")
ax3.set_xlabel("Depth (cm)")

out1 = t2[:,1,6]
out2 = t2[:,1,7]
out3 = t2[:,1,8]


'''Method 2: rm0 is only from fidx=0,3,6,9, which means signal from first-run (before water) is used for subtraction
'''
#loop through subjects
for sub in range(0,num_subject):
    #mean for this run across all breath holds
    rm0 = np.mean(t0[:,:,sub*3],1) #fidx=0,3,6,9 ... (before water)
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
'''
out_txt.append("C:\\Users\\jerem\\Dropbox (Partners HealthCare)\\BWH_Dropbox\\MR\\OCM\\PancreasProject\\transducer0.txt")
out_txt.append("C:\\Users\\jerem\\Dropbox (Partners HealthCare)\\BWH_Dropbox\\MR\\OCM\\PancreasProject\\transducer1.txt")
out_txt.append("C:\\Users\\jerem\\Dropbox (Partners HealthCare)\\BWH_Dropbox\\MR\\OCM\\PancreasProject\\transducer2.txt")
out_txt.append("C:\\Users\\jerem\\Dropbox (Partners HealthCare)\\BWH_Dropbox\\MR\\OCM\\PancreasProject\\transducer3.txt")
'''

#Jihun Local
out_txt.append("C:\\OCM_Data\\Panc_OCM\\transducer0.txt")
out_txt.append("C:\\OCM_Data\\Panc_OCM\\transducer1.txt")
out_txt.append("C:\\OCM_Data\\Panc_OCM\\transducer2.txt")
out_txt.append("C:\\OCM_Data\\Panc_OCM\\transducer3.txt")


#np.savetxt(out_txt[2],d0,fmt='%0.04f',delimiter=' ',newline='\n')

np.savetxt(out_txt[0],d0,fmt='%0.04f',delimiter=' ',newline='\n')
np.savetxt(out_txt[1],d1,fmt='%0.04f',delimiter=' ',newline='\n')
np.savetxt(out_txt[2],d2,fmt='%0.04f',delimiter=' ',newline='\n')
np.savetxt(out_txt[3],d3,fmt='%0.04f',delimiter=' ',newline='\n')