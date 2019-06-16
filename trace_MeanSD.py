import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pickle
import matplotlib.style as style
style.available

#style.use('ggplot')

with open('ocm012_s1r1_10per.pkl', 'rb') as f:
    ocm0_all_r1, ocm1_all_r1, ocm2_all_r1 = pickle.load(f)

print(ocm0_all_r1.shape) # (350, 5000, 3)

ocm0_m = np.zeros([ocm0_all_r1.shape[0], 5, 2]) # 3rd dimension is before and after water
ocm0_sd = np.zeros([ocm0_all_r1.shape[0], 5, 2])
ocm0_bef_m_sd = np.zeros([ocm0_all_r1.shape[0], 5, 2]) # 3rd dimension is mean+SD and mean-SD
ocm0_aft_m_sd = np.zeros([ocm0_all_r1.shape[0], 5, 2])

test = np.zeros([ocm0_all_r1.shape[0], 5])

for tp in range(0, 5): # timepoint 0 to 4
    ocm0_m[:,tp,0] = np.mean(ocm0_all_r1[:,tp*1000:(tp+1)*1000,0], axis=1)
    ocm0_m[:,tp,1] = np.mean(ocm0_all_r1[:,tp*1000:(tp+1)*1000,1], axis=1)
    ocm0_sd[:,tp,0] = np.std(ocm0_all_r1[:,tp*1000:(tp+1)*1000,0], axis=1)
    ocm0_sd[:,tp,1] = np.std(ocm0_all_r1[:,tp*1000:(tp+1)*1000,1], axis=1)

    ocm0_bef_m_sd[:,tp,0] = ocm0_m[:,tp,0] + ocm0_sd[:,tp,0]
    ocm0_bef_m_sd[:,tp,1] = ocm0_m[:,tp,0] - ocm0_sd[:,tp,0]
    ocm0_aft_m_sd[:,tp,0] = ocm0_m[:,tp,1] + ocm0_sd[:,tp,1]
    ocm0_aft_m_sd[:,tp,1] = ocm0_m[:,tp,1] - ocm0_sd[:,tp,1]


s = ocm0_m.shape[0]
r = range(s)
print(r)
r2 = np.arange(s)
# ========================Visualize===========================================
# This part shows how the signal changed after the filtering.

depth = np.linspace(2.3,6.2,s)
fig = plt.figure(figsize=(5,9))

for tp in range(5):
    ax0 = fig.add_subplot(5,1,tp+1)
    a0 = ax0.plot(depth,ocm0_m[:,tp,0],'b', linewidth=2) # Before water
    a1 = ax0.plot(depth,ocm0_m[:,tp,1],'r', linewidth=2) #After water
    for i in range(s):
        if i % 10 == 0:
            ax0.errorbar(depth[i], ocm0_m[i,tp,0], ocm0_sd[i,tp,0], fmt='none', capsize=2, ecolor='b', linestyle="dashed")
            ax0.errorbar(depth[i], ocm0_m[i,tp,1], ocm0_sd[i,tp,1], fmt='none', capsize=2, ecolor='r', linestyle="dashed")

    #plt.fill_between(np.arange(2.3,6.2,s),ocm0_bef_m_sd[:,tp,0],ocm0_bef_m_sd[:,tp,1],alpha=.3)

    ax0.set_title('BH %i' % tp)
    ax0.set_xlim(2,6)
fig.tight_layout()
fig.show()
plt.savefig('Trace_s1r1.png')
# =============================================================================
