import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pickle
import matplotlib.style as style
style.available

plt.rcParams['font.family'] ='sans-serif'#使用するフォント
plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')

with open('ocm012_s1r1_10per_350.pkl', 'rb') as f:
    ocm0_all_r1, ocm1_all_r1, ocm2_all_r1 = pickle.load(f)
print(ocm0_all_r1.shape) # (350, 5000, 3)

ocm0_m = np.zeros([ocm0_all_r1.shape[0], 5, 2]) # 3rd dimension is before and after water
ocm0_sd = np.zeros([ocm0_all_r1.shape[0], 5, 2])
ocm0_bef_m_sd = np.zeros([ocm0_all_r1.shape[0], 5, 2]) # 3rd dimension is mean+SD and mean-SD
ocm0_aft_m_sd = np.zeros([ocm0_all_r1.shape[0], 5, 2])

test = np.zeros([ocm0_all_r1.shape[0], 5])

index = int(ocm0_all_r1.shape[1]/5) # Depth of iteration
print(index)

for tp in range(0, 5): # timepoint 0 to 4
    ocm0_m[:,tp,0] = np.mean(ocm0_all_r1[:,tp*index:(tp+1)*index,0], axis=1)
    ocm0_m[:,tp,1] = np.mean(ocm0_all_r1[:,tp*index:(tp+1)*index,1], axis=1)
    ocm0_sd[:,tp,0] = np.std(ocm0_all_r1[:,tp*index:(tp+1)*index,0], axis=1)
    ocm0_sd[:,tp,1] = np.std(ocm0_all_r1[:,tp*index:(tp+1)*index,1], axis=1)

    ocm0_bef_m_sd[:,tp,0] = ocm0_m[:,tp,0] + ocm0_sd[:,tp,0]
    ocm0_bef_m_sd[:,tp,1] = ocm0_m[:,tp,0] - ocm0_sd[:,tp,0]
    ocm0_aft_m_sd[:,tp,0] = ocm0_m[:,tp,1] + ocm0_sd[:,tp,1]
    ocm0_aft_m_sd[:,tp,1] = ocm0_m[:,tp,1] - ocm0_sd[:,tp,1]

s = ocm0_m.shape[0]
r = range(s)
print(r)
r2 = np.arange(s)
'''
# ========================Visualize===========================================
# This part shows how the signal changed after the filtering.

#depth = np.linspace(2.3,6.2,s) # Jeremy Depth
depth = np.linspace(2.3,4.9,s) # My Depth
fig = plt.figure(figsize=(5,9))

for tp in range(5):
    ax0 = fig.add_subplot(5,1,tp+1)
    a0 = ax0.plot(depth,ocm0_m[:,tp,0],'b', linewidth=2, label="Before") # Before water
    a1 = ax0.plot(depth,ocm0_m[:,tp,1],'r', linewidth=2, label="After") #After water
    for i in range(s):
        if i % 10 == 0:
            ax0.errorbar(depth[i], ocm0_m[i,tp,0], ocm0_sd[i,tp,0], fmt='none', capsize=2, ecolor='b', linestyle="dashed")
            ax0.errorbar(depth[i], ocm0_m[i,tp,1], ocm0_sd[i,tp,1], fmt='none', capsize=2, ecolor='r', linestyle="dashed")

    #plt.fill_between(np.arange(2.3,6.2,s),ocm0_bef_m_sd[:,tp,0],ocm0_bef_m_sd[:,tp,1],alpha=.3)

    plt.xlabel('Depth (cm)')
    plt.ylabel('Intensity (a.u.)')
    plt.legend(loc='upper right')

    ax0.set_title('BH %i' % (tp+1))
    ax0.set_xlim(2,5)
fig.tight_layout()
fig.show()
plt.savefig('Trace_.png')
# =============================================================================
'''

# ========================Visualize only one trace for figure===========================================
# This part shows how the signal changed after the filtering.

#depth = np.linspace(2.3,6.2,s) # Jeremy Depth
depth = np.linspace(2.3,4.9,s) # My Depth
fig = plt.figure(figsize=(5,5))

# Set wich timepoint to visualize
tp = 4

# Before water
ax1 = fig.add_subplot(2,1,1)
a1 = ax1.plot(depth, ocm0_m[:,tp,0],'b', linewidth=2)
ax1.set_xlim(2,5)
ax1.set_ylim(0,1)
plt.ylabel('Intensity (a.u.)', fontsize=13)
plt.gca().xaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
plt.gca().yaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
# Plot SD
for i in range(s):
    if i % 10 == 0:
        ax1.errorbar(depth[i], ocm0_m[i,tp,0], ocm0_sd[i,tp,0], fmt='none', capsize=2, ecolor='b', linestyle="dashed")


# After water
# shared axis X
ax2 = fig.add_subplot(2,1,2, sharex = ax1)
a2 = ax2.plot(depth, ocm0_m[:,tp,1],'r', linewidth=2)
plt.setp(ax1.get_xticklabels(), visible=False)
# Plot SD
for i in range(s):
    if i % 10 == 0:
        ax2.errorbar(depth[i], ocm0_m[i,tp,1], ocm0_sd[i,tp,1], fmt='none', capsize=2, ecolor='r', linestyle="dashed")

ax2.set_xlim(2,5)
ax2.set_ylim(0,1)
plt.xlabel('Depth (cm)', fontsize=13)
plt.ylabel('Intensity (a.u.)', fontsize=13)
# remove last tick label for the second subplot
yticks = ax2.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)
plt.gca().xaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
plt.gca().yaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)

plt.subplots_adjust(hspace=.0)
#fig.tight_layout()
plt.subplots_adjust(left=0.12, right=0.97, bottom=0.1, top=0.95)
fig.show()
plt.savefig('Trace_one.png')
# =============================================================================
