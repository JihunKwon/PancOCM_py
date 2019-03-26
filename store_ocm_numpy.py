# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 21:20:51 2018

@author: jerem
"""

import numpy as np
import matplotlib.pyplot as plt
import load_ocm as ocm

#this entire script's goal is to find the offset between the optical and breathing trace data

plt.close("all")


ocm_list = []
'''
ocm_list.append("C:\\Users\\jerem\\Dropbox (Partners HealthCare)\\BWH_Dropbox\\MR\\OCM\\PancreasProject\\Subject_01_20180928\\OCM\\run1.bin")
ocm_list.append("C:\\Users\\jerem\\Dropbox (Partners HealthCare)\\BWH_Dropbox\\MR\\OCM\\PancreasProject\\Subject_01_20180928\\OCM\\run2.bin")
ocm_list.append("C:\\Users\\jerem\\Dropbox (Partners HealthCare)\\BWH_Dropbox\\MR\\OCM\\PancreasProject\\Subject_01_20180928\\OCM\\run3.bin")
ocm_list.append("C:\\Users\\jerem\\Dropbox (Partners HealthCare)\\BWH_Dropbox\\MR\\OCM\\PancreasProject\\Subject_01_20181102\\OCM\\run1.bin")
ocm_list.append("C:\\Users\\jerem\\Dropbox (Partners HealthCare)\\BWH_Dropbox\\MR\\OCM\\PancreasProject\\Subject_01_20181102\\OCM\\run2.bin")
ocm_list.append("C:\\Users\\jerem\\Dropbox (Partners HealthCare)\\BWH_Dropbox\\MR\\OCM\\PancreasProject\\Subject_01_20181102\\OCM\\run3.bin")
ocm_list.append("C:\\Users\\jerem\\Dropbox (Partners HealthCare)\\BWH_Dropbox\\MR\\OCM\\PancreasProject\\Subject_02_20181102\\OCM\\run1.bin")
ocm_list.append("C:\\Users\\jerem\\Dropbox (Partners HealthCare)\\BWH_Dropbox\\MR\\OCM\\PancreasProject\\Subject_02_20181102\\OCM\\run2.bin")
ocm_list.append("C:\\Users\\jerem\\Dropbox (Partners HealthCare)\\BWH_Dropbox\\MR\\OCM\\PancreasProject\\Subject_02_20181102\\OCM\\run3.bin")
'''
#Jihun Local
'''
ocm_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_01_20180928\\run1.bin")
ocm_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_01_20180928\\run2.bin")
ocm_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_01_20180928\\run3.bin")
ocm_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_01_20181102\\runb1.bin")
ocm_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_01_20181102\\runb2.bin")
ocm_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_01_20181102\\runb3.bin")
ocm_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_02_20181102\\run1.bin")
ocm_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_02_20181102\\run2.bin")
ocm_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_02_20181102\\run3.bin")
ocm_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_02_20181220\\run1_ocm.bin")
ocm_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_02_20181220\\run2_ocm.bin")
ocm_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_02_20181220\\run3_ocm.bin")
'''
ocm_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_03_20190228\\run1.bin")
ocm_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_03_20190228\\run2a.bin")
ocm_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_03_20190228\\run2b.bin")


out_list = []
'''
out_list.append("C:\\Users\\jerem\\Dropbox (Partners HealthCare)\\BWH_Dropbox\\MR\\OCM\\PancreasProject\\Subject_01_20180928\\OCM\\run1.npy")
out_list.append("C:\\Users\\jerem\\Dropbox (Partners HealthCare)\\BWH_Dropbox\\MR\\OCM\\PancreasProject\\Subject_01_20180928\\OCM\\run2.npy")
out_list.append("C:\\Users\\jerem\\Dropbox (Partners HealthCare)\\BWH_Dropbox\\MR\\OCM\\PancreasProject\\Subject_01_20180928\\OCM\\run3.npy")
out_list.append("C:\\Users\\jerem\\Dropbox (Partners HealthCare)\\BWH_Dropbox\\MR\\OCM\\PancreasProject\\Subject_01_20181102\\OCM\\run1.npy")
out_list.append("C:\\Users\\jerem\\Dropbox (Partners HealthCare)\\BWH_Dropbox\\MR\\OCM\\PancreasProject\\Subject_01_20181102\\OCM\\run2.npy")
out_list.append("C:\\Users\\jerem\\Dropbox (Partners HealthCare)\\BWH_Dropbox\\MR\\OCM\\PancreasProject\\Subject_01_20181102\\OCM\\run3.npy")
out_list.append("C:\\Users\\jerem\\Dropbox (Partners HealthCare)\\BWH_Dropbox\\MR\\OCM\\PancreasProject\\Subject_02_20181102\\OCM\\run1.npy")
out_list.append("C:\\Users\\jerem\\Dropbox (Partners HealthCare)\\BWH_Dropbox\\MR\\OCM\\PancreasProject\\Subject_02_20181102\\OCM\\run2.npy")
out_list.append("C:\\Users\\jerem\\Dropbox (Partners HealthCare)\\BWH_Dropbox\\MR\\OCM\\PancreasProject\\Subject_02_20181102\\OCM\\run3.npy")
'''
#Jihun Local
'''
out_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_01_20180928\\run1.npy")
out_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_01_20180928\\run2.npy")
out_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_01_20180928\\run3.npy")
out_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_01_20181102\\run1.npy")
out_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_01_20181102\\run2.npy")
out_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_01_20181102\\run3.npy")
out_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_02_20181102\\run1.npy")
out_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_02_20181102\\run2.npy")
out_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_02_20181102\\run3.npy")
out_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_02_20181220\\run1.npy")
out_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_02_20181220\\run2.npy")
out_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_02_20181220\\run3.npy")
'''
out_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_03_20190228\\run1.npy")
out_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_03_20190228\\run2.npy")
out_list.append("C:\\OCM_Data\\Panc_OCM\\Subject_03_20190228\\run3.npy")

for fidx in range(0,3):
    ocm_filename = ocm_list[fidx]
    ocm_data = ocm.load_ocm(ocm_filename)
    
    out_filename = out_list[fidx]
    f = open(out_filename,"wb")
    np.save(f,ocm_data)
