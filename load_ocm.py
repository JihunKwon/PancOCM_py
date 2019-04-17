import struct
import numpy as np
import matplotlib.pyplot as plt
import cmath as math

def load_ocm(filename):
    f = open(filename,"rb")

    #read header
    samples_per_trace = np.fromfile(f, dtype=np.int, count=1)[0]
    bytes_per_sample = np.fromfile(f, dtype=np.int, count=1)[0]
    sys_ref_time = np.fromfile(f, dtype=np.float64, count=1)[0]
    ni_ref_time = np.fromfile(f, dtype=np.float64, count=1)[0]

    #find how many full traces are in the file
    cnt = 0
    while 1:
        t1 = np.fromfile(f, dtype=np.float64, count=1)
        if t1.size<1:
            break
        t2 = np.fromfile(f, dtype=np.float64, count=1)
        d = np.fromfile(f, dtype=np.short, count=samples_per_trace)
        if ((cnt>1 and d.size<samples_per_trace) or t1.size<1):
            break
        cnt = cnt+1
        
    #rewind and read the whole file into ram
    ocm = np.zeros((samples_per_trace, cnt), dtype=np.short)
    ts1_us = np.zeros((1,cnt), dtype=np.float64)
    ts2_us = np.zeros((1,cnt), dtype=np.float64)
    f.seek(24)
    for i in range(0,cnt):
        ts1_us[0,i] = np.fromfile(f, dtype=np.float64, count=1)[0]
        ts2_us[0,i] = np.fromfile(f, dtype=np.float64, count=1)[0]
        ocm[:,i] = np.fromfile(f, dtype=np.short, count=samples_per_trace)

    f.close()
    return ocm


