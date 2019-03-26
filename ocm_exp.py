import numpy as np
import cmath as math

class ocm_exp(object):
    """
    Opens a binary OCM file.

    Attributes:
        ocm_filename (str): path and file name combined
        samples_per_trace (int): number of samples for one pulse
        cnt (int): number of traces in the file
        ocm (short matrix): ocm samples, size = samples per trace X cnt
        ts1_us (float array): system timestamp
        ts2_us (float array): NI timestamp

    """

    def __init__(self,ocm_filename):
        self.ocm_filename = ocm_filename
        
        f = open(self.ocm_filename,"rb")

        #read header
        self.samples_per_trace = np.fromfile(f, dtype=np.int, count=1)[0]
        bytes_per_sample = np.fromfile(f, dtype=np.int, count=1)[0]
        sys_ref_time = np.fromfile(f, dtype=np.float64, count=1)[0]
        ni_ref_time = np.fromfile(f, dtype=np.float64, count=1)[0]

        #find how many full traces are in the file
        self.cnt = 0
        while 1:
            t1 = np.fromfile(f, dtype=np.float64, count=1)
            if t1.size<1:
                break
            t2 = np.fromfile(f, dtype=np.float64, count=1)
            d = np.fromfile(f, dtype=np.short, count=self.samples_per_trace)
            if ((self.cnt>1 and d.size<self.samples_per_trace) or t1.size<1):
                break
            self.cnt = self.cnt+1
            
        #rewind and read the whole file into ram
        self.ocm = np.zeros((self.samples_per_trace, self.cnt), dtype=np.short)
        self.ts1_us = np.zeros((1,self.cnt), dtype=np.float64)
        self.ts2_us = np.zeros((1,self.cnt), dtype=np.float64)
        f.seek(24)
        for i in range(0,self.cnt):
            self.ts1_us[0,i] = np.fromfile(f, dtype=np.float64, count=1)[0]
            self.ts2_us[0,i] = np.fromfile(f, dtype=np.float64, count=1)[0]
            self.ocm[:,i] = np.fromfile(f, dtype=np.short, count=self.samples_per_trace)

        f.close()


