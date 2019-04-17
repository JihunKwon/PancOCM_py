#import io
import numpy as np
import struct
from matplotlib import pyplot as plt
import array

#from alma import dsp
from dsp import fermi #This instead of above line works!
'''
###### This is from dsp. Temporary moved to here. ######
def fermi(n, tr_wdth, edge, type):
    """
    Fermi filter.

    fermi(N, tr_wdth, edge, type) is a Fermi filter made of N
    elements (i.e. # of pixels). tr_wdth is the width, in pixels, of the transition
    region where the filter passes from 1% to 99% edge is the position of the end
    of the transition region where filter is 1%), in pixels. type is either
    'low pass' or 'high pass'
    """
    if type[0:3] == 'low':
        sgn = 1
    elif type[0:3] == 'high':
        sgn = -1
    else:
        raise ValueError("Wrong value for type.")

    # It takes a `distance' 9.2*kT to go from 99% to 1%.
    # This is made in tr_wdth pixels.
    kT = tr_wdth/9.2
    Ef = edge - sgn*4.6*kT
    i = np.arange(0, n)
    f = 1./(1+np.exp(sgn*(i-Ef)/kT))

    return f
############################################################
'''


def load(fn, format='frankie_acq_v2'):
    'Loads raw OCM .bin file'
    if format != 'frankie_acq_v2':
        raise ValueError('OCM format {} not supported.'.format(format))

    print('Probing OCM file size...')
    count = 0
    format_string = ''
    with open(fn, 'rb') as f:

        dims = f.read(8)
        tlen = struct.unpack('i', dims[0:4])[0]
        width = struct.unpack('i', dims[4:])[0]

        ts = f.read(8)
        ts1 = struct.unpack('d', ts)    # system time
        ts = f.read(8)
        ts2 = struct.unpack('d', ts)    # NI time (absoluteInitialX)

        format_string = 'h' * tlen
        while True:

            ts = f.read(8)
            if len(ts) < 8:
                break  # one possible file ending
            ts1 = struct.unpack('d', ts)    # system time
            ts = f.read(8)
            ts2 = struct.unpack('d', ts)    # NI time (absoluteInitialX)

            # trace = array.array('h')
            # trace.fromfile(f, tlen)
            # if count and (len(trace) < tlen or ts1 is None):
            #     break   # can also be one of the possible file endings
            f.read(tlen * 2)  # instead just skip

            count += 1

        f.close()

    print('Loading OCM binary data...')
    ocm = np.zeros((tlen, count))
    t1 = np.zeros(count)
    t2 = np.zeros(count)

    with open(fn, 'rb') as f:

        dims = f.read(8)
        tlen = struct.unpack('i', dims[0:4])[0]
        width = struct.unpack('i', dims[4:])[0]

        ts = f.read(8)
        ts1 = struct.unpack('d', ts)    # system time
        ts = f.read(8)
        ts2 = struct.unpack('d', ts)    # NI time (absoluteInitialX)

        for i in range(count):
            ts = f.read(8)
            t1[i] = struct.unpack('d', ts)[0]    # system time
            ts = f.read(8)
            t2[i] = struct.unpack('d', ts)[0]    # NI time (absoluteInitialX)

            trace = array.array('h')
            trace.fromfile(f, tlen)
            ocm[:, i] = trace

    f.close()

    # align ext and ocm
    timecodes = np.where(t2 < 0)[0]
    # revert the inverted timestamps
    t2[timecodes] *= -1
    print("Done")
    return ocm, t1, t2, timecodes


def align_timecodes(tc_from, tc_to):

    d_tc_from = np.diff(tc_from)
    d_tc_to = np.diff(tc_to)
    ref = int(len(tc_to) / 2)
    min_diff = np.inf
    best_pos = -1
    for pos in range(len(tc_to) - 4):
        diff = d_tc_from[pos:pos+3] - d_tc_to[ref:ref+3]
        diff = np.sum(np.abs(diff))
        if diff < min_diff:
            min_diff = diff
            best_pos = pos

    inds_align = np.arange(best_pos-ref, len(tc_from))
    return tc_from[inds_align], inds_align


def phase_transform(ocm_real, F0, bw, dT, n_avg):

    print('Phase transforming OCM signal...')
    attn = 0.5          # Attenuation, in dB/cm/Mhz
    u = 1540            # Speed of sound, in m/s
    period = 1/F0       # Period of one oscillation, in s
    lambda_ = period*u  # Wavelength, in m
    dt_raw = 1/bw       # Dwell time

    #approx_target_framerate = 100
    # There is really nothing to do with the first ~250 points, just crop them
    Nt_raw = ocm_real.shape[0]  # Should be 5000-250, presumably
    NT = ocm_real.shape[1]

    T = np.arange(0, NT)*dT

    # Perform Transmit Gain Compensation (TGC)
    print('\tPerforming Transmit Gain Compensation (TGC)...')
    t = dt_raw * np.arange(0, Nt_raw)          # time axis
    tgc_dB = attn*(F0/1e6)*(u*100)*t        # TGC, in dB. u*100 is in cm/s
    tgc = 10**(tgc_dB/10)	                # TGC, as a multiplicative factor
    ocm_real = ocm_real * np.tile(tgc.reshape(-1, 1), [1, NT])

    # Turn the ocm signal into a complex entity, and filter out
    # some noise along the way
    tmp = np.fft.fft(ocm_real, axis=0)

    # There seems to be no point in keepiung signals beyond about 8*F0. A fermi
    # filter is used, which transitions from 0.99 at 5*F0, down to 0.01 at
    # 8*F0. Beyond 10*F0, and for negative frequencies, it essentially zeros
    # everything.
    fltr = dsp.fermi(Nt_raw, np.round(Nt_raw*5*F0/bw),
                     np.round(Nt_raw*8*F0/bw), 'low pass')
    tmp = np.multiply(np.tile(fltr.reshape(-1, 1), (1, NT)), tmp)
    # Now that it has been zeroed anyway, might as well truncate these zeros
    # and deal with smaller matrix sizes
    fac = 1
    if bw == 100e6:
        fac = 1/5
        warning('\tReducing sampling rate of 5Mhz input to make it better ' +
                'comparable to 1Mhz data - MICCAI 2018')

    inds = np.s_[int(fac*Nt_raw*8*F0/bw):tmp.shape[0]]
    tmp = np.delete(tmp, inds, axis=0)

    # Go back to t space, and redefine Nt and dt for this new matrix size
    ocm_complex = np.fft.ifft(tmp, axis=0)
    Nt = ocm_complex.shape[0]

    # As done above for NT, ensure that Nt is a multiple of 4.
    Nt = int(4*np.floor(Nt/4))
    inds = np.s_[Nt:ocm_complex.shape[0]]
    ocm_complex = np.delete(ocm_complex, inds, axis=0)
    dt = dt_raw*Nt_raw/Nt
    t = np.arange(0, Nt) * dt

    # Get the main parameters there are to get from the ocm signal: its
    # magnitude, its phase (wrapped or unwrapped), and derivatives of both
    # magnitude and phase along both t and T.
    # Start with the magnitude
    OCM = abs(ocm_complex)
    dOCM_dt = np.zeros_like(OCM)
    dOCM_dt[1:, :] = OCM[1:, :] - OCM[0:-1, :]
    dOCM_dT = np.zeros_like(OCM)
    dOCM_dT[:, 1:] = OCM[:, 1:] - OCM[:, 0:-1]
    # Get the unwrapped phase by integrating the derivative of the phase along
    # t (assumes all phase increments from one time point to the next remain
    # within -pi to pi).
    wTHETA = np.angle(ocm_complex)
    dTHETA_dt = np.zeros_like(ocm_complex)
    dTHETA_dt[1:, :] = (180/np.pi) * np.angle(ocm_complex[1:, :]
                                              * np.conj(ocm_complex[0:-1, :]))
    # Integrate along t to get the unwrapped phase
    THETA = np.cumsum(dTHETA_dt, axis=0)
    # Now the derivative along T
    dTHETA_dT = np.zeros_like(OCM)
    dTHETA_dT[:, 1:] = (180/np.pi)*np.angle(ocm_complex[:, 1:]
                                            * np.conj(ocm_complex[:, 0:-1]))
    # Using the unwrapped phase one can try to reverse engineer the
    # oscilloscope card sampling rate, even if just for fun.
    ncyc = np.max(np.mean(THETA, 1))/360    # Number of cycles in a whole trace
    t_trace = ncyc*period                   # Time for whole trace
    bw_meas = 1/(t_trace/Nt_raw)    # Bandwith is inverse of time increment
    print('Assumed sampling rate = ' +
          '{0:.2f}MHz, measured one = {0:.2f}MHz'.format(bw/1e6, bw_meas/1e6))
    # Might as well correct dTHETA_dt for this linear trend, thus making
    # positive vs. negative values more meaningful
    dTHETA_dt = dTHETA_dt - 360*ncyc/Nt

    vz = (0.5*lambda_/360)*dTHETA_dT/dT  # Speed along z, in m/s
    #if n_avg > 0:
        # f_avg = fspecial('average',[1,n_avg*2]);
        # f_avg(length(f_avg)/2+1:end)=0;
        # f_avg = f_avg * 2;
        # cumsum(f_avg);
        # vz = imfilter(vz,f_avg);

    return vz, T
