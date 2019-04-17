import numpy as np

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
