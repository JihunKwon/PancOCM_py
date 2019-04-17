'''
Created on Mar 6, 2019

@author: jihun
'''
from scipy import signal

envelope = abs(signal.hilbert(data))