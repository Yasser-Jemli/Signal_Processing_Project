

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
def nextpow2(N):
    """ Function for finding the next power of 2 """
    n = 1
    while n < N: n *= 2
    return n
# ============================================================

"""
App sig cos() ech

"""
f0 = 10
Fe = 80
T0 = 1/Fe


N = 32
t = np.arange(0,N)*T0
x = np.cos(2*np.pi*f0*t)

plt.figure()
plt.subplot(211); plt.stem(t,x);
plt.xlabel('temps(s)')
plt.ylabel('amplitude')
plt.show()

N = len(x)
Nf = nextpow2(N)
X = np.fft.fft(x,Nf)/N
X = np.fft.fftshift(X)
Xa = np.abs(X)

fp=np.arange(-Nf/2,Nf/2)*Fe/Nf

#-Représentation fréquentiel
plt.subplot(212)
plt.stem(fp,Xa)
plt.xlabel('fréquence (Hz)')
plt.ylabel('spectre d''amplitude')
plt.show()