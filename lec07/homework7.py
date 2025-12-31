import numpy as np

def major_chord(f, Fs):
    '''
    Generate a one-half-second major chord.
    '''
    duration = 0.5  # seconds
    N = int(Fs * duration)
    n = np.arange(N)

    # Frequencies of a major chord
    f_root = f
    f_major_third = f * 2**(4/12)
    f_major_fifth = f * 2**(7/12)

    # Generate chord (sum of three tones)
    x = (np.cos(2*np.pi*f_root*n/Fs) +
         np.cos(2*np.pi*f_major_third*n/Fs) +
         np.cos(2*np.pi*f_major_fifth*n/Fs))

    return x


def dft_matrix(N):
    '''
    Create a DFT transform matrix of size NxN.
    '''
    k = np.arange(N).reshape((N, 1))
    n = np.arange(N).reshape((1, N))
    W = np.cos(2*np.pi*k*n/N) - 1j*np.sin(2*np.pi*k*n/N)
    return W


def spectral_analysis(x, Fs):
    '''
    Find the three loudest frequencies in x.
    '''
    N = len(x)

    # Compute DFT using FFT
    X = np.fft.fft(x)
    magnitude = np.abs(X)

    # Frequency axis
    freqs = np.fft.fftfreq(N, d=1/Fs)

    # Use only positive frequencies
    positive = freqs > 0
    freqs = freqs[positive]
    magnitude = magnitude[positive]

    # Find indices of three largest peaks
    idx = np.argsort(magnitude)[-3:]
    loudest_freqs = np.sort(freqs[idx])

    return loudest_freqs[0], loudest_freqs[1], loudest_freqs[2]
