import numpy as np

def voiced_excitation(duration, F0, Fs):
    """
    Create voiced speech excitation (impulse train).
    """
    T0 = int(np.round(Fs / F0))   # pitch period in samples
    excitation = np.zeros(duration)
    excitation[::T0] = -1
    return excitation


def resonator(x, F, BW, Fs):
    """
    Generate output of a second-order resonator (formant filter).
    """
    N = len(x)
    y = np.zeros(N)

    r = np.exp(-np.pi * BW / Fs)
    theta = 2 * np.pi * F / Fs

    a1 = 2 * r * np.cos(theta)
    a2 = -r**2

    for n in range(N):
        y[n] = x[n]
        if n >= 1:
            y[n] += a1 * y[n-1]
        if n >= 2:
            y[n] += a2 * y[n-2]

    return y


def synthesize_vowel(duration, F0,
                     F1, F2, F3, F4,
                     BW1, BW2, BW3, BW4,
                     Fs):
    """
    Synthesize vowel using source-filter model.
    """

    # Generate excitation
    excitation = voiced_excitation(duration, F0, Fs)

    # Cascade formant resonators
    y1 = resonator(excitation, F1, BW1, Fs)
    y2 = resonator(y1, F2, BW2, Fs)
    y3 = resonator(y2, F3, BW3, Fs)
    speech = resonator(y3, F4, BW4, Fs)

    return speech


# ===== Example Usage =====
if __name__ == "__main__":
    Fs = 8000
    duration = 8000  # 1 second
    F0 = 100

    # Example vowel (/a/)
    F1, F2, F3, F4 = 730, 1090, 2440, 3400
    BW1, BW2, BW3, BW4 = 80, 90, 120, 200

    speech = synthesize_vowel(duration, F0,
                              F1, F2, F3, F4,
                              BW1, BW2, BW3, BW4,
                              Fs)

    print("Synthesis complete. Signal length:", len(speech))
