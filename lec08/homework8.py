import numpy as np

def waveform_to_frames(waveform, frame_length, step):
    '''
    Chop a waveform into overlapping frames.
    '''
    N = len(waveform)
    num_frames = 1 + (N - frame_length) // step

    frames = np.zeros((num_frames, frame_length))

    for i in range(num_frames):
        start = i * step
        frames[i, :] = waveform[start:start + frame_length]

    return frames


def frames_to_mstft(frames):
    '''
    Take the magnitude FFT of every row of the frames matrix.
    '''
    mstft = np.abs(np.fft.fft(frames, axis=1))
    return mstft


def mstft_to_spectrogram(mstft):
    '''
    Convert magnitude STFT to decibel spectrogram with dynamic range limit.
    '''
    # Floor to avoid log of zero
    floor = 0.001 * np.amax(mstft)
    mstft_safe = np.maximum(mstft, floor)

    spectrogram = 20 * np.log10(mstft_safe)

    # Ensure dynamic range ≤ 60 dB
    max_val = np.amax(spectrogram)
    spectrogram = np.maximum(spectrogram, max_val - 60)

    return spectrogram
