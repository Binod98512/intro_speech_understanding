import numpy as np
import librosa
from scipy.signal import lfilter


def lpc(speech, frame_length, frame_skip, order):
    """
    Perform linear predictive analysis.
    """

    nframes = 1 + (len(speech) - frame_length) // frame_skip
    A = np.zeros((nframes, order + 1))
    excitation = np.zeros((nframes, frame_length))

    window = np.hamming(frame_length)

    for i in range(nframes):
        start = i * frame_skip
        frame = speech[start:start + frame_length]

        frame_win = frame * window

        # LPC coefficients (a[0] = 1)
        a = librosa.lpc(frame_win, order)
        A[i, :] = a

        # residual (prediction error)
        e = lfilter(a, [1.0], frame_win)
        excitation[i, :] = e

    return A, excitation


def synthesize(e, A, frame_skip):
    """
    Synthesize speech from LPC residual.
    """

    nframes, order_plus_one = A.shape
    order = order_plus_one - 1

    duration = nframes * frame_skip
    synthesis = np.zeros(duration)

    for i in range(nframes):
        start = i * frame_skip
        a = A[i]

        # take only valid part of excitation
        e_frame = e[start:start + frame_skip]

        # synthesis filter: 1/A(z)
        y = lfilter([1.0], a, e_frame)

        synthesis[start:start + frame_skip] = y

    return synthesis


def robot_voice(excitation, T0, frame_skip):
    """
    Create robot voice excitation (periodic impulse train with same frame gain).
    """

    nframes, frame_length = excitation.shape

    gain = np.zeros(nframes)
    e_robot = np.zeros(nframes * frame_skip)

    for i in range(nframes):
        # Compute gain from residual energy
        gain[i] = np.sqrt(np.mean(excitation[i, -frame_skip:] ** 2))

        start = i * frame_skip

        # Create impulse train for this frame
        for n in range(frame_skip):
            if n % T0 == 0:
                e_robot[start + n] = gain[i]

    return gain, e_robot
