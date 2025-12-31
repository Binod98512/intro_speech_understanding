import numpy as np

def VAD(waveform, Fs):
    frame_length = int(0.025 * Fs)
    step = int(0.010 * Fs)

    N = len(waveform)
    num_frames = 1 + (N - frame_length) // step

    frames = []
    energies = []

    for i in range(num_frames):
        start = i * step
        frame = waveform[start:start + frame_length]
        frames.append(frame)
        energies.append(np.sum(frame ** 2))

    energies = np.array(energies)
    threshold = 0.1 * np.max(energies)

    segments = []
    current = []

    for i, energy in enumerate(energies):
        if energy > threshold:
            current.extend(frames[i])
        else:
            if len(current) > 0:
                segments.append(np.array(current))
                current = []

    if len(current) > 0:
        segments.append(np.array(current))

    return segments


def segments_to_models(segments, Fs):
    models = []

    frame_length = int(0.004 * Fs)
    step = int(0.002 * Fs)

    for segment in segments:
        emphasized = np.append(segment[0], segment[1:] - 0.97 * segment[:-1])

        N = len(emphasized)
        num_frames = 1 + (N - frame_length) // step

        spectra = []

        for i in range(num_frames):
            start = i * step
            frame = emphasized[start:start + frame_length]

            spectrum = np.abs(np.fft.fft(frame))
            spectrum = spectrum[:len(spectrum)//2]
            spectra.append(np.log(spectrum + 1e-6))

        model = np.mean(np.array(spectra), axis=0)
        models.append(model)

    return models


def recognize_speech(testspeech, Fs, models, labels):
    test_segments = VAD(testspeech, Fs)
    test_models = segments_to_models(test_segments, Fs)

    Y = len(models)
    K = len(test_models)

    sims = np.zeros((Y, K))
    test_outputs = []

    for k, test_model in enumerate(test_models):
        for y, model in enumerate(models):
            sims[y, k] = np.dot(test_model, model) / (
                np.linalg.norm(test_model) * np.linalg.norm(model)
            )

        best = np.argmax(sims[:, k])
        test_outputs.append(labels[best])

    return sims, test_outputs
