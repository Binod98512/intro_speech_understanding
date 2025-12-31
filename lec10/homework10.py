import numpy as np
import torch
import torch.nn as nn



def get_features(waveform, Fs):
    # ---------- Pre-emphasis ----------
    waveform = np.append(waveform[0], waveform[1:] - 0.97 * waveform[:-1])

    # ---------- Spectrogram (4ms frame, 2ms step) ----------
    frame_length = int(0.004 * Fs)
    step = int(0.002 * Fs)

    N = len(waveform)
    num_frames = 1 + (N - frame_length) // step

    features = []
    for i in range(num_frames):
        start = i * step
        frame = waveform[start:start + frame_length]
        spectrum = np.abs(np.fft.fft(frame))
        spectrum = spectrum[:len(spectrum)//2]   # low-frequency half
        features.append(np.log(spectrum + 1e-6))

    features = np.array(features)

    # ---------- VAD (25ms frame, 10ms step) ----------
    vad_frame = int(0.025 * Fs)
    vad_step = int(0.010 * Fs)

    vad_frames = 1 + (N - vad_frame) // vad_step
    energies = []

    for i in range(vad_frames):
        start = i * vad_step
        frame = waveform[start:start + vad_frame]
        energies.append(np.sum(frame ** 2))

    energies = np.array(energies)
    threshold = 0.1 * np.max(energies)

    labels = -1 * np.ones(num_frames, dtype=int)
    label_id = 0
    repeat = 5
    idx = 0

    for energy in energies:
        if energy > threshold:
            labels[idx:idx + repeat] = label_id
            label_id += 1
        idx += repeat

    labels[labels < 0] = 0

    return features, labels



def train_neuralnet(features, labels, iterations):
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    NFEATS = X.shape[1]
    NLABELS = int(labels.max()) + 1

    model = nn.Sequential(
        nn.LayerNorm(NFEATS),
        nn.Linear(NFEATS, NLABELS)
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    lossvalues = np.zeros(iterations)

    for i in range(iterations):
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        lossvalues[i] = loss.item()

    return model, lossvalues



def test_neuralnet(model, features):
    X = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(X)
        probabilities = torch.softmax(outputs, dim=1)

    return probabilities.detach().numpy()
