import numpy as np
from scipy import signal


def apply_filters(data, srate, highpass=None, lowpass=None, notch=None):
    filtered = data.copy()
    nyquist = srate / 2

    if highpass is not None and highpass > 0:
        sos_high = signal.butter(4, highpass / nyquist, btype="high", output="sos")
        filtered = signal.sosfilt(sos_high, filtered)

    if lowpass is not None and lowpass < nyquist:
        sos_low = signal.butter(4, lowpass / nyquist, btype="low", output="sos")
        filtered = signal.sosfilt(sos_low, filtered)

    if notch is not None and notch > 0:
        b_notch, a_notch = signal.iirnotch(notch, 30, srate)
        filtered = signal.filtfilt(b_notch, a_notch, filtered)

    return filtered


def detect_markers(emg, srate, threshold, time_after, time_before, after_a, before_a, duration):
    emg_rect = np.abs(emg)
    rng = np.max(emg_rect) - np.min(emg_rect)
    if rng == 0:
        return np.array([], dtype=int), emg_rect

    emg_scaled = (emg_rect - np.min(emg_rect)) / rng
    emg_binary = emg_scaled > threshold
    emg_diff = np.diff(emg_binary.astype(int))
    onsets = np.where(emg_diff == 1)[0]

    if len(onsets) == 0:
        return np.array([], dtype=int), emg_scaled

    samples_after = int(time_after * srate)
    samples_before = int(time_before * srate)
    validated = []

    for onset in onsets:
        if onset < samples_before + 20:
            continue
        if onset + samples_after >= len(emg_scaled):
            continue
        after_mean = np.mean(emg_scaled[onset : onset + samples_after])
        before_mean = np.mean(emg_scaled[onset - samples_before : onset])
        if after_mean > after_a and before_mean < before_a:
            validated.append(onset)

    if not validated:
        return np.array([], dtype=int), emg_scaled

    min_samples = int(duration * srate)
    final = [validated[0]]
    for v in validated[1:]:
        if (v - final[-1]) > min_samples:
            final.append(v)

    return np.array(final, dtype=int), emg_scaled


def segment_data(eeg, emg, markers, window, onset, srate):
    win_samples = int(window * srate)
    onset_samples = int(onset * srate)

    eeg_epochs = []
    emg_epochs = []

    for m in markers:
        beg = m - onset_samples
        end = beg + win_samples
        if beg >= 0 and end <= len(eeg):
            eeg_epochs.append(eeg[beg:end])
            emg_epochs.append(emg[beg:end])

    if not eeg_epochs:
        return np.array([]), np.array([])

    return np.array(eeg_epochs), np.array(emg_epochs)


def epoch_and_average(eeg_epochs, emg_epochs, srate, baseline=0.1):
    if eeg_epochs.size == 0 or emg_epochs.size == 0:
        return None, None, None

    baseline_samples = int(baseline * srate)
    eeg_corrected = eeg_epochs.copy()
    for i in range(len(eeg_corrected)):
        eeg_corrected[i] -= np.mean(eeg_corrected[i, :baseline_samples])

    return np.mean(eeg_corrected, axis=0), np.mean(emg_epochs, axis=0), eeg_corrected


def reorder_and_split(eeg_epochs, n_groups=2, rng=None):
    if eeg_epochs is None or eeg_epochs.size == 0:
        return []

    rng = rng if rng is not None else np.random.default_rng()
    n_trials = eeg_epochs.shape[0]
    shuffled = eeg_epochs[rng.permutation(n_trials)]
    groups = np.array_split(shuffled, n_groups)
    return [(np.mean(g, axis=0), len(g)) for g in groups if len(g) > 0]
