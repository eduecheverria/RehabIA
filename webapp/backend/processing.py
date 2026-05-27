import numpy as np
import pywt
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

_trapz = getattr(np, "trapezoid", None) or np.trapz


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


# ── Clustering: comparación BacAv vs K-Means ─────────────────────────────────

FEATURE_NAMES = [
    "RMS", "MAV", "STD", "Skewness", "Kurtosis", "ZCR", "WaveformLen",
    "Wavelet_D1", "Wavelet_D2", "Wavelet_D3", "Wavelet_D4", "BandPow_20-150",
]


def extract_window_features(emg, srate, win_s=0.05, hop_s=0.01):
    """Extrae 12 features por ventana corta, completamente vectorizado.

    Devuelve (feats [n_win, 12], times [n_win]) — porteado del notebook
    laboratory/exploracion_inicial.ipynb (sección 11) pero ~1000x más rápido.
    """
    nwin = int(win_s * srate)
    nhop = int(hop_s * srate)
    x = emg - np.mean(emg)

    wv = sliding_window_view(x, nwin)[::nhop]   # (n_win, nwin)
    n = wv.shape[1]
    times = (np.arange(wv.shape[0]) * nhop + nwin // 2) / srate

    rms = np.sqrt(np.mean(wv ** 2, axis=1))
    mav = np.mean(np.abs(wv), axis=1)
    std = np.std(wv, axis=1)
    sk = skew(wv, axis=1)
    ku = kurtosis(wv, axis=1)
    zcr = np.sum(np.diff(np.sign(wv), axis=1) != 0, axis=1) / n
    wl = np.sum(np.abs(np.diff(wv, axis=1)), axis=1)

    coeffs = pywt.wavedec(wv, "db2", level=4, axis=1)
    e_d1 = np.sum(coeffs[4] ** 2, axis=1) / n
    e_d2 = np.sum(coeffs[3] ** 2, axis=1) / n
    e_d3 = np.sum(coeffs[2] ** 2, axis=1) / n
    e_d4 = np.sum(coeffs[1] ** 2, axis=1) / n

    freqs = np.fft.rfftfreq(nwin, 1 / srate)
    hann = np.hanning(nwin)
    pw = np.abs(np.fft.rfft(wv * hann, axis=1)) ** 2
    band = (freqs >= 20) & (freqs <= 150)
    bandpow = _trapz(pw[:, band], freqs[band], axis=1)

    feats = np.column_stack([rms, mav, std, sk, ku, zcr, wl, e_d1, e_d2, e_d3, e_d4, bandpow])
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats, times


def cluster_kmeans_2(feats_2d, rms_col):
    """K-Means(2) sobre las 2 features seleccionadas (estandarizadas).

    `feats_2d`  : (n_win, 2) — las 2 features elegidas por el usuario.
    `rms_col`   : (n_win,)   — RMS de cada ventana, para etiquetar qué cluster
                  es "burst" (el de mayor RMS medio).

    Devuelve (labels, burst_cluster, centroids_2d).
    """
    scaler = StandardScaler()
    feats_sc = scaler.fit_transform(feats_2d)
    labels = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(feats_sc)

    rms_means = [rms_col[labels == i].mean() if np.any(labels == i) else -np.inf for i in range(2)]
    burst_cluster = int(np.argmax(rms_means))

    centroids = np.array([feats_2d[labels == i].mean(axis=0) for i in range(2)])
    return labels, burst_cluster, centroids


def kmeans_onsets(labels, burst_cluster, times, srate):
    """Onsets = transiciones reposo→burst en la máscara de cluster."""
    burst_mask = (labels == burst_cluster).astype(int)
    transitions = np.where(np.diff(burst_mask) == 1)[0]
    return (times[transitions] * srate).astype(int)


def compare_onsets(markers_a, markers_b, srate, tol_s=0.2):
    """Cuántos marcadores de A tienen al menos un marcador de B dentro de ±tol_s."""
    if len(markers_a) == 0 or len(markers_b) == 0:
        return 0
    b = np.asarray(markers_b)
    return int(sum(1 for m in markers_a if np.any(np.abs(b - m) / srate <= tol_s)))
