import numpy as np
import pandas as pd
import streamlit as st
from scipy import signal
from scipy.stats import kurtosis, skew
import warnings
warnings.filterwarnings('ignore')

def com_sin(wtime, freq):
    """
    Genera señales sinusoidales complejas para análisis wavelet

    Parameters:
    -----------
    wtime : array
        Vector de tiempo
    freq : array
        Frecuencias a generar

    Returns:
    --------
    array : Señales sinusoidales complejas
    """
    f1 = np.repeat(freq, len(wtime))
    csin = np.exp(1j * 2 * np.pi * f1 * np.tile(wtime, len(freq)))
    return csin.reshape(len(freq), len(wtime))

def com_gau(wtime, s):
    """
    Genera ventanas gaussianas para análisis wavelet

    Parameters:
    -----------
    wtime : array
        Vector de tiempo
    s : array
        Parámetros de escala

    Returns:
    --------
    array : Ventanas gaussianas
    """
    s1 = np.repeat(s, len(wtime))
    gaus_win = np.exp(-(np.tile(wtime, len(s))**2) / (2 * np.tile(s, len(wtime))**2))
    return gaus_win.reshape(len(s), len(wtime))

def apply_filters(data, srate, highpass=None, lowpass=None, notch=None):
    """
    Aplica filtros digitales a la señal

    Parameters:
    -----------
    data : array
        Señal de entrada
    srate : float
        Frecuencia de muestreo
    highpass : float, optional
        Frecuencia de corte del filtro pasa-alto
    lowpass : float, optional
        Frecuencia de corte del filtro pasa-bajo
    notch : float, optional
        Frecuencia del filtro notch

    Returns:
    --------
    array : Señal filtrada
    """
    filtered_data = data.copy()
    nyquist = srate / 2

    # Filtro pasa-alto
    if highpass is not None and highpass > 0:
        sos_high = signal.butter(4, highpass/nyquist, btype='high', output='sos')
        filtered_data = signal.sosfilt(sos_high, filtered_data)

    # Filtro pasa-bajo
    if lowpass is not None and lowpass < nyquist:
        sos_low = signal.butter(4, lowpass/nyquist, btype='low', output='sos')
        filtered_data = signal.sosfilt(sos_low, filtered_data)

    # Filtro notch
    if notch is not None:
        # Crear filtro notch para eliminar interferencia de línea eléctrica
        Q = 30  # Factor de calidad
        b_notch, a_notch = signal.iirnotch(notch, Q, srate)
        filtered_data = signal.filtfilt(b_notch, a_notch, filtered_data)

    return filtered_data

def calculate_features(data, srate):
    """
    Calcula características estadísticas y espectrales de la señal

    Parameters:
    -----------
    data : array
        Señal de entrada
    srate : float
        Frecuencia de muestreo

    Returns:
    --------
    dict : Diccionario con características calculadas
    """
    features = {}

    # Características temporales
    features['mean'] = np.mean(data)
    features['std'] = np.std(data)
    features['var'] = np.var(data)
    features['rms'] = np.sqrt(np.mean(data**2))
    features['max'] = np.max(data)
    features['min'] = np.min(data)
    features['range'] = features['max'] - features['min']
    features['skewness'] = skew(data)
    features['kurtosis'] = kurtosis(data)

    # Características espectrales básicas
    freqs, psd = signal.welch(data, srate, nperseg=min(len(data)//4, 1024))
    features['total_power'] = np.trapz(psd, freqs)
    features['mean_freq'] = np.trapz(freqs * psd, freqs) / features['total_power']
    features['median_freq'] = freqs[np.cumsum(psd) >= features['total_power']/2][0]

    # Bandas de frecuencia (para EEG)
    if srate >= 100:  # Solo si la frecuencia de muestreo es suficiente
        delta_band = (0.5, 4)
        theta_band = (4, 8)
        alpha_band = (8, 13)
        beta_band = (13, 30)
        gamma_band = (30, 100)

        for band_name, (low, high) in [('delta', delta_band), ('theta', theta_band),
                                       ('alpha', alpha_band), ('beta', beta_band),
                                       ('gamma', gamma_band)]:
            mask = (freqs >= low) & (freqs <= high)
            if np.any(mask):
                features[f'{band_name}_power'] = np.trapz(psd[mask], freqs[mask])
                features[f'{band_name}_relative'] = features[f'{band_name}_power'] / features['total_power']

    return features

def spectral_analysis(data, srate, nperseg=None):
    """
    Realiza análisis espectral completo de la señal

    Parameters:
    -----------
    data : array
        Señal de entrada
    srate : float
        Frecuencia de muestreo
    nperseg : int, optional
        Longitud de segmento para Welch

    Returns:
    --------
    dict : Resultados del análisis espectral
    """
    if nperseg is None:
        nperseg = min(len(data)//4, 1024)

    # Densidad espectral de potencia usando método de Welch
    freqs, psd = signal.welch(data, srate, nperseg=nperseg)

    # Espectrograma para análisis tiempo-frecuencia
    f_spec, t_spec, Sxx = signal.spectrogram(data, srate, nperseg=nperseg//2)

    results = {
        'freqs': freqs,
        'psd': psd,
        'f_spec': f_spec,
        't_spec': t_spec,
        'spectrogram': Sxx,
        'total_power': np.trapz(psd, freqs),
        'peak_freq': freqs[np.argmax(psd)]
    }

    return results

def detect_markers(emg, srate, threshold, time_after, time_before, after_a, before_a, duration):
    """
    Detecta marcadores de activación muscular en señal EMG

    Parameters:
    -----------
    emg : array
        Señal EMG
    srate : float
        Frecuencia de muestreo
    threshold : float
        Umbral de detección
    time_after : float
        Tiempo después para validación (segundos)
    time_before : float
        Tiempo antes para validación (segundos)
    after_a : float
        Amplitud mínima después
    before_a : float
        Amplitud máxima antes
    duration : float
        Duración mínima entre activaciones (segundos)

    Returns:
    --------
    array : Índices de los marcadores detectados
    """
    # Rectificar
    emg_rect = np.abs(emg)
    # Normalizar señal EMG entre 0 y 1
    emg_scaled = (emg_rect - np.min(emg_rect)) / (np.max(emg_rect) - np.min(emg_rect))

    # Detectar cruces por umbral
    emg_binary = emg_scaled > threshold
    emg_diff = np.diff(emg_binary.astype(int))

    # Encontrar inicios de activación (transiciones de 0 a 1)
    onset_candidates = np.where(emg_diff == 1)[0]

    if len(onset_candidates) == 0:
        return np.array([])

    # Validar candidatos con criterios de amplitud
    validated_onsets = []

    for onset in onset_candidates:
        # Verificar que no estemos muy cerca del inicio del archivo
        if onset < int(time_before * srate) + 20:
            continue

        # Verificar que no estemos muy cerca del final
        if onset + int(time_after * srate) >= len(emg_scaled):
            continue

        # Calcular amplitudes promedio antes y después
        samples_after = int(time_after * srate)
        samples_before = int(time_before * srate)

        after_mean = np.mean(emg_scaled[onset:onset + samples_after])
        before_mean = np.mean(emg_scaled[onset - samples_before:onset])

        # Aplicar criterios de validación
        if after_mean > after_a and before_mean < before_a:
            validated_onsets.append(onset)

    if len(validated_onsets) == 0:
        return np.array([])

    # Eliminar detecciones muy cercanas (criterio de duración mínima)
    final_onsets = [validated_onsets[0]]  # Siempre incluir el primero
    min_samples = int(duration * srate)

    for i in range(1, len(validated_onsets)):
        if (validated_onsets[i] - final_onsets[-1]) > min_samples:
            final_onsets.append(validated_onsets[i])

    return np.array(final_onsets)

def detect_artifacts(data, srate, method='amplitude', **kwargs):
    """
    Detecta artefactos en la señal

    Parameters:
    -----------
    data : array
        Señal de entrada
    srate : float
        Frecuencia de muestreo
    method : str
        Método de detección ('amplitude', 'gradient', 'statistical')
    **kwargs : dict
        Parámetros adicionales según el método

    Returns:
    --------
    array : Índices de muestras con artefactos
    """
    artifacts = []

    if method == 'amplitude':
        # Detectar por amplitud extrema
        threshold = kwargs.get('amplitude_threshold', 5)  # múltiplos de desviación estándar
        std_data = np.std(data)
        mean_data = np.mean(data)
        artifacts = np.where(np.abs(data - mean_data) > threshold * std_data)[0]

    elif method == 'gradient':
        # Detectar por cambios abruptos
        threshold = kwargs.get('gradient_threshold', 10)
        gradient = np.abs(np.gradient(data))
        artifacts = np.where(gradient > threshold * np.std(gradient))[0]

    elif method == 'statistical':
        # Detectar usando criterios estadísticos
        window_size = kwargs.get('window_size', int(srate))  # ventana de 1 segundo
        hop_size = window_size // 2

        for i in range(0, len(data) - window_size, hop_size):
            window = data[i:i + window_size]

            # Calcular estadísticas de la ventana
            window_kurt = kurtosis(window)
            window_std = np.std(window)

            # Criterios para artefactos
            kurt_threshold = kwargs.get('kurtosis_threshold', 5)
            std_threshold = kwargs.get('std_threshold', 3 * np.std(data))

            if window_kurt > kurt_threshold or window_std > std_threshold:
                artifacts.extend(range(i, i + window_size))

    return np.unique(artifacts)

def epoch_data(data, markers, pre_time, post_time, srate):
    """
    Segmenta la señal en épocas basadas en marcadores

    Parameters:
    -----------
    data : array
        Señal continua
    markers : array
        Índices de marcadores
    pre_time : float
        Tiempo antes del marcador (segundos)
    post_time : float
        Tiempo después del marcador (segundos)
    srate : float
        Frecuencia de muestreo

    Returns:
    --------
    array : Épocas (trials x samples)
    """
    pre_samples = int(pre_time * srate)
    post_samples = int(post_time * srate)
    epoch_length = pre_samples + post_samples

    epochs = []

    for marker in markers:
        start_idx = marker - pre_samples
        end_idx = marker + post_samples

        # Verificar límites
        if start_idx >= 0 and end_idx <= len(data):
            epoch = data[start_idx:end_idx]
            epochs.append(epoch)

    return np.array(epochs) if epochs else np.array([])

def baseline_correction(epochs, baseline_start, baseline_end, srate):
    """
    Aplica corrección de línea base a las épocas

    Parameters:
    -----------
    epochs : array
        Épocas (trials x samples)
    baseline_start : float
        Inicio de línea base (segundos relativos al marcador)
    baseline_end : float
        Final de línea base (segundos relativos al marcador)
    srate : float
        Frecuencia de muestreo

    Returns:
    --------
    array : Épocas corregidas
    """
    if len(epochs) == 0:
        return epochs

    # Convertir tiempos a muestras (asumiendo que el marcador está en el centro)
    pre_samples = epochs.shape[1] // 2  # aproximación
    baseline_start_idx = int((baseline_start * srate) + pre_samples)
    baseline_end_idx = int((baseline_end * srate) + pre_samples)

    # Asegurar índices válidos
    baseline_start_idx = max(0, baseline_start_idx)
    baseline_end_idx = min(epochs.shape[1], baseline_end_idx)

    corrected_epochs = epochs.copy()

    for i in range(len(epochs)):
        baseline_mean = np.mean(epochs[i, baseline_start_idx:baseline_end_idx])
        corrected_epochs[i] = epochs[i] - baseline_mean

    return corrected_epochs

def calculate_erp(epochs):
    """
    Calcula el potencial relacionado con eventos (ERP) promedio

    Parameters:
    -----------
    epochs : array
        Épocas (trials x samples)

    Returns:
    --------
    dict : ERP promedio y estadísticas
    """
    if len(epochs) == 0:
        return {'mean': np.array([]), 'std': np.array([]), 'n_trials': 0}

    erp_mean = np.mean(epochs, axis=0)
    erp_std = np.std(epochs, axis=0)
    erp_sem = erp_std / np.sqrt(len(epochs))  # Error estándar de la media

    return {
        'mean': erp_mean,
        'std': erp_std,
        'sem': erp_sem,
        'n_trials': len(epochs)
    }

def interpolate_artifacts(data, artifact_indices, method='linear'):
    """
    Interpola segmentos con artefactos

    Parameters:
    -----------
    data : array
        Señal con artefactos
    artifact_indices : array
        Índices de muestras con artefactos
    method : str
        Método de interpolación ('linear', 'cubic', 'nearest')

    Returns:
    --------
    array : Señal interpolada
    """
    if len(artifact_indices) == 0:
        return data

    clean_data = data.copy()
    good_indices = np.setdiff1d(np.arange(len(data)), artifact_indices)

    if len(good_indices) < 2:
        return data  # No hay suficientes puntos buenos para interpolar

    # Usar interpolación scipy
    from scipy.interpolate import interp1d

    try:
        if method == 'linear':
            f = interp1d(good_indices, data[good_indices], kind='linear',
                        bounds_error=False, fill_value='extrapolate')
        elif method == 'cubic':
            f = interp1d(good_indices, data[good_indices], kind='cubic',
                        bounds_error=False, fill_value='extrapolate')
        else:  # nearest
            f = interp1d(good_indices, data[good_indices], kind='nearest',
                        bounds_error=False, fill_value='extrapolate')

        clean_data[artifact_indices] = f(artifact_indices)
    except:
        # Si falla la interpolación, usar interpolación lineal simple
        clean_data[artifact_indices] = np.interp(artifact_indices, good_indices, data[good_indices])

    return clean_data

def create_emg_timeseries_with_markers(emg_signal, markers, srate, include_filtered=True, include_scaled=True):
    """
    Crea un DataFrame con la serie de tiempo completa del EMG incluyendo las marcas
    
    Parameters:
    - emg_signal: señal EMG procesada
    - markers: array con los índices de los marcadores
    - srate: frecuencia de muestreo
    - include_filtered: incluir señal filtrada
    - include_scaled: incluir señal escalada/normalizada
    
    Returns:
    - DataFrame con columnas: Tiempo, EMG_Crudo, EMG_Filtrado, EMG_Escalado, Marcadores
    """
    
    # Vector de tiempo
    time_vector = np.arange(len(emg_signal)) / srate
    
    # Crear array de marcadores binario (0 o 1)
    marker_binary = np.zeros(len(emg_signal), dtype=int)
    
    # Marcar las posiciones donde hay marcadores
    valid_markers = markers[markers < len(emg_signal)]  # Asegurar que los marcadores estén en rango
    marker_binary[valid_markers] = 1
    
    # Crear DataFrame base
    timeseries_df = pd.DataFrame({
        'Tiempo_s': time_vector,
        'EMG_Filtrado': emg_signal,
        'Marcadores': marker_binary
    })
    
    # Agregar señal escalada si se solicita
    if include_scaled:
        emg_rect = np.abs(emg_signal)
        if np.max(emg_rect) - np.min(emg_rect) > 0:
            emg_scaled = (emg_rect - np.min(emg_rect)) / (np.max(emg_rect) - np.min(emg_rect))
        else:
            emg_scaled = emg_rect
        
        timeseries_df['EMG_Escalado'] = emg_scaled
    
    # Agregar información adicional de contexto de marcadores
    # Crear columnas que indican proximidad a marcadores
    marker_context = np.zeros(len(emg_signal), dtype=int)
    
    for marker in valid_markers:
        # Marcar región alrededor del marcador (±50 muestras como ejemplo)
        context_window = min(50, int(0.05 * srate))  # 50ms de contexto
        start_idx = max(0, marker - context_window)
        end_idx = min(len(emg_signal), marker + context_window + 1)
        marker_context[start_idx:end_idx] = 1
    
    timeseries_df['Contexto_Marcador'] = marker_context
    
    # Agregar columna con número de marcador (0 si no hay marcador, N si es el marcador N)
    marker_numbers = np.zeros(len(emg_signal), dtype=int)
    for i, marker in enumerate(valid_markers):
        marker_numbers[marker] = i + 1
    
    timeseries_df['Numero_Marcador'] = marker_numbers
    
    return timeseries_df

def create_synced_controls(param_name, display_name, min_val, max_val, step, format_str="%.3f", help_text=""):
                col_slider, col_input = st.columns([2, 1])
                
                with col_slider:
                    slider_val = st.slider(
                        f"{display_name} (Slider)",
                        min_value=min_val,
                        max_value=max_val,
                        value=st.session_state[f'{param_name}_value'],
                        step=step,
                        key=f"{param_name}_slider",
                        help=help_text
                    )
                
                with col_input:
                    input_val = st.number_input(
                        "Entrada directa",
                        min_value=min_val,
                        max_value=max_val,
                        value=st.session_state[f'{param_name}_value'],
                        step=step,
                        format=format_str,
                        key=f"{param_name}_input"
                    )
                
                # Sincronización
                if st.session_state[f'{param_name}_slider'] != st.session_state[f'{param_name}_value']:
                    st.session_state[f'{param_name}_value'] = st.session_state[f'{param_name}_slider']
                    st.rerun()
                elif st.session_state[f'{param_name}_input'] != st.session_state[f'{param_name}_value']:
                    st.session_state[f'{param_name}_value'] = st.session_state[f'{param_name}_input']
                    st.rerun()
                
                return st.session_state[f'{param_name}_value']

# processing.py

def segment_data(eeg, emg, markers, window, onset, srate):
    """
    Segmenta EEG y EMG alrededor de marcadores.
    
    Parameters:
    -----------
    eeg, emg : array
        Señales EEG y EMG
    markers : array
        Índices de los marcadores detectados
    window : float
        Duración de la ventana en segundos
    onset : float
        Tiempo (en segundos) que define la posición del "0" dentro de la ventana
    srate : float
        Frecuencia de muestreo

    Returns:
    --------
    eeg_epochs, emg_epochs : arrays (n_trials x samples)
    """
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

    if len(eeg_epochs) == 0:
        return np.array([]), np.array([])

    return np.array(eeg_epochs), np.array(emg_epochs)


def epoch_and_average(eeg_epochs, emg_epochs, srate, baseline=0.1):
    """
    Aplica baseline correction al EEG y calcula promedios.
    - EEG: se resta la media de los primeros baseline segundos
    - EMG: se promedia sin baseline correction
    """
    if eeg_epochs.size == 0 or emg_epochs.size == 0:
        return None, None

    eeg_corrected = eeg_epochs.copy()
    baseline_samples = int(baseline * srate)

    for i in range(len(eeg_corrected)):
        d1 = np.mean(eeg_corrected[i, :baseline_samples])
        eeg_corrected[i] -= d1

    eeg_avg = np.mean(eeg_corrected, axis=0)
    emg_avg = np.mean(emg_epochs, axis=0)

    return eeg_avg, emg_avg


def reorder_and_split(eeg_epochs, n_groups=2):
    """
    Reordena aleatoriamente los trials de EEG y calcula promedios en grupos.
    """
    if eeg_epochs is None or eeg_epochs.size == 0:
        return None

    n_trials = eeg_epochs.shape[0]
    shuffled = eeg_epochs[np.random.permutation(n_trials)]

    groups = np.array_split(shuffled, n_groups)
    averages = [np.mean(g, axis=0) for g in groups if len(g) > 0]

    return averages
