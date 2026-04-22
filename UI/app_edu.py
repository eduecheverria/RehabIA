import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.signal as signal
from scipy import stats
import io

# Asegúrate de que las siguientes funciones estén definidas en un archivo llamado 'processing.py'
# en el mismo directorio:
# def detect_markers(signal, srate, threshold, time_after, time_before, after_a, before_a, duration): ...
# def apply_filters(signal, srate, highpass_freq, lowpass_freq, notch_freq): ...
# def calculate_features(signal, srate): ...
# def spectral_analysis(signal, srate): ...
# def detect_bp(eeg_signal, emg_markers, srate, bp_amplitude_threshold, window_duration, onset_bp): ...

try:
    # Nota: Se importa detect_bp pero no se usa en esta versión del código.
    from processing import detect_markers, apply_filters, calculate_features, spectral_analysis
except ImportError:
    st.error("Error: No se pudo importar el archivo 'processing.py'. "
             "Asegúrate de que está en el mismo directorio y contiene las funciones necesarias.")
    st.stop()


# Configuración de la página
st.set_page_config(
    page_title="EEG/EMG Signal Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar la apariencia
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 20px 0 10px 0;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .burst-config-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    .parameter-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
        transition: all 0.3s ease;
    }
    .parameter-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .preset-button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        border: none;
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
        margin: 5px;
    }
    .preset-button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .quality-metric {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 5px;
    }
    .threshold-indicator {
        position: relative;
        padding: 10px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        border-left: 3px solid;
    }
    .threshold-high {
        border-left-color: #e74c3c;
        background-color: rgba(231, 76, 60, 0.1);
    }
    .threshold-medium {
        border-left-color: #f39c12;
        background-color: rgba(243, 156, 18, 0.1);
    }
    .threshold-low {
        border-left-color: #27ae60;
        background-color: rgba(39, 174, 96, 0.1);
    }
    .interactive-plot-container {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        background: white;
        margin: 15px 0;
    }
    .config-summary {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #e0e0e0;
    }
    .slider-container {
        background: rgba(255, 255, 255, 0.8);
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
    }
    .validation-window {
        opacity: 0.3;
        border: 2px dashed;
    }
    .validation-before {
        border-color: #27ae60;
        background-color: rgba(39, 174, 96, 0.1);
    }
    .validation-after {
        border-color: #f39c12;
        background-color: rgba(243, 156, 18, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">🧠 EEG & EMG Signal Analyzer</h1>', unsafe_allow_html=True)

# Sidebar para configuración
with st.sidebar:
    st.header("⚙ Configuración")

    # Sección de carga de archivos
    st.subheader("📂 Cargar Datos")
    uploaded_file = st.file_uploader(
        "Selecciona archivo de datos",
        type=["csv", "txt", "tsv"],
        help="Formatos soportados: CSV, TXT, TSV. Los datos deben estar en columnas separadas por espacios, tabs o comas."
    )

    # Configuración de separador
    separator = st.selectbox(
        "Separador de columnas",
        options=[None, ",", "\t", " ", ";"],
        format_func=lambda x: "Auto-detect" if x is None else f"'{x}'"
    )

# Función para cargar y procesar datos
@st.cache_data
def load_data(file, sep):
    """Carga y procesa los datos del archivo"""
    try:
        if sep is None:
            df = pd.read_csv(file, sep=None, engine='python')
        else:
            df = pd.read_csv(file, sep=sep)
        return df, None
    except Exception as e:
        return None, str(e)

# Main content
if uploaded_file is not None:
    # Cargar datos
    df, error = load_data(uploaded_file, separator)

    if error:
        st.error(f"Error al cargar el archivo: {error}")
        st.stop()

    # Renombrar columnas del DataFrame de forma permanente
    cols = ['Tiempo']
    if df.shape[1] > 1:
        cols.append('EEG-1')
    if df.shape[1] > 2:
        cols.append('EEG-2')
    if df.shape[1] > 3:
        eeg_cols = [f'EMG-{i}' for i in range(1, df.shape[1] - 2)]
        cols.extend(eeg_cols)
    
    df.columns = cols[:df.shape[1]]

    # Información del dataset
    st.markdown('<div class="section-header"><h3>📊 Información del Dataset</h3></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Filas", df.shape[0])
    with col2:
        st.metric("Columnas", df.shape[1])
    with col3:
        st.metric("Tamaño (MB)", f"{uploaded_file.size / (1024*1024):.2f}")
    with col4:
        st.metric("Tipo de archivo", uploaded_file.type)

    # Vista previa de los datos
    with st.expander("🔍 Vista previa de los datos"):
        st.subheader("Vista previa con encabezados personalizados")
        st.dataframe(df.head(10))
        st.subheader("Estadísticas básicas")
        st.dataframe(df.describe())

    # Configuración de canales y parámetros
    st.markdown('<div class="section-header"><h3>🎛 Configuración de Análisis</h3></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Configuración de Canales")
        eeg_channel_name = st.selectbox(
            "Canal EEG",
            options=df.columns,
            index=0
        )
        emg_channel_name = st.selectbox(
            "Canal EMG",
            options=df.columns,
            index=1
        )
        srate = st.number_input(
            "Frecuencia de muestreo (Hz)",
            min_value=1, max_value=10000, value=1000, step=1
        )

    with col2:
        st.subheader("Filtros de Señal")
        apply_filtering = st.checkbox("Aplicar filtros", value=True)
        if apply_filtering:
            highpass_freq = st.number_input("Filtro pasa-alto (Hz)", min_value=0.1, max_value=100.0, value=1.0, step=0.1)
            lowpass_freq = st.number_input("Filtro pasa-bajo (Hz)", min_value=1.0, max_value=500.0, value=100.0, step=1.0)
            notch_freq = st.number_input("Filtro notch (Hz)", min_value=40.0, max_value=70.0, value=50.0, step=1.0)

    # Extraer y procesar señales
    try:
        eeg_raw = df[eeg_channel_name].values
        emg_raw = df[emg_channel_name].values

        if apply_filtering:
            eeg_filtered = apply_filters(eeg_raw, srate, highpass_freq, lowpass_freq, notch_freq)
            emg_filtered = apply_filters(emg_raw, srate, highpass_freq, lowpass_freq, notch_freq)
        else:
            eeg_filtered = eeg_raw
            emg_filtered = emg_raw

        emg_filtered = emg_filtered - np.mean(emg_filtered)
        
        ttime = np.arange(0, len(eeg_filtered)/srate, 1/srate)[:len(eeg_filtered)]
        total_duration = ttime[-1]

        # Visualización de señales
        st.markdown('<div class="section-header"><h3>📈 Visualización de Señales</h3></div>', unsafe_allow_html=True)
        st.info("Utiliza el selector de rango de tiempo para enfocar un área específica.")

        col_eeg, col_emg = st.columns(2)
        with col_eeg:
            st.subheader(f"Señal {eeg_channel_name}")
            fig_eeg = go.Figure(data=go.Scatter(x=ttime, y=eeg_filtered, mode='lines', name=eeg_channel_name, line=dict(color='blue')))
            fig_eeg.update_layout(height=300, title=f"Señal {eeg_channel_name} Completa", xaxis_title="Tiempo (s)", yaxis_title="Amplitud (µV)")
            st.plotly_chart(fig_eeg, use_container_width=True)

        with col_emg:
            st.subheader(f"Señal {emg_channel_name}")
            fig_emg = go.Figure(data=go.Scatter(x=ttime, y=emg_filtered, mode='lines', name=emg_channel_name, line=dict(color='green')))
            fig_emg.update_layout(height=300, title=f"Señal {emg_channel_name} Completa", xaxis_title="Tiempo (s)", yaxis_title="Amplitud (µV)")
            st.plotly_chart(fig_emg, use_container_width=True)

        zoom_range = st.slider(
            "Selecciona el rango de tiempo (s) para hacer zoom:",
            min_value=0.0,
            max_value=total_duration,
            value=(0.0, min(total_duration, 10.0)),
            step=0.1
        )

        start_time, end_time = zoom_range
        start_idx = int(start_time * srate)
        end_idx = int(end_time * srate)
        
        ttime_zoomed = ttime[start_idx:end_idx]
        eeg_zoomed = eeg_filtered[start_idx:end_idx]
        emg_zoomed = emg_filtered[start_idx:end_idx]

        fig_zoomed = go.Figure()
        fig_zoomed.add_trace(go.Scatter(x=ttime_zoomed, y=eeg_zoomed, mode='lines', name=eeg_channel_name, line=dict(color='blue')))
        fig_zoomed.add_trace(go.Scatter(x=ttime_zoomed, y=emg_zoomed, mode='lines', name=emg_channel_name, line=dict(color='green')))
        fig_zoomed.update_layout(
            height=450,
            title=f"Ventana de Análisis: {start_time:.1f}s - {end_time:.1f}s",
            xaxis_title="Tiempo (s)",
            yaxis_title="Amplitud (µV)",
            legend=dict(x=0, y=1.1, orientation='h')
        )
        st.plotly_chart(fig_zoomed, use_container_width=True)
        
        if len(eeg_zoomed) > 0 and len(emg_zoomed) > 0:
            eeg_features_zoomed = calculate_features(eeg_zoomed, srate)
            emg_features_zoomed = calculate_features(emg_zoomed, srate)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"Características {eeg_channel_name} (Ventana)")
                st.metric("RMS", f"{eeg_features_zoomed['rms']:.3f}")
                st.metric("Media", f"{eeg_features_zoomed['mean']:.3f}")
                st.metric("Desv. Estándar", f"{eeg_features_zoomed['std']:.3f}")
            with col2:
                st.subheader(f"Características {emg_channel_name} (Ventana)")
                st.metric("RMS", f"{emg_features_zoomed['rms']:.3f}")
                st.metric("Media", f"{emg_features_zoomed['mean']:.3f}")
                st.metric("Desv. Estándar", f"{emg_features_zoomed['std']:.3f}")
        else:
            st.warning("La ventana de tiempo seleccionada no contiene datos.")

        # Análisis de características
        st.markdown('<div class="section-header"><h3>🔬 Análisis de Características</h3></div>', unsafe_allow_html=True)
        eeg_features = calculate_features(eeg_filtered, srate)
        emg_features = calculate_features(emg_filtered, srate)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"📊 Características {eeg_channel_name}")
            st.metric("RMS", f"{eeg_features['rms']:.3f}")
            st.metric("Media", f"{eeg_features['mean']:.3f}")
            st.metric("Desv. Estándar", f"{eeg_features['std']:.3f}")
            st.metric("Potencia Total", f"{eeg_features['total_power']:.3f}")
        with col2:
            st.subheader(f"📊 Características {emg_channel_name}")
            st.metric("RMS", f"{emg_features['rms']:.3f}")
            st.metric("Media", f"{emg_features['mean']:.3f}")
            st.metric("Desv. Estándar", f"{emg_features['std']:.3f}")
            st.metric("Potencia Total", f"{emg_features['total_power']:.3f}")

        # Análisis espectral
        st.markdown('<div class="section-header"><h3>🌊 Análisis Espectral</h3></div>', unsafe_allow_html=True)
        if st.button("🔍 Realizar Análisis Espectral"):
            eeg_spectrum = spectral_analysis(eeg_filtered, srate)
            emg_spectrum = spectral_analysis(emg_filtered, srate)
            fig_spectrum = make_subplots(
                rows=1, cols=2,
                subplot_titles=(f"Espectro {eeg_channel_name}", f"Espectro {emg_channel_name}")
            )
            fig_spectrum.add_trace(go.Scatter(x=eeg_spectrum['freqs'], y=eeg_spectrum['psd'], mode='lines', name=f"PSD {eeg_channel_name}"), row=1, col=1)
            fig_spectrum.add_trace(go.Scatter(x=emg_spectrum['freqs'], y=emg_spectrum['psd'], mode='lines', name=f"PSD {emg_channel_name}"), row=1, col=2)
            fig_spectrum.update_layout(height=400, title_text="Análisis de Densidad Espectral de Potencia")
            fig_spectrum.update_xaxes(title_text="Frecuencia (Hz)")
            fig_spectrum.update_yaxes(title_text="PSD (µV²/Hz)", type="log")
            st.plotly_chart(fig_spectrum, use_container_width=True)

        # Configuración interactiva de parámetros de burst
        st.markdown('<div class="section-header"><h3>⚙ Configuración Interactiva de Parámetros de Burst</h3></div>', unsafe_allow_html=True)
        def create_burst_visualization(threshold, after_a, before_a, time_after, time_before, duration, emg_sample, srate_sample, window_start_time=0):
            window_duration = 3.0
            samples_window = int(window_duration * srate_sample)
            start_idx = int(window_start_time * srate_sample)
            end_idx = min(len(emg_sample), start_idx + samples_window)
            if end_idx == len(emg_sample):
                start_idx = max(0, end_idx - samples_window)
            emg_window = emg_sample[start_idx:end_idx]
            emg_rect = np.abs(emg_window)
            if np.max(emg_rect) - np.min(emg_rect) == 0:
                emg_scaled = emg_rect
            else:
                emg_scaled = (emg_rect - np.min(emg_rect)) / (np.max(emg_rect) - np.min(emg_rect))
            time_window = (np.arange(len(emg_window)) / srate_sample) + window_start_time
            fig_burst = make_subplots(rows=2, cols=1, subplot_titles=('Señal EMG con Parámetros de Burst', 'Patrón de Burst Ideal'), vertical_spacing=0.15, row_heights=[0.6, 0.4])
            fig_burst.add_trace(go.Scatter(x=time_window, y=emg_scaled, mode='lines', name='EMG Normalizado', line=dict(color='lightblue', width=1.5)), row=1, col=1)
            fig_burst.add_hline(y=threshold, line_dash="solid", line_color="red", line_width=2, annotation_text=f"Umbral Principal ({threshold:.2f})", row=1, col=1)
            fig_burst.add_hline(y=after_a, line_dash="dash", line_color="orange", line_width=2, annotation_text=f"Amplitud Después ({after_a:.2f})", row=1, col=1)
            fig_burst.add_hline(y=before_a, line_dash="dot", line_color="green", line_width=2, annotation_text=f"Amplitud Antes ({before_a:.2f})", row=1, col=1)
            burst_time_total = time_before + duration + time_after
            burst_samples = int(burst_time_total * srate_sample)
            burst_time = np.linspace(0, burst_time_total, burst_samples)
            burst_pattern = np.zeros(burst_samples)
            before_samples = int(time_before * srate_sample)
            burst_pattern[:before_samples] = before_a * 0.8
            burst_samples_active = int(duration * srate_sample)
            burst_start = before_samples
            burst_end = burst_start + burst_samples_active
            ramp_samples = min(burst_samples_active // 4, int(0.05 * srate_sample))
            burst_pattern[burst_start:burst_start + ramp_samples] = np.linspace(before_a * 0.8, threshold * 1.5, ramp_samples)
            burst_pattern[burst_start + ramp_samples:burst_end - ramp_samples] = threshold * 1.5
            burst_pattern[burst_end - ramp_samples:burst_end] = np.linspace(threshold * 1.5, after_a * 1.2, ramp_samples)
            after_samples = int(time_after * srate_sample)
            if burst_end < len(burst_pattern):
                burst_pattern[burst_end:] = after_a * 1.2
            fig_burst.add_trace(go.Scatter(x=burst_time, y=burst_pattern, mode='lines', name='Burst Ideal', line=dict(color='purple', width=3), fill='tonexty'), row=2, col=1)
            fig_burst.add_hline(y=threshold, line_dash="solid", line_color="red", line_width=1, row=2, col=1)
            fig_burst.add_hline(y=after_a, line_dash="dash", line_color="orange", line_width=1, row=2, col=1)
            fig_burst.add_hline(y=before_a, line_dash="dot", line_color="green", line_width=1, row=2, col=1)
            fig_burst.add_vline(x=time_before, line_dash="dashdot", line_color="gray", annotation_text="Inicio Burst", row=2, col=1)
            fig_burst.add_vline(x=time_before + duration, line_dash="dashdot", line_color="gray", annotation_text="Fin Burst", row=2, col=1)
            fig_burst.update_layout(height=600, title_text="Configuración Visual de Parámetros de Burst", showlegend=True)
            fig_burst.update_xaxes(title_text="Tiempo (s)")
            fig_burst.update_yaxes(title_text="Amplitud Normalizada", row=1, col=1)
            fig_burst.update_yaxes(title_text="Amplitud", row=2, col=1)
            return fig_burst

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("🎯 Umbrales de Amplitud")
            threshold_slider = st.slider("Umbral Principal", min_value=0.01, max_value=0.8, value=0.2, step=0.01, help="Amplitud mínima para considerar inicio de burst")
            after_a_slider = st.slider("Amplitud Después", min_value=0.01, max_value=0.8, value=0.15, step=0.01, help="Amplitud mínima promedio después del onset")
            before_a_slider = st.slider("Amplitud Antes", min_value=0.01, max_value=0.8, value=0.20, step=0.01, help="Amplitud máxima promedio antes del onset")
        with col2:
            st.subheader("⏱ Parámetros Temporales")
            time_after_slider = st.slider("Tiempo Después (ms)", min_value=1, max_value=100, value=20, step=1, help="Ventana temporal después del onset para validación") / 1000
            time_before_slider = st.slider("Tiempo Antes (ms)", min_value=1, max_value=100, value=20, step=1, help="Ventana temporal antes del onset para validación") / 1000
            duration_slider = st.slider("Duración Mínima Burst (ms)", min_value=100, max_value=2000, value=400, step=10, help="Duración mínima entre bursts consecutivos") / 1000
        with col3:
            st.subheader("🎨 Opciones de Visualización")
            show_burst_pattern = st.checkbox("Mostrar Patrón de Burst", value=True)
            update_realtime = st.checkbox("Actualización en Tiempo Real", value=True)
            st.subheader("📍 Navegación Temporal")
            total_duration_emg = len(emg_filtered) / srate
            window_duration = 3.0
            max_start_time = max(0, total_duration_emg - window_duration)
            window_start_time = st.slider("Posición en el tiempo (s)", min_value=0.0, max_value=max_start_time, value=0.0, step=0.1, help=f"Desliza para navegar por los {total_duration_emg:.1f}s de datos")
            window_end_time = min(total_duration_emg, window_start_time + window_duration)
            st.caption(f"Mostrando: {window_start_time:.1f}s - {window_end_time:.1f}s")
            col_nav1, col_nav2, col_nav3 = st.columns(3)
            with col_nav1:
                if st.button("⏮ Inicio"):
                    st.session_state.window_start_time = 0.0
                    st.experimental_rerun()
            with col_nav2:
                if st.button("⏯ Centro"):
                    st.session_state.window_start_time = max(0, (total_duration_emg - window_duration) / 2)
                    st.experimental_rerun()
            with col_nav3:
                if st.button("⏭ Final"):
                    st.session_state.window_start_time = max_start_time
                    st.experimental_rerun()
            if 'window_start_time' in st.session_state:
                window_start_time = st.session_state.window_start_time
            if st.button("🔄 Actualizar Visualización"):
                update_realtime = True
        
        if show_burst_pattern and (update_realtime or st.button("Ver Configuración")):
            try:
                fig_interactive = create_burst_visualization(threshold_slider, after_a_slider, before_a_slider, time_after_slider, time_before_slider, duration_slider, emg_filtered, srate, window_start_time)
                st.plotly_chart(fig_interactive, use_container_width=True)
                st.subheader("📊 Estadísticas Predictivas")
                emg_rect = np.abs(emg_filtered)
                emg_scaled = (emg_rect - np.min(emg_rect)) / (np.max(emg_rect) - np.min(emg_rect))
                above_threshold = np.sum(emg_scaled > threshold_slider)
                percentage_above = (above_threshold / len(emg_scaled)) * 100
                emg_binary = emg_scaled > threshold_slider
                emg_diff = np.diff(emg_binary.astype(int))
                potential_onsets = len(np.where(emg_diff == 1)[0])
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("% Señal > Umbral", f"{percentage_above:.1f}%")
                with col2: st.metric("Cruces Potenciales", potential_onsets)
                with col3: st.metric("Sensibilidad", "Alta" if threshold_slider < 0.3 else "Media" if threshold_slider < 0.5 else "Baja")
                with col4: st.metric("Especificidad", "Baja" if before_a_slider > after_a_slider else "Media" if abs(before_a_slider - after_a_slider) < 0.1 else "Alta")
            except Exception as e:
                st.error(f"Error al crear la visualización: {str(e)}")

        configured_params = {
            'threshold': threshold_slider,
            'time_after': time_after_slider,
            'time_before': time_before_slider,
            'after_a': after_a_slider,
            'before_a': before_a_slider,
            'duration': duration_slider
        }

        st.success("✅ Parámetros configurados. Usa estos valores en la detección de marcadores.")
        st.markdown("---")

        st.markdown('<div class="section-header"><h3>🎯 Detección de Marcadores EMG</h3></div>', unsafe_allow_html=True)
        use_configured_params = st.checkbox("🔗 Usar parámetros de configuración interactiva", value=True, help="Si está marcado, usará los parámetros configurados arriba. Si no, permite configuración manual.")
        if use_configured_params and 'configured_params' in locals():
            threshold = configured_params['threshold']
            time_after = configured_params['time_after']
            time_before = configured_params['time_before']
            after_a = configured_params['after_a']
            before_a = configured_params['before_a']
            duration = configured_params['duration']
            st.info(f"""
            📋 *Usando parámetros configurados:*
            Umbral: {threshold:.3f} | Después: {after_a:.3f} | Antes: {before_a:.3f} |
            T.Después: {time_after*1000:.0f}ms | T.Antes: {time_before*1000:.0f}ms | Duración: {duration*1000:.0f}ms
            """)
        else:
            col1, col2, col3 = st.columns(3)
            with col1: threshold = st.number_input("Umbral", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
            with col2: after_a = st.number_input("Amplitud después >", min_value=0.01, max_value=1.0, value=0.15, step=0.01)
            with col3: before_a = st.number_input("Amplitud antes <", min_value=0.01, max_value=1.0, value=0.20, step=0.01)
            time_after = st.number_input("Tiempo después (seg)", min_value=0.001, max_value=0.1, value=0.02, step=0.001)
            time_before = st.number_input("Tiempo antes (seg)", min_value=0.001, max_value=0.1, value=0.02, step=0.001)
            duration = st.number_input("Duración burst (seg)", min_value=0.1, max_value=2.0, value=0.40, step=0.01)

        if st.button("🔍 Detectar Marcadores EMG"):
            with st.spinner("Detectando marcadores..."):
                emg_markers = detect_markers(emg_filtered, srate, threshold, time_after, time_before, after_a, before_a, duration)
                st.session_state.emg_markers = emg_markers
            st.success(f"✅ Se detectaron {len(emg_markers)} marcadores")

            if len(emg_markers) > 0:
                st.subheader("📋 Resumen de Estadísticas de Bursts EMG")
                marker_times = np.array(emg_markers) / srate
                intervals = np.diff(marker_times)
                summary_data = {
                    'Métrica': ['Total de Bursts Detectados', 'Tiempo Total de la Señal', 'Intervalo Promedio entre Bursts', 'Intervalo Mínimo entre Bursts', 'Intervalo Máximo entre Bursts'],
                    'Valor': [len(emg_markers), f"{len(emg_filtered)/srate:.2f}s", f"{np.mean(intervals):.3f}s" if len(intervals) > 0 else "N/A", f"{np.min(intervals):.3f}s" if len(intervals) > 0 else "N/A", f"{np.max(intervals):.3f}s" if len(intervals) > 0 else "N/A"]
                }
                summary_df = pd.DataFrame(summary_data)
                st.table(summary_df)

                st.subheader("📈 Análisis de Calidad de Detección")
                emg_rect = np.abs(emg_filtered)
                emg_scaled = (emg_rect - np.min(emg_rect)) / (np.max(emg_rect) - np.min(emg_rect))
                quality_metrics = []
                for i, marker in enumerate(emg_markers):
                    samples_after = int(time_after * srate)
                    samples_before = int(time_before * srate)
                    if marker - samples_before >= 0 and marker + samples_after < len(emg_scaled):
                        after_mean = np.mean(emg_scaled[marker:marker + samples_after])
                        before_mean = np.mean(emg_scaled[marker - samples_before:marker])
                        quality_score = (after_mean - before_mean) / threshold
                        quality_metrics.append({'marker': i + 1, 'time': marker / srate, 'after_mean': after_mean, 'before_mean': before_mean, 'quality_score': quality_score})
                if quality_metrics:
                    quality_df = pd.DataFrame(quality_metrics)
                    st.write("---")
                    st.subheader("Top 10 Detecciones de Mayor Calidad")
                    quality_df_sorted = quality_df.sort_values(by='quality_score', ascending=False)
                    st.dataframe(quality_df_sorted.head(10).round(4))
                    st.write("---")
                    st.subheader("Distribución de Scores de Calidad")
                    st.line_chart(quality_df['quality_score'])
            else:
                st.warning("⚠ No se detectaron marcadores con los parámetros actuales.")

    except Exception as e:
        st.error(f"Error inesperado en el procesamiento: {e}")

else:
    st.info("👆 Por favor, carga un archivo para empezar el análisis.")
