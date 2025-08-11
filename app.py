import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.signal as signal
from scipy import stats
import io
from processing import detect_markers, apply_filters, calculate_features, spectral_analysis

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
    st.header("⚙️ Configuración")

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
        st.dataframe(df.head(10))

        # Estadísticas básicas
        st.subheader("Estadísticas básicas")
        st.dataframe(df.describe())

    # Configuración de canales y parámetros
    st.markdown('<div class="section-header"><h3>🎛️ Configuración de Análisis</h3></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Configuración de Canales")
        max_col = df.shape[1] - 1
        eeg_channel = st.number_input(
            "Canal EEG (índice de columna)",
            min_value=0, max_value=max_col, value=0, step=1
        )
        emg_channel = st.number_input(
            "Canal EMG (índice de columna)",
            min_value=0, max_value=max_col, value=min(1, max_col), step=1
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
        eeg_raw = df.iloc[:, int(eeg_channel)].values
        emg_raw = df.iloc[:, int(emg_channel)].values

        # Aplicar filtros si está habilitado
        if apply_filtering:
            eeg = apply_filters(eeg_raw, srate, highpass_freq, lowpass_freq, notch_freq)
            emg = apply_filters(emg_raw, srate, highpass_freq, lowpass_freq, notch_freq)
        else:
            eeg = eeg_raw
            emg = emg_raw

        # Remover media del EMG
        emg = emg - np.mean(emg)

        # Vector de tiempo
        ttime = np.arange(0, len(eeg)/srate, 1/srate)[:len(eeg)]

        # Visualización de señales
        st.markdown('<div class="section-header"><h3>📈 Visualización de Señales</h3></div>', unsafe_allow_html=True)

        # Controles de visualización
        col1, col2, col3 = st.columns(3)
        with col1:
            show_raw = st.checkbox("Mostrar señales sin filtrar", value=False)
        with col2:
            time_window = st.slider("Ventana de tiempo (seg)", min_value=1, max_value=min(500, int(len(eeg)/srate)), value=min(10, int(len(eeg)/srate)))
        with col3:
            start_time = st.slider("Tiempo de inicio (seg)", min_value=0, max_value=max(0, int(len(eeg)/srate) - time_window), value=0)

        # Crear gráfico
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Señal EEG', 'Señal EMG'),
            vertical_spacing=0.1
        )

        # Filtrar datos para la ventana de tiempo
        start_idx = int(start_time * srate)
        end_idx = int((start_time + time_window) * srate)
        time_slice = slice(start_idx, end_idx)

        # EEG plot
        fig.add_trace(
            go.Scatter(
                x=ttime[time_slice],
                y=eeg[time_slice],
                mode='lines',
                name='EEG Filtrado',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )

        if show_raw:
            fig.add_trace(
                go.Scatter(
                    x=ttime[time_slice],
                    y=eeg_raw[time_slice],
                    mode='lines',
                    name='EEG Crudo',
                    line=dict(color='yellow', width=1),
                    opacity=0.6
                ),
                row=1, col=1
            )

        # EMG plot
        fig.add_trace(
            go.Scatter(
                x=ttime[time_slice],
                y=emg[time_slice],
                mode='lines',
                name='EMG Filtrado',
                line=dict(color='red', width=1)
            ),
            row=2, col=1
        )

        if show_raw:
            fig.add_trace(
                go.Scatter(
                    x=ttime[time_slice],
                    y=emg_raw[time_slice],
                    mode='lines',
                    name='EMG Crudo',
                    line=dict(color='pink', width=1),
                    opacity=0.6
                ),
                row=2, col=1
            )

        fig.update_layout(
            height=600,
            title_text="Análisis de Señales EEG/EMG",
            showlegend=True
        )

        fig.update_xaxes(title_text="Tiempo (s)")
        fig.update_yaxes(title_text="Amplitud (µV)", row=1, col=1)
        fig.update_yaxes(title_text="Amplitud (µV)", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # Análisis de características
        st.markdown('<div class="section-header"><h3>🔬 Análisis de Características</h3></div>', unsafe_allow_html=True)

        eeg_features = calculate_features(eeg, srate)
        emg_features = calculate_features(emg, srate)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Características EEG")
            st.metric("RMS", f"{eeg_features['rms']:.3f}")
            st.metric("Media", f"{eeg_features['mean']:.3f}")
            st.metric("Desv. Estándar", f"{eeg_features['std']:.3f}")
            st.metric("Potencia Total", f"{eeg_features['total_power']:.3f}")

        with col2:
            st.subheader("📊 Características EMG")
            st.metric("RMS", f"{emg_features['rms']:.3f}")
            st.metric("Media", f"{emg_features['mean']:.3f}")
            st.metric("Desv. Estándar", f"{emg_features['std']:.3f}")
            st.metric("Potencia Total", f"{emg_features['total_power']:.3f}")

        # Análisis espectral
        st.markdown('<div class="section-header"><h3>🌊 Análisis Espectral</h3></div>', unsafe_allow_html=True)

        if st.button("🔍 Realizar Análisis Espectral"):
            eeg_spectrum = spectral_analysis(eeg, srate)
            emg_spectrum = spectral_analysis(emg, srate)

            fig_spectrum = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Espectro EEG', 'Espectro EMG')
            )

            fig_spectrum.add_trace(
                go.Scatter(
                    x=eeg_spectrum['freqs'],
                    y=eeg_spectrum['psd'],
                    mode='lines',
                    name='PSD EEG'
                ),
                row=1, col=1
            )

            fig_spectrum.add_trace(
                go.Scatter(
                    x=emg_spectrum['freqs'],
                    y=emg_spectrum['psd'],
                    mode='lines',
                    name='PSD EMG'
                ),
                row=1, col=2
            )

            fig_spectrum.update_layout(height=400, title_text="Análisis de Densidad Espectral de Potencia")
            fig_spectrum.update_xaxes(title_text="Frecuencia (Hz)")
            fig_spectrum.update_yaxes(title_text="PSD (µV²/Hz)", type="log")

            st.plotly_chart(fig_spectrum, use_container_width=True)

        


        # Configuración interactiva de parámetros de burst
        st.markdown('<div class="section-header"><h3>⚙️ Configuración Interactiva de Parámetros de Burst</h3></div>', unsafe_allow_html=True)
        
        # Función para crear visualización interactiva del burst
        def create_burst_visualization(threshold, after_a, before_a, time_after, time_before, duration, emg_sample, srate_sample, window_start_time=0):
            """
            Crea una visualización interactiva del burst ideal y la señal EMG
            """
                # Configuración de ventana deslizante
            window_duration = 3.0  # 3 segundos de visualización (aumentado para mejor vista)
            samples_window = int(window_duration * srate_sample)
            
            # Calcular índices basados en el tiempo de inicio seleccionado
            start_idx = int(window_start_time * srate_sample)
            end_idx = min(len(emg_sample), start_idx + samples_window)
            
            # Ajustar start_idx si end_idx alcanza el final
            if end_idx == len(emg_sample):
                start_idx = max(0, end_idx - samples_window)
            
            emg_window = emg_sample[start_idx:end_idx]
            emg_rect = np.abs(emg_window)
            
            # Evitar división por cero
            if np.max(emg_rect) - np.min(emg_rect) == 0:
                emg_scaled = emg_rect
            else:
                emg_scaled = (emg_rect - np.min(emg_rect)) / (np.max(emg_rect) - np.min(emg_rect))
            
            # Vector de tiempo ajustado al tiempo real
            time_window = (np.arange(len(emg_window)) / srate_sample) + window_start_time
    
            
            # Crear figura con subplots
            fig_burst = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Señal EMG con Parámetros de Burst', 'Patrón de Burst Ideal'),
                vertical_spacing=0.15,
                row_heights=[0.6, 0.4]
            )
            
            # Plot 1: Señal EMG real con líneas de referencia
            fig_burst.add_trace(
                go.Scatter(
                    x=time_window,
                    y=emg_scaled,
                    mode='lines',
                    name='EMG Normalizado',
                    line=dict(color='lightblue', width=1.5)
                ),
                row=1, col=1
            )
            
            # Líneas de umbral
            fig_burst.add_hline(
                y=threshold, line_dash="solid", line_color="red", line_width=2,
                annotation_text=f"Umbral Principal ({threshold:.2f})",
                row=1, col=1
            )
            
            fig_burst.add_hline(
                y=after_a, line_dash="dash", line_color="orange", line_width=2,
                annotation_text=f"Amplitud Después ({after_a:.2f})",
                row=1, col=1
            )
            
            fig_burst.add_hline(
                y=before_a, line_dash="dot", line_color="green", line_width=2,
                annotation_text=f"Amplitud Antes ({before_a:.2f})",
                row=1, col=1
            )
            
            # Plot 2: Patrón de burst ideal
            burst_time_total = time_before + duration + time_after
            burst_samples = int(burst_time_total * srate_sample)
            burst_time = np.linspace(0, burst_time_total, burst_samples)
            
            # Crear patrón de burst idealizado
            burst_pattern = np.zeros(burst_samples)
            
            # Fase antes del burst (baja amplitud)
            before_samples = int(time_before * srate_sample)
            burst_pattern[:before_samples] = before_a * 0.8  # Ligeramente por debajo del umbral
            
            # Fase de burst (alta amplitud)
            burst_samples_active = int(duration * srate_sample)
            burst_start = before_samples
            burst_end = burst_start + burst_samples_active
            
            # Crear forma de burst (rampa ascendente, plateau, rampa descendente)
            ramp_samples = min(burst_samples_active // 4, int(0.05 * srate_sample))  # 50ms o 1/4 del burst
            
            # Rampa ascendente
            burst_pattern[burst_start:burst_start + ramp_samples] = np.linspace(
                before_a * 0.8, threshold * 1.5, ramp_samples
            )
            
            # Plateau
            burst_pattern[burst_start + ramp_samples:burst_end - ramp_samples] = threshold * 1.5
            
            # Rampa descendente
            burst_pattern[burst_end - ramp_samples:burst_end] = np.linspace(
                threshold * 1.5, after_a * 1.2, ramp_samples
            )
            
            # Fase después del burst
            after_samples = int(time_after * srate_sample)
            if burst_end < len(burst_pattern):
                burst_pattern[burst_end:] = after_a * 1.2  # Ligeramente por encima del umbral after_a
            
            fig_burst.add_trace(
                go.Scatter(
                    x=burst_time,
                    y=burst_pattern,
                    mode='lines',
                    name='Burst Ideal',
                    line=dict(color='purple', width=3),
                    fill='tonexty'
                ),
                row=2, col=1
            )
            
            # Añadir líneas de referencia en el burst ideal
            fig_burst.add_hline(
                y=threshold, line_dash="solid", line_color="red", line_width=1,
                row=2, col=1
            )
            
            fig_burst.add_hline(
                y=after_a, line_dash="dash", line_color="orange", line_width=1,
                row=2, col=1
            )
            
            fig_burst.add_hline(
                y=before_a, line_dash="dot", line_color="green", line_width=1,
                row=2, col=1
            )
            
            # Añadir anotaciones de tiempo en el burst ideal
            fig_burst.add_vline(
                x=time_before, line_dash="dashdot", line_color="gray",
                annotation_text="Inicio Burst", row=2, col=1
            )
            
            fig_burst.add_vline(
                x=time_before + duration, line_dash="dashdot", line_color="gray",
                annotation_text="Fin Burst", row=2, col=1
            )
            
            # Configuración del layout
            fig_burst.update_layout(
                height=600,
                title_text="Configuración Visual de Parámetros de Burst",
                showlegend=True
            )
            
            fig_burst.update_xaxes(title_text="Tiempo (s)")
            fig_burst.update_yaxes(title_text="Amplitud Normalizada", row=1, col=1)
            fig_burst.update_yaxes(title_text="Amplitud", row=2, col=1)
            
            return fig_burst
        
        # Crear columnas para los sliders
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🎯 Umbrales de Amplitud")
            threshold_slider = st.slider(
                "Umbral Principal",
                min_value=0.01, max_value=0.8, value=0.2, step=0.01,
                help="Amplitud mínima para considerar inicio de burst"
            )
            
            after_a_slider = st.slider(
                "Amplitud Después",
                min_value=0.01, max_value=0.8, value=0.15, step=0.01,
                help="Amplitud mínima promedio después del onset"
            )
            
            before_a_slider = st.slider(
                "Amplitud Antes",
                min_value=0.01, max_value=0.8, value=0.20, step=0.01,
                help="Amplitud máxima promedio antes del onset"
            )
        
        with col2:
            st.subheader("⏱️ Parámetros Temporales")
            time_after_slider = st.slider(
                "Tiempo Después (ms)",
                min_value=1, max_value=100, value=20, step=1,
                help="Ventana temporal después del onset para validación"
            ) / 1000  # Convertir a segundos
            
            time_before_slider = st.slider(
                "Tiempo Antes (ms)",
                min_value=1, max_value=100, value=20, step=1,
                help="Ventana temporal antes del onset para validación"
            ) / 1000  # Convertir a segundos
            
            duration_slider = st.slider(
                "Duración Mínima Burst (ms)",
                min_value=100, max_value=2000, value=400, step=10,
                help="Duración mínima entre bursts consecutivos"
            ) / 1000  # Convertir a segundos
        
        with col3:
            st.subheader("🎨 Opciones de Visualización")
            show_burst_pattern = st.checkbox("Mostrar Patrón de Burst", value=True)
            update_realtime = st.checkbox("Actualización en Tiempo Real", value=True)
            
            # NUEVO: Controles de ventana deslizante
            st.subheader("📍 Navegación Temporal")
            
            # Calcular duración total de la señal
            total_duration = len(emg) / srate
            window_duration = 3.0  # duración de la ventana de visualización
            max_start_time = max(0, total_duration - window_duration)
            
            # Slider para posición temporal
            window_start_time = st.slider(
                "Posición en el tiempo (s)",
                min_value=0.0,
                max_value=max_start_time,
                value=0.0,
                step=0.1,
                help=f"Desliza para navegar por los {total_duration:.1f}s de datos"
            )
            
            # Información de la ventana actual
            window_end_time = min(total_duration, window_start_time + window_duration)
            st.caption(f"Mostrando: {window_start_time:.1f}s - {window_end_time:.1f}s")
            
            # Botones de navegación rápida
            col_nav1, col_nav2, col_nav3 = st.columns(3)
            with col_nav1:
                if st.button("⏮️ Inicio"):
                    st.session_state.window_start_time = 0.0
                    st.experimental_rerun()
            with col_nav2:
                if st.button("⏯️ Centro"):
                    st.session_state.window_start_time = max(0, (total_duration - window_duration) / 2)
                    st.experimental_rerun()
            with col_nav3:
                if st.button("⏭️ Final"):
                    st.session_state.window_start_time = max_start_time
                    st.experimental_rerun()
            
            # Usar el valor del session_state si existe
            if 'window_start_time' in st.session_state:
                window_start_time = st.session_state.window_start_time
            
            if st.button("🔄 Actualizar Visualización"):
                update_realtime = True
        
        # Crear y mostrar la visualización interactiva
        if show_burst_pattern and (update_realtime or st.button("Ver Configuración")):
            try:
                fig_interactive = create_burst_visualization(
                    threshold_slider, after_a_slider, before_a_slider,
                    time_after_slider, time_before_slider, duration_slider,
                    emg, srate, window_start_time
                )
                st.plotly_chart(fig_interactive, use_container_width=True)
                
                # Mostrar estadísticas predictivas
                st.subheader("📊 Estadísticas Predictivas")
                
                # Simular detección con parámetros actuales
                emg_rect = np.abs(emg)
                emg_scaled = (emg_rect - np.min(emg_rect)) / (np.max(emg_rect) - np.min(emg_rect))
                
                # Contar cruces de umbral
                above_threshold = np.sum(emg_scaled > threshold_slider)
                percentage_above = (above_threshold / len(emg_scaled)) * 100
                
                # Estimación aproximada de detecciones
                emg_binary = emg_scaled > threshold_slider
                emg_diff = np.diff(emg_binary.astype(int))
                potential_onsets = len(np.where(emg_diff == 1)[0])
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("% Señal > Umbral", f"{percentage_above:.1f}%")
                with col2:
                    st.metric("Cruces Potenciales", potential_onsets)
                with col3:
                    st.metric("Sensibilidad", "Alta" if threshold_slider < 0.3 else "Media" if threshold_slider < 0.5 else "Baja")
                with col4:
                    st.metric("Especificidad", "Baja" if before_a_slider > after_a_slider else "Media" if abs(before_a_slider - after_a_slider) < 0.1 else "Alta")
                
            except Exception as e:
                st.error(f"Error al crear la visualización: {str(e)}")
        
        # Botones de configuración preestablecida
        st.subheader("🎛️ Configuraciones Preestablecidas")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔧 Sensible"):
                st.session_state.update({
                    'threshold': 0.15, 'after_a': 0.12, 'before_a': 0.18,
                    'time_after': 0.015, 'time_before': 0.015, 'duration': 0.3
                })
                st.experimental_rerun()
        
        with col2:
            if st.button("⚖️ Balanceado"):
                st.session_state.update({
                    'threshold': 0.25, 'after_a': 0.18, 'before_a': 0.22,
                    'time_after': 0.02, 'time_before': 0.02, 'duration': 0.4
                })
                st.experimental_rerun()
        
        with col3:
            if st.button("🎯 Conservador"):
                st.session_state.update({
                    'threshold': 0.35, 'after_a': 0.25, 'before_a': 0.3,
                    'time_after': 0.03, 'time_before': 0.03, 'duration': 0.5
                })
                st.experimental_rerun()
        
        with col4:
            if st.button("🔬 Personalizado"):
                st.info("Ajusta los sliders manualmente para configuración personalizada")
        
        # Guardar parámetros configurados para usar en la detección
        configured_params = {
            'threshold': threshold_slider,
            'time_after': time_after_slider,
            'time_before': time_before_slider,
            'after_a': after_a_slider,
            'before_a': before_a_slider,
            'duration': duration_slider
        }
        
        st.success("✅ Parámetros configurados. Usa estos valores en la detección de marcadores.")
        
        # Separador visual
        st.markdown("---")    

        # Reemplaza la sección "Detección de marcadores EMG" existente con esta versión modificada:

        # Detección de marcadores EMG
        st.markdown('<div class="section-header"><h3>🎯 Detección de Marcadores EMG</h3></div>', unsafe_allow_html=True)

        # Opción para usar parámetros configurados o manuales
        use_configured_params = st.checkbox(
            "🔗 Usar parámetros de configuración interactiva", 
            value=True,
            help="Si está marcado, usará los parámetros configurados arriba. Si no, permite configuración manual."
        )

        if use_configured_params and 'configured_params' in locals():
            # Usar parámetros configurados
            threshold = configured_params['threshold']
            time_after = configured_params['time_after']
            time_before = configured_params['time_before']
            after_a = configured_params['after_a']
            before_a = configured_params['before_a']
            duration = configured_params['duration']
            
            # Mostrar valores actuales en forma compacta
            st.info(f"""
            📋 **Usando parámetros configurados:**
            Umbral: {threshold:.3f} | Después: {after_a:.3f} | Antes: {before_a:.3f} | 
            T.Después: {time_after*1000:.0f}ms | T.Antes: {time_before*1000:.0f}ms | Duración: {duration*1000:.0f}ms
            """)
            
        else:
            # Configuración manual tradicional
            col1, col2, col3 = st.columns(3)

            with col1:
                threshold = st.number_input("Umbral", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
                time_after = st.number_input("Tiempo después (seg)", min_value=0.001, max_value=0.1, value=0.02, step=0.001)
                time_before = st.number_input("Tiempo antes (seg)", min_value=0.001, max_value=0.1, value=0.02, step=0.001)

            with col2:
                after_a = st.number_input("Amplitud después >", min_value=0.01, max_value=1.0, value=0.15, step=0.01)
                before_a = st.number_input("Amplitud antes <", min_value=0.01, max_value=1.0, value=0.20, step=0.01)
                duration = st.number_input("Duración burst (seg)", min_value=0.1, max_value=2.0, value=0.40, step=0.01)

        # Opciones de visualización (mantener la sección existente)
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Opciones de visualización")
                show_scaled_emg = st.checkbox("Mostrar EMG escalado", value=True)
                marker_color = st.color_picker("Color de marcadores", "#FF0000")
            
            with col2:
                st.subheader("Opciones avanzadas")
                show_validation_windows = st.checkbox("Mostrar ventanas de validación", value=False)
                highlight_rejected = st.checkbox("Destacar detecciones rechazadas", value=False)

        # Detección de marcadores
        if st.button("🔍 Detectar Marcadores EMG"):
            with st.spinner("Detectando marcadores..."):
                markers = detect_markers(emg, srate, threshold, time_after, time_before, after_a, before_a, duration)

            st.success(f"✅ Se detectaron {len(markers)} marcadores")

            if len(markers) > 0:
                # Mostrar estadísticas de marcadores (mantener sección existente)
                marker_times = markers / srate
                intervals = np.diff(marker_times)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Marcadores", len(markers))
                with col2:
                    st.metric("Intervalo Promedio", f"{np.mean(intervals):.3f}s" if len(intervals) > 0 else "N/A")
                with col3:
                    st.metric("Intervalo Mín", f"{np.min(intervals):.3f}s" if len(intervals) > 0 else "N/A")
                with col4:
                    st.metric("Intervalo Máx", f"{np.max(intervals):.3f}s" if len(intervals) > 0 else "N/A")

                # Análisis adicional de calidad de detección
                st.subheader("📈 Análisis de Calidad de Detección")
                
                # Calcular métricas de calidad
                emg_rect = np.abs(emg)
                emg_scaled = (emg_rect - np.min(emg_rect)) / (np.max(emg_rect) - np.min(emg_rect))
                
                quality_metrics = []
                for i, marker in enumerate(markers):
                    # Verificar ventanas de validación
                    samples_after = int(time_after * srate)
                    samples_before = int(time_before * srate)
                    
                    if marker - samples_before >= 0 and marker + samples_after < len(emg_scaled):
                        after_mean = np.mean(emg_scaled[marker:marker + samples_after])
                        before_mean = np.mean(emg_scaled[marker - samples_before:marker])
                        
                        quality_score = (after_mean - before_mean) / threshold
                        quality_metrics.append({
                            'marker': i + 1,
                            'time': marker / srate,
                            'after_mean': after_mean,
                            'before_mean': before_mean,
                            'quality_score': quality_score
                        })
                
                if quality_metrics:
                    quality_df = pd.DataFrame(quality_metrics)
                    avg_quality = np.mean(quality_df['quality_score'])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Calidad Promedio", f"{avg_quality:.2f}")
                    with col2:
                        st.metric("Mejor Detección", f"{np.max(quality_df['quality_score']):.2f}")
                    with col3:
                        st.metric("Peor Detección", f"{np.min(quality_df['quality_score']):.2f}")

            # Gráfico con marcadores (versión mejorada)
            fig_markers = go.Figure()

            if show_scaled_emg:
                emg_rect = abs(emg)
                emg_scaled = (emg_rect - np.min(emg_rect)) / (np.max(emg_rect) - np.min(emg_rect))
                fig_markers.add_trace(
                    go.Scatter(
                        x=ttime,
                        y=emg_scaled,
                        mode='lines',
                        name='EMG Escalado',
                        line=dict(color='lightblue', width=1)
                    )
                )

                # Líneas de umbral
                fig_markers.add_hline(
                    y=threshold, line_dash="solid", line_color="red", line_width=2,
                    annotation_text=f"Umbral ({threshold:.3f})"
                )
                
                fig_markers.add_hline(
                    y=after_a, line_dash="dash", line_color="orange", line_width=1,
                    annotation_text=f"Después ({after_a:.3f})"
                )
                
                fig_markers.add_hline(
                    y=before_a, line_dash="dot", line_color="green", line_width=1,
                    annotation_text=f"Antes ({before_a:.3f})"
                )
            else:
                fig_markers.add_trace(
                    go.Scatter(
                        x=ttime,
                        y=emg,
                        mode='lines',
                        name='EMG',
                        line=dict(color='lightblue', width=1)
                    )
                )

            # Añadir marcadores con información adicional
            for i, m in enumerate(markers):
                marker_time = m / srate
                
                # Marcador principal
                fig_markers.add_vline(
                    x=marker_time,
                    line_width=3,
                    line_color=marker_color,
                    annotation_text=f"M{i+1}",
                    annotation_position="top"
                )
                
                # Mostrar ventanas de validación si está habilitado
                if show_validation_windows and show_scaled_emg:
                    # Ventana antes
                    fig_markers.add_vrect(
                        x0=marker_time - time_before,
                        x1=marker_time,
                        fillcolor="green",
                        opacity=0.1,
                        layer="below",
                        line_width=0
                    )
                    
                    # Ventana después
                    fig_markers.add_vrect(
                        x0=marker_time,
                        x1=marker_time + time_after,
                        fillcolor="orange",
                        opacity=0.1,
                        layer="below",
                        line_width=0
                    )

            fig_markers.update_layout(
                title="Detección de Marcadores en Señal EMG",
                xaxis_title="Tiempo (s)",
                yaxis_title="Amplitud",
                height=500,
                showlegend=True
            )

            st.plotly_chart(fig_markers, use_container_width=True)

            # Opción de exportar marcadores (mantener sección existente)
            if len(markers) > 0:
                marker_df = pd.DataFrame({
                    'Marcador': range(1, len(markers) + 1),
                    'Muestra': markers,
                    'Tiempo (s)': markers / srate
                })

                if len(markers) > 1:
                    intervals_list = [0] + list(np.diff(markers / srate))
                    marker_df['Intervalo (s)'] = intervals_list
                
                # Añadir métricas de calidad si están disponibles
                if 'quality_df' in locals():
                    marker_df = marker_df.merge(
                        quality_df[['marker', 'quality_score', 'after_mean', 'before_mean']],
                        left_on='Marcador', right_on='marker', how='left'
                    ).drop('marker', axis=1)

                st.subheader("📋 Tabla de Marcadores")
                st.dataframe(marker_df)

                # Botón de descarga
                csv = marker_df.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar Marcadores (CSV)",
                    data=csv,
                    file_name=f"marcadores_{uploaded_file.name.split('.')[0]}.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"Error al procesar las señales: {str(e)}")
        st.error("Verifica que los índices de canal sean correctos y que el archivo contenga datos numéricos.")

else:
    # Página de inicio cuando no hay archivo cargado
    st.markdown("""
    ## 👋 Bienvenido al Analizador de Señales EEG/EMG

    Esta aplicación te permite:

    ### 📂 **Cargar y Visualizar Datos**
    - Soporta archivos CSV, TXT y TSV
    - Detección automática de separadores
    - Vista previa y estadísticas básicas

    ### 🔧 **Procesamiento de Señales**
    - Filtros pasa-alto, pasa-bajo y notch
    - Eliminación de artefactos
    - Escalado y normalización

    ### 📊 **Análisis Avanzado**
    - Cálculo de características en dominio del tiempo
    - Análisis espectral (PSD)
    - Detección automática de marcadores EMG

    ### 📈 **Visualización Interactiva**
    - Gráficos interactivos con Plotly
    - Zoom, pan y selección de ventanas temporales
    - Exportación de resultados

    ---

    **Para comenzar, carga un archivo de datos usando el panel lateral.**
    """)

    # Información adicional sobre formatos de archivo
    with st.expander("ℹ️ Información sobre formatos de archivo"):
        st.markdown("""
        ### Formatos Soportados

        - **CSV**: Valores separados por comas
        - **TXT**: Datos tabulares separados por espacios o tabs
        - **TSV**: Valores separados por tabs

        ### Estructura de Datos Esperada

        - Cada fila representa un punto temporal
        - Cada columna representa un canal de señal
        - Sin encabezados (o serán tratados como datos)
        - Datos numéricos únicamente

        ### Ejemplo de formato correcto:
        ```
        0.123  -0.456  0.789
        0.234  -0.567  0.890
        0.345  -0.678  0.901
        ...
        ```
        """)
