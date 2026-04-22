import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.signal as signal
from scipy import stats
import io
from processing import (
    detect_markers, apply_filters, calculate_features, spectral_analysis, 
    create_emg_timeseries_with_markers, create_synced_controls,
    segment_data, epoch_and_average, reorder_and_split
)
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
        background: white;
        color: black;
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
    .synchronized-plots {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .plot-instructions {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 10px 15px;
        border-radius: 8px;
        margin: 10px 0;
        font-size: 14px;
    }
    .plot-instructions strong {
        color: #ffd700;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">EEG & EMG Signal Analyzer</h1>', unsafe_allow_html=True)

# Sidebar para configuración
with st.sidebar:
    st.header("Configuración")

    # Sección de carga de archivos
    st.subheader("Cargar Datos")
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

    rename_columns = True

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

    if rename_columns:
        cols = ['Tiempo']
        if df.shape[1] > 1:
            cols.append('EEG-1')
        if df.shape[1] > 2:
            cols.append('EEG-2')
        if df.shape[1] > 3:
            eeg_cols = [f'EMG-{i}' for i in range(1, df.shape[1] - 2)]
            cols.extend(eeg_cols)

        df.columns = cols[:df.shape[1]]

    # Vista previa de los datos
    with st.expander("Vista previa de los datos"):
        if rename_columns:
            st.subheader("Vista previa con encabezados personalizados")
        st.dataframe(df.head(10))

        # Estadísticas básicas
        st.subheader("Estadísticas básicas")
        st.dataframe(df.describe())

    # PESTAÑAS
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Visualización de Señales",
    "Análisis Espectral",
    "Detección de Marcadores",
    "Average",
    "Reorder and Split"
    ])

    with tab1:
        # Configuración de canales y parámetros
        st.markdown('<div class="section-header"><h3>Configuración de Análisis</h3></div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Configuración de Canales")

            if rename_columns and len(df.columns) > 1:
                eeg_channel_name = st.selectbox(
                    "Canal EEG",
                    options=df.columns,
                    index=2
                )
                emg_channel_name = st.selectbox(
                    "Canal EMG",
                    options=df.columns,
                    index=3
                )
                eeg_channel = df.columns.get_loc(eeg_channel_name)
                emg_channel = df.columns.get_loc(emg_channel_name)
            else:
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
                min_value=1, max_value=10000, value=1000, step=1)
            apply_filtering = st.checkbox("Aplicar filtros", value=True)

        with col2:
            if apply_filtering:
                st.subheader("Filtros para EMG")
                emg_highpass = st.number_input("EMG Pasa-alto (Hz)", min_value=1.0, max_value=20.0, value=10.0, step=1.0, key="emg_hp")
                emg_lowpass = st.number_input("EMG Pasa-bajo (Hz)", min_value=100.0, max_value=500.0, value=250.0, step=1.0, key="emg_lp")
                emg_notch = st.number_input("EMG Filtro notch (Hz)", min_value=40.0, max_value=70.0, value=50.0, step=1.0, key="emg_notch")

        with col3:
            if apply_filtering:
                st.subheader("Filtros para EEG")
                eeg_highpass = st.number_input("EEG Pasa-alto (Hz)", min_value=0.01, max_value=5.0, value=0.01, step=0.01, key="eeg_hp")
                eeg_lowpass = st.number_input("EEG Pasa-bajo (Hz)", min_value=10.0, max_value=100.0, value=50.0, step=1.0, key="eeg_lp")
                eeg_notch = st.number_input("EEG Filtro notch (Hz)", min_value=40.0, max_value=70.0, value=50.0, step=1.0, key="eeg_notch")

        # Extraer y procesar señales
        try:
            eeg_raw = df[eeg_channel_name].values
            emg_raw = df[emg_channel_name].values

            if apply_filtering:
                emg_filtered = apply_filters(emg_raw, srate, emg_highpass, emg_lowpass, emg_notch)
                eeg_filtered = apply_filters(eeg_raw, srate, eeg_highpass, eeg_lowpass, eeg_notch)
            else:
                eeg_filtered = eeg_raw
                emg_filtered = emg_raw

            emg_filtered = emg_filtered - np.mean(emg_filtered)
            ttime = np.arange(0, len(eeg_filtered)/srate, 1/srate)[:len(eeg_filtered)]
            total_duration = ttime[-1]

            st.session_state['emg_filtered'] = emg_filtered
            st.session_state['eeg_filtered'] = eeg_filtered
            st.session_state['srate'] = srate
            st.session_state['ttime'] = ttime
            st.session_state['eeg_channel_name'] = eeg_channel_name
            st.session_state['emg_channel_name'] = emg_channel_name
            st.session_state['uploaded_file_name'] = uploaded_file.name


            # Visualización de señales (adaptada)
            st.markdown('<div class="section-header"><h3>Visualización de Señales</h3></div>', unsafe_allow_html=True)
            st.info("Haz zoom o pan en cualquiera de los tres subplots para sincronizar la vista. Arrastra el mouse sobre un área para hacer zoom.")

            fig_all = make_subplots(
                rows=3, cols=1,
                subplot_titles=(f"Señal {eeg_channel_name}", f"Señal {emg_channel_name}", "Ambas Señales"),
                vertical_spacing=0.15,
                shared_xaxes=True
            )

            fig_all.add_trace(go.Scatter(x=ttime, y=eeg_filtered, mode='lines', name=eeg_channel_name, line=dict(color='blue')), row=1, col=1)
            fig_all.update_yaxes(title_text="Amplitud (µV)", row=1, col=1)
            fig_all.add_trace(go.Scatter(x=ttime, y=emg_filtered, mode='lines', name=emg_channel_name, line=dict(color='green')), row=2, col=1)
            fig_all.update_yaxes(title_text="Amplitud (µV)", row=2, col=1)
            fig_all.add_trace(go.Scatter(x=ttime, y=eeg_filtered, mode='lines', name=eeg_channel_name, line=dict(color='blue')), row=3, col=1)
            fig_all.add_trace(go.Scatter(x=ttime, y=emg_filtered, mode='lines', name=emg_channel_name, line=dict(color='green')), row=3, col=1)
            fig_all.update_yaxes(title_text="Amplitud", row=3, col=1)
            fig_all.update_xaxes(title_text="Tiempo (s)", row=3, col=1)

            fig_all.update_layout(
                height=800,
                title_text="Visualización de Señales Sincronizadas",
                showlegend=False,
                hovermode='x unified',
            )
            st.plotly_chart(fig_all, use_container_width=True)

        except Exception as e:
            st.error(f"Error al procesar las señales: {e}")
            st.stop()

    with tab2:
        if 'emg_filtered' in st.session_state and 'eeg_filtered' in st.session_state:
            emg_filtered = st.session_state['emg_filtered']
            eeg_filtered = st.session_state['eeg_filtered']
            srate = st.session_state['srate']
            eeg_channel_name = st.session_state['eeg_channel_name']
            emg_channel_name = st.session_state['emg_channel_name']

            # Sección de análisis espectral
            st.markdown('<div class="section-header"><h3>Análisis Espectral</h3></div>', unsafe_allow_html=True)
            if st.button("Realizar Análisis Espectral"):
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
        else:
            st.warning("Por favor, carga un archivo y configura los canales en la pestaña 'Visualización de Señales' para realizar el análisis espectral.")

    with tab3:
        if 'emg_filtered' in st.session_state and 'srate' in st.session_state:
            emg_filtered = st.session_state['emg_filtered']
            srate = st.session_state['srate']
            ttime = st.session_state['ttime']
            uploaded_file_name = st.session_state['uploaded_file_name']

            # Función para crear visualización interactiva del burst
            def create_burst_visualization(threshold, after_a, before_a, time_after, time_before, duration, emg_sample, srate_sample, window_start_time=0):
                """
                Crea una visualización de la señal EMG con parámetros de burst
                Versión simplificada con solo el gráfico de la señal EMG
                """
                # Configuración de ventana deslizante
                window_duration = 20  # segundos de visualización
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

                # Crear figura simple con un solo subplot
                fig_burst = go.Figure()
                
                # Plot: Señal EMG real con líneas de referencia
                fig_burst.add_trace(
                    go.Scatter(
                        x=time_window,
                        y=emg_scaled,
                        mode='lines',
                        name='EMG Normalizado',
                        line=dict(color='lightblue', width=1.5),
                        showlegend=True
                    )
                )
                
                # Líneas de umbral
                fig_burst.add_hline(
                    y=threshold, line_dash="solid", line_color="red", line_width=2,
                    annotation_text=f"Umbral Principal ({threshold:.2f})",
                    annotation_position="bottom right"
                )
                
                fig_burst.add_hline(
                    y=after_a, line_dash="dash", line_color="orange", line_width=2,
                    annotation_text=f"Amplitud Después ({after_a:.2f})",
                    annotation_position="top left"
                )
                
                fig_burst.add_hline(
                    y=before_a, line_dash="dot", line_color="green", line_width=2,
                    annotation_text=f"Amplitud Antes ({before_a:.2f})",
                    annotation_position="top right"
                )
                fig_burst.update_layout(
                    height=400,
                    title_text="Señal EMG con Parámetros de Burst",
                    showlegend=True,
                    hovermode='x unified',
                    template='plotly_white',
                    font=dict(size=12),
                    
                    # Configurar márgenes para mejor visualización
                    margin=dict(l=50, r=50, t=60, b=50),
                    
                    # Configuración del eje X
                    xaxis=dict(
                        title="Tiempo (s)",
                        range=[window_start_time, window_start_time + window_duration],
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightgray'
                    ),
                    
                    # Configuración del eje Y
                    yaxis=dict(
                        title="Amplitud Normalizada",
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightgray',
                        range=[-0.05, 1.05]  # Rango fijo para mejor comparación
                    )
                )
                
                return fig_burst



            #Parametros Burst
            col1, col2= st.columns(2)

            with col1:
                # Inicializar todos los valores
                param_defaults = {
                    'threshold_value': 0.2,
                    'after_a_value': 0.15,
                    'before_a_value': 0.20,
                    'time_after_value': 20,
                    'time_before_value': 20,
                    'duration_value': 400
                }

                for key, default_val in param_defaults.items():
                    if key not in st.session_state:
                        st.session_state[key] = default_val

                # Función helper para crear controles sincronizados
                

                # USO:
                st.subheader("Umbrales de Amplitud")

                threshold_slider = create_synced_controls(
                    "threshold", "Umbral Principal", 
                    0.01, 0.8, 0.01, "%.3f",
                    "Amplitud mínima para considerar inicio de burst"
                )

                after_a_slider = create_synced_controls(
                    "after_a", "Amplitud Después", 
                    0.01, 0.8, 0.01, "%.3f",
                    "Amplitud mínima promedio después del onset"
                )

                before_a_slider = create_synced_controls(
                    "before_a", "Amplitud Antes", 
                    0.01, 0.8, 0.01, "%.3f", 
                    "Amplitud máxima promedio antes del onset"
                )

                
            
            with col2:
                st.subheader("Parámetros Temporales")

                time_after_slider = create_synced_controls(
                    "time_after", "Tiempo Después (ms)", 
                    1, 300, 1, "%d",
                    "Ventana temporal después del onset para validación"
                ) / 1000  # Convertir a segundos

                time_before_slider = create_synced_controls(
                    "time_before", "Tiempo Antes (ms)", 
                    1, 300, 1, "%d",
                    "Ventana temporal antes del onset para validación"  
                ) / 1000  # Convertir a segundos

                duration_slider = create_synced_controls(
                    "duration", "Duración Mínima Burst (ms)", 
                    100, 2000, 10, "%d",
                    "Duración mínima entre bursts consecutivos"
                ) / 1000  # Convertir a segundos
        
            
                
                
            
            
            # Crear y mostrar la visualización interactiva
            window_start_time = 0
            
            try:
                with st.spinner("Generando visualización sincronizada..."):
                    fig_interactive = create_burst_visualization(
                        threshold_slider, after_a_slider, before_a_slider,
                        time_after_slider, time_before_slider, duration_slider,
                        emg_filtered, srate, window_start_time
                    )


            except Exception as e:
                st.error(f"Error al procesar las señales: {str(e)}")
                st.error("Verifica que los índices de canal sean correctos y que el archivo contenga datos numéricos.")
            
            # Guardar parámetros configurados para usar en la detección
            configured_params = {
                'threshold': threshold_slider,
                'time_after': time_after_slider,
                'time_before': time_before_slider,
                'after_a': after_a_slider,
                'before_a': before_a_slider,
                'duration': duration_slider
            }
            # Usar parámetros configurados
            threshold = configured_params['threshold']
            time_after = configured_params['time_after']
            time_before = configured_params['time_before']
            after_a = configured_params['after_a']
            before_a = configured_params['before_a']
            duration = configured_params['duration']

            with st.spinner("Detectando marcadores..."):
                markers = detect_markers(emg_filtered, srate, threshold, time_after, time_before, after_a, before_a, duration)

            

            # Gráfico con marcadores (versión mejorada)
            # Crear figura una sola vez
            CHART_ID = "emg_markers_chart"  # Cambia esto por cada gráfico diferente

            # Inicializar estado de zoom ESPECÍFICO para este gráfico
            zoom_key = f'plot_ranges_{CHART_ID}'
            if zoom_key not in st.session_state:
                st.session_state[zoom_key] = {
                    'xaxis.range[0]': float(ttime[0]),
                    'xaxis.range[1]': float(ttime[-1]),
                    'yaxis.range[0]': None,
                    'yaxis.range[1]': None
                }

            if "xrange" not in st.session_state:
                st.session_state.xrange = None
            
            fig_markers = go.Figure()

            # Optimización 1: Reducir puntos de datos si es necesario
            def downsample_data(x, y, max_points=10000):
                """Reduce puntos si hay demasiados para acelerar rendering"""
                if len(x) > max_points:
                    step = len(x) // max_points
                    return x[::step], y[::step]
                return x, y

            # Aplicar downsampling si es necesario
            ttime_opt, emg_opt = downsample_data(ttime, emg_filtered)

            show_scaled_emg = 1
            if show_scaled_emg:
                emg_rect = abs(emg_filtered)
                emg_scaled = (emg_rect - np.min(emg_rect)) / (np.max(emg_rect) - np.min(emg_rect))
                
                # Usar datos optimizados
                ttime_plot, emg_plot = downsample_data(ttime, emg_scaled)
                
                fig_markers.add_trace(
                    go.Scatter(
                        x=ttime_plot,
                        y=emg_plot,
                        mode='lines',
                        name='EMG Escalado',
                        line=dict(color='lightblue', width=1),
                        hovertemplate='Tiempo: %{x:.3f}s<br>Amplitud: %{y:.3f}<extra></extra>'
                    )
                )
            else:
                ttime_plot, emg_plot = downsample_data(ttime, emg_filtered)
                fig_markers.add_trace(
                    go.Scatter(
                        x=ttime_plot,
                        y=emg_plot,
                        mode='lines',
                        name='EMG',
                        line=dict(color='white', width=1),
                        hovertemplate='Tiempo: %{x:.3f}s<br>Amplitud: %{y:.3f}<extra></extra>'
                    )
                )
            # Optimización 2: Añadir todas las líneas horizontales de una vez
            fig_markers.add_hline(y=threshold, line_dash="solid", line_color="red", line_width=2, annotation_text="")
            fig_markers.add_hline(y=after_a, line_dash="dash", line_color="orange", line_width=1, annotation_text="")
            fig_markers.add_hline(y=before_a, line_dash="dot", line_color="green", line_width=1, annotation_text="")

            # Optimización 3: Usar add_shape() en batch para todas las líneas verticales
            shapes = []

            for i, m in enumerate(markers):
                marker_time = m / srate
                
                # Marcador principal (rojo)
                shapes.append({
                    'type': 'line',
                    'x0': marker_time, 'x1': marker_time,
                    'y0': 0, 'y1': 1,
                    'yref': 'paper',
                    'line': {'color': 'red', 'width': 3}
                })
                
                # Línea ANTES (verde)
                time_before_line = marker_time - time_before
                if time_before_line >= 0:
                    shapes.append({
                        'type': 'line',
                        'x0': time_before_line, 'x1': time_before_line,
                        'y0': 0, 'y1': 1,
                        'yref': 'paper',
                        'line': {'color': 'green', 'width': 2, 'dash': 'dash'}
                    })
                
                # Línea DESPUÉS (naranja)
                time_after_line = marker_time + time_after
                if time_after_line <= ttime[-1]:
                    shapes.append({
                        'type': 'line',
                        'x0': time_after_line, 'x1': time_after_line,
                        'y0': 0, 'y1': 1,
                        'yref': 'paper',
                        'line': {'color': 'orange', 'width': 2, 'dash': 'dash'}
                    })
                
                # Línea de duración (morado)
                duration_line = marker_time + duration
                if duration_line <= ttime[-1]:
                    shapes.append({
                        'type': 'line',
                        'x0': duration_line, 'x1': duration_line,
                        'y0': 0, 'y1': 1,
                        'yref': 'paper',
                        'line': {'color': 'purple', 'width': 2, 'dash': 'dot'}
                    })

            # Optimización 4: Configurar layout una sola vez con todas las shapes

            if st.session_state.xrange:
                fig_markers.update_xaxes(range=st.session_state.xrange)

            fig_markers.update_layout(
            title="",
            xaxis_title="Tiempo (s)",
            yaxis_title="Amplitud",
            height=500,
            showlegend=True,
            shapes=shapes,
            hovermode='x unified',
            dragmode='pan',
            template= 'plotly_dark',
            paper_bgcolor= 'rgba(0,0,0,0)',  # Fondo transparente
            plot_bgcolor= '#0E1117',         # Fondo del gráfico (color Streamlit dark)
            # IMPORTANTE: uirevision ESPECÍFICO para este gráfico
            uirevision="constant"#f'zoom_state_{CHART_ID}',
            
        )

            

            

        
            st.plotly_chart(
                fig_markers, 
                use_container_width=True,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
                    'responsive': True
                }
                
            )

            

            st.subheader("Configurar rango X")

            x_min = st.text_input("x mínimo", value="0")
            x_max = st.text_input("x máximo", value="10")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Aplicar rango"):
                    try:
                        xmin = float(x_min)
                        xmax = float(x_max)
                        if xmin < xmax:
                            st.session_state.xrange = [xmin, xmax]
                            st.rerun()
                    except ValueError:
                        st.warning("Introduce valores numéricos válidos.")
            with col2:
                if st.button("Liberar rango"):
                    st.session_state.xrange = None
                    st.rerun()
        
            


            if st.button("🔍Exportar"):
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
                    
                    
                    st.subheader("Tabla de Marcadores")
                    st.dataframe(marker_df)

                    # Botón de descarga
                    csv = marker_df.to_csv(index=False)
                    st.download_button(
                        label="Descargar Marcadores (CSV)",
                        data=csv,
                        file_name=f"marcadores_{uploaded_file.name.split('.')[0]}.csv",
                        mime="text/csv",
                        key = "dwbt2"
                    )
                    st.subheader("📥 Opciones de Exportación")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Exportar Tabla de Marcadores**")
                        csv_markers = marker_df.to_csv(index=False)
                        st.download_button(
                            label="Descargar Marcadores (CSV)",
                            data=csv_markers,
                            file_name=f"marcadores_{uploaded_file.name.split('.')[0]}.csv",
                            mime="text/csv",
                            key = "dwnbt3"
                        )

                    with col2:
                        st.write("**📈 Exportar Serie de Tiempo Completa**")
                        
                        # Opciones de exportación para serie de tiempo
                        with st.expander("Configurar Exportación de Serie de Tiempo"):
                            export_raw = st.checkbox("Incluir señal cruda (sin filtrar)", value=False)
                            export_scaled = st.checkbox("Incluir señal escalada/normalizada", value=True)
                            
                            # Opción de submuestreo para archivos grandes
                            downsample_factor = st.selectbox(
                                "Factor de submuestreo (para reducir tamaño de archivo)",
                                options=[1, 2, 5, 10, 20],
                                index=0,
                                format_func=lambda x: f"Sin submuestreo" if x == 1 else f"1 de cada {x} muestras"
                            )
                            
                            # Mostrar información sobre el tamaño estimado del archivo
                            estimated_rows = len(emg_filtered) // downsample_factor
                            estimated_size_mb = estimated_rows * 6 * 8 / (1024 * 1024)  # 6 columnas, 8 bytes por float aprox
                            st.caption(f"Filas estimadas: {estimated_rows:,} | Tamaño estimado: {estimated_size_mb:.1f} MB")
                        
                        
                                

                    # Botón de descarga para serie de tiempo (aparece solo si se ha generado)
                    if 'timeseries_export' in st.session_state:
                        timeseries_csv = st.session_state.timeseries_export.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Descargar Serie de Tiempo EMG (CSV)",
                            data=timeseries_csv,
                            file_name=f"serie_tiempo_emg_{uploaded_file.name.split('.')[0]}.csv",
                            mime="text/csv",
                            help="Descarga la serie de tiempo completa con marcadores incluidos", 
                            key = "dwnbt4"
                        )
                        
                        # Información adicional sobre el archivo
                        st.success(f"""
                        ✅ **Serie de tiempo lista para descarga:**
                        - Duración: {st.session_state.timeseries_export['Tiempo_s'].max():.2f} segundos
                        - Muestras: {len(st.session_state.timeseries_export):,}
                        - Marcadores: {st.session_state.timeseries_export['Marcadores'].sum()}
                        - Columnas: {', '.join(st.session_state.timeseries_export.columns)}
                        """)

                    # Opción adicional: Exportar solo segmentos alrededor de marcadores
                    if len(markers) > 0:
                        st.write("**Exportar Segmentos de Marcadores**")
                        
                        with st.expander("Configurar Exportación de Segmentos"):
                            segment_duration = st.slider(
                                "Duración del segmento alrededor de cada marcador (segundos)",
                                min_value=0.1, max_value=5.0, value=1.0, step=0.1
                            )
                            
                            segment_before = st.slider(
                                "Tiempo antes del marcador (%)",
                                min_value=10, max_value=90, value=50, step=5
                            ) / 100
                            
                            if st.button("Generar Segmentos de Marcadores", key = "markers-segments"):
                                segments_data = []
                                
                                segment_samples = int(segment_duration * srate)
                                before_samples = int(segment_samples * segment_before)
                                after_samples = segment_samples - before_samples
                                
                                for i, marker in enumerate(markers):
                                    start_idx = max(0, marker - before_samples)
                                    end_idx = min(len(emg_filtered), marker + after_samples)
                                    
                                    if end_idx > start_idx:
                                        segment_time = np.arange(start_idx, end_idx) / srate
                                        segment_emg = emg_filtered[start_idx:end_idx]
                                        
                                        # Tiempo relativo al marcador
                                        relative_time = segment_time - (marker / srate)
                                        
                                        segment_df = pd.DataFrame({
                                            'Marcador_ID': i + 1,
                                            'Tiempo_Absoluto_s': segment_time,
                                            'Tiempo_Relativo_s': relative_time,
                                            'EMG_Filtrado': segment_emg,
                                            'Es_Marcador': (np.arange(start_idx, end_idx) == marker).astype(int)
                                        })
                                        
                                        if export_scaled:
                                            segment_rect = np.abs(segment_emg)
                                            if np.max(segment_rect) - np.min(segment_rect) > 0:
                                                segment_scaled = (segment_rect - np.min(segment_rect)) / (np.max(segment_rect) - np.min(segment_rect))
                                            else:
                                                segment_scaled = segment_rect
                                            segment_df['EMG_Escalado'] = segment_scaled
                                        
                                        segments_data.append(segment_df)
                                
                                if segments_data:
                                    all_segments_df = pd.concat(segments_data, ignore_index=True)
                                    
                                    st.write(f"**Vista Previa - Segmentos de {len(markers)} Marcadores:**")
                                    st.dataframe(all_segments_df.head(20))
                                    
                                    segments_csv = all_segments_df.to_csv(index=False)
                                    st.download_button(
                                        label="Descargar Segmentos de Marcadores (CSV)",
                                        data=segments_csv,
                                        file_name=f"segmentos_marcadores_{uploaded_file.name.split('.')[0]}.csv",
                                        mime="text/csv", 
                                        key = "dwb1"
                                    )

        # except Exception as e:
        #     st.error(f"Error al procesar las señales: {str(e)}")
        #     st.error("Verifica que los índices de canal sean correctos y que el archivo contenga datos numéricos.")    


    with tab4:
        if 'eeg_filtered' in st.session_state and 'emg_filtered' in st.session_state and 'srate' in st.session_state:
            eeg = st.session_state['eeg_filtered']
            emg = st.session_state['emg_filtered']
            srate = st.session_state['srate']

            # Detectar marcadores con parámetros configurados en Time Domain
            markers = detect_markers(
                emg, srate,
                st.session_state['threshold_value'],
                st.session_state['time_after_value']/1000,
                st.session_state['time_before_value']/1000,
                st.session_state['after_a_value'],
                st.session_state['before_a_value'],
                st.session_state['duration_value']/1000
            )

            st.subheader("Parámetros de Segmentación")
            window = st.number_input("Window (s)", 0.1, 5.0, 1.2, 0.1)
            onset = st.number_input("Onset (s)", 0.0, 2.0, 1.0, 0.1)
            baseline = st.number_input("Baseline correction (s)", 0.01, 1.0, 0.1, 0.01)

            if st.button("RUN Average"):
                eeg_epochs, emg_epochs = segment_data(eeg, emg, markers, window, onset, srate)
                eeg_avg, emg_avg = epoch_and_average(eeg_epochs, emg_epochs, srate, baseline)

                if eeg_avg is not None:
                    t_window = np.linspace(-onset, window-onset, eeg_epochs.shape[1])
                    fig = make_subplots(rows=2, cols=1, subplot_titles=("EEG Average", "EMG Average"), shared_xaxes=True)
                    fig.add_trace(go.Scatter(x=t_window, y=eeg_avg, mode="lines", name="EEG"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=t_window, y=emg_avg, mode="lines", name="EMG"), row=2, col=1)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No se generaron segmentos válidos.")

    # --- Reorder and Split ---
    with tab5:
        if 'eeg_filtered' in st.session_state and 'emg_filtered' in st.session_state and 'srate' in st.session_state:
            eeg = st.session_state['eeg_filtered']
            emg = st.session_state['emg_filtered']
            srate = st.session_state['srate']

            markers = detect_markers(
                emg, srate,
                st.session_state['threshold_value'],
                st.session_state['time_after_value']/1000,
                st.session_state['time_before_value']/1000,
                st.session_state['after_a_value'],
                st.session_state['before_a_value'],
                st.session_state['duration_value']/1000
            )

            st.subheader("Parámetros de Segmentación")
            window = st.number_input("Window (s)", 0.1, 5.0, 1.2, 0.1, key="re_window")
            onset = st.number_input("Onset (s)", 0.0, 2.0, 1.0, 0.1, key="re_onset")

            if st.button("RUN Reorder & Split"):
                eeg_epochs, _ = segment_data(eeg, emg, markers, window, onset, srate)
                groups = reorder_and_split(eeg_epochs, n_groups=2)

                if groups is not None:
                    t_window = np.linspace(-onset, window-onset, eeg_epochs.shape[1])
                    fig = make_subplots(rows=2, cols=1, subplot_titles=("Group 1", "Group 2"), shared_xaxes=True)
                    fig.add_trace(go.Scatter(x=t_window, y=groups[0], mode="lines", name="Group 1"), row=1, col=1)
                    if len(groups) > 1:
                        fig.add_trace(go.Scatter(x=t_window, y=groups[1], mode="lines", name="Group 2"), row=2, col=1)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No hay suficientes segmentos para dividir.")
    
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
