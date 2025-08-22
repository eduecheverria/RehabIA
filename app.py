import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.signal as signal
from scipy import stats
import io
from processing import detect_markers, apply_filters, calculate_features, spectral_analysis, create_emg_timeseries_with_markers

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
    
    # Opción de renombrar columnas (del segundo archivo)
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

    with st.expander("🔍 Vista previa de los datos"):
        if rename_columns:
            st.subheader("Vista previa con encabezados personalizados")
        st.dataframe(df.head(10))

        # Estadísticas básicas
        st.subheader("Estadísticas básicas")
        st.dataframe(df.describe())

        # Configuración de canales y parámetros
    st.markdown('<div class="section-header"><h3>🎛 Configuración de Análisis</h3></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Configuración de Canales")
        
        # Opción mejorada del segundo archivo con selectbox
        if rename_columns and len(df.columns) > 1:
            eeg_channel_name = st.selectbox(
                "Canal EEG",
                options=df.columns,
                index=0
            )
            emg_channel_name = st.selectbox(
                "Canal EMG",
                options=df.columns,
                index=min(1, len(df.columns)-1)
            )
            # Obtener índices de las columnas seleccionadas
            eeg_channel = df.columns.get_loc(eeg_channel_name)
            emg_channel = df.columns.get_loc(emg_channel_name)
        else:
            # Configuración original de app_rafa.py
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

      
        


        fig_zoomed = go.Figure()
        fig_zoomed.add_trace(go.Scatter(x=ttime, y=eeg_filtered, mode='lines', name=eeg_channel_name, line=dict(color='blue')))
        fig_zoomed.add_trace(go.Scatter(x=ttime, y=emg_filtered, mode='lines', name=emg_channel_name, line=dict(color='green')))
        fig_zoomed.update_layout(
            height=450,
            title=f"Ventana de Análisis",
            xaxis_title="Tiempo (s)",
            yaxis_title="Amplitud (µV)",
            legend=dict(x=0, y=1.1, orientation='h')
        )
        st.plotly_chart(fig_zoomed, use_container_width=True)

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
        st.markdown('<div class="section-header"><h3> Burst</h3></div>', unsafe_allow_html=True)
        
        # Función para crear visualización interactiva del burst
        def create_burst_visualization(threshold, after_a, before_a, time_after, time_before, duration, emg_sample, srate_sample, window_start_time=0):
            """
            Crea una visualización interactiva del burst ideal y la señal EMG con ejes sincronizados
            Incluye rectángulos para mostrar las ventanas temporales de validación
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

            # Crear figura con subplots - LA CLAVE ES shared_xaxes=True
            fig_burst = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Señal EMG con Parámetros de Burst', 'Patrón de Burst Ideal con Ventanas de Validación'),
                vertical_spacing=0.12,
                row_heights=[0.6, 0.4],
                shared_xaxes=True,  # ESTO SINCRONIZA AUTOMÁTICAMENTE EL ZOOM Y PAN
                x_title="Tiempo (s)"
            )
            
            # Plot 1: Señal EMG real con líneas de referencia
            fig_burst.add_trace(
                go.Scatter(
                    x=time_window,
                    y=emg_scaled,
                    mode='lines',
                    name='EMG Normalizado',
                    line=dict(color='lightblue', width=1.5),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Líneas de umbral en el primer subplot
            fig_burst.add_hline(
                y=threshold, line_dash="solid", line_color="red", line_width=2,
                annotation_text=f"Umbral Principal ({threshold:.2f})",
                annotation_position="bottom right",
                row=1, col=1
            )
            
            fig_burst.add_hline(
                y=after_a, line_dash="dash", line_color="orange", line_width=2,
                annotation_text=f"Amplitud Después ({after_a:.2f})",
                annotation_position="top left",
                row=1, col=1
            )
            
            fig_burst.add_hline(
                y=before_a, line_dash="dot", line_color="green", line_width=2,
                annotation_text=f"Amplitud Antes ({before_a:.2f})",
                annotation_position="top right",
                row=1, col=1
            )
            
            # Plot 2: Patrón de burst ideal - ALINEADO CON LA VENTANA DE TIEMPO
            burst_time_total = min(time_before + duration + time_after, window_duration)
            burst_samples = int(burst_time_total * srate_sample)
            
            # Tiempo del burst alineado con la ventana actual
            burst_time_start = window_start_time + (window_duration - burst_time_total) / 2  # Centrar el burst
            burst_time = np.linspace(burst_time_start, burst_time_start + burst_time_total, burst_samples)
            
            # Crear patrón de burst idealizado
            burst_pattern = np.zeros(burst_samples)
            
            # Fase antes del burst (baja amplitud)
            before_samples = int(time_before * srate_sample)
            burst_pattern[:before_samples] = before_a * 0.8
            
            # Fase de burst (alta amplitud)
            burst_samples_active = int(duration * srate_sample)
            burst_start = before_samples
            burst_end = min(burst_start + burst_samples_active, len(burst_pattern))
            
            # Crear forma de burst (rampa ascendente, plateau, rampa descendente)
            ramp_samples = min(burst_samples_active // 4, int(0.05 * srate_sample))
            
            if ramp_samples > 0 and burst_end > burst_start:
                # Rampa ascendente
                end_ramp_up = min(burst_start + ramp_samples, burst_end)
                if end_ramp_up > burst_start:
                    burst_pattern[burst_start:end_ramp_up] = np.linspace(
                        before_a * 0.8, threshold * 1.5, end_ramp_up - burst_start
                    )
                
                # Plateau
                start_plateau = end_ramp_up
                end_plateau = max(burst_end - ramp_samples, start_plateau)
                if end_plateau > start_plateau:
                    burst_pattern[start_plateau:end_plateau] = threshold * 1.5
                
                # Rampa descendente
                if burst_end > end_plateau:
                    burst_pattern[end_plateau:burst_end] = np.linspace(
                        threshold * 1.5, after_a * 1.2, burst_end - end_plateau
                    )
            else:
                # Si no hay suficiente espacio para rampas, solo plateau
                burst_pattern[burst_start:burst_end] = threshold * 1.5
            
            # Fase después del burst
            if burst_end < len(burst_pattern):
                burst_pattern[burst_end:] = after_a * 1.2
            
            # NUEVO: Agregar rectángulos para las ventanas temporales
            
            # Definir posiciones temporales del burst
            burst_actual_start = burst_time_start + time_before
            burst_actual_end = burst_actual_start + duration
            
            # Ventana "ANTES" del burst - Rectángulo verde semitransparente
            rect_before_start = burst_actual_start - time_before
            rect_before_end = burst_actual_start
            
            if rect_before_start >= burst_time_start and rect_before_end <= burst_time_start + burst_time_total:
                fig_burst.add_shape(
                    type="rect",
                    x0=rect_before_start,
                    x1=rect_before_end,
                    y0=0,
                    y1=before_a,
                    fillcolor="rgba(46, 204, 113, 0.3)",  # Verde semitransparente
                    line=dict(color="rgba(46, 204, 113, 0.8)", width=2),
                    row=2, col=1
                )
                
                # Etiqueta para la ventana "antes"
                fig_burst.add_annotation(
                    x=(rect_before_start + rect_before_end) / 2,
                    y=before_a / 2,
                    text=f"ANTES<br>{time_before*1000:.0f}ms",
                    showarrow=False,
                    font=dict(color="white", size=10, family="Arial Black"),
                    bgcolor="rgba(46, 204, 113, 0.8)",
                    bordercolor="white",
                    borderwidth=1,
                    row=2, col=1
                )
            
            # Ventana "DESPUÉS" del burst - Rectángulo naranja semitransparente
            rect_after_start = burst_actual_end
            rect_after_end = burst_actual_end + time_after
            
            if rect_after_start >= burst_time_start and rect_after_end <= burst_time_start + burst_time_total:
                fig_burst.add_shape(
                    type="rect",
                    x0=rect_after_start,
                    x1=rect_after_end,
                    y0=0,
                    y1=after_a,
                    fillcolor="rgba(243, 156, 18, 0.3)",  # Naranja semitransparente
                    line=dict(color="rgba(243, 156, 18, 0.8)", width=2),
                    row=2, col=1
                )
                
                # Etiqueta para la ventana "después"
                fig_burst.add_annotation(
                    x=(rect_after_start + rect_after_end) / 2,
                    y=after_a / 2,
                    text=f"DESPUÉS<br>{time_after*1000:.0f}ms",
                    showarrow=False,
                    font=dict(color="white", size=10, family="Arial Black"),
                    bgcolor="rgba(243, 156, 18, 0.8)",
                    bordercolor="white",
                    borderwidth=1,
                    row=2, col=1
                )
            
            # NUEVO: Rectángulo para la duración del burst - Rojo semitransparente
            if burst_actual_start >= burst_time_start and burst_actual_end <= burst_time_start + burst_time_total:
                fig_burst.add_shape(
                    type="rect",
                    x0=burst_actual_start,
                    x1=burst_actual_end,
                    y0=threshold,
                    y1=threshold * 1.5,
                    fillcolor="rgba(231, 76, 60, 0.2)",  # Rojo semitransparente
                    line=dict(color="rgba(231, 76, 60, 0.8)", width=2, dash="dash"),
                    row=2, col=1
                )
                
                # Etiqueta para la duración del burst
                fig_burst.add_annotation(
                    x=(burst_actual_start + burst_actual_end) / 2,
                    y=threshold * 1.25,
                    text=f"BURST<br>{duration*1000:.0f}ms",
                    showarrow=False,
                    font=dict(color="white", size=10, family="Arial Black"),
                    bgcolor="rgba(231, 76, 60, 0.8)",
                    bordercolor="white",
                    borderwidth=1,
                    row=2, col=1
                )
            
            # Agregar el patrón de burst como antes
            fig_burst.add_trace(
                go.Scatter(
                    x=burst_time,
                    y=burst_pattern,
                    mode='lines',
                    name='Burst Ideal',
                    line=dict(color='purple', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(128, 0, 128, 0.1)',  # Reducir opacidad para que no tape los rectángulos
                    showlegend=True
                ),
                row=2, col=1
            )
            
            # Líneas de referencia en el burst ideal
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
            
            # Líneas verticales para marcar inicio y fin del burst
            if burst_actual_start >= burst_time_start and burst_actual_start <= burst_time_start + burst_time_total:
                fig_burst.add_vline(
                    x=burst_actual_start, line_dash="dashdot", line_color="gray",
                    annotation_text="Inicio Burst", 
                    annotation_position="top",
                    row=2, col=1
                )
            
            if burst_actual_end >= burst_time_start and burst_actual_end <= burst_time_start + burst_time_total:
                fig_burst.add_vline(
                    x=burst_actual_end, line_dash="dashdot", line_color="gray",
                    annotation_text="Fin Burst",
                    annotation_position="top", 
                    row=2, col=1
                )
            
            # Configuración del layout optimizada para sincronización
            fig_burst.update_layout(
                height=600,
                title_text="Configuración Visual de Parámetros de Burst - Zoom Sincronizado ⚡",
                showlegend=True,
                hovermode='x unified',  # Hover unificado en ambos subplots
                template='plotly_white',
                font=dict(size=12),
                
                # Configurar márgenes para mejor visualización
                margin=dict(l=50, r=50, t=80, b=50),
                
                # Rango inicial sincronizado
                xaxis=dict(
                    showticklabels=False,  # Ocultar labels del primer subplot
                    range=[window_start_time, window_start_time + window_duration],
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray'
                ),
                xaxis2=dict(
                    title="Tiempo (s)",
                    range=[window_start_time, window_start_time + window_duration],
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray'
                )
            )
            
            # Configurar los ejes Y con mejor formato
            fig_burst.update_yaxes(
                title_text="Amplitud Normalizada", 
                row=1, col=1,
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                range=[-0.05, 1.05]  # Rango fijo para mejor comparación
            )
            
            fig_burst.update_yaxes(
                title_text="Amplitud", 
                row=2, col=1,
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                range=[0, max(threshold * 1.8, after_a * 1.5, before_a * 1.5)]  # Rango dinámico basado en parámetros
            )
            
            return fig_burst
        
        def add_zoom_controls(fig_burst, window_start_time, window_duration):
            """
            Agrega controles de zoom predefinidos como botones
            """
            # Agregar botones de zoom personalizados
            fig_burst.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="left",
                        buttons=list([
                            dict(
                                args=[{"xaxis.range": [window_start_time, window_start_time + window_duration],
                                    "xaxis2.range": [window_start_time, window_start_time + window_duration]}],
                                label="Vista Completa",
                                method="relayout"
                            ),
                            dict(
                                args=[{"xaxis.range": [window_start_time, window_start_time + window_duration/2],
                                    "xaxis2.range": [window_start_time, window_start_time + window_duration/2]}],
                                label="Zoom 2x",
                                method="relayout"
                            ),
                            dict(
                                args=[{"xaxis.range": [window_start_time, window_start_time + window_duration/4],
                                    "xaxis2.range": [window_start_time, window_start_time + window_duration/4]}],
                                label="Zoom 4x",
                                method="relayout"
                            ),
                        ]),
                        pad={"r": 10, "t": 10},
                        showactive=True,
                        x=0.0,
                        xanchor="left",
                        y=1.15,
                        yanchor="top"
                    ),
                ]
            )
            return fig_burst
        
        #Parametros Burst
        col1, col2= st.columns(2)

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
                min_value=1, max_value=300, value=20, step=1,
                help="Ventana temporal después del onset para validación"
            ) / 1000  # Convertir a segundos
            
            time_before_slider = st.slider(
                "Tiempo Antes (ms)",
                min_value=1, max_value=300, value=20, step=1,
                help="Ventana temporal antes del onset para validación"
            ) / 1000  # Convertir a segundos
            
            duration_slider = st.slider(
                "Duración Mínima Burst (ms)",
                min_value=100, max_value=2000, value=400, step=10,
                help="Duración mínima entre bursts consecutivos"
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
                
                # Agregar controles de zoom si se desea
                add_zoom_controls(fig_interactive, window_start_time, 3.0)
            
            
            
            st.plotly_chart(fig_interactive, use_container_width=True, key=f"burst_viz_{window_start_time}")
            
            # Mostrar estadísticas predictivas mejoradas
            st.markdown("### Análisis en Tiempo Real")
            
            # Analizar solo la ventana actual
            window_samples = int(3.0 * srate)
            start_idx = int(window_start_time * srate)
            end_idx = min(len(emg_filtered), start_idx + window_samples)
            
            if end_idx > start_idx:
                emg_window_analysis = emg_filtered[start_idx:end_idx]
                emg_rect_analysis = np.abs(emg_window_analysis)
                
                if np.max(emg_rect_analysis) - np.min(emg_rect_analysis) > 0:
                    emg_scaled_analysis = (emg_rect_analysis - np.min(emg_rect_analysis)) / (np.max(emg_rect_analysis) - np.min(emg_rect_analysis))
                    
                    # Estadísticas de la ventana actual
                    above_threshold_window = np.sum(emg_scaled_analysis > threshold_slider)
                    percentage_above_window = (above_threshold_window / len(emg_scaled_analysis)) * 100
                    
                    # Detectar cruces de umbral en la ventana
                    emg_binary_window = emg_scaled_analysis > threshold_slider
                    emg_diff_window = np.diff(emg_binary_window.astype(int))
                    potential_onsets_window = len(np.where(emg_diff_window == 1)[0])
                    
                    # Calcular calidad promedio en la ventana
                    mean_amplitude = np.mean(emg_scaled_analysis)
                    std_amplitude = np.std(emg_scaled_analysis)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(
                            "% > Umbral (Ventana)", 
                            f"{percentage_above_window:.1f}%",
                            delta=f"vs {threshold_slider*100:.1f}% esperado"
                        )
                    with col2:
                        st.metric(
                            "Cruces Detectados", 
                            potential_onsets_window,
                            delta="en ventana actual"
                        )
                    with col3:
                        st.metric(
                            "Amplitud Media", 
                            f"{mean_amplitude:.3f}",
                            delta=f"±{std_amplitude:.3f}"
                        )
                    with col4:
                        # Calcular un índice de calidad
                        if percentage_above_window > 0:
                            quality_index = min(100, (potential_onsets_window / max(1, percentage_above_window/20)) * 100)
                        else:
                            quality_index = 0
                        
                        st.metric(
                            "Índice Calidad", 
                            f"{quality_index:.0f}%",
                            delta="configuración actual"
                        )
                    
                    # Alertas y recomendaciones
                    if percentage_above_window > 50:
                        st.warning("⚠️ **Umbral muy bajo:** Considera aumentar el umbral principal")
                    elif percentage_above_window < 5:
                        st.warning("⚠️ **Umbral muy alto:** Considera reducir el umbral principal")
                    elif 10 <= percentage_above_window <= 30:
                        st.success("✅ **Configuración óptima:** El umbral parece adecuado")
                    
                    # Recomendaciones dinámicas
                    if potential_onsets_window == 0:
                        st.info("💡 **Sugerencia:** Reduce el umbral o verifica los parámetros temporales")
                    elif potential_onsets_window > 10:
                        st.info("💡 **Sugerencia:** Considera aumentar la duración mínima del burst")
                

        except Exception as e:
            st.error(f"Error al procesar las señales: {str(e)}")
            st.error("Verifica que los índices de canal sean correctos y que el archivo contenga datos numéricos.")
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
                # Crear DataFrame con serie de tiempo
                timeseries_df = create_emg_timeseries_with_markers(
                    emg, markers, srate, 
                    include_filtered=True, 
                    include_scaled=True
                )          
                
            
                
                # Estadísticas de la serie exportada
                col1_stats, col2_stats, col3_stats = st.columns(3)
                with col1_stats:
                    st.metric("Total Muestras", f"{len(timeseries_df):,}")
                with col2_stats:
                    st.metric("Duración", f"{timeseries_df['Tiempo_s'].max():.2f}s")
                with col3_stats:
                    st.metric("Marcadores Incluidos", f"{timeseries_df['Marcadores'].sum()}")
                
                # Guardar en session_state para descarga
                st.session_state.timeseries_export = timeseries_df

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
                    mime="text/csv",
                    key = "dwbt2"
                )
                st.subheader("📥 Opciones de Exportación")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**📋 Exportar Tabla de Marcadores**")
                    csv_markers = marker_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Descargar Marcadores (CSV)",
                        data=csv_markers,
                        file_name=f"marcadores_{uploaded_file.name.split('.')[0]}.csv",
                        mime="text/csv",
                        key = "dwnbt3"
                    )

                with col2:
                    st.write("**📈 Exportar Serie de Tiempo Completa**")
                    
                    # Opciones de exportación para serie de tiempo
                    with st.expander("⚙️ Configurar Exportación de Serie de Tiempo"):
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
                        estimated_rows = len(emg) // downsample_factor
                        estimated_size_mb = estimated_rows * 6 * 8 / (1024 * 1024)  # 6 columnas, 8 bytes por float aprox
                        st.caption(f"📊 Filas estimadas: {estimated_rows:,} | Tamaño estimado: {estimated_size_mb:.1f} MB")
                    
                    
                            

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
                    - 🕐 Duración: {st.session_state.timeseries_export['Tiempo_s'].max():.2f} segundos
                    - 📊 Muestras: {len(st.session_state.timeseries_export):,}
                    - 🎯 Marcadores: {st.session_state.timeseries_export['Marcadores'].sum()}
                    - 📋 Columnas: {', '.join(st.session_state.timeseries_export.columns)}
                    """)

                # Opción adicional: Exportar solo segmentos alrededor de marcadores
                if len(markers) > 0:
                    st.write("**✂️ Exportar Segmentos de Marcadores**")
                    
                    with st.expander("⚙️ Configurar Exportación de Segmentos"):
                        segment_duration = st.slider(
                            "Duración del segmento alrededor de cada marcador (segundos)",
                            min_value=0.1, max_value=5.0, value=1.0, step=0.1
                        )
                        
                        segment_before = st.slider(
                            "Tiempo antes del marcador (%)",
                            min_value=10, max_value=90, value=50, step=5
                        ) / 100
                        
                        if st.button("📋 Generar Segmentos de Marcadores", key = "markers-segments"):
                            segments_data = []
                            
                            segment_samples = int(segment_duration * srate)
                            before_samples = int(segment_samples * segment_before)
                            after_samples = segment_samples - before_samples
                            
                            for i, marker in enumerate(markers):
                                start_idx = max(0, marker - before_samples)
                                end_idx = min(len(emg), marker + after_samples)
                                
                                if end_idx > start_idx:
                                    segment_time = np.arange(start_idx, end_idx) / srate
                                    segment_emg = emg[start_idx:end_idx]
                                    
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
                                
                                st.write(f"**📊 Vista Previa - Segmentos de {len(markers)} Marcadores:**")
                                st.dataframe(all_segments_df.head(20))
                                
                                segments_csv = all_segments_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Descargar Segmentos de Marcadores (CSV)",
                                    data=segments_csv,
                                    file_name=f"segmentos_marcadores_{uploaded_file.name.split('.')[0]}.csv",
                                    mime="text/csv", 
                                    key = "dwb1"
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
