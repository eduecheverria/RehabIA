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
            time_window = st.slider("Ventana de tiempo (seg)", min_value=1, max_value=min(60, int(len(eeg)/srate)), value=min(10, int(len(eeg)/srate)))
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
                line=dict(color='blue', width=1)
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
                    line=dict(color='lightblue', width=1),
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

        # Detección de marcadores EMG
        st.markdown('<div class="section-header"><h3>🎯 Detección de Marcadores EMG</h3></div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            threshold = st.number_input("Umbral", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
            time_after = st.number_input("Tiempo después (seg)", min_value=0.001, max_value=0.1, value=0.02, step=0.001)
            time_before = st.number_input("Tiempo antes (seg)", min_value=0.001, max_value=0.1, value=0.02, step=0.001)

        with col2:
            after_a = st.number_input("Amplitud después >", min_value=0.01, max_value=1.0, value=0.15, step=0.01)
            before_a = st.number_input("Amplitud antes <", min_value=0.01, max_value=1.0, value=0.20, step=0.01)
            duration = st.number_input("Duración burst (seg)", min_value=0.1, max_value=2.0, value=0.40, step=0.01)

        with col3:
            st.subheader("Opciones de visualización")
            show_scaled_emg = st.checkbox("Mostrar EMG escalado", value=True)
            marker_color = st.color_picker("Color de marcadores", "#FF0000")

        if st.button("🔍 Detectar Marcadores EMG"):
            markers = detect_markers(emg, srate, threshold, time_after, time_before, after_a, before_a, duration)

            st.success(f"✅ Se detectaron {len(markers)} marcadores")

            if len(markers) > 0:
                # Mostrar estadísticas de marcadores
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

            # Gráfico con marcadores
            fig_markers = go.Figure()

            if show_scaled_emg:
                emg_scaled = (emg - np.min(emg)) / (np.max(emg) - np.min(emg))
                fig_markers.add_trace(
                    go.Scatter(
                        x=ttime,
                        y=emg_scaled,
                        mode='lines',
                        name='EMG Escalado',
                        line=dict(color='blue', width=1)
                    )
                )

                # Línea de umbral
                fig_markers.add_hline(
                    y=threshold,
                    line_dash="dash",
                    line_color="green",
                    annotation_text=f"Umbral ({threshold})"
                )
            else:
                fig_markers.add_trace(
                    go.Scatter(
                        x=ttime,
                        y=emg,
                        mode='lines',
                        name='EMG',
                        line=dict(color='blue', width=1)
                    )
                )

            # Añadir marcadores
            for i, m in enumerate(markers):
                fig_markers.add_vline(
                    x=m/srate,
                    line_width=2,
                    line_color=marker_color,
                    annotation_text=f"M{i+1}",
                    annotation_position="top"
                )

            fig_markers.update_layout(
                title="Detección de Marcadores en Señal EMG",
                xaxis_title="Tiempo (s)",
                yaxis_title="Amplitud",
                height=500
            )

            st.plotly_chart(fig_markers, use_container_width=True)

            # Opción de exportar marcadores
            if len(markers) > 0:
                marker_df = pd.DataFrame({
                    'Marcador': range(1, len(markers) + 1),
                    'Muestra': markers,
                    'Tiempo (s)': markers / srate
                })

                if len(markers) > 1:
                    intervals_list = [0] + list(np.diff(markers / srate))
                    marker_df['Intervalo (s)'] = intervals_list

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
