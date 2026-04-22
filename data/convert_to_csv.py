"""
Convierte Data.txt a CSV con nombres de columnas intuitivos.

Estructura del archivo (9 columnas, separadas por espacios):
  Col 0 : Tiempo (s)      — vector de tiempo, incrementos de 0.001 s → 1000 Hz
  Col 1 : EEG_1           — canal EEG frontal/central (amplitud ~1–2 µV)
  Col 2 : EEG_2           — segundo canal EEG
  Col 3 : EMG_1           — canal EMG muscular (amplitud ~0.02 µV, menor escala)
  Col 4 : EMG_2
  Col 5 : EMG_3
  Col 6 : EMG_4
  Col 7 : EMG_5
  Col 8 : EMG_6

Canales clave para el análisis (según app_edu.py):
  - EEG_1 y EEG_2 → señal cerebral para detectar Bereitschaftspotential (BP)
  - EMG_1..6      → actividad muscular; se usa uno (o varios) para detectar bursts
                     y definir marcadores de movimiento

Uso:
  python convert_to_csv.py
  python convert_to_csv.py --input Data.txt --output datos_eeg_emg.csv
"""

import argparse
import pandas as pd
from pathlib import Path

COLUMN_NAMES = [
    "Tiempo_s",
    "EEG_1",
    "EEG_2",
    "EMG_1",
    "EMG_2",
    "EMG_3",
    "EMG_4",
    "EMG_5",
    "EMG_6",
]

SRATE = 1000  # Hz


def convert(input_path: Path, output_path: Path) -> None:
    print(f"Leyendo {input_path} ...")
    df = pd.read_csv(input_path, sep=r"\s+", header=None, engine="python")

    if df.shape[1] != len(COLUMN_NAMES):
        print(
            f"Advertencia: el archivo tiene {df.shape[1]} columnas, "
            f"se esperaban {len(COLUMN_NAMES)}. "
            "Ajustando nombres automáticamente."
        )
        n = df.shape[1]
        auto_names = ["Tiempo_s"]
        # primeras 2 columnas no-tiempo son EEG
        eeg_count = min(2, n - 1)
        for i in range(1, eeg_count + 1):
            auto_names.append(f"EEG_{i}")
        # resto son EMG
        for i in range(1, n - eeg_count):
            auto_names.append(f"EMG_{i}")
        df.columns = auto_names
    else:
        df.columns = COLUMN_NAMES

    duration_s = df["Tiempo_s"].iloc[-1]
    print(f"  Muestras  : {len(df):,}")
    print(f"  Duración  : {duration_s:.1f} s  ({duration_s/60:.1f} min)")
    print(f"  Canales   : {list(df.columns[1:])}")
    print(f"  Frec.     : {SRATE} Hz")

    df.to_csv(output_path, index=False)
    size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"\nGuardado en {output_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convierte Data.txt a CSV.")
    parser.add_argument(
        "--input", default="Data.txt", help="Archivo de entrada (default: Data.txt)"
    )
    parser.add_argument(
        "--output",
        default="datos_eeg_emg.csv",
        help="Archivo CSV de salida (default: datos_eeg_emg.csv)",
    )
    args = parser.parse_args()

    base = Path(__file__).parent
    convert(base / args.input, base / args.output)
