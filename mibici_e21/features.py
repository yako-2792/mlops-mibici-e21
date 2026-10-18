import pandas as pd
from pathlib import Path
from loguru import logger
from mibici_e21.config import PROCESSED_DATA_DIR

def main():
    input_path = PROCESSED_DATA_DIR / "mibici-trips-2024.parquet"
    output_features = PROCESSED_DATA_DIR / "features_2024.parquet"
    output_demanda = PROCESSED_DATA_DIR / "demanda_agrupada_2024.parquet"
    
    logger.info(f"Cargando archivo {input_path}")
    df = pd.read_parquet(input_path)
    
    logger.info("Transformando fechas y horas")
    df['inicio_del_viaje'] = pd.to_datetime(df['inicio_del_viaje'])
    df['fin_del_viaje'] = pd.to_datetime(df['fin_del_viaje'])
    
    # Variables de tiempo: hora del día, día de la semana y si es fin de semana
    df['hora'] = df['inicio_del_viaje'].dt.hour
    df['dia_semana'] = df['inicio_del_viaje'].dt.dayofweek
    # 0 a 4 es lunes a viernes, 5 y 6 son finde
    df['fin_de_semana'] = (df['dia_semana'] >= 5).astype(int)
    
    logger.info("Variables de edad y duración del viaje")
    df['duracion_minutos'] = (df['fin_del_viaje'] - df['inicio_del_viaje']).dt.total_seconds() / 60.0
    
    # Variable de edad
    df['año_de_nacimiento'] = pd.to_numeric(df['año_de_nacimiento'], errors='coerce')
    df['edad'] = 2024 - df['año_de_nacimiento']
    

    df.loc[(df['edad'] > 100) | (df['edad'] < 16), 'edad'] = None
    
    logger.info("dataset features a nivel viaje")
    df.to_parquet(output_features, index=False)
    
    logger.info("Agrupar datos para predecir demanda")
    df_demanda = df.groupby(['origen_id', 'dia_semana', 'hora']).size().reset_index(name='total_viajes')
    
    logger.info("dataset de demanda")
    df_demanda.to_parquet(output_demanda, index=False)
    
    logger.success("Nuevos datasets generados.")

if __name__ == "__main__":
    main()