import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger
from mibici_e21.config import PROCESSED_DATA_DIR, FIGURES_DIR

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Cargando datos para las gráficas/artefactos ")
    df_demanda = pd.read_parquet(PROCESSED_DATA_DIR / "demanda_agrupada_2024.parquet")
    df_features = pd.read_parquet(PROCESSED_DATA_DIR / "features_2024.parquet")
    

    #Artefacto 1 
    logger.info("Generando gráfica para P1")
    plt.figure(figsize=(10, 6))
    demanda_hora = df_demanda.groupby('hora')['total_viajes'].mean().reset_index()
    sns.barplot(data=demanda_hora, x='hora', y='total_viajes', color="#6D94D3")
    plt.title('Promedio de Viajes por Hora')
    plt.xlabel('Hora')
    plt.ylabel('Promedio de Viajes')
    plt.savefig(FIGURES_DIR / "demanda_hora.png")
    plt.close()
    

    #Artefacto 2 
    logger.info("Generando gráfica para P2")
    plt.figure(figsize=(8, 5))
    dias = ['Lun', 'Mar', 'Mie', 'Jue', 'Vie', 'Sab', 'Dom']
    # Contamos cuántos viajes hay por día 
    viajes_dia = df_features['dia_semana'].value_counts().sort_index()
    sns.barplot(x=dias, y=viajes_dia.values, color="#7CDBD3")
    plt.title('Total de Viajes por Semana')
    plt.ylabel('Total de Viajes')
    plt.savefig(FIGURES_DIR / "total_dias.png")
    plt.close()
    
    #Artefacto 3 
    logger.info("Generando gráfica para P3")
    plt.figure(figsize=(10, 6))
    # Sacamos las 10 estaciones con más tráfico 
    top_estaciones = df_features['origen_id'].value_counts().head(10)
    sns.barplot(x=top_estaciones.index.astype(str), y=top_estaciones.values, color="#E2555A")
    plt.title('Top 10 Estaciones con Más Tráfico')
    plt.xlabel('ID de Estación')
    plt.ylabel('Total de Viajes')
    plt.savefig(FIGURES_DIR / "top_estaciones.png")
    plt.close()
    
    logger.success("Gráficas guardadas")

if __name__ == "__main__":
    main()