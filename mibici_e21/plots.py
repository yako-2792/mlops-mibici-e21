import matplotlib
import pandas as pd

matplotlib.use("Agg")

from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns

from mibici_e21.config import FEATURE_STORE_SETTINGS, FIGURES_DIR
from mibici_e21.schemas.columns import GoldColumns, TripColumns


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Cargando datos para las gráficas/artefactos ")
    df_demanda = pd.read_parquet(
        FEATURE_STORE_SETTINGS.PROCESSED_DIR / "demand_aggregated_2024.parquet", engine="pyarrow"
    )
    df_features = pd.read_parquet(
        FEATURE_STORE_SETTINGS.PROCESSED_DIR / "features_2024.parquet", engine="pyarrow"
    )

    # Artefacto 1
    logger.info("Generando gráfica para P1")
    plt.figure(figsize=(10, 6))
    demanda_hora = (
        df_demanda.groupby(GoldColumns.HOUR)[GoldColumns.DEPARTURES_COUNT].mean().reset_index()
    )
    sns.barplot(
        data=demanda_hora, x=GoldColumns.HOUR, y=GoldColumns.DEPARTURES_COUNT, color="#6D94D3"
    )
    plt.title("Promedio de Viajes por Hora")
    plt.xlabel("Hora")
    plt.ylabel("Promedio de Viajes")
    plt.savefig(FIGURES_DIR / "demanda_hora.png")
    plt.close()

    # Artefacto 2
    logger.info("Generando gráfica para P2")
    plt.figure(figsize=(8, 5))
    dias = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"]
    # Contamos cuántos viajes hay por día
    viajes_dia = df_features[GoldColumns.DAY_OF_WEEK].value_counts().sort_index()
    sns.barplot(x=dias, y=viajes_dia.values, color="#7CDBD3")
    plt.title("Total de Viajes por Semana")
    plt.ylabel("Total de Viajes")
    plt.savefig(FIGURES_DIR / "total_dias.png")
    plt.close()

    # Artefacto 3
    logger.info("Generando gráfica para P3")
    plt.figure(figsize=(10, 6))
    # Sacamos las 10 estaciones con más tráfico
    top_estaciones = df_features[TripColumns.ORIGIN_ID].value_counts().head(10)
    sns.barplot(x=top_estaciones.index.astype(str), y=top_estaciones.values, color="#E2555A")
    plt.title("Top 10 Estaciones con Más Tráfico")
    plt.xlabel("ID de Estación")
    plt.ylabel("Total de Viajes")
    plt.savefig(FIGURES_DIR / "top_estaciones.png")
    plt.close()

    logger.success("Gráficas guardadas")


if __name__ == "__main__":
    main()
