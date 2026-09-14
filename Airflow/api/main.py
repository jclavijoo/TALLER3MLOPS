from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(
    title="API de Penguins",
    description="Predice la especie a partir de cuatro medidas.",
    version="1.0",
)

RUTA_MODELO = Path("/opt/airflow/models/penguins_model.pkl")


class PenguinEntrada(BaseModel):
    bill_length_mm: float = Field(..., gt=0, example=39.1)
    bill_depth_mm: float = Field(..., gt=0, example=18.7)
    flipper_length_mm: float = Field(..., gt=0, example=181)
    body_mass_g: float = Field(..., gt=0, example=3750)

    class Config:
        allow_inf_nan = False
        extra = "forbid"


@app.get("/")
def inicio():
    return {
        "mensaje": "API de Penguins activa",
        "archivo_modelo_disponible": RUTA_MODELO.is_file(),
    }


@app.post("/predict")
def predecir(datos: PenguinEntrada):
    # 1. Cargar el modelo guardado por t4.
    # Se lee en cada petición para recoger nuevos entrenamientos.
    try:
        paquete = joblib.load(RUTA_MODELO)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Todavía no hay modelo. Ejecuta el DAG completo.",
        )

    modelo = paquete["modelo"]
    columnas = paquete["columnas"]
    scaler = paquete["scaler"]

    # 2. Ordenar las medidas como durante el entrenamiento.
    entrada = datos.dict()

    valores = np.array(
        [[entrada[columna] for columna in columnas]],
        dtype=float,
    )

    # 3. Aplicar la misma normalización que se ajustó en t3.
    # MinMaxScaler transforma mediante: x * scale_ + min_.
    valores_normalizados = (
        valores * np.array(scaler["scale"], dtype=float)
        + np.array(scaler["offset"], dtype=float)
    )

    X = pd.DataFrame(
        valores_normalizados,
        columns=columnas,
    )

    # 4. Obtener la especie y las probabilidades del modelo.
    especie = modelo.predict(X)[0]
    probabilidades = modelo.predict_proba(X)[0]

    # 5. Responder al usuario.
    return {
        "especie_predicha": str(especie),
        "probabilidades": {
            str(clase): float(probabilidad)
            for clase, probabilidad in zip(
                modelo.classes_, probabilidades
            )
        },
        "medidas_normalizadas": {
            columna: float(valor)
            for columna, valor in zip(
                columnas, valores_normalizados[0]
            )
        },
    }