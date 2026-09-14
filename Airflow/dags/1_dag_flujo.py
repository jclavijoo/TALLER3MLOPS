import os

from airflow import DAG
from datetime import datetime
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook

import csv
import time
import pandas as pd
import json
import platform
import joblib
import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def borrar_datos():
    conexion = MySqlHook(mysql_conn_id="mysql_penguins")

    # Crear Tabla penguins_raw si no existe
    conexion.run("""
        CREATE TABLE IF NOT EXISTS penguins_raw (
            rowid VARCHAR(20),
            species VARCHAR(50),
            island VARCHAR(50),
            bill_length_mm VARCHAR(30),
            bill_depth_mm VARCHAR(30),
            flipper_length_mm VARCHAR(30),
            body_mass_g VARCHAR(30),
            sex VARCHAR(20),
            year VARCHAR(10)
        )
    """)
    # Elimina los registros, conservando la estructura de la tabla.
    conexion.run("DELETE FROM penguins_raw")
    print("La tabla penguins_raw se ha vaciado y creado.")


def cargar_penguins():
    conexion = MySqlHook(mysql_conn_id="mysql_penguins")
    time.sleep(10)  # Espera 5 segundos antes de cargar los datos
    columnas = [
        "rowid",
        "species",
        "island",
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "sex",
        "year",
    ]

    ruta = "/opt/airflow/data/penguins.csv"

    with open(ruta, newline="", encoding="utf-8-sig") as archivo:
        lector = csv.DictReader(archivo)

        filas = [
            tuple(fila[columna] for columna in columnas)
            for fila in lector
        ]

    conexion.insert_rows(
        table="penguins_raw",
        rows=filas,
        target_fields=columnas,
        commit_every=0,
    )

    print(f"Se cargaron {len(filas)} filas en penguins_raw")


def preprocesar_datos():
    conexion = MySqlHook(mysql_conn_id="mysql_penguins")
    time.sleep(10)  # Espera 10 segundos antes de cargar los datos
    medidas = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]

    columnas = ["rowid", "species"] + medidas

    # 1. Leer los datos originales.
    registros = conexion.get_records("""
        SELECT
            rowid,
            species,
            bill_length_mm,
            bill_depth_mm,
            flipper_length_mm,
            body_mass_g
        FROM penguins_raw
        ORDER BY CAST(rowid AS UNSIGNED)
    """)

    datos = pd.DataFrame(registros, columns=columnas)
    cantidad_original = len(datos)

    # 2. Convertir las medidas y eliminar faltantes.
    for columna in medidas:
        datos[columna] = pd.to_numeric(
            datos[columna], errors="coerce"
        )

    datos["species"] = datos["species"].replace(
        {"NA": None, "": None}
    )

    datos = datos.dropna(
        subset=medidas + ["species"]
    ).copy()

    if datos.empty:
        raise ValueError("No quedaron datos válidos.")

    datos["rowid"] = pd.to_numeric(
        datos["rowid"], errors="raise"
    ).astype(int)

    # 3. Separar 80 % para entrenamiento y 20 % para prueba.
    entrenamiento, prueba = train_test_split(
        datos,
        test_size=0.2,
        random_state=42,
        stratify=datos["species"],
    )

    entrenamiento = entrenamiento.copy()
    prueba = prueba.copy()

    # 4. Ajustar el escalador SOLO con entrenamiento.
    escalador = MinMaxScaler()
    escalador.fit(entrenamiento[medidas])

    entrenamiento[medidas] = escalador.transform(
        entrenamiento[medidas]
    )
    prueba[medidas] = escalador.transform(prueba[medidas])

    # Guardar a qué conjunto pertenece cada registro.
    entrenamiento["conjunto"] = "train"
    prueba["conjunto"] = "test"

    procesados = pd.concat(
        [entrenamiento, prueba], ignore_index=True
    )

    columnas_salida = columnas + ["conjunto"]

    # 5. Crear la tabla de datos procesados.
    conexion.run("""
        CREATE TABLE IF NOT EXISTS penguins_processed (
            rowid INT PRIMARY KEY,
            species VARCHAR(50) NOT NULL,
            bill_length_mm DOUBLE NOT NULL,
            bill_depth_mm DOUBLE NOT NULL,
            flipper_length_mm DOUBLE NOT NULL,
            body_mass_g DOUBLE NOT NULL,
            conjunto VARCHAR(10) NOT NULL
        )
    """)

    # Adaptar la tabla si ya existe con la estructura anterior.
    tiene_conjunto = conexion.get_first("""
        SHOW COLUMNS FROM penguins_processed LIKE 'conjunto'
    """)

    if tiene_conjunto is None:
        conexion.run("""
            ALTER TABLE penguins_processed
            ADD COLUMN conjunto VARCHAR(10)
        """)

    # Guardar los parámetros para transformar datos nuevos en la API.
    conexion.run("""
        CREATE TABLE IF NOT EXISTS penguins_scaler (
            nombre VARCHAR(50) PRIMARY KEY,
            parametros JSON NOT NULL
        )
    """)

    parametros = json.dumps({
        "tipo": "MinMaxScaler",
        "columnas": medidas,
        "scale": escalador.scale_.tolist(),
        "offset": escalador.min_.tolist(),
        "data_min": escalador.data_min_.tolist(),
        "data_max": escalador.data_max_.tolist(),
    })

    filas = list(
        procesados[columnas_salida].itertuples(
            index=False, name=None
        )
    )

    # 6. Guardar datos y parámetros en una sola transacción.
    db = conexion.get_conn()
    cursor = None

    try:
        conexion.set_autocommit(db, False)
        cursor = db.cursor()

        cursor.execute("DELETE FROM penguins_processed")

        cursor.executemany("""
            INSERT INTO penguins_processed (
                rowid,
                species,
                bill_length_mm,
                bill_depth_mm,
                flipper_length_mm,
                body_mass_g,
                conjunto
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, filas)

        cursor.execute("""
            INSERT INTO penguins_scaler (nombre, parametros)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE parametros = VALUES(parametros)
        """, ("minmax", parametros))

        db.commit()

    except Exception:
        db.rollback()
        raise

    finally:
        if cursor is not None:
            cursor.close()
        db.close()

    print(f"Filas originales: {cantidad_original}")
    print(f"Filas válidas: {len(procesados)}")
    print(f"Entrenamiento: {len(entrenamiento)}")
    print(f"Prueba: {len(prueba)}")
    print("Datos normalizados y parámetros guardados en MySQL.")

def entrenar_modelo():
    conexion = MySqlHook(mysql_conn_id="mysql_penguins")

    medidas = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]

    # 1. Leer los datos normalizados desde MySQL.
    registros = conexion.get_records("""
        SELECT
            bill_length_mm,
            bill_depth_mm,
            flipper_length_mm,
            body_mass_g,
            species,
            conjunto
        FROM penguins_processed
        ORDER BY rowid
    """)

    datos = pd.DataFrame(
        registros,
        columns=medidas + ["species", "conjunto"],
    )

    # 2. Usar la separación creada en t3.
    entrenamiento = datos[datos["conjunto"] == "train"]
    prueba = datos[datos["conjunto"] == "test"]

    if entrenamiento.empty or prueba.empty:
        raise ValueError("Faltan datos de entrenamiento o prueba.")

    X_train = entrenamiento[medidas].astype(float)
    y_train = entrenamiento["species"]

    X_test = prueba[medidas].astype(float)
    y_test = prueba["species"]

    # 3. Entrenar el modelo.
    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_train, y_train)

    # 4. Evaluar usando el conjunto de prueba.
    predicciones = modelo.predict(X_test)

    metricas = {
        "accuracy": float(accuracy_score(y_test, predicciones)),
        "f1_macro": float(
            f1_score(
                y_test,
                predicciones,
                average="macro",
                zero_division=0,
            )
        ),
        "filas_train": len(entrenamiento),
        "filas_test": len(prueba),
    }

    print(f"Accuracy: {metricas['accuracy']:.4f}")
    print(f"F1 macro: {metricas['f1_macro']:.4f}")

    # 5. Recuperar los parámetros de normalización.
    registro_scaler = conexion.get_first("""
        SELECT parametros
        FROM penguins_scaler
        WHERE nombre = 'minmax'
    """)

    if registro_scaler is None:
        raise ValueError("No se encontraron los parámetros Min-Max.")

    parametros_scaler = json.loads(registro_scaler[0])

    if parametros_scaler["columnas"] != medidas:
        raise ValueError("Las columnas del escalador no coinciden.")

    # 6. Reunir lo necesario para hacer predicciones desde la API.
    paquete = {
        "modelo": modelo,
        "columnas": medidas,
        "scaler": parametros_scaler,
        "metricas": metricas,
        "versiones": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
    }

    # 7. Guardar en el volumen compartido.
    ruta_temporal = "/opt/airflow/models/penguins_model.tmp"
    ruta_modelo = "/opt/airflow/models/penguins_model.pkl"

    joblib.dump(paquete, ruta_temporal)

    # Reemplazar el archivo final cuando terminó de escribirse.
    os.replace(ruta_temporal, ruta_modelo)

    print(f"Modelo guardado correctamente en: {ruta_modelo}")



#EJECUCION
#=================

with DAG(
    dag_id="penguins_pipeline",
    description="Carga, preprocesamiento y entrenamiento de penguins",
    start_date=datetime(2026, 9, 12),
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
) as dag:

    t1 = PythonOperator(task_id="borrar_datos",
          python_callable=borrar_datos)

    t2 = PythonOperator(task_id="cargar_penguins",
          python_callable=cargar_penguins)

    t3 = PythonOperator(task_id="preprocesar_datos",
          python_callable=preprocesar_datos)

    t4 = PythonOperator(task_id="entrenar_modelo",
          python_callable=entrenar_modelo)

    t1 >> t2 >> t3 >> t4 