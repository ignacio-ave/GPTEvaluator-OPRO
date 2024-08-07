# Librerías de datos
import pandas as pd # type: ignore
import numpy as np # type: ignore

# Librerías de aprendizaje automático
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score, confusion_matrix # type: ignore

# Librerías de visualización
import matplotlib.pyplot as plt # type: ignore

# Librerías de sistema
import os
import subprocess
from typing import Optional, List, Tuple

# Librerías de fecha y hora
from datetime import datetime, timedelta

# Librerías de compresión
import zipfile

# Librerías de GPT
import openai # type: ignore
from GPTEvaluator import get_gpt_answers, chat_gpt_multiple
from openai_multi_client import OpenAIMultiClient # type: ignore

# Librerías de expresiones regulares
import re



def descargar_archivo(url_archivo: str, nombre_archivo: str):
    comando = f"wget -O {nombre_archivo} '{url_archivo}'"
    subprocess.run(comando, shell=True, check=True)

def crear_archivo(contenido: str, nombre_archivo: str):
    with open(nombre_archivo, 'w', encoding='utf-8') as archivo:
        archivo.write(contenido)

def leer_archivo(nombre_archivo: str):
    try:
        with open(nombre_archivo, 'r', encoding='utf-8') as archivo:
            contenido = archivo.read()
        return contenido
    except FileNotFoundError:
        return ""
    
#### 

def crear_archivos(
    campo_respuesta_correcta: str = "Contexto",
    columna_eval_real: str = "EvalGPT",
    url_archivo_correctas: str = "https://drive.google.com/uc?id=1CGUl5kgmQpigzktIx0feGM3EFUcfSjE-",
    nombre_archivo_correctas: str = "guidelines.xlsx",
    url_archivo_eval: str = "https://drive.google.com/uc?id=1NWNEYlRX_wAqiaIY2iKJB6cBEZWkM8Wc",
    nombre_archivo_eval: str = "real_evaluations.xlsx",
    tamano_muestra: Optional[int] = None,
    semilla_aleatoria: Optional[int] = None,
):
    ruta_datos = 'datos/'
    nombre_archivo_correctas = 'guidelines.xlsx'
    nombre_archivo_eval ='real_evaluations.xlsx'

    # Descargar el archivo con las respuestas correctas
    descargar_archivo(url_archivo_correctas, os.path.join(ruta_datos, nombre_archivo_correctas))

    # Descargar el archivo con las evaluaciones reales
    descargar_archivo(url_archivo_eval, os.path.join(ruta_datos, nombre_archivo_eval))

    # Cargar las respuestas correctas
    respuestas_correctas_df = pd.read_excel(os.path.join(ruta_datos, nombre_archivo_correctas))

    # Cargar las evaluaciones reales
    columnas_a_cargar = ['fullname', 'id_control', 'id_pregunta', 'Pregunta', 'Respuesta Estudiante', columna_eval_real]
    df = pd.read_excel(os.path.join(ruta_datos, nombre_archivo_eval), usecols=columnas_a_cargar)
    df.rename(columns={columna_eval_real: 'Eval'}, inplace=True)

    # Combinar los DataFrames
    respuestas_df = df.merge(respuestas_correctas_df[['Pregunta', campo_respuesta_correcta]], on='Pregunta', how='left')
    respuestas_df.rename(columns={campo_respuesta_correcta: 'Respuesta'}, inplace=True)

    # Mezclar y seleccionar una muestra
    if tamano_muestra and semilla_aleatoria:
        respuestas_df = respuestas_df.sample(n=tamano_muestra, random_state=semilla_aleatoria).reset_index(drop=True)
    elif tamano_muestra:
        respuestas_df = respuestas_df.sample(n=tamano_muestra).reset_index(drop=True)

    return respuestas_df


def verificar_archivos(ruta: str, archivos_necesarios: List[str]):
    archivos_verificados = []
    for archivo in archivos_necesarios:
        ruta_archivo = os.path.join(ruta, archivo)
        archivo_presente = os.path.isfile(ruta_archivo)
        archivos_verificados.append((archivo, archivo_presente, ruta_archivo))
    return archivos_verificados

####

def obtener_feedbacks(responses_df, template, api):
    api = OpenAIMultiClient(endpoint="chats", data_template={"model": "gpt-3.5-turbo", "temperature": 0.1, "n": 1, "timeout":10}, concurrency=50, wait_interval=1, max_retries=3, retry_max=10, retry_multiplier=1)

    texts = []
    ids = []
    for i, row in responses_df.iterrows():
        text = template.format(Pregunta=row['Pregunta'], RespuestaA=row['Respuesta Estudiante'], RespuestaC=row['Respuesta'])
        texts.append(text)
        ids.append((row['fullname'], row['id_control'], row['id_pregunta']))
    feedbacks_gpt = chat_gpt_multiple(api, texts)
    return feedbacks_gpt, ids


def obtener_scores(feedbacks_gpt, template, api):
    api = OpenAIMultiClient(endpoint="chats", data_template={"model": "gpt-3.5-turbo", "temperature": 0.1, "n": 1, "timeout":10}, concurrency=50, wait_interval=1, max_retries=3, retry_max=10, retry_multiplier=1)

    texts = []
    for feedback in feedbacks_gpt:
        text = template.format(Feedback=feedback[0])
        texts.append(text)
    gpt_scores = chat_gpt_multiple(api, texts)
    return gpt_scores


"""

def obtener_scores(feedbacks_gpt, template, api):
    texts = []
    for feedback in feedbacks_gpt:
        # Use the 'get' method with a default value of None
        respuesta_c = feedback.get('RespuestaC', None)
        respuesta_a = feedback.get('RespuestaA', None)
        pregunta = feedback.get('Pregunta', None)

        # Check for each possible combination of variables
        if respuesta_c is not None and respuesta_a is None and pregunta is None:
            text = template.format(Feedback=respuesta_c)
        elif respuesta_c is not None and respuesta_a is not None and pregunta is None:
            text = template.format(Feedback=respuesta_c, RespuestaA=respuesta_a)
        elif respuesta_c is not None and respuesta_a is None and pregunta is not None:
            text = template.format(Feedback=respuesta_c, Pregunta=pregunta)
        elif respuesta_c is not None and respuesta_a is not None and pregunta is not None:
            text = template.format(Feedback=respuesta_c, RespuestaA=respuesta_a, Pregunta=pregunta)
        else:
            continue
        
        texts.append(text)
    gpt_scores = chat_gpt_multiple(api, texts)
    return gpt_scores

"""

####

def procesar_scores(gpt_scores, feedbacks_gpt, ids, responses_df):
    answers = []
    for i in range(len(ids)):
        match = re.search(r'\d+', gpt_scores[i][0])
        score = int(match.group())
        
        score = 3 if score >= 8 else (2 if score >= 5 else (1 if score >= 3 else 0))
        
        mask = (responses_df['fullname'] == ids[i][0]) & (responses_df['id_control'] == ids[i][1]) & (responses_df['id_pregunta'] == ids[i][2])
        q = responses_df.loc[mask, 'Pregunta'].tolist()[0]
        a = responses_df.loc[mask, 'Respuesta Estudiante'].tolist()[0]
        real_ev = responses_df.loc[mask, 'Eval'].tolist()[0]
        answers.append([ids[i], q, a, feedbacks_gpt[i][0], score, real_ev])
    return answers

def crear_dataframe(answers):
    nombres_columnas = ['id', 'pregunta','respuesta', 'feedback', 'evalGPT', 'evalReal']
    df = pd.DataFrame(data=answers, columns=nombres_columnas)
    df = df.sort_values(by='evalGPT', ascending=True)
    return df

####
def calcular_metricas(real_array, pred_array):
    # Contar número de valores NaN en cada array
    num_nan_real = np.isnan(real_array).sum()
    num_nan_pred = np.isnan(pred_array).sum()

    # Calcular porcentaje de valores válidos
    total_values = len(real_array)
    valid_values_real = total_values - num_nan_real
    valid_values_pred = total_values - num_nan_pred

    percent_valid_real = (valid_values_real / total_values) * 100
    percent_valid_pred = (valid_values_pred / total_values) * 100

    print(f"Porcentaje de valores válidos en real_array: {percent_valid_real:.2f}%")
    print(f"Porcentaje de valores válidos en pred_array: {percent_valid_pred:.2f}%")

    print("Cantidad de NaN valores en real_array:", num_nan_real)
    print("Cantidad de NaN valores en pred_array:", num_nan_pred)

    mask = ~(np.isnan(real_array) | np.isnan(pred_array))
    real_array = real_array[mask]
    pred_array = pred_array[mask]

    matriz_confusion = confusion_matrix(real_array, pred_array)
    print("Matriz de Confusión:")
    print(matriz_confusion)

    reporte_clasificacion = classification_report(real_array, pred_array, target_names=['Clase 0', 'Clase 1', 'Clase 2', 'Clase 3'])
    print("Reporte de Clasificación:")
    print(reporte_clasificacion)

    mae = mean_absolute_error(real_array, pred_array)
    print("Mean Absolute Error (MAE):", mae)

    mse = mean_squared_error(real_array, pred_array)
    print("Mean Squared Error (MSE):", mse)

    rmse = np.sqrt(mse)
    print("Root Mean Squared Error (RMSE):", rmse)

    r2 = r2_score(real_array, pred_array)
    print("R^2 Score:", r2)

    return mae, mse, rmse, r2, matriz_confusion

####

def crear_histograma(real_array, pred_array):
    bins = np.arange(-0.5, 4, 1)
    width = 0.35

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    plt.bar(bin_centers - width/2, np.histogram(real_array, bins)[0], width=width, label='Real Eval', align='center')
    plt.bar(bin_centers + width/2, np.histogram(pred_array, bins)[0], width=width, label='GPT Eval', align='center')

    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')
    plt.title('Histograma')
    plt.legend()

    plt.show()

####

def guardar_archivos(df, metrics_df , formatted_date_time):
    ruta_datos = 'datos/'
    ruta_temp = 'datos/temp/'
    ruta_evals = 'datos/evals/'
    
    df.to_excel(os.path.join(ruta_datos, 'gpt_evaluations.xlsx'), index=False)
    metrics_df.to_excel(os.path.join(ruta_temp,'metrics.xlsx'), index=False)
     
    
    with zipfile.ZipFile(os.path.join(ruta_evals, f'{formatted_date_time}-prompt_results.zip'), 'w') as mi_zip:
        mi_zip.write(os.path.join(ruta_datos, 'gpt_evaluations.xlsx'))
        mi_zip.write(os.path.join(ruta_datos, 'guidelines.xlsx'))
        mi_zip.write(os.path.join(ruta_datos,'real_evaluations.xlsx'))
        mi_zip.write(os.path.join(ruta_temp, 'template_feedback.txt'))
        mi_zip.write(os.path.join(ruta_temp, 'template_scores.txt'))
        mi_zip.write(os.path.join(ruta_temp,'metrics.xlsx'))

    print("Archivo zip creado con éxito.")
    



def procesar_prompts(text_context, text_feedback_rewiever):
    # Crear archivos de template
    crear_archivo(text_context, os.path.join('datos/temp/', 'template_feedback.txt'))
    crear_archivo(text_feedback_rewiever, os.path.join('datos/temp/', 'template_scores.txt'))
    # Cargar datos de respuestas
    responses_df = crear_archivos(tamano_muestra=100, semilla_aleatoria=42)

    api = OpenAIMultiClient(endpoint="chats", data_template={"model": "gpt-3.5-turbo", "temperature": 0.1, "n": 1, "timeout":10}, concurrency=50, wait_interval=1, max_retries=3, retry_max=10, retry_multiplier=1)

    # Obtener feedbacks con OpenAI
    with open(os.path.join('datos/temp/', 'template_feedback.txt'), 'r', encoding='utf-8') as file:
        template_feedback = file.read()
    feedbacks_gpt, ids = obtener_feedbacks(responses_df, template_feedback, api)

    
    # Obtener scores con OpenAI
    with open(os.path.join('datos/temp/', 'template_scores.txt'), 'r', encoding='utf-8') as file:
        template_scores = file.read()
    gpt_scores = obtener_scores(feedbacks_gpt, template_scores, api)

    # Procesar scores
    answers = procesar_scores(gpt_scores,feedbacks_gpt, ids, responses_df)

    # Crear dataframe
    df = crear_dataframe(answers)

    # Calcular métricas
    new_scores = [sublista[-2] for sublista in answers]
    real_array = np.array(responses_df['Eval'])
    pred_array = np.array(new_scores)
    mae, mse, rmse, r2, matriz_confusion = calcular_metricas(real_array, pred_array)
    metrics_df = pd.DataFrame({
        'Metrica': ['MAE', 'MSE', 'RMSE', 'R2'],
        'Valor': [mae, mse, rmse, r2]
    })
    metrics_df.to_excel(os.path.join('datos/temp/', 'metrics.xlsx'), index=False)

    # Crear histograma
    # crear_histograma(real_array, pred_array)

    prompts = [text_context, text_feedback_rewiever]
    return prompts, metrics_df, df


"""
# crear template_feedback.txt
# crear templates_scores.txt
text_context= ''' '''

text_feedback_rewiever = ''' '''

prompts, metrics_df, df = procesar_prompts(text_context, text_feedback_rewiever)

# Guardar archivos
now = datetime.now()-timedelta(hours=4)
formatted_date_time = now.strftime('%Y%m%d-%H%M')
guardar_archivos(df, metrics_df, formatted_date_time)

print("Fin del programa")
print("Prompts:")
print(prompts)
print("Métricas:")
print(metrics_df)

"""
