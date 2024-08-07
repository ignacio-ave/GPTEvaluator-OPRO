#%% 
from GPT_validator import *
from GPT_validator import procesar_prompts, guardar_archivos

from utils import guardar_json, read_json_files, extract_info
import openai # type: ignore
import json
import os
from datetime import datetime, timedelta
import pandas as df
import getpass

# Función para generar metaprompt usando openai
def solicitar_respuesta_chat(modelo, metaprompt):
    """
    Envía una solicitud de chat a la API de OpenAI y devuelve la respuesta.

    :param modelo: El modelo de OpenAI a utilizar.
    :param metaprompt: El texto del prompt para el chat.
    :return: Respuesta del modelo de chat.
    """
    try:
        respuesta_chat = openai.ChatCompletion.create(
            model=modelo,
            messages=[{"role": "user", "content": metaprompt}]
        )
        return respuesta_chat.choices[0].message.content
    except Exception as e:
        print(" \u2717 Error al solicitar la respuesta del chat:", e)
        return None


def crear_prompt_scores(data_list):
    """ 
    Crea un string con los prompts y las métricas de evaluación de los datos de la lista.
    """
    prompts_scores = ""
    for item in data_list:
        prompt1, prompt2, metrics = extract_info(item)
        prompts_scores += f"Prompt 1:( {prompt1} )\nPrompt 2:( {prompt2} )\n"
        prompts_scores += "########\nMetrics:"
        for metric in metrics:
            prompts_scores += f"\n{metric['name']}: {metric['value']}"
        prompts_scores += "\n########"
    return prompts_scores



def crear_metaprompt(prompts_scores):
    """
    Crea un metaprompt a partir de un template y los prompts_scores.
    """
    try:
        with open("metaprompt_template.txt", "r") as file:
            metaprompt = file.read()
            metaprompt = metaprompt.format(prompts_scores=prompts_scores)
            return metaprompt
    
    except Exception as e:
        print("Error al leer el archivo o formatear metaprompt", e)
        return None    
    
def es_primer_prompt_valido(texto):
    """
    Verifica si el texto contiene los marcadores {Pregunta} y {RespuestaA}.
    """
    return re.search(r'\{Pregunta\}', texto) and re.search(r'\{RespuestaA\}', texto)


def es_segundo_prompt_valido(texto):
    """ 
    Verifica si el texto contiene el marcador {Feedback}.
    """
    return re.search(r'\{Feedback\}', texto)



def extraer_texto_entre_parentesis(text):
    """
    Utiliza una pila para manejar de manera adecuada la extracción de texto
    entre paréntesis, incluyendo el manejo correcto de paréntesis anidados.
    """
    
    stack = []  # Pila para manejar los índices de los paréntesis abiertos
    matches = []  # Lista para almacenar los textos extraídos
    result = []  # Almacena los pares de índices de paréntesis abiertos y cerrados

    # Itera sobre el texto para identificar paréntesis abiertos y cerrados
    for i, char in enumerate(text):
        if char == '(':
            stack.append(i)  
        elif char == ')' and stack:
            start_index = stack.pop()  
            result.append((start_index, i))  

    result.sort(key=lambda x: x[0])

    for start, end in result:
        matches.append(text[start+1:end])

    filtered_matches = list(dict.fromkeys([m for m in matches if m]))

    return filtered_matches


def extraer_prompts_validos(text):
    """
    Extrae los prompts válidos de un texto que contiene múltiples prompts.
    """
    if es_primer_prompt_valido(text) and es_segundo_prompt_valido(text):
        matches = extraer_texto_entre_parentesis(text)
        
        primer_prompt_validos = sorted([match for match in matches if es_primer_prompt_valido(match)], key=len)
        segundo_prompt_validos = sorted([match for match in matches if es_segundo_prompt_valido(match)], key=len)
        
        if primer_prompt_validos and segundo_prompt_validos:
            return primer_prompt_validos[0].strip(), segundo_prompt_validos[0].strip()
        
    return False, False

def cargar_data_prompts_evaluados(dir_path = "datos/evals/jsons"):
    """ 
    Carga los datos de los prompts evaluados en una lista.
    
    
    
    La lista principal contiene n tuplas.
    Cada tupla contiene dos elementos:
        El primer elemento es un string que representa los prompts evaluados.
        El segundo elemento es una lista de diccionarios que representan métricas de evaluación. Cada diccionario tiene dos claves: name y value, que representan el nombre de la métrica y su valor respectivamente.

    """
    data_list = read_json_files(dir_path)
    return data_list


def crear_prompts_con_metaprompt(data_list, metapromptalt=None, iteraciones_maximas=15):
    """ 
    Crea prompts a partir de los datos de la lista y un metaprompt.
    El metaprompt puede ser proporcionado como argumento o generado automáticamente a partir del template default.
    """
    
    prompts_scores = crear_prompt_scores(data_list)
    
    if metapromptalt is not None:
        metaprompt = metapromptalt
        metaprompt = metaprompt.format(prompts_scores=prompts_scores)
     
    else: 
        metaprompt = crear_metaprompt(prompts_scores)
    
    response = solicitar_respuesta_chat("gpt-3.5-turbo", metaprompt)
    prompt1, prompt2 = extraer_prompts_validos(response)

    if prompt1 == False or prompt2 == False:
        for i in range(iteraciones_maximas):
            response = solicitar_respuesta_chat("gpt-3.5-turbo", metaprompt)
            prompt1, prompt2 = extraer_prompts_validos(response)
            if prompt1 != False and prompt2 != False:
                return prompt1, prompt2
        
        return None, None
            
    return prompt1, prompt2


def seleccionar_mejores_r2(datos, num_seleccion=3):
    """
    Selecciona los mejores datos según R2
    """
    return sorted(datos, key=lambda x: x[1][3]['value'], reverse=True)[:num_seleccion]

def seleccionar_mejores_mse(datos, num_seleccion=3):
    """
    Selecciona los mejores datos según MSE
    """
    return sorted(datos, key=lambda x: x[1][1]['value'])[:num_seleccion]




























""" USO EN LOOP


openai.api_key = getpass.getpass("api-key: ")

n_prompts = input("¿Cuántos prompts desea crear? ")


for i in range(int(n_prompts)):
    dir_path = "datos/evals/jsons"
    data_list = cargar_data_prompts_evaluados(dir_path)
    data_list_filtered = seleccionar_mejores_r2(data_list, num_seleccion=3)  # o seleccionar_mejores_mse(data_list, num_seleccion=3)
    prompt1, prompt2 = crear_prompts_con_metaprompt(data_list_filtered, iteraciones_maximas=15)

    print("Prompts:")
    print(prompt1)
    print(prompt2)

    if prompt1 is None or prompt2 is None:
        print("No se pudo obtener los prompts")
        print("Fin del programa")
        exit()
        
    prompts, metrics_df, df = procesar_prompts(prompt1, prompt2)


    # Guardar archivos
    now = datetime.now()-timedelta(hours=4)
    formatted_date_time = now.strftime('%Y%m%d-%H%M')
    # Guardar en .zip
    guardar_archivos(df, metrics_df, formatted_date_time)
    # Guardar en json (solo información necesaria para construir el meta-prompt)
    guardar_json(metrics_df, prompts, formatted_date_time)

    print("\n\n\n")

print("Fin del programa")

    
dir_path = "datos/evals/jsons"
data_list = cargar_data_prompts_evaluados(dir_path)
data_list_filtered = seleccionar_mejores_r2(data_list, num_seleccion=3)  
print(data_list_filtered)

"""

""" USO SIMPLE


openai.api_key = getpass.getpass("api-key: ")


dir_path = "datos/evals/jsons"
data_list = cargar_data_prompts_evaluados(dir_path)
prompt1, prompt2 = crear_prompts_con_metaprompt(data_list, iteraciones_maximas=15)

print("Prompts:")
print(prompt1)
print(prompt2)

if prompt1 is None or prompt2 is None:
    print("No se pudo obtener los prompts")
    print("Fin del programa")
    exit()
    
prompts, metrics_df, df = procesar_prompts(prompt1, prompt2)

# Guardar archivos
now = datetime.now()-timedelta(hours=4)
formatted_date_time = now.strftime('%Y%m%d-%H%M')
# Guardar en .zip
guardar_archivos(df, metrics_df, formatted_date_time)
# Guardar en json (solo información necesaria para construir el meta-prompt)
guardar_json(metrics_df, prompts, formatted_date_time)

print("\n\n\n")
print("Fin del programa")
"""

