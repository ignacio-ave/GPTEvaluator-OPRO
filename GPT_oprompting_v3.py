"""

███████ ███    ██      ██████  ██████  ███    ██ ███████ ████████ ██████  ██    ██  ██████  ██████ ██  ██████  ███    ██              
██      ████   ██     ██      ██    ██ ████   ██ ██         ██    ██   ██ ██    ██ ██      ██      ██ ██    ██ ████   ██              
█████   ██ ██  ██     ██      ██    ██ ██ ██  ██ ███████    ██    ██████  ██    ██ ██      ██      ██ ██    ██ ██ ██  ██              
██      ██  ██ ██     ██      ██    ██ ██  ██ ██      ██    ██    ██   ██ ██    ██ ██      ██      ██ ██    ██ ██  ██ ██              
███████ ██   ████      ██████  ██████  ██   ████ ███████    ██    ██   ██  ██████   ██████  ██████ ██  ██████  ██   ████     ██ ██ ██ 
                                                                                                                                      
                                                                                                                                      
"""

"""

# OPROMPTING V3 Documentación

## Mini PromptsLos 
# **mini prompts** son componentes específicos y delimitados dentro de un prompt más amplio, diseñados para realizar tareas concretas en la generación o análisis de texto. La utilización de mini prompts permite descomponer tareas complejas en subtareas manejables y especializadas, facilitando así una mayor precisión en la interacción con modelos de lenguaje.

### Definición y Función

Un **mini prompt** es una unidad de instrucción que aborda una faceta concreta del problema o tarea en cuestión. A través de la combinación de múltiples mini prompts, se puede construir un prompt completo que aborde de manera integral un objetivo o consulta más amplia.

## Conceptos Clave

### Mini Prompts

- **Definición:** Son partes pequeñas y específicas de un prompt más grande, diseñadas para realizar tareas concretas en la generación o análisis de texto. Cada mini prompt tiene una función definida y puede combinarse con otros mini prompts para formar un prompt completo. Se almacenan en archivos `.txt` y utilizan placeholders como `{context}`, `{question}`, y `{answer}` para insertar información específica.

### Prompt

- **Definición:** Es una combinación de varios mini prompts que trabajan juntos para crear una instrucción completa. Cada mini prompt en un prompt tiene una función específica y contribuye a la formación de una tarea más amplia o a la respuesta a una consulta.

## Categorías de Mini Prompts

1. **Examples:** Proporciona ejemplos detallados y pasos para realizar una tarea específica.
2. **Context:** Ofrece información adicional relevante para ayudar al modelo a entender o procesar la tarea.
   - Placeholder Obligatorio: `{context}`
3. **Question:** Contiene los datos relacionados con la pregunta planteada al estudiante.
   - Placeholder Obligatorio: `{question}`
4. **Answer:** Presenta los datos derivados de la respuesta del estudiante.
   - Placeholder Obligatorio: `{answer}`
5. **Analysis:** Guía el proceso de análisis del modelo mediante preguntas y directrices sobre el uso del contexto o conocimiento.
6. **Feedback:** Ofrece comentarios sobre el desempeño del estudiante, destacando fortalezas y áreas de mejora.
7. **Score:** Instrucciones para asignar una puntuación basada en la respuesta del estudiante y la retroalimentación generada.






## Pasos a Realizar

1. Seleccionar la categoría para crear el mini prompt.
2. Verificar si están evaluados con el template default.
3. Evaluar los faltantes.
4. Crear el leaderboard de mini prompts en la categoría.
5. Crear el meta prompt.
6. Crear mini prompts utilizando el meta prompt.

## Funciones a Obtener

- Solicitud de respuesta de chat.
- Cargar prompts evaluados.
- Evaluar prompts.
- Seleccionar los mejores mini prompts basados en el score.
- Crear leaderboard.
- Crear meta prompt.
- Crear mini prompts utilizando el meta prompt.
- Validar prompts.


## Funciones de oscar a importar

- `read_miniprompts` : Retorna un diccionario con el contenido de cada miniprompt
- `generate_prompt` : Genera un prompt a partir de un diccionario con el contenido de cada miniprompt
- `get_gpt_dicts` : Convierte la respuesta de GPT en un diccionario
- `get_stats` : Calcula las métricas de evaluación
- `save_results` : Guarda los resultados 
- `evaluate_prompt` : Evalúa un prompt y retorna las estadísticas obtenidas
- `generate_prompts` : Genera los prompts a partir de un objeto prompt_data
- `experiment` :  Evalúa varios prompts a la vez con N repeticiones
- `generate_prompts` : Genera una lista con los prompts a evaluar 

## Funciones a Crear


- `select_category` : Selecciona la categoría de mini prompts a evaluar
- `create_template` : Crea un template default para evaluar mini prompts en la categoria seleccionada. Los otros compontentes son seleccionados de manera default. 
- `extract_data` : Extrae los datos de los prompts evaluados
- `create_leaderboard` : Crea un leaderboard con los mejores mini prompts en la categoría seleccionada
- `create_metaprompt` : Crea un metaprompt a partir de los mini prompts seleccionados
- `create_miniprompts` : Crea los mini prompts utilizando el metaprompt
- `validate_prompts` : Valida los prompts generados


- `save_prompts` : Guarda los prompts generados **(No es necesario)**
- `load_prompts` : Carga los prompts generados **(No es necesario)**

"""




from framework import read_miniprompts, generate_prompt, get_gpt_dicts, get_stats, save_results, evaluate_prompt, generate_prompts, experiment, generate_prompts

from framework import show_dataset_info, load_dataset 
from framework import generate_prompts 

import openai
import getpass 
import re
import os
import json




prompt_data = {
    "examples": "*",
    "context": "knowledge_1.txt",
    "question": "question_1.txt",
    "answer": "answer_1.txt",
    "instructions": {
        "analysis": "analysis_1.txt",
        "feedback": "feedback_1.txt",
        "score": "score_1.txt",
    }
}

prompt_folder = "Experiments/Miniprompts"

prompts = generate_prompts(prompt_data, prompt_folder)

print(f"Prompts: {prompts}")



def create_list_of_folders(dir_path):
    """
    creates a list of folders from the specified directory.
    """
    folders_list = []
    for folder_name in os.listdir(dir_path):
        folder_path = os.path.join(dir_path, folder_name)
        if os.path.isdir(folder_path):
            folders_list.append(folder_path)
    return folders_list

# crearemos una lista de las carpetas y accederemos a results.json de cada carpeta.
dir_path = "Experiments/Results"
folders_list = create_list_of_folders(dir_path)

print("\n \n \n")

print(folders_list)

def read_json_files(dir_path):
    """
    Reads the JSON files from the specified directory.
    """
    json_files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
    data_list = []
    for file in json_files:
        file_path = os.path.join(dir_path, file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            prompts = tuple(data['prompts'])
            metrics = data['metrics']
            data_list.append((prompts, metrics))
    return data_list

# loop para leer todos los jsons de las carpetas

def read_json_from_folders(folders_list):
    """
    Reads the JSON files from the specified folders.
    """
    data_list = []
    for folder_path in folders_list:
        folder_data = read_json_files(folder_path)
        data_list.extend(folder_data)
    return data_list

data_list = read_json_from_folders(folders_list)

print("\n \n \n")

print(data_list)




def create_json_list(dir_path):
    """
    Creates a list with the JSON results from the specified directory.
    """
    json_list = []
    for file_name in os.listdir(dir_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(dir_path, file_name)
            with open(file_path, "r") as file:
                json_data = json.load(file)
                json_list.append(json_data)
    return json_list





print("\n \n \n")





def request_chat_response(model, metaprompt):
    """
    Requests a chat response from the OpenAI Chat API.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": metaprompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print("\u2717 Error requesting chat response:", e)
        return None



def create_leaderboard(data_list, n=5):
    """
    Creates a leaderboard of the top n prompts based on evaluation metrics.
    """
    sorted_data = sorted(data_list, key=lambda x: x[1]['score'], reverse=True)
    leaderboard = sorted_data[:n]
    return leaderboard




"""
### 

# COMO CREAR Y VISUALIZAR PROMPT.
prompt_data = {
    "examples": "examples_3.txt",
    "context": "context_1.txt",
    "question": "question_1.txt",
    "answer": "answer_1.txt",
    "instructions": {
        "analysis": "analysis_1.txt",
        "feedback": "feedback_1.txt",
        "score": "score_1.txt",
    }
}

column_data = {
    "context": "Contexto",
    "question": "Pregunta",
    "answer": "Respuesta",
    "real_eval": "EvalProfe"
}

criteria = ["correctness", "completeness", "clarity"]

'''
experiment(
    dataset="test.xlsx",
    sheet_name="C2-claim",
    column_data=column_data,
    prompt_data=prompt_data,
    criteria=criteria,
    repetitions=1,
    eval_function="map"
)
'''

visualize_prompt(prompt_data, criteria)


show_results("Experiments/Results/20240723-1518")



###
"""