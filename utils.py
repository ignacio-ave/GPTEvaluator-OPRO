import os 
import json
import pandas as pd


def save_metrics(metrics_df):
    metrics = []
    for index, row in metrics_df.iterrows():
        metrics.append({"name": row["Metrica"], "value": row["Valor"]})
    return metrics

def save_prompts(prompts):
    return prompts

def guardar_json(metrics_df, prompts, formatted_data_time):
    data = {
        "description": "Metrics and prompts for model evaluation",
        "metrics": save_metrics(metrics_df),
        "prompts": save_prompts(prompts)
    }
    dir_path = "datos/evals/jsons"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_path = os.path.join(dir_path, f"{formatted_data_time}.json")
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
        
def read_json_files(dir_path):
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

def extract_info(item):
    prompts, metrics = item
    prompt1, prompt2 = prompts
    return prompt1, prompt2, metrics
 