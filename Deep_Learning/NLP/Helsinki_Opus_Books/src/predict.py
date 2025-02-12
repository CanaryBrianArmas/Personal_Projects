# src/predict.py

import torch
from transformers import MarianTokenizer, MarianMTModel

# Cargar el modelo entrenado
model_path = "./final_model"  # Asegúrate de que esta ruta es correcta
model = MarianMTModel.from_pretrained(model_path)
tokenizer = MarianTokenizer.from_pretrained(model_path)

# Función para traducir un texto
def translate(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(**inputs)

    translated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return translated_text

if __name__ == "__main__":
    # Solicitar input del usuario
    text = input("Introduce el texto a traducir: ")
    
    # Traducir y mostrar resultado
    translation = translate(text)
    print("Traducción:", translation)
