from datasets import load_dataset
from transformers import MarianMTModel, MarianTokenizer

# Cargar dataset
dataset = load_dataset("Helsinki-NLP/opus_books", "en-es")  # Cambia "en-es" por el par de idiomas que elijas
print(dataset["train"][0])  # Ver una muestra del dataset

# Cargar modelo y tokenizador preentrenados
model_name = "Helsinki-NLP/opus-mt-en-es"  # Modelo para inglés-español
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Traducir una frase de ejemplo
input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt")
translated = model.generate(**inputs)
output_text = tokenizer.decode(translated[0], skip_special_tokens=True)

print(f"Input: {input_text}")
print(f"Translated: {output_text}")