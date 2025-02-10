# src/model.py

from transformers import MarianMTModel, MarianTokenizer
import config

def load_model_and_tokenizer():
    """
    Loads the MarianMT model and its tokenizer.
    """
    tokenizer = MarianTokenizer.from_pretrained(config.MODEL_NAME)
    model = MarianMTModel.from_pretrained(config.MODEL_NAME)
    return model, tokenizer
